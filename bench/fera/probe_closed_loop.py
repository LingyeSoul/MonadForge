#!/usr/bin/env python
"""Closed-loop FeRA router probe.

Companion to ``probe_fei.py``. The original probe runs the *base* DiT,
captures ``z_t``, and computes 3-band FEI offline. This script instead:

  - loads a trained FeRA checkpoint onto the DiT (in-place Linear →
    FeRALinear),
  - runs the same CFG-batched inference trajectory with FeRA *live*,
  - calls ``set_fera_zt(anima, latents)`` before every model forward, and
  - reads ``network._last_fei`` and ``network._last_gates`` after each
    prepare_forward — i.e. the FEI and per-expert gate weights the actual
    deployed router produced.

Question answered: on the production inference distribution (CFG=4,
28-step, default-prompts), does the trained router actually use all
experts, and does it differentiate by prompt content?

Default config matches `make test` (1024×1024, CFG=4, 28 steps) and the
DEFAULT_PROMPTS set from ``probe_fei.py``. Single seed by default — the
question is "are experts dead across prompts," not "how seed-stable are
gates."

Usage
-----
    uv run python bench/fera/probe_closed_loop.py \\
        --adapter output/ckpt/anima_lora-000005.safetensors \\
        --label fera-closed-loop
"""

from __future__ import annotations

import argparse
import csv
import gc
import logging
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bench._common import make_run_dir, write_result  # noqa: E402
from library.anima import weights as anima_utils  # noqa: E402
from library.inference import sampling as inference_utils  # noqa: E402
from library.inference.adapters import (  # noqa: E402
    clear_fera_zt,
    clear_hydra_sigma,
    set_fera_zt,
    set_hydra_sigma,
)

# Reuse probe_fei.py's shared helpers — same band convention.
from bench.fera.probe_fei import (  # noqa: E402
    DEFAULT_PROMPTS,
    _encode_prompt,
    _setup_text_strategies,
    _TextEncoderArgs,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fera-closed")


@torch.no_grad()
def run_trajectory_closed_loop(
    anima,
    network,
    embed: torch.Tensor,
    embed_uncond: torch.Tensor | None,
    sigmas: torch.Tensor,
    h_lat: int,
    w_lat: int,
    seed: int,
    device: torch.device,
    *,
    cfg_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one denoising trajectory with FeRA live.

    Returns ``(fei[n_steps, num_bands], gates[n_steps, num_experts])`` —
    the actual router outputs from each step's ``prepare_forward``.

    FeRA's ``_last_fei`` band order is ``[high, mid, low]`` (see
    ``networks/methods/fera.py:128``). We pass that ordering through
    unchanged.
    """
    n_steps = sigmas.numel() - 1
    latent_c = 16
    g = torch.Generator(device="cpu").manual_seed(seed)
    latents = torch.randn(
        (1, latent_c, 1, h_lat, w_lat), generator=g, dtype=torch.bfloat16
    ).to(device)
    pad = torch.zeros(1, 1, h_lat, w_lat, dtype=torch.bfloat16, device=device)
    timesteps = sigmas[:-1].to(device, dtype=torch.bfloat16)
    sigmas_d = sigmas.to(device)

    fei_traj = np.zeros((n_steps, network.num_bands), dtype=np.float32)
    gate_traj = np.zeros((n_steps, network.num_experts), dtype=np.float32)

    try:
        for i in range(n_steps):
            # Route on z_t (BEFORE the forward) — exactly how generation.py uses it.
            set_fera_zt(anima, latents)
            fei = network._last_fei  # (1, num_bands)
            gates = network._last_gates  # (1, num_experts)
            fei_traj[i] = fei[0].float().cpu().numpy()
            gate_traj[i] = gates[0].float().cpu().numpy()

            t = timesteps[i].expand(latents.shape[0])
            set_hydra_sigma(anima, t)  # harmless no-op if no Hydra attached

            if cfg_scale == 1.0 or embed_uncond is None:
                v = anima(latents, t, embed, padding_mask=pad)
            else:
                B = latents.shape[0]
                e_u = (
                    embed_uncond
                    if embed_uncond.shape[0] == B
                    else embed_uncond.expand(B, -1, -1).contiguous()
                )
                x2 = torch.cat([latents, latents], dim=0)
                t2 = torch.cat([t, t], dim=0)
                e2 = torch.cat([e_u, embed], dim=0)
                p2 = torch.cat([pad, pad], dim=0)
                v_pair = anima(x2, t2, e2, padding_mask=p2)
                v_u, v_c = v_pair[:B], v_pair[B:]
                v = v_u + cfg_scale * (v_c - v_u)

            latents = inference_utils.step(latents, v, sigmas_d, i).to(latents.dtype)
    finally:
        clear_hydra_sigma(anima)
        clear_fera_zt(anima)

    return fei_traj, gate_traj


def _parse_buckets(s: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        w, h = tok.lower().split("x")
        out.append((int(w), int(h)))
    return out


def _attach_fera(model, adapter_path: str, device: torch.device):
    """Load FeRA checkpoint as a router-live adapter on ``model``.

    Mirrors ``library/inference/models.py::load_dit_model``'s FeRA branch
    — we replicate it here so we don't have to construct the full
    inference argparse to call into the loader.
    """
    from networks.methods import fera as fera_module

    log.info(f"loading FeRA adapter {adapter_path}")
    # Use the distilled lora_down/lora_up path (fera_ortho=False, default).
    # Ortho-mode Cayley solve uses torch.linalg.solve which has no bf16 CUDA
    # kernel. The checkpoint stores distilled split q/k/v keys for the
    # fused qkv/kv layers; FeRANetwork.load_state_dict re-fuses on load.
    network, weights_sd = fera_module.create_network_from_weights(
        multiplier=1.0,
        file=adapter_path,
        ae=None,
        text_encoders=[],
        unet=model,
        weights_sd=None,
        for_inference=True,
    )
    network.apply_to([], model, apply_text_encoder=False, apply_unet=True)
    info = network.load_state_dict(weights_sd, strict=False)
    if info.unexpected_keys:
        log.warning(f"FeRA: unexpected keys: {info.unexpected_keys[:5]}...")
    if info.missing_keys:
        log.warning(f"FeRA: missing keys: {info.missing_keys[:5]}...")
    network.to(device, dtype=torch.bfloat16)
    # Router runs in fp32 (matches training-side ``e_t.float()`` invariant);
    # FeRALinear forward casts gate weights back to caller dtype.
    network.router.to(torch.float32)
    network.eval().requires_grad_(False)
    model._fera_network = network
    model._fera_networks = [network]
    log.info(
        f"FeRA attached: {len(network.fera_layers)} modules, "
        f"{network.num_experts} experts × rank {network.rank}, "
        f"num_bands={network.num_bands}"
    )
    return network


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dit", default="models/diffusion_models/anima-preview3-base.safetensors")
    p.add_argument("--text_encoder", default="models/text_encoders/qwen_3_06b_base.safetensors")
    p.add_argument(
        "--adapter",
        default="output/ckpt/anima_lora-000005.safetensors",
        help="FeRA checkpoint to load (router-live).",
    )
    p.add_argument(
        "--buckets",
        default="1024x1024",
        help="Comma-sep WxH list (pixel). Default: 1024x1024 only (matches make test).",
    )
    p.add_argument("--prompts", nargs="*", default=None)
    p.add_argument("--n_prompts", type=int, default=4)
    p.add_argument("--n_seeds", type=int, default=1)
    p.add_argument("--seed_base", type=int, default=1234)
    p.add_argument("--infer_steps", type=int, default=28)
    p.add_argument("--flow_shift", type=float, default=1.0)
    p.add_argument("--guidance_scale", type=float, default=4.0)
    p.add_argument(
        "--negative_prompt",
        default="worst quality, low quality, score_1, score_2, score_3",
    )
    p.add_argument("--attn_mode", default="flash")
    p.add_argument("--label", default=None)
    args = p.parse_args()

    buckets = _parse_buckets(args.buckets)
    prompts = (args.prompts if args.prompts else DEFAULT_PROMPTS)[: args.n_prompts]
    if not prompts:
        raise SystemExit("no prompts after --n_prompts truncation")

    out_dir = make_run_dir("fera", label=args.label)
    log.info(f"output → {out_dir}")
    log.info(
        f"adapter={args.adapter}, buckets={buckets}, prompts={len(prompts)}, "
        f"seeds={args.n_seeds}, steps={args.infer_steps}, CFG={args.guidance_scale}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    log.info("loading DiT…")
    anima = anima_utils.load_anima_model(
        device=device,
        dit_path=args.dit,
        attn_mode=args.attn_mode,
        split_attn=False,
        loading_device=device,
        dit_weight_dtype=dtype,
    )
    anima.to(device, dtype=dtype).eval().requires_grad_(False)
    anima.reset_mod_guidance()

    network = _attach_fera(anima, args.adapter, device)

    _setup_text_strategies(args.text_encoder)
    from library.inference.models import load_text_encoder

    log.info("loading text encoder (transient)…")
    te = load_text_encoder(_TextEncoderArgs(args.text_encoder), dtype=dtype, device=device)
    te.eval()

    embeds_cond = [_encode_prompt(anima, te, p, device) for p in prompts]
    embed_uncond: torch.Tensor | None = None
    if args.guidance_scale != 1.0:
        embed_uncond = _encode_prompt(anima, te, args.negative_prompt, device)

    del te
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("text encoder released")

    _, sigmas = inference_utils.get_timesteps_sigmas(
        args.infer_steps, args.flow_shift, device
    )
    sigmas = sigmas.cpu()
    n_steps = int(sigmas.numel() - 1)
    num_bands = network.num_bands
    num_experts = network.num_experts

    rows: list[dict] = []
    per_bucket_fei: dict = {b: [] for b in buckets}
    per_bucket_gates: dict = {b: [] for b in buckets}

    for (w_pix, h_pix) in buckets:
        h_lat, w_lat = h_pix // 8, w_pix // 8
        log.info(f"bucket {w_pix}x{h_pix} (latent {w_lat}x{h_lat})")

        for pi, embed in enumerate(embeds_cond):
            for si in range(args.n_seeds):
                seed = args.seed_base + 1000 * pi + si
                fei_traj, gate_traj = run_trajectory_closed_loop(
                    anima,
                    network,
                    embed,
                    embed_uncond,
                    sigmas,
                    h_lat,
                    w_lat,
                    seed,
                    device,
                    cfg_scale=args.guidance_scale,
                )
                per_bucket_fei[(w_pix, h_pix)].append(fei_traj)
                per_bucket_gates[(w_pix, h_pix)].append(gate_traj)
                for i in range(n_steps):
                    row = {
                        "bucket_w": w_pix,
                        "bucket_h": h_pix,
                        "prompt_idx": pi,
                        "seed": seed,
                        "step": i,
                        "sigma": float(sigmas[i]),
                    }
                    # FEI band order from FeRANetwork: [high, mid, low] for num_bands=3.
                    band_names = (
                        ["high", "mid", "low"]
                        if num_bands == 3
                        else ([f"band{k}" for k in range(num_bands)])
                    )
                    for k, name in enumerate(band_names):
                        row[f"fei_{name}"] = float(fei_traj[i, k])
                    for k in range(num_experts):
                        row[f"gate_{k}"] = float(gate_traj[i, k])
                    rows.append(row)

    csv_path = out_dir / "fei_gates_per_step.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"wrote {csv_path} ({len(rows)} rows)")

    # Aggregate diagnostics — argmax counts + between/within-prompt dispersion.
    log.info("=== aggregate diagnostics ===")
    diag: dict = {"per_bucket": {}}
    for (w_pix, h_pix) in buckets:
        G = np.stack(per_bucket_gates[(w_pix, h_pix)], axis=0)
        # G: [n_prompts * n_seeds, n_steps, num_experts]
        flat = G.reshape(-1, num_experts)
        am = np.argmax(flat, axis=-1)
        counts = np.bincount(am, minlength=num_experts).tolist()
        log.info(
            f"bucket {w_pix}x{h_pix}  argmax counts {counts}  "
            f"(N={flat.shape[0]} samples)"
        )
        diag["per_bucket"][f"{w_pix}x{h_pix}"] = {
            "argmax_counts": counts,
            "mean_gate": flat.mean(0).tolist(),
            "std_gate": flat.std(0).tolist(),
        }

        # Reshape: [n_prompts, n_seeds, n_steps, num_experts]
        Gp = G.reshape(len(prompts), args.n_seeds, n_steps, num_experts)
        # avg over seeds first, then std-across-prompts vs std-across-steps
        Gp_mean_seed = Gp.mean(axis=1)  # [n_prompts, n_steps, num_experts]
        between = Gp_mean_seed.std(axis=0).mean(axis=0)  # [num_experts]
        within = Gp_mean_seed.std(axis=1).mean(axis=0)  # [num_experts]
        log.info(f"  between-prompt std: {between.tolist()}")
        log.info(f"  within-prompt  std: {within.tolist()}")
        diag["per_bucket"][f"{w_pix}x{h_pix}"]["between_prompt_std"] = between.tolist()
        diag["per_bucket"][f"{w_pix}x{h_pix}"]["within_prompt_std"] = within.tolist()

        # Per-prompt argmax counts
        per_prompt_argmax = []
        for pi in range(len(prompts)):
            am_p = np.argmax(Gp_mean_seed[pi], axis=-1)
            per_prompt_argmax.append(np.bincount(am_p, minlength=num_experts).tolist())
        diag["per_bucket"][f"{w_pix}x{h_pix}"]["per_prompt_argmax_counts"] = per_prompt_argmax
        for pi, c in enumerate(per_prompt_argmax):
            log.info(f"  prompt {pi} argmax counts over {n_steps} steps: {c}")

    metrics = {
        "adapter": args.adapter,
        "n_buckets": len(buckets),
        "n_prompts": len(prompts),
        "n_seeds": args.n_seeds,
        "n_steps": n_steps,
        "num_bands": num_bands,
        "num_experts": num_experts,
        **diag,
    }
    write_result(
        out_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        artifacts=[csv_path.name],
        label=args.label,
        device=device,
    )
    log.info("done")


if __name__ == "__main__":
    main()
