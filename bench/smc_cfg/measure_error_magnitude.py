#!/usr/bin/env python3
"""Per-step measurement of the CFG semantic-error magnitude ``|e| = |v_cond − v_uncond|``.

Standalone diagnostic for SMC-CFG sanity (Wang et al., arXiv:2603.03281).
SMC's switching term injects ``Δe = −k · φ(s/ε)`` into the cond/uncond combine
with ``|Δe| ≤ k`` per element. If ``k`` is comparable to ``|e|`` at our
production CFG (4), the controller dominates the actual semantic correction
rather than refining it — i.e. SMC at low guidance behaves more like noise
injection than feedback control. The paper sells SMC at high CFG (≥7),
which we suspect is the wrong regime fit for us.

This bench does *not* run SMC. It runs a normal CFG reverse trajectory and
captures ``|e|`` per step, then computes the SMC quantities offline:

    e_t          = v_cond − v_uncond                  (measured)
    s_t          = (e_t − e_{t-1}) + λ · e_{t-1}      (derived, λ default 5.0)
    |Δe|         = k                                  (per-element bound)
    Δe/e ratio   = k / |e|.mean()                     (per k, per step)

Sweeps CFG ∈ {2, 4, 6, 8} by default so the magnitude→CFG curve is visible.

Reuses the cache layout + helpers from ``scripts.dcw`` (cached x_0 latents +
crossattn embeds under ``post_image_dataset/lora``), so no VAE / T5 is loaded.

Output (bench/smc_cfg/results/<YYYYMMDD-HHMM>[-<label>]/)
---------------------------------------------------------
    per_step.csv    long-form: cfg, prompt_idx, step, sigma, e_mean, e_p50,
                    e_p95, s_mean, k_over_e_{k}, dE_over_e_{k}
    result.json     standard bench envelope + summary stats
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from bench._common import make_run_dir, write_result  # noqa: E402
from library.anima import weights as anima_utils  # noqa: E402
from library.inference import sampling as inference_utils  # noqa: E402
from library.inference.adapters import (  # noqa: E402
    clear_hydra_fei,
    clear_hydra_sigma,
    compute_and_set_hydra_fei,
    set_hydra_sigma,
)
from scripts.dcw.adapters import attach_loras  # noqa: E402
from scripts.dcw.cache import load_cached, pick_cached_samples  # noqa: E402
from scripts.dcw.trajectory import encode_uncond_embed  # noqa: E402

log = logging.getLogger("smc-cfg-bench")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _latest_lora() -> Path | None:
    ckpt = REPO_ROOT / "output" / "ckpt"
    if not ckpt.is_dir():
        return None
    cands = [
        p for p in ckpt.glob("*.safetensors")
        if "_moe" not in p.name
        and "fusion_head" not in p.name
        and "pooled_text_proj" not in p.name  # mod-guidance head, not a LoRA
    ]
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dit", type=str,
                   default="models/diffusion_models/anima-base-v1.0.safetensors")
    p.add_argument("--text_encoder", type=str,
                   default="models/text_encoders/qwen_3_06b_base.safetensors")
    p.add_argument("--attn_mode", type=str, default="flash")
    p.add_argument("--lora_weight", type=str, default=None,
                   help="Default: latest under output/ckpt/. Pass empty to disable.")
    p.add_argument("--lora_multiplier", type=float, default=1.0)
    p.add_argument("--dataset_dir", type=str, default="post_image_dataset/lora")
    p.add_argument("--text_variant", type=int, default=2,
                   help="crossattn_emb_v{N} key in cached _anima_te.safetensors")
    p.add_argument("--image_h", type=int, default=1024)
    p.add_argument("--image_w", type=int, default=1024)
    p.add_argument("--n_prompts", type=int, default=10)
    p.add_argument("--shuffle_seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0,
                   help="Single noise seed shared across all (prompt, cfg) cells.")
    p.add_argument("--negative_prompt", type=str, default="")
    p.add_argument("--cfgs", nargs="+", type=float,
                   default=[4.0])
    p.add_argument("--infer_steps", type=int, default=28)
    p.add_argument("--flow_shift", type=float, default=1.0)
    p.add_argument("--lam", type=float, default=5.0,
                   help="SMC λ used to derive |s| offline (paper-best default).")
    p.add_argument("--ks", nargs="+", type=float, default=[0.02, 0.1],
                   help="k values for SMC ratio reporting. 0.02 = our default, "
                   "0.1 = paper-best.")
    p.add_argument("--label", type=str, default=None)
    return p.parse_args()


@torch.no_grad()
def _measure_one_trajectory(
    anima,
    *,
    embed: torch.Tensor,        # (1, 512, 1024) bf16
    embed_uncond: torch.Tensor,  # (1, 512, 1024) bf16
    h_latent: int,
    w_latent: int,
    seed: int,
    sigmas: torch.Tensor,
    cfg_scale: float,
    lam: float,
    device: torch.device,
) -> list[dict[str, float]]:
    """Run a full reverse-denoise loop, return per-step |e| stats."""
    n_steps = len(sigmas) - 1
    pad = torch.zeros(1, 1, h_latent, w_latent, dtype=torch.bfloat16, device=device)
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(
        (1, 16, 1, h_latent, w_latent), generator=g, dtype=torch.float32
    ).to(device, dtype=torch.bfloat16)

    rows: list[dict[str, float]] = []
    e_prev: torch.Tensor | None = None  # element-wise prior step's e (fp32)
    for i in range(n_steps):
        sigma_i = float(sigmas[i])
        t = torch.full((1,), sigma_i, device=device, dtype=torch.bfloat16)
        set_hydra_sigma(anima, t)
        compute_and_set_hydra_fei(anima, x)

        v_cond = anima(x, t, embed, padding_mask=pad)
        v_uncond = anima(x, t, embed_uncond, padding_mask=pad)
        e = (v_cond - v_uncond).float()

        e_abs = e.abs()
        e_flat = e_abs.flatten()
        e_mean = float(e_abs.mean())
        e_p50 = float(e_flat.quantile(0.5))
        e_p95 = float(e_flat.quantile(0.95))

        # Sliding surface, element-wise per the paper:
        #   s_t = (e_t − e_{t-1}) + λ · e_{t-1}
        # Bootstrap (i=0) matches `if e(t+1) is None then e(t+1) ← e(t)`:
        #   s_0 = 0 + λ · e_0 = λ · e_0.
        if e_prev is None:
            s = lam * e
        else:
            s = (e - e_prev) + lam * e_prev
        s_abs = s.abs()
        s_mean = float(s_abs.mean())
        s_p50 = float(s_abs.flatten().quantile(0.5))

        rows.append({
            "step": i,
            "sigma": sigma_i,
            "e_mean": e_mean,
            "e_p50": e_p50,
            "e_p95": e_p95,
            "s_mean": s_mean,
            "s_p50": s_p50,
        })
        e_prev = e

        # Standard CFG combine — drives the trajectory forward so |e| at the
        # next step is measured on a realistic x_{i+1}, not a forward-noised
        # sample. Matches library/inference/generation.py.
        v = v_uncond + cfg_scale * (v_cond - v_uncond)
        x = inference_utils.step(x, v, sigmas, i).to(x.dtype)

    return rows


def main() -> None:
    args = _parse_args()
    if args.lora_weight is None:
        lora = _latest_lora()
        if lora is None:
            log.warning("no LoRA under output/ckpt/ and --lora_weight unset; "
                        "running on base DiT only")
    elif args.lora_weight == "":
        lora = None
    else:
        lora = Path(args.lora_weight)
        if not lora.exists():
            raise SystemExit(f"--lora_weight not found: {lora}")

    out_dir = make_run_dir("smc_cfg", label=args.label)
    log.info(f"output → {out_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    log.info(f"loading DiT: {args.dit}")
    anima = anima_utils.load_anima_model(
        device=device,
        dit_path=args.dit,
        attn_mode=args.attn_mode,
        split_attn=False,
        loading_device=device,
        dit_weight_dtype=dtype,
    )
    anima.to(device, dtype=dtype)
    anima.reset_mod_guidance()
    anima.eval().requires_grad_(False)

    # Prime text strategies + load TE once to encode the uncond crossattn,
    # then free. Mirrors scripts/dcw/measure_bias.py:174-227.
    from library.anima import strategy as strategy_anima, text_strategies
    from library.inference.models import load_text_encoder
    from library.inference.text import MAX_CROSSATTN_TOKENS

    text_strategies.TokenizeStrategy.set_strategy(
        strategy_anima.AnimaTokenizeStrategy(
            qwen3_path=args.text_encoder,
            t5_tokenizer_path=None,
            qwen3_max_length=MAX_CROSSATTN_TOKENS,
            t5_max_length=MAX_CROSSATTN_TOKENS,
        )
    )
    text_strategies.TextEncodingStrategy.set_strategy(
        strategy_anima.AnimaTextEncodingStrategy()
    )

    log.info(f"encoding uncond (negative_prompt={args.negative_prompt!r})…")
    text_encoder = load_text_encoder(args, dtype=torch.bfloat16, device=device)
    text_encoder.eval()
    embed_uncond = encode_uncond_embed(anima, text_encoder, args.negative_prompt, device)
    del text_encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if lora is not None:
        attach_loras(anima, [str(lora)], [args.lora_multiplier], device, dtype)

    samples = pick_cached_samples(
        Path(args.dataset_dir),
        args.n_prompts,
        image_h=args.image_h,
        image_w=args.image_w,
        shuffle_seed=args.shuffle_seed,
    )
    if not samples:
        raise SystemExit(
            f"no cached samples matching {args.image_h}x{args.image_w} under "
            f"{args.dataset_dir}. Expected *_anima.npz + *_anima_te.safetensors "
            "pairs (make preprocess)."
        )
    log.info(f"sampled {len(samples)} prompts: {[s[0] for s in samples]}")

    _, sigmas = inference_utils.get_timesteps_sigmas(
        args.infer_steps, args.flow_shift, device
    )
    sigmas = sigmas.cpu()

    h_latent = args.image_h // 8
    w_latent = args.image_w // 8

    rows: list[dict] = []
    try:
        for cfg in args.cfgs:
            for p_idx, (stem, npz_path, te_path) in enumerate(samples):
                _, embed = load_cached(npz_path, te_path, args.text_variant, device)
                # embed is (1, 512, 1024) already; pad if needed (load_cached
                # leaves it at the cached length — typically 512).
                if embed.shape[1] < MAX_CROSSATTN_TOKENS:
                    embed = torch.nn.functional.pad(
                        embed, (0, 0, 0, MAX_CROSSATTN_TOKENS - embed.shape[1])
                    )
                log.info(f"cfg={cfg:.1f} prompt[{p_idx}]={stem}")
                step_rows = _measure_one_trajectory(
                    anima,
                    embed=embed,
                    embed_uncond=embed_uncond,
                    h_latent=h_latent,
                    w_latent=w_latent,
                    seed=args.seed + p_idx,
                    sigmas=sigmas,
                    cfg_scale=cfg,
                    lam=args.lam,
                    device=device,
                )
                for r in step_rows:
                    for k in args.ks:
                        r[f"k_over_e_{k}"] = float(k / max(r["e_mean"], 1e-12))
                    r["cfg"] = float(cfg)
                    r["prompt_idx"] = int(p_idx)
                    r["stem"] = stem
                    rows.append(r)
    finally:
        clear_hydra_sigma(anima)
        clear_hydra_fei(anima)

    # Write per-step CSV.
    fieldnames = (
        ["cfg", "prompt_idx", "stem", "step", "sigma",
         "e_mean", "e_p50", "e_p95", "s_mean", "s_p50"]
        + [f"k_over_e_{k}" for k in args.ks]
    )
    csv_path = out_dir / "per_step.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # Aggregate per CFG: mean / median / quartile |e| across (prompts × steps),
    # plus k/|e| at the configured k values.
    summary: dict[str, dict] = {}
    for cfg in args.cfgs:
        cfg_rows = [r for r in rows if r["cfg"] == cfg]
        es = np.array([r["e_mean"] for r in cfg_rows], dtype=np.float64)
        cell: dict[str, float | dict] = {
            "n_steps": int(len(es)),
            "e_mean_mean": float(es.mean()) if es.size else float("nan"),
            "e_mean_p10": float(np.quantile(es, 0.10)) if es.size else float("nan"),
            "e_mean_p50": float(np.quantile(es, 0.50)) if es.size else float("nan"),
            "e_mean_p90": float(np.quantile(es, 0.90)) if es.size else float("nan"),
            "e_mean_min": float(es.min()) if es.size else float("nan"),
            "e_mean_max": float(es.max()) if es.size else float("nan"),
        }
        for k in args.ks:
            ratios = np.array([r[f"k_over_e_{k}"] for r in cfg_rows], dtype=np.float64)
            cell[f"k_over_e_mean_k{k}"] = float(ratios.mean()) if ratios.size else float("nan")
            cell[f"k_over_e_p90_k{k}"] = float(np.quantile(ratios, 0.90)) if ratios.size else float("nan")
            # Fraction of steps where the switching bound dominates |e|.
            cell[f"frac_k_ge_e_k{k}"] = float((ratios >= 1.0).mean()) if ratios.size else float("nan")
        summary[f"cfg={cfg}"] = cell

    metrics = {
        "n_prompts": len(samples),
        "n_steps": args.infer_steps,
        "cfgs": list(args.cfgs),
        "ks": list(args.ks),
        "lam": args.lam,
        "lora_weight": str(lora) if lora is not None else None,
        "per_cfg": summary,
    }
    write_result(
        out_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label,
        artifacts=["per_step.csv"],
    )
    log.info(f"wrote {out_dir / 'result.json'}")

    # Tight console summary for the user.
    print("\n=== |e| summary (per CFG) ===")
    print(f"{'cfg':>5} {'|e|_p10':>10} {'|e|_p50':>10} {'|e|_p90':>10}", end="")
    for k in args.ks:
        print(f" {'k=' + str(k) + ' k/|e|_mean':>16}", end="")
    print()
    for cfg in args.cfgs:
        c = summary[f"cfg={cfg}"]
        print(f"{cfg:>5.1f} {c['e_mean_p10']:>10.4f} {c['e_mean_p50']:>10.4f} "
              f"{c['e_mean_p90']:>10.4f}", end="")
        for k in args.ks:
            print(f" {c[f'k_over_e_mean_k{k}']:>16.3f}", end="")
        print()
    print()
    print("Interpretation: k/|e| ≪ 1 → SMC is a small refinement of CFG.")
    print("                k/|e| ≈ 1 → SMC dominates the combine (noise regime).")
    print("                k/|e| ≫ 1 → controller saturates the bound, signal-free.")


if __name__ == "__main__":
    main()
