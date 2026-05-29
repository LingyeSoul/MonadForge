#!/usr/bin/env python
"""Capture per-step DiT block features for offline Spectrum-forecaster analysis.

Live successor to `_archive/bench/spectrum/capture_features.py`, kept faithful to
the shipped Spectrum path but with one addition: a `--from_synth_pool` mode that
draws prompts + native buckets straight from the mod-guidance distillation pool
(`post_image_dataset/distill_mod_synth/`). That pool is already a large,
aspect-balanced corpus of er_sde@cfg2.5 rollouts, so reusing its captions gives
representative trajectories without re-curating a prompt set (see
`bench/spectrum_pareto/README.md`).

What it captures
----------------
The input to `final_layer` (the block-stack output `x_B_T_H_W_D` the forecaster
predicts), per denoising step, for the conditional stream (and the
unconditional stream with `--capture_uncond`). Saved as fp16.

Faithfulness: a custom runner mirrors `spectrum_denoise`'s actual-forward branch
(same `set_hydra_sigma`, same `final_layer` pre-hook, same sampler step) but
forces an actual forward at every step. The forecaster buffer only ever ingests
actual-forward features, so one all-actual capture supports replaying *any* cache
schedule offline (`replay_forecaster.py`) — no DiT in the loop.

`--sampler` controls the latent path the forecaster must predict. euler routes
through the generic `inference_utils.step` (sampler=None branch); er_sde / lcm
use their stochastic `.step`. This is the axis `compare_samplers.py` sweeps.

Run from the repo root (model/config paths resolve under anima_home()).
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Importing networks.spectrum registers the production runner; we override it
# per-capture below so generate() dispatches into our all-actual capture loop.
import networks.spectrum  # noqa: E402,F401
from library.inference import sampling as inference_utils  # noqa: E402
from library.inference.adapters import clear_hydra_sigma, set_hydra_sigma  # noqa: E402
from library.inference.generation import register_spectrum_runner  # noqa: E402
from bench._common import make_run_dir, write_result  # noqa: E402

# Filled by the capture runner on each generate() call, drained by the caller.
_CAPTURE: dict = {}

POOL_DIR = "post_image_dataset/distill_mod_synth"
CAPTION_DIR = "image_dataset"

# Inference needs explicit model paths (base.toml is training-side and isn't
# merged into the inference arg namespace). Mirror configs/base.toml; override
# via the CLI flags or ANIMA_DIT / ANIMA_VAE / ANIMA_TEXT_ENCODER.
DEFAULT_DIT = "models/diffusion_models/anima-base-v1.0.safetensors"
DEFAULT_VAE = "models/vae/qwen_image_vae.safetensors"
DEFAULT_TEXT_ENCODER = "models/text_encoders/qwen_3_06b_base.safetensors"


def _make_capture_runner(capture_uncond: bool):
    """A spectrum-runner-shaped fn that forces all-actual forwards and records
    the per-step `final_layer` input feature(s) into `_CAPTURE`."""

    def capture_runner(
        anima,
        latents,
        timesteps,
        sigmas,
        embed,
        negative_embed,
        padding_mask,
        guidance_scale,
        sampler,
        device,
        ctx,
        **_spectrum_kwargs,  # window_size/flex_window/warmup/m/lam/w/... — ignored
    ):
        do_cfg = guidance_scale != 1.0
        num_steps = len(timesteps)
        captured: dict = {}

        def _pre_hook(module, args):
            # args[0] = x_B_T_H_W_D (block-stack output, post static-unpad).
            captured["feat"] = args[0].detach().to(torch.float16).cpu().numpy()

        hook = anima.final_layer.register_forward_pre_hook(_pre_hook)
        cond_feats: list[np.ndarray] = []
        uncond_feats: list[np.ndarray] = []
        try:
            for i, t in enumerate(timesteps):
                t_exp = t.expand(latents.shape[0])
                set_hydra_sigma(anima, t_exp)
                with torch.no_grad():
                    noise_pred = anima(latents, t_exp, embed, padding_mask=padding_mask)
                cond_feats.append(captured["feat"])
                if do_cfg:
                    with torch.no_grad():
                        uncond_noise_pred = anima(
                            latents, t_exp, negative_embed, padding_mask=padding_mask
                        )
                    if capture_uncond:
                        uncond_feats.append(captured["feat"])
                    noise_pred = uncond_noise_pred + guidance_scale * (
                        noise_pred - uncond_noise_pred
                    )

                # Sampler step — identical to spectrum_denoise's tail. euler when
                # sampler is None (generation.py builds er_sde only for er_sde/lcm).
                denoised = latents.float() - sigmas[i] * noise_pred.float()
                if sampler is not None:
                    new_latents = sampler.step(latents, denoised, i)
                else:
                    new_latents = inference_utils.step(latents, noise_pred, sigmas, i)
                latents = new_latents.to(latents.dtype)
        finally:
            clear_hydra_sigma(anima)
            hook.remove()

        _CAPTURE.clear()
        _CAPTURE.update(
            cond=np.stack(cond_feats),  # (num_steps, B, T, H, W, D)
            uncond=np.stack(uncond_feats) if uncond_feats else None,
            sigmas=sigmas.detach().float().cpu().numpy(),
            timesteps=timesteps.detach().float().cpu().numpy(),
            num_steps=num_steps,
            do_cfg=do_cfg,
        )
        return latents

    return capture_runner


def _slug(text: str, n: int = 40) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip()).strip("_").lower()
    return (s[:n] or "prompt").rstrip("_")


def _capture_payload(seed: int, guidance_scale: float, capture_uncond: bool) -> dict:
    """Turn the most recent `_CAPTURE` into a saveable payload (B=1 squeezed)."""
    cond_s = _CAPTURE["cond"][:, 0]  # (num_steps, T, H, W, D)
    payload = dict(
        cond=cond_s.astype(np.float16),
        sigmas=_CAPTURE["sigmas"].astype(np.float32),
        timesteps=_CAPTURE["timesteps"].astype(np.float32),
        num_steps=np.int64(_CAPTURE["num_steps"]),
        feat_shape=np.asarray(cond_s.shape[1:], dtype=np.int64),  # (T,H,W,D)
        guidance_scale=np.float32(guidance_scale),
        seed=np.int64(seed),
    )
    if capture_uncond and _CAPTURE.get("uncond") is not None:
        payload["uncond"] = _CAPTURE["uncond"][:, 0].astype(np.float16)
    return payload


def capture_one(
    prompt: str,
    seed: int,
    hw: tuple[int, int],
    *,
    steps: int,
    guidance_scale: float,
    flow_shift: float,
    sampler: str,
    negative_prompt: str = "",
    lora: str | None = None,
    device: str = "cuda",
    dit: str | None = None,
    vae: str | None = None,
    text_encoder: str | None = None,
    capture_uncond: bool = False,
    shared_models: dict | None = None,
    compile_blocks: bool = False,
    compile_inductor_mode: str | None = None,
) -> dict:
    """Run one all-actual capture and return the saveable payload dict.

    Registers the capture runner (idempotent per call), so callers can sweep the
    `sampler` axis across calls in one process. `hw` is pixel (H, W). Pass a
    persistent `shared_models` dict to reuse the loaded DiT/text encoder across
    calls (the dominant cost for a multi-prompt probe).

    `compile_blocks` enables `model.compile_blocks()` at load (graphs keyed on
    token count → reused across captures sharing a bucket). It only takes effect
    on the first load; once the DiT is cached in `shared_models` it is reused
    already-compiled.
    """
    from anima_lora import GenerationRequest, generate, get_generation_settings

    register_spectrum_runner(_make_capture_runner(capture_uncond))
    h, w = hw
    extra_argv = ["--spectrum"]  # dispatch into our capture runner
    if compile_blocks:
        extra_argv.append("--compile_blocks")
        if compile_inductor_mode:
            extra_argv += ["--compile_inductor_mode", compile_inductor_mode]
    kw = dict(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image_size=(h, w),
        infer_steps=steps,
        guidance_scale=guidance_scale,
        flow_shift=flow_shift,
        sampler=sampler,
        seed=seed,
        device=device,
        lora_weight=[lora] if lora else None,
        output_type="latent",  # generate() returns the latent; no decode/save
        no_metadata=True,
        extra_argv=tuple(extra_argv),
    )
    kw["dit"] = dit or DEFAULT_DIT
    kw["vae"] = vae or DEFAULT_VAE
    kw["text_encoder"] = text_encoder or DEFAULT_TEXT_ENCODER
    req = GenerationRequest(**kw)
    args_ns = req.to_args()
    # With a non-None shared_models, generate()/prepare_text_inputs expect the
    # caller to have pre-loaded the text encoder (it only auto-loads when
    # shared_models is None). Populate it once; the DiT is cached on first
    # generate(). load_shared_models loads the TE to CPU — it's moved on demand.
    if shared_models is not None and "text_encoder" not in shared_models:
        from library.inference.models import load_shared_models

        shared_models.update(load_shared_models(args_ns))
    generate(args_ns, get_generation_settings(args_ns), shared_models=shared_models)
    return _capture_payload(seed, guidance_scale, capture_uncond)


# ---------------------------------------------------------------------------
# Synth-pool prompt sampling
# ---------------------------------------------------------------------------

_POOL_RE = re.compile(r"^(?P<stem>.+)_(?P<hl>\d+)x(?P<wl>\d+)_anima\.npz$")


def pool_samples(
    n: int,
    *,
    pool_dir: str = POOL_DIR,
    caption_dir: str = CAPTION_DIR,
    buckets: list[str] | None = None,
    seed: int = 0,
) -> list[dict]:
    """Sample N (caption, native-bucket) pairs from the distill_mod_synth pool.

    Each entry: {prompt, hw (pixel H,W), latent_bucket, artist, stem, npz}.
    `buckets` (optional) restricts to latent-dim strings like "150x112".
    Pixel size comes from the stored `original_size_*` ([W_pix, H_pix]); falls
    back to latent×8 if absent.
    """
    root = Path(pool_dir)
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[2] / pool_dir
    cap_root = Path(caption_dir)
    if not cap_root.is_absolute():
        cap_root = Path(__file__).resolve().parents[2] / caption_dir

    candidates: list[dict] = []
    for npz in root.rglob("*_anima.npz"):
        m = _POOL_RE.match(npz.name)
        if not m:
            continue
        latent_bucket = f"{m['hl']}x{m['wl']}"
        if buckets and latent_bucket not in buckets:
            continue
        artist = npz.parent.name
        stem = m["stem"]
        cap_path = cap_root / artist / f"{stem}.txt"
        if not cap_path.exists():
            continue
        prompt = " ".join(cap_path.read_text(errors="ignore").split()).strip()
        if not prompt:
            continue
        # Pixel size from the npz original_size; fall back to latent×8.
        hl, wl = int(m["hl"]), int(m["wl"])
        h_pix, w_pix = hl * 8, wl * 8
        try:
            d = np.load(npz)
            key = f"original_size_{hl}x{wl}"
            if key in d.files:
                w_pix, h_pix = (int(x) for x in d[key])  # stored [W_pix, H_pix]
        except Exception:
            pass
        candidates.append(
            {
                "prompt": prompt,
                "hw": (h_pix, w_pix),
                "latent_bucket": latent_bucket,
                "artist": artist,
                "stem": stem,
                "npz": str(npz),
            }
        )
    if not candidates:
        raise SystemExit(
            f"no pool samples under {root} with captions under {cap_root}"
            + (f" for buckets {buckets}" if buckets else "")
        )
    candidates.sort(key=lambda c: (c["artist"], c["stem"]))  # deterministic order
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[: min(n, len(candidates))]


# ---------------------------------------------------------------------------
# Standalone CLI (single-sampler capture; compare_samplers.py drives the probe)
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--prompt", action="append", default=[], help="Prompt (repeatable)."
    )
    ap.add_argument(
        "--from_synth_pool",
        type=int,
        default=0,
        help="Draw N (caption, native bucket) pairs from the distill_mod_synth pool.",
    )
    ap.add_argument(
        "--buckets",
        type=str,
        nargs="+",
        default=None,
        help="Restrict --from_synth_pool to latent-dim buckets, e.g. 128x128 150x112.",
    )
    ap.add_argument("--pool_dir", type=str, default=POOL_DIR)
    ap.add_argument("--caption_dir", type=str, default=CAPTION_DIR)
    ap.add_argument("--pool_seed", type=int, default=0, help="Pool sampling RNG seed.")
    ap.add_argument("--negative_prompt", type=str, default="")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance_scale", type=float, default=2.5)
    ap.add_argument("--flow_shift", type=float, default=3.0)
    ap.add_argument("--sampler", type=str, default="er_sde")
    ap.add_argument(
        "--image_size",
        type=str,
        default="1024x1024",
        help="HxW pixels, for --prompt mode (pool mode uses each sample's bucket).",
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--capture_uncond", action="store_true")
    ap.add_argument("--lora", type=str, default=None)
    ap.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    ap.add_argument("--dit", type=str, default=None)
    ap.add_argument("--vae", type=str, default=None)
    ap.add_argument("--text_encoder", type=str, default=None)
    ap.add_argument("--label", type=str, default="cap")
    args = ap.parse_args()

    if args.from_synth_pool > 0:
        samples = pool_samples(
            args.from_synth_pool,
            pool_dir=args.pool_dir,
            caption_dir=args.caption_dir,
            buckets=args.buckets,
            seed=args.pool_seed,
        )
        jobs = [(s["prompt"], s["hw"], s) for s in samples]
        print(f"[{len(jobs)} pool sample(s) from {args.pool_dir}]")
    else:
        prompts = args.prompt or [
            "1girl, solo, masterpiece, best quality, detailed background, soft light",
            "a red fox sitting in a snowy pine forest, watercolor, soft palette",
            "cyberpunk city street at night, neon signs, rain, cinematic lighting",
        ]
        h, w = (int(x) for x in args.image_size.lower().split("x"))
        jobs = [(p, (h, w), {"prompt": p}) for p in prompts]

    out_dir = make_run_dir("spectrum_pareto", label=args.label)
    cap_dir = out_dir / "captures"
    cap_dir.mkdir(parents=True, exist_ok=True)
    print(f"out_dir = {out_dir}")

    shared_models: dict = {}  # reuse the loaded DiT/text encoder across captures
    manifest: list[dict] = []
    for idx, (prompt, hw, prov) in enumerate(jobs):
        slug = _slug(prompt)
        for seed in args.seeds:
            print(
                f"\n=== capture [{args.sampler}]: '{prompt[:60]}' {hw} seed={seed} ==="
            )
            payload = capture_one(
                prompt,
                seed,
                hw,
                steps=args.steps,
                guidance_scale=args.guidance_scale,
                flow_shift=args.flow_shift,
                sampler=args.sampler,
                negative_prompt=args.negative_prompt,
                lora=args.lora,
                device=args.device,
                dit=args.dit,
                vae=args.vae,
                text_encoder=args.text_encoder,
                capture_uncond=args.capture_uncond,
                shared_models=shared_models,
            )
            fname = f"p{idx:02d}_{slug}_seed{seed}.npz"
            np.savez(cap_dir / fname, **payload)
            meta = {
                "file": fname,
                "prompt": prompt,
                "seed": seed,
                "hw": list(hw),
                "num_steps": int(payload["num_steps"]),
                "feat_shape": [int(x) for x in payload["feat_shape"]],
                "mb": round((cap_dir / fname).stat().st_size / 1e6, 1),
                **{
                    k: prov[k] for k in ("latent_bucket", "artist", "stem") if k in prov
                },
            }
            manifest.append(meta)
            print(f"  saved captures/{fname}  ({meta['mb']} MB, {meta['feat_shape']})")

    metrics = {
        "n_captures": len(manifest),
        "sampler": args.sampler,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "from_synth_pool": args.from_synth_pool,
        "captures": manifest,
        "captures_dir": "captures",
    }
    result_path = write_result(
        out_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics=metrics,
        artifacts=["captures"],
        device=args.device,
    )
    print(f"\nresult → {result_path}")
    print(
        f"replay:  python -m bench.spectrum_pareto.replay_forecaster --captures_dir {cap_dir}"
    )


if __name__ == "__main__":
    main()
