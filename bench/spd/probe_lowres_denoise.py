"""Phase-2 go/no-go probe for Spectral Progressive Diffusion (SPD).

THE REAL FALSIFICATION TEST (see ``bench/spd/plan.md`` Phase 2). Phase 0 already
showed the *static* spectral premise holds for Anima latents (β=2.26). What that
does NOT prove is the load-bearing assumption of *training-free* SPD: that the
**bare DiT can denoise a lower-resolution latent** for the early (noise-dominated)
steps, then accept a spectral-noise-expansion handoff to full resolution and
finish coherently. Anima trains only at the ~4096-token bucket and the inference
loop pins one static shape — so whether the model produces anything but mush at
low res is untested. If it's mush, training-free SPD is dead (only the fine-tune
recipe, Phase 4, could survive).

This is a throwaway probe — no pipeline changes. It loads the **bare** DiT (no
LoRA), encodes one prompt, and runs two generations from the same seed noise:

  * **baseline**  — plain Euler full-res denoise (the reference).
  * **spd**       — first stage(s) at low res, DCT low-pass init, then spectral
                    noise expansion (paper Eq. i–iii) + timestep alignment
                    (Eq. 5–6) to full res, finish normally.

The SPD math mirrors the community ``SamplerSPEED`` node
(``../comfy/custom_nodes/comfyui-speed/speed_sampler.py``) so the exact schedule
shipped in the ``GJ5Rt3Xz`` workflow can be reproduced via ``--community`` and
judged directly, alongside the cleaner single-stage falsification default.

Verdict is fundamentally a *visual* call (coherent vs mush) — the script saves
baseline/spd PNGs + side-by-side montages per seed. The auto-metrics only flag
*hard* divergence (NaN/Inf, latent-std blow-up, washed-out or grain-blasted
sharpness); they cannot certify coherence.

Usage:
  uv run python -m bench.spd.probe_lowres_denoise                 # single-stage 0.5x, handoff at 30% of steps
  uv run python -m bench.spd.probe_lowres_denoise --community     # exact GJ5Rt3Xz schedule (0.5->0.75->1.0 @ σ 0.8/0.6)
  uv run python -m bench.spd.probe_lowres_denoise --start_scale 0.5 --lowres_frac 0.3 --seeds 40 41 42
  uv run python -m bench.spd.probe_lowres_denoise --stages 0.5 0.75 1.0 --transition_sigmas 0.8 0.6
"""

from __future__ import annotations

import argparse
import logging
import math
import sys

import numpy as np
import torch
from PIL import Image

from bench._common import make_run_dir, write_result

log = logging.getLogger("bench.spd.probe")
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Canonical model paths + test settings (mirror scripts/tasks/_common.INFERENCE_BASE).
DEFAULT_DIT = "models/diffusion_models/anima-base-v1.0.safetensors"
DEFAULT_TE = "models/text_encoders/qwen_3_06b_base.safetensors"
DEFAULT_VAE = "models/vae/qwen_image_vae.safetensors"
DEFAULT_PROMPT = (
    "masterpiece, best quality, score_7, safe. An anime girl wearing a black tank-top"
    " and denim shorts is standing outdoors. She's holding a rectangular sign out in"
    ' front of her that reads "ANIMA". She\'s looking at the viewer with a smile. The'
    " background features some trees and blue sky with clouds."
)
DEFAULT_NEG = (
    "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia"
)


# ── DCT helpers (2D separable, type-II, pure PyTorch — matches comfyui-speed) ──

def _dct_matrix(n: int, device, dtype) -> torch.Tensor:
    nr = torch.arange(n, device=device, dtype=dtype)
    k = nr.unsqueeze(1)
    m = torch.cos(torch.pi * k * (2 * nr + 1) / (2 * n))
    m[0] *= 1.0 / math.sqrt(n)
    m[1:] *= math.sqrt(2.0 / n)
    return m


def dct2(x: torch.Tensor) -> torch.Tensor:
    """2D type-II DCT over the last two dims of a (B, C, H, W) tensor."""
    B, C, H, W = x.shape
    Dh = _dct_matrix(H, x.device, x.dtype)
    Dw = _dct_matrix(W, x.device, x.dtype)
    y = x.reshape(B * C, H, W)
    y = Dh @ y
    y = y @ Dw.T
    return y.reshape(B, C, H, W)


def idct2(x: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    Dh = _dct_matrix(H, x.device, x.dtype)
    Dw = _dct_matrix(W, x.device, x.dtype)
    y = x.reshape(B * C, H, W)
    y = Dh.T @ y
    y = y @ Dw
    return y.reshape(B, C, H, W)


def _snap(v: float, mult: int) -> int:
    """Round to nearest positive multiple of ``mult`` (DiT patch_spatial)."""
    return max(mult, int(round(v / mult)) * mult)


# ── SPD spectral primitives ───────────────────────────────────────────────────

def dct_lowpass_init(x5: torch.Tensor, scale: float, patch: int) -> torch.Tensor:
    """DCT low-pass of a (B,C,1,H,W) latent down to a (B,C,1,h,w) grid (paper T_Φ)."""
    B, C, T, H, W = x5.shape
    x4 = x5.squeeze(2).float()
    xi = dct2(x4)
    h = min(_snap(H * scale, patch), H)
    w = min(_snap(W * scale, patch), W)
    x_low = idct2(xi[:, :, :h, :w])
    return x_low.unsqueeze(2).to(x5.dtype)


def spectral_expand(
    x5: torch.Tensor, sigma_val: float, scale_lo: float, scale_hi: float,
    H_full: int, W_full: int, patch: int, gen: torch.Generator,
) -> tuple[torch.Tensor, float]:
    """Embed the current low-res DCT block into a larger grid, fill HF slots with
    σ-scaled noise, iDCT, scale by κ (Eq. iii) and align the timestep (Eq. 5–6).

    Returns (expanded (B,C,1,h_hi,w_hi) latent, sigma_aligned).
    """
    B, C, T, h_lo, w_lo = x5.shape
    x4 = x5.squeeze(2).float()
    xi = dct2(x4)

    h_hi = max(_snap(H_full * scale_hi, patch), h_lo)
    w_hi = max(_snap(W_full * scale_hi, patch), w_lo)

    r = scale_hi / scale_lo
    sigma_aligned = (r * sigma_val) / (1.0 + (r - 1.0) * sigma_val)
    kappa = r / (1.0 + (r - 1.0) * sigma_val)

    xi_new = torch.zeros(B, C, h_hi, w_hi, device=x5.device, dtype=torch.float32)
    xi_new[:, :, :h_lo, :w_lo] = xi
    noise = torch.randn(xi_new.shape, generator=gen, device=x5.device, dtype=torch.float32)
    mask = torch.zeros_like(xi_new)
    mask[:, :, h_lo:, :] = 1.0
    mask[:, :, :h_lo, w_lo:] = 1.0
    xi_new = xi_new + mask * sigma_val * noise

    x4_new = idct2(xi_new) * kappa
    return x4_new.unsqueeze(2).to(x5.dtype), float(sigma_aligned)


# ── Denoise loop (Euler, velocity form, CFG; optional SPD schedule) ─────────────

@torch.no_grad()
def denoise(
    anima, x5_init: torch.Tensor, embed: torch.Tensor, neg_embed: torch.Tensor,
    sigmas: torch.Tensor, guidance: float, device, patch: int,
    stages: list[float], transition_sigmas: list[float], gen: torch.Generator,
) -> torch.Tensor:
    """``stages`` is ascending resolution scales (e.g. [0.5, 0.75, 1.0]);
    ``transition_sigmas`` (len = len(stages)-1) are the σ thresholds at which to
    expand to the next stage. ``stages=[1.0]`` + ``[]`` is the full-res baseline.
    """
    H_full, W_full = x5_init.shape[-2], x5_init.shape[-1]
    sigmas = sigmas.clone().float()

    cur_scale = stages[0]
    x5 = x5_init
    if cur_scale < 1.0:
        x5 = dct_lowpass_init(x5, cur_scale, patch)
    stage_idx = 0

    def velocity(x: torch.Tensor, sigma_scalar: float) -> torch.Tensor:
        t = x.new_full((x.shape[0],), float(sigma_scalar))  # timestep == σ in [0,1]
        pad = torch.zeros(x.shape[0], 1, x.shape[-2], x.shape[-1],
                          dtype=x.dtype, device=device)
        v_c = anima(x, t, embed, padding_mask=pad)
        if guidance != 1.0:
            v_u = anima(x, t, neg_embed, padding_mask=pad)
            return v_u + guidance * (v_c - v_u)
        return v_c

    n = len(sigmas) - 1
    for i in range(n):
        sigma = float(sigmas[i])
        while stage_idx < len(transition_sigmas) and sigma <= transition_sigmas[stage_idx]:
            nxt = stages[stage_idx + 1]
            if nxt > cur_scale:
                orig = float(sigmas[i])
                x5, sigma_new = spectral_expand(
                    x5, sigma, cur_scale, nxt, H_full, W_full, patch, gen
                )
                cur_scale = nxt
                if orig > 0 and sigma_new != orig:  # re-space remaining σ (Sec 4.3)
                    sigmas[i + 1:] = sigma_new * (sigmas[i + 1:] / orig)
                sigma = sigma_new
            stage_idx += 1

        v = velocity(x5, sigma).float()
        dt = float(sigmas[i + 1]) - sigma
        x5 = (x5.float() + v * dt).to(torch.bfloat16)

    if cur_scale < 1.0:  # never handed off to full res — bicubic rescue so decode works
        import torch.nn.functional as F
        x5 = F.interpolate(
            x5.squeeze(2).float(), size=(H_full, W_full), mode="bicubic",
            align_corners=False,
        ).unsqueeze(2).to(torch.bfloat16)
    return x5


# ── Cheap image metrics (flag hard divergence only) ─────────────────────────────

def _to_pil(pixels: torch.Tensor) -> Image.Image:
    arr = pixels.clamp(-1, 1).add(1).mul(127.5).round().byte()  # (C,H,W) [-1,1]->[0,255]
    return Image.fromarray(arr.permute(1, 2, 0).cpu().numpy())


def _gray96(img: Image.Image) -> np.ndarray:
    g = np.asarray(img.convert("L").resize((96, 96), Image.LANCZOS), dtype=np.float32)
    return (g - g.mean()) / (g.std() + 1e-6)


def _laplacian_var(img: Image.Image) -> float:
    g = np.asarray(img.convert("L"), dtype=np.float32)
    lap = (
        -4 * g
        + np.roll(g, 1, 0) + np.roll(g, -1, 0)
        + np.roll(g, 1, 1) + np.roll(g, -1, 1)
    )
    return float(lap[1:-1, 1:-1].var())


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dit", default=DEFAULT_DIT)
    ap.add_argument("--text_encoder", default=DEFAULT_TE)
    ap.add_argument("--vae", default=DEFAULT_VAE)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--negative_prompt", default=DEFAULT_NEG)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--infer_steps", type=int, default=28)
    ap.add_argument("--flow_shift", type=float, default=1.0)
    ap.add_argument("--guidance_scale", type=float, default=4.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[40, 41])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # SPD schedule knobs
    ap.add_argument("--start_scale", type=float, default=0.5,
                    help="Single-stage low-res fraction (used with --lowres_frac).")
    ap.add_argument("--lowres_frac", type=float, default=0.3,
                    help="Fraction of steps run at low res before handoff (single-stage).")
    ap.add_argument("--stages", type=float, nargs="+", default=None,
                    help="Explicit ascending resolution stages, e.g. 0.5 0.75 1.0.")
    ap.add_argument("--transition_sigmas", type=float, nargs="+", default=None,
                    help="σ thresholds to expand to each next stage (len = len(stages)-1).")
    ap.add_argument("--community", action="store_true",
                    help="Reproduce the GJ5Rt3Xz workflow schedule: stages 0.5/0.75/1.0 @ σ 0.8/0.6.")
    ap.add_argument("--label", default="lowres-probe")
    args = ap.parse_args()

    device = torch.device(args.device)

    # ── Resolve the SPD schedule ──
    if args.community:
        stages = [0.5, 0.75, 1.0]
        transition_sigmas = [0.8, 0.6]
        sched_desc = "community GJ5Rt3Xz: 0.5→0.75→1.0 @ σ 0.8/0.6"
    elif args.stages is not None:
        stages = list(args.stages)
        if stages[-1] != 1.0:
            stages.append(1.0)
        transition_sigmas = list(args.transition_sigmas or [])
        if len(transition_sigmas) != len(stages) - 1:
            raise SystemExit(
                f"--transition_sigmas needs {len(stages) - 1} values for stages {stages}"
            )
        sched_desc = f"explicit: {stages} @ σ {transition_sigmas}"
    else:
        stages = [args.start_scale, 1.0]
        transition_sigmas = None  # filled per-run from step fraction (needs σ schedule)
        sched_desc = f"single-stage: {args.start_scale}→1.0 @ {args.lowres_frac:.0%} of steps"

    log.info(f"SPD schedule — {sched_desc}")

    # ── Build a fully-populated inference args namespace, then load the bare DiT ──
    import inference as inference_mod
    from library.anima import strategy as strategy_anima, text_strategies
    from library.inference.models import load_dit_model
    from library.inference.text import prepare_text_inputs, MAX_CROSSATTN_TOKENS
    from library.inference.output import decode_latent
    from library.models import qwen_vae
    from diffusers.utils.torch_utils import randn_tensor
    from library.inference import sampling as inference_utils

    infer_argv = [
        "--dit", args.dit,
        "--text_encoder", args.text_encoder,
        "--vae", args.vae,
        "--vae_chunk_size", "64", "--vae_disable_cache",
        "--attn_mode", "flash",
        "--lora_multiplier", "1.0",
        "--prompt", args.prompt,
        "--negative_prompt", args.negative_prompt,
        "--image_size", str(args.height), str(args.width),
        "--infer_steps", str(args.infer_steps),
        "--flow_shift", str(args.flow_shift),
        "--guidance_scale", str(args.guidance_scale),
        "--seed", str(args.seeds[0]),
        "--device", str(device),
        "--save_path", "output/tests",  # required by parse_args; probe writes its own artifacts
    ]
    _saved_argv = sys.argv
    try:
        sys.argv = ["inference.py", *infer_argv]
        iargs = inference_mod.parse_args()
    finally:
        sys.argv = _saved_argv
    iargs.lora_weight = None  # BARE DiT — Phase 2 tests the unadapted model
    iargs.sampler = "euler"   # plain Euler (not er_sde) so baseline == SPD step code

    # Tokenize/encoding strategies (inference.py sets these before generate()).
    text_strategies.TokenizeStrategy.set_strategy(
        strategy_anima.AnimaTokenizeStrategy(
            qwen3_path=args.text_encoder, t5_tokenizer_path=None,
            qwen3_max_length=MAX_CROSSATTN_TOKENS, t5_max_length=MAX_CROSSATTN_TOKENS,
        )
    )
    text_strategies.TextEncodingStrategy.set_strategy(
        strategy_anima.AnimaTextEncodingStrategy()
    )

    log.info("Loading bare DiT (no LoRA, eager / dynamic shape — no torch.compile) ...")
    anima = load_dit_model(iargs, device, torch.bfloat16)
    patch = anima.patch_spatial

    log.info("Encoding prompt + negative prompt ...")
    context, context_null = prepare_text_inputs(iargs, device, anima, shared_models=None)
    embed = context["embed"][0].to(device, torch.bfloat16)
    neg_embed = context_null["embed"][0].to(device, torch.bfloat16)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    log.info("Loading VAE ...")
    vae = qwen_vae.load_vae(args.vae, device="cpu", spatial_chunk_size=64)
    vae.to(torch.bfloat16)
    vae.eval()

    timesteps, sigmas = inference_utils.get_timesteps_sigmas(
        args.infer_steps, args.flow_shift, device
    )
    sigmas = sigmas.to(device)  # (n+1,), σ ∈ [0,1]

    # Single-stage transition σ derived from step fraction, if not explicit.
    if transition_sigmas is None:
        k = max(1, min(args.infer_steps - 1, int(round(args.infer_steps * args.lowres_frac))))
        transition_sigmas = [float(sigmas[k])]
        log.info(f"  single-stage handoff at step {k}/{args.infer_steps} → σ={transition_sigmas[0]:.4f}")

    run_dir = make_run_dir("spd", label=args.label)
    h_lat, w_lat = args.height // 8, args.width // 8
    per_seed = []

    for seed in args.seeds:
        log.info(f"\n=== seed {seed} ===")
        seed_g = torch.Generator(device="cpu").manual_seed(seed)
        init = randn_tensor(
            (1, anima.LATENT_CHANNELS, 1, h_lat, w_lat),
            generator=seed_g, device=device, dtype=torch.bfloat16,
        )
        spd_gen = torch.Generator(device=device).manual_seed(seed + 10_000)

        def _timed(fn_stages, fn_trans):
            if device.type == "cuda":
                torch.cuda.synchronize()
            import time
            t0 = time.perf_counter()
            lat = denoise(anima, init, embed, neg_embed, sigmas,
                          args.guidance_scale, device, patch, fn_stages, fn_trans, spd_gen)
            if device.type == "cuda":
                torch.cuda.synchronize()
            return lat, time.perf_counter() - t0

        log.info("  baseline (full-res Euler) ...")
        base_lat, base_t = _timed([1.0], [])
        log.info("  spd (low-res → spectral-expand → full-res) ...")
        spd_lat, spd_t = _timed(stages, transition_sigmas)
        speedup = base_t / spd_t if spd_t > 0 else float("nan")

        base_nan = bool(torch.isnan(base_lat).any() or torch.isinf(base_lat).any())
        spd_nan = bool(torch.isnan(spd_lat).any() or torch.isinf(spd_lat).any())
        base_std = float(base_lat.float().std())
        spd_std = float(spd_lat.float().std())

        base_px = decode_latent(vae, base_lat, device)
        spd_px = decode_latent(vae, spd_lat, device)
        base_img, spd_img = _to_pil(base_px), _to_pil(spd_px)
        base_img.save(run_dir / f"baseline_seed{seed}.png")
        spd_img.save(run_dir / f"spd_seed{seed}.png")
        montage = Image.new("RGB", (base_img.width + spd_img.width, max(base_img.height, spd_img.height)), "white")
        montage.paste(base_img, (0, 0))
        montage.paste(spd_img, (base_img.width, 0))
        montage.save(run_dir / f"compare_seed{seed}.png")

        base_sharp, spd_sharp = _laplacian_var(base_img), _laplacian_var(spd_img)
        lowfreq_mse = float(np.mean((_gray96(base_img) - _gray96(spd_img)) ** 2))
        sharp_ratio = spd_sharp / base_sharp if base_sharp > 0 else float("nan")
        std_ratio = spd_std / base_std if base_std > 0 else float("nan")

        log.info(
            f"  time base/spd={base_t:.1f}s/{spd_t:.1f}s (×{speedup:.2f} faster)  "
            f"std ×{std_ratio:.2f}  sharp ×{sharp_ratio:.2f}  "
            f"lowfreq_mse={lowfreq_mse:.3f}  nan={base_nan or spd_nan}"
        )
        per_seed.append({
            "seed": seed,
            "base_time_s": base_t, "spd_time_s": spd_t, "speedup": speedup,
            "base_nan_inf": base_nan, "spd_nan_inf": spd_nan,
            "base_latent_std": base_std, "spd_latent_std": spd_std, "std_ratio": std_ratio,
            "base_sharpness": base_sharp, "spd_sharpness": spd_sharp, "sharp_ratio": sharp_ratio,
            "lowfreq_mse": lowfreq_mse,
        })

    # ── Aggregate auto-verdict (hard divergence only; coherence is visual) ──
    any_nan = any(s["spd_nan_inf"] or s["base_nan_inf"] for s in per_seed)
    std_ratios = [s["std_ratio"] for s in per_seed if math.isfinite(s["std_ratio"])]
    sharp_ratios = [s["sharp_ratio"] for s in per_seed if math.isfinite(s["sharp_ratio"])]
    mean_std_ratio = float(np.mean(std_ratios)) if std_ratios else float("nan")
    mean_sharp_ratio = float(np.mean(sharp_ratios)) if sharp_ratios else float("nan")
    mean_lowfreq_mse = float(np.mean([s["lowfreq_mse"] for s in per_seed]))
    mean_speedup = float(np.mean([s["speedup"] for s in per_seed if math.isfinite(s["speedup"])]))

    hard_fail = (
        any_nan
        or (math.isfinite(mean_std_ratio) and (mean_std_ratio > 3.0 or mean_std_ratio < 0.33))
        or (math.isfinite(mean_sharp_ratio) and (mean_sharp_ratio < 0.2 or mean_sharp_ratio > 5.0))
    )
    if hard_fail:
        verdict = (
            "HARD_FAIL (MUSH): auto-metrics show divergence (NaN/Inf, latent-std "
            "blow-up, or sharpness collapse/grain). Training-free SPD likely dead on "
            "Anima — inspect compare_*.png to confirm, then see plan.md Phase 4."
        )
    else:
        verdict = (
            "NO_HARD_DIVERGENCE: outputs decoded without instability. Coherence is a "
            "VISUAL call — open compare_seed*.png. If SPD images are coherent (subject "
            "intact, no smear/double-image at the handoff), Phase 2 PASSES → Phase 3."
        )

    metrics = {
        "schedule_description": sched_desc,
        "stages": stages,
        "transition_sigmas": transition_sigmas,
        "infer_steps": args.infer_steps,
        "flow_shift": args.flow_shift,
        "guidance_scale": args.guidance_scale,
        "resolution_hw": [args.height, args.width],
        "n_seeds": len(args.seeds),
        "mean_speedup": mean_speedup,
        "mean_std_ratio_spd_over_base": mean_std_ratio,
        "mean_sharpness_ratio_spd_over_base": mean_sharp_ratio,
        "mean_lowfreq_mse": mean_lowfreq_mse,
        "any_nan_inf": any_nan,
        "per_seed": per_seed,
        "verdict": verdict,
    }
    artifacts = [p.name for p in sorted(run_dir.glob("*.png"))]
    write_result(run_dir, script=__file__, args=args, metrics=metrics,
                 artifacts=artifacts, device=device)

    log.info("\n" + "=" * 70)
    log.info(f"  schedule: {sched_desc}")
    log.info(f"  speedup (base/spd time):   ×{mean_speedup:.2f}")
    log.info(f"  std ratio (spd/base):      {mean_std_ratio:.2f}")
    log.info(f"  sharpness ratio (spd/base): {mean_sharp_ratio:.2f}")
    log.info(f"  low-freq MSE (spd vs base): {mean_lowfreq_mse:.3f}")
    log.info(f"  {verdict}")
    log.info(f"  → {run_dir}  (open compare_seed*.png)")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
