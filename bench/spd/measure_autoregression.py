"""Phase-1 autoregression-dynamics probe for Spectral Progressive Diffusion (SPD).

See ``bench/spd/plan.md`` Phase 1. Phase 0 showed the *static* premise (clean
latents decay as a power law, β=2.26). That says HF *carries less signal* — it
does NOT prove HF stays *noise-dominated until late* in Anima's actual
denoising trajectory. That dynamic (the paper's Fig 2b "spectral
autoregression") is what justifies running early steps at low resolution.

This probe answers it two ways:

  HALF A — empirical eyeball (no δ).
    Run N full-res generations, capture x_t and the running clean estimate
    x0_pred = x_t − σ·v at every step, compute the per-channel-standardized
    radial-band power, and plot the *resolved fraction* R_band(σ) =
    power(σ) / power(σ→0) per band. Autoregression ⇔ low-freq bands reach
    R≈1 at high σ (early) while high-freq bands stay near 0 until small σ.
    You read the curves. The script also reports each band's σ_resolve (the σ
    below which R *stays* ≥ resolve_threshold — a last-crossing metric, immune to
    the x0_pred-from-noise transient that spikes high bands in the first ~2 steps);
    a monotone-decreasing σ_resolve across bands is the PASS.

  HALF B — δ-optimal schedule (Prop 1/2 cross-check).
    Measure P_ω from real clean latents (ortho-FFT on unit-variance-
    standardized latents so the paper's x0^(ω)~N(0,P_ω), ε^(ω)~N(0,1)
    assumption holds), then:
      Prop 1:  t_ω = 1 / (1 + sqrt( δ / (P_ω (1 + P_ω − δ)) ))   (Eq. 9)
      Prop 2:  t*_i = t_ω at ω = s_i·ω_Nyquist  (= t_ω(k=s_i))   (Eq. 10)
    For S∈{2,3} with geometric scales, map t*_i onto the real 28-step σ
    schedule, count steps per resolution stage, and report the predicted
    token / attention-FLOP saving vs the full-res baseline. The derived t*_i
    are overlaid as vertical lines on the Half-A plot — they should land where
    the band of matching Nyquist (k=s_i) actually resolves.

Usage:
  uv run python -m bench.spd.measure_autoregression
  uv run python -m bench.spd.measure_autoregression --delta 0.01 --scales_s3 0.5 0.75
  uv run python -m bench.spd.measure_autoregression --n_pomega 128 --seeds 40 41 42
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from bench._common import make_run_dir, write_result
from bench.spd.measure_latent_spectrum import (
    discover_images,
    radial_profile,
    vae_dims,
)
from library.datasets.image_utils import IMAGE_TRANSFORMS

log = logging.getLogger("bench.spd.autoreg")
logging.basicConfig(level=logging.INFO, format="%(message)s")

DEFAULT_DIT = "models/diffusion_models/anima-base-v1.0.safetensors"
DEFAULT_TE = "models/text_encoders/qwen_3_06b_base.safetensors"
DEFAULT_VAE = "models/vae/qwen_image_vae.safetensors"
DEFAULT_NEG = "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia"
# A small, varied prompt set — flat-fill, detailed, and text-bearing so the
# trajectory isn't dominated by one spectral regime.
DEFAULT_PROMPTS = [
    "masterpiece, best quality, score_7, safe. An anime girl with long flowing hair "
    "standing in a field of flowers under a bright blue sky, soft lighting.",
    "masterpiece, best quality, score_7, safe. A highly detailed cyberpunk city street "
    "at night, neon signs, rain, dense reflections and intricate background.",
    "masterpiece, best quality, score_7, safe. A minimalist flat-color illustration of "
    "a red apple on a plain pastel background, cel shading, large flat fills.",
    "masterpiece, best quality, score_7, safe. A close-up portrait of a young woman, "
    "freckles, detailed eyes, fine hair strands, shallow depth of field.",
]


# ── per-channel-standardized radial band power ──────────────────────────────────


def band_power(lat: np.ndarray, n_bands: int) -> np.ndarray:
    """Mean radial-band power of a per-channel-standardized latent.

    ``lat``: (C, H, W). Standardize each channel to zero-mean / unit-variance
    (so the measurement is *shape*, comparable across timesteps regardless of
    the latent's overall scale), ortho-FFT, |F|², radial-bin into ``n_bands``,
    average over channels. Returns (n_bands,).
    """
    C = lat.shape[0]
    acc = np.zeros(n_bands)
    cnt = np.zeros(n_bands)
    for c in range(C):
        x = lat[c]
        x = (x - x.mean()) / (x.std() + 1e-8)
        f = np.fft.fftshift(np.fft.fft2(x, norm="ortho"))
        power = f.real**2 + f.imag**2
        prof = radial_profile(power, n_bands)
        good = np.isfinite(prof)
        acc[good] += prof[good]
        cnt[good] += 1
    return acc / np.where(cnt > 0, cnt, np.nan)


def _to_chw(lat5: torch.Tensor) -> np.ndarray:
    """(1,C,1,H,W) bf16 latent → (C,H,W) float32 numpy."""
    return lat5.squeeze(0).squeeze(1).float().cpu().numpy()


# ── Prop 1 / Prop 2 ─────────────────────────────────────────────────────────────


def activation_time(p_omega: np.ndarray, delta: float) -> np.ndarray:
    """Prop 1, Eq. 9: t_ω = 1/(1 + sqrt(δ/(P_ω(1+P_ω−δ)))).

    Monotone increasing in P_ω, so low-freq (large P_ω) → t_ω near 1 (resolves
    early as t:1→0), high-freq (small P_ω) → t_ω near 0 (resolves only late).
    """
    p = np.clip(p_omega, 1e-12, None)
    denom = p * (1.0 + p - delta)
    return 1.0 / (1.0 + np.sqrt(delta / np.clip(denom, 1e-12, None)))


def transition_times(k_grid, p_k, scales, delta):
    """Prop 2: t*_i = t_ω(k = s_i) for each intermediate scale s_i."""
    t_k = activation_time(p_k, delta)
    return [float(np.interp(s, k_grid, t_k)) for s in scales], t_k


def stage_budget(sigmas, scales_full, transitions, base_tokens):
    """Map transition σ thresholds onto the real step schedule and tally cost.

    ``scales_full`` ascending incl. 1.0 (e.g. [0.5,1.0]); ``transitions`` are
    the σ at which to expand from scale i to i+1 (len = len(scales_full)-1).
    Returns (steps_per_stage, attn_cost, mlp_cost, attn_speedup, mlp_speedup).
    """
    sig = sigmas.detach().cpu().numpy()
    n = len(sig) - 1
    steps = [0] * len(scales_full)
    stage = 0
    for i in range(n):
        s = float(sig[i])
        while stage < len(transitions) and s <= transitions[stage]:
            stage += 1
        steps[stage] += 1
    attn = mlp = 0.0
    for st, sc in zip(steps, scales_full):
        toks = (sc**2) * base_tokens
        attn += st * toks**2
        mlp += st * toks
    base_attn = n * base_tokens**2
    base_mlp = n * base_tokens
    return (
        steps,
        attn,
        mlp,
        base_attn / attn if attn > 0 else float("nan"),
        base_mlp / mlp if mlp > 0 else float("nan"),
    )


# ── main ────────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dit", default=DEFAULT_DIT)
    ap.add_argument("--text_encoder", default=DEFAULT_TE)
    ap.add_argument("--vae", default=DEFAULT_VAE)
    ap.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    ap.add_argument("--negative_prompt", default=DEFAULT_NEG)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--infer_steps", type=int, default=28)
    ap.add_argument("--flow_shift", type=float, default=1.0)
    ap.add_argument("--guidance_scale", type=float, default=4.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[40, 41])
    ap.add_argument(
        "--n_bands",
        type=int,
        default=6,
        help="Coarse radial bands for the eyeball plot.",
    )
    ap.add_argument(
        "--resolve_threshold",
        type=float,
        default=0.8,
        help="A band is 'resolved' once R stays ≥ this (last-crossing σ_resolve).",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Half B
    ap.add_argument(
        "--delta",
        type=float,
        default=0.01,
        help="SPD error tolerance (paper default 0.01).",
    )
    ap.add_argument(
        "--n_pomega", type=int, default=96, help="Real images encoded to estimate P_ω."
    )
    ap.add_argument("--pomega_bins", type=int, default=96)
    ap.add_argument("--image_dir", default="image_dataset")
    ap.add_argument(
        "--scales_s2",
        type=float,
        nargs="+",
        default=[0.5],
        help="Intermediate scale(s) for S=2.",
    )
    ap.add_argument(
        "--scales_s3",
        type=float,
        nargs="+",
        default=[0.5, 0.75],
        help="Intermediate scales for S=3.",
    )
    ap.add_argument("--label", default="autoregression")
    args = ap.parse_args()

    device = torch.device(args.device)

    # ── Load bare DiT + TE + VAE (mirror probe_lowres_denoise) ──
    import inference as inference_mod
    from diffusers.utils.torch_utils import randn_tensor

    from library.inference import sampling as inference_utils
    from library.inference.models import load_dit_model
    from library.inference.text import (
        MAX_CROSSATTN_TOKENS,
        ensure_text_strategies,
        prepare_text_inputs,
    )
    from library.models import qwen_vae

    infer_argv = [
        "--dit",
        args.dit,
        "--text_encoder",
        args.text_encoder,
        "--vae",
        args.vae,
        "--vae_chunk_size",
        "64",
        "--vae_disable_cache",
        "--attn_mode",
        "flash",
        "--lora_multiplier",
        "1.0",
        "--prompt",
        args.prompts[0],
        "--negative_prompt",
        args.negative_prompt,
        "--image_size",
        str(args.height),
        str(args.width),
        "--infer_steps",
        str(args.infer_steps),
        "--flow_shift",
        str(args.flow_shift),
        "--guidance_scale",
        str(args.guidance_scale),
        "--seed",
        str(args.seeds[0]),
        "--device",
        str(device),
        "--save_path",
        "output/tests",
    ]
    _saved = sys.argv
    try:
        sys.argv = ["inference.py", *infer_argv]
        iargs = inference_mod.parse_args()
    finally:
        sys.argv = _saved
    iargs.lora_weight = None
    iargs.sampler = "euler"

    ensure_text_strategies(args.text_encoder, MAX_CROSSATTN_TOKENS)

    log.info("Loading bare DiT (no LoRA, eager / dynamic shape) ...")
    anima = load_dit_model(iargs, device, torch.bfloat16)

    log.info(f"Encoding {len(args.prompts)} prompt(s) + negative ...")
    embeds = []
    for pr in args.prompts:
        iargs.prompt = pr
        ctx, ctx_null = prepare_text_inputs(iargs, device, anima, shared_models=None)
        embeds.append(
            (
                ctx["embed"][0].to(device, torch.bfloat16),
                ctx_null["embed"][0].to(device, torch.bfloat16),
            )
        )
    if device.type == "cuda":
        torch.cuda.empty_cache()

    timesteps, sigmas = inference_utils.get_timesteps_sigmas(
        args.infer_steps, args.flow_shift, device
    )
    sigmas = sigmas.to(device)
    n_steps = len(sigmas) - 1
    sig_at_step = sigmas[:n_steps].detach().cpu().numpy()  # σ before each Euler step

    h_lat, w_lat = args.height // 8, args.width // 8

    # ── HALF A: capture per-band power over the trajectory ──
    # accumulate (n_steps, n_bands) for x_t and x0_pred, averaged over runs.
    xt_acc = np.zeros((n_steps, args.n_bands))
    x0_acc = np.zeros((n_steps, args.n_bands))
    n_runs = 0

    @torch.no_grad()
    def velocity(x5, sigma_scalar, embed, neg):
        t = x5.new_full((x5.shape[0],), float(sigma_scalar))
        pad = torch.zeros(
            x5.shape[0], 1, x5.shape[-2], x5.shape[-1], dtype=x5.dtype, device=device
        )
        v_c = anima(x5, t, embed, padding_mask=pad)
        if args.guidance_scale != 1.0:
            v_u = anima(x5, t, neg, padding_mask=pad)
            return v_u + args.guidance_scale * (v_c - v_u)
        return v_c

    for pi, (embed, neg) in enumerate(embeds):
        for seed in args.seeds:
            log.info(f"  gen prompt {pi} seed {seed} ...")
            g = torch.Generator(device="cpu").manual_seed(seed)
            x5 = randn_tensor(
                (1, anima.LATENT_CHANNELS, 1, h_lat, w_lat),
                generator=g,
                device=device,
                dtype=torch.bfloat16,
            )
            for i in range(n_steps):
                sigma = float(sigmas[i])
                v = velocity(x5, sigma, embed, neg).float()
                x0_pred = x5.float() - sigma * v  # clean estimate (x0 = x_t − t·v)
                xt_acc[i] += band_power(_to_chw(x5), args.n_bands)
                x0_acc[i] += band_power(_to_chw(x0_pred), args.n_bands)
                dt = float(sigmas[i + 1]) - sigma
                x5 = (x5.float() + v * dt).to(torch.bfloat16)
            n_runs += 1

    xt_mean = xt_acc / n_runs
    x0_mean = x0_acc / n_runs

    # Resolved fraction R_band(σ) = power(step) / power(final step). Use x0_pred
    # (the emerging clean image); x_t kept in the CSV for reference.
    final = x0_mean[-1]
    R = x0_mean / np.where(final > 0, final, np.nan)[None, :]

    band_centers = (np.arange(args.n_bands) + 0.5) / args.n_bands  # k/Nyquist
    # σ_resolve per band: the σ BELOW WHICH R stays ≥ resolve_threshold (the LAST
    # upward crossing). A "first crossing" metric is fooled by the x0_pred-from-
    # noise transient in the first ~2 steps — at σ≈1 the one-shot, CFG-amplified
    # x0_pred has spurious HF energy, so high bands spike then fall back. Last-
    # crossing measures *genuine* resolution (power reaches and holds near final)
    # and is immune to that transient. Bands with no sub-threshold step (e.g. the
    # DC band, R≥1 throughout) count as resolved from σ=σ_max.
    thr = args.resolve_threshold
    sigma_resolve = []
    for b in range(args.n_bands):
        below = np.where(R[:, b] < thr)[0]
        if len(below) == 0:
            sigma_resolve.append(float(sig_at_step[0]))
        else:
            j = min(int(below[-1]) + 1, len(sig_at_step) - 1)
            sigma_resolve.append(float(sig_at_step[j]))

    monotone = all(
        sigma_resolve[b] >= sigma_resolve[b + 1] - 1e-9 for b in range(args.n_bands - 1)
    )

    # ── HALF B: P_ω from real latents → t_ω → t*_i → budget ──
    log.info(f"Estimating P_ω from {args.n_pomega} real latents ...")
    vae = qwen_vae.load_vae(args.vae, device=str(device), spatial_chunk_size=64)
    vae.eval()
    picks = discover_images(Path(args.image_dir), args.n_pomega, seed=0)
    pk_acc = np.zeros(args.pomega_bins)
    pk_cnt = np.zeros(args.pomega_bins)
    for j, p in enumerate(picks):
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        w, h = img.size
        nw, nh = vae_dims(max(w, h), w, h, 1024)
        img = img.resize((nw, nh), Image.LANCZOS)
        x = IMAGE_TRANSFORMS(np.array(img)).unsqueeze(0).to(device, torch.bfloat16)
        with torch.no_grad():
            lat = vae.encode_pixels_to_latents(x).float().cpu().numpy()
        lat = lat.reshape(16, lat.shape[-2], lat.shape[-1])  # 16 latent channels
        prof = band_power(lat, args.pomega_bins)  # already unit-var + ortho normalized
        good = np.isfinite(prof)
        pk_acc[good] += prof[good]
        pk_cnt[good] += 1
        if (j + 1) % 25 == 0:
            log.info(f"    {j + 1}/{len(picks)} ...")
    p_k = pk_acc / np.where(pk_cnt > 0, pk_cnt, np.nan)
    k_grid = (np.arange(args.pomega_bins) + 0.5) / args.pomega_bins
    # renormalize so the mode-count-weighted mean = 1 (paper's per-mode unit noise).
    valid = np.isfinite(p_k)
    wmean = np.average(p_k[valid], weights=k_grid[valid])
    p_k = p_k / wmean

    base_tokens = (h_lat // anima.patch_spatial) * (w_lat // anima.patch_spatial)

    schedules = {}
    for name, inter in (("S2", args.scales_s2), ("S3", args.scales_s3)):
        scales_full = sorted(set([*inter, 1.0]))
        t_star, t_k = transition_times(k_grid, p_k, inter, args.delta)
        # transitions are the σ thresholds (descending order matches scale order)
        steps, attn, mlp, attn_su, mlp_su = stage_budget(
            sigmas, scales_full, sorted(t_star, reverse=True), base_tokens
        )
        schedules[name] = {
            "intermediate_scales": list(inter),
            "scales_full": scales_full,
            "t_star": t_star,
            "steps_per_stage": steps,
            "attn_speedup": attn_su,
            "mlp_speedup": mlp_su,
        }
        log.info(
            f"  {name}: scales {scales_full}  t*={[round(t, 3) for t in t_star]}  "
            f"steps/stage={steps}  attn ×{attn_su:.2f}  mlp ×{mlp_su:.2f}"
        )

    # ── verdict ──
    spread = (
        np.nanmax(sigma_resolve) - np.nanmin(sigma_resolve)
        if np.isfinite(sigma_resolve).any()
        else float("nan")
    )
    # Best δ-schedule saving across S∈{2,3} (S=3 usually wins on attn FLOPs).
    best_su = max(s["attn_speedup"] for s in schedules.values())
    half_a_pass = bool(monotone and np.isfinite(spread) and spread > 0.1)
    half_b_pass = bool(np.isfinite(best_su) and best_su > 1.3)
    if half_a_pass and half_b_pass:
        verdict = (
            f"PASS: bands resolve in frequency order (σ_resolve monotone, spread "
            f"{spread:.2f}) AND the δ={args.delta} schedule predicts non-trivial "
            f"saving (best attn ×{best_su:.2f}). Autoregression dynamics hold → Phase 3."
        )
    elif half_a_pass:
        verdict = (
            f"WEAK: autoregression IS present (σ_resolve monotone, spread {spread:.2f}) "
            f"but the δ={args.delta} schedule's saving is thin (best attn ×{best_su:.2f}) "
            f"— t_ω is conservative, so the principled schedule transitions early. Sweep "
            f"--delta to match the hand-tuned σ≈0.5 knee; this is the regime where the "
            f"fine-tune (docs/proposal/spd_finetune_lora.md) earns its keep."
        )
    else:
        verdict = (
            f"FAIL: bands do NOT resolve in clean frequency order (σ_resolve "
            f"{[round(s, 2) for s in sigma_resolve]}, monotone={monotone}). The static "
            f"power law does not translate into trajectory autoregression on Anima — "
            f"early-low-res is not justified; lean on Spectrum/Turbo instead."
        )

    # ── artifacts ──
    run_dir = make_run_dir("spd", label=args.label)
    # CSV: per-step σ + per-band x_t / x0_pred power + resolved fraction
    hdr = ["step", "sigma"]
    for b in range(args.n_bands):
        hdr += [f"xt_b{b}", f"x0_b{b}", f"R_b{b}"]
    lines = [",".join(hdr)]
    for i in range(n_steps):
        row = [str(i), f"{sig_at_step[i]:.5f}"]
        for b in range(args.n_bands):
            row += [f"{xt_mean[i, b]:.6f}", f"{x0_mean[i, b]:.6f}", f"{R[i, b]:.4f}"]
        lines.append(",".join(row))
    (run_dir / "band_power.csv").write_text("\n".join(lines) + "\n")
    artifacts = ["band_power.csv"]

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))
        cmap = plt.get_cmap("viridis")
        for b in range(args.n_bands):
            axA.plot(
                sig_at_step,
                R[:, b],
                "o-",
                ms=3,
                color=cmap(b / max(1, args.n_bands - 1)),
                label=f"band {b}  k≈{band_centers[b]:.2f}",
            )
        for nm, sch in schedules.items():
            for ts in sch["t_star"]:
                axA.axvline(ts, ls="--", lw=1, color="0.4")
                axA.text(
                    ts, 1.02, f"{nm} t*={ts:.2f}", rotation=90, fontsize=7, va="bottom"
                )
        axA.axhline(thr, color="0.7", lw=0.8, ls=":")
        axA.invert_xaxis()  # trajectory runs σ:1→0
        axA.set_xlabel("σ (= t),  trajectory →")
        axA.set_ylabel("resolved fraction  R_band(σ) = P(σ)/P(σ→0)")
        axA.set_title("Half A — band resolution vs trajectory (x0_pred)")
        axA.legend(fontsize=7)
        axA.grid(alpha=0.3)

        axB.plot(
            k_grid, p_k, color="#1f4e8c", label="P_ω (real latents, ortho/unit-var)"
        )
        axB.set_yscale("log")
        axB.set_xlabel("k / k_Nyquist")
        axB.set_ylabel("P_ω")
        axB2 = axB.twinx()
        t_k = activation_time(p_k, args.delta)
        axB2.plot(k_grid, t_k, color="#c0392b", label=f"t_ω  (Prop 1, δ={args.delta})")
        axB2.set_ylabel("activation time  t_ω")
        for nm, sch in schedules.items():
            for s, ts in zip(sch["intermediate_scales"], sch["t_star"]):
                axB2.plot([s], [ts], "ks", ms=5)
                axB2.text(s, ts, f" {nm} s={s}", fontsize=7)
        axB.set_title("Half B — P_ω → t_ω → t*_i")
        lines_, labs_ = axB.get_legend_handles_labels()
        l2, la2 = axB2.get_legend_handles_labels()
        axB.legend(lines_ + l2, labs_ + la2, fontsize=7, loc="lower left")
        fig.tight_layout()
        fig.savefig(run_dir / "autoregression.png", dpi=130)
        artifacts.append("autoregression.png")
    except Exception as e:
        log.warning(f"plot skipped: {e}")

    metrics = {
        "resolution_hw": [args.height, args.width],
        "infer_steps": args.infer_steps,
        "flow_shift": args.flow_shift,
        "guidance_scale": args.guidance_scale,
        "n_runs": n_runs,
        "n_bands": args.n_bands,
        "band_centers_k": band_centers.tolist(),
        "resolve_threshold": thr,
        "sigma_resolve_per_band": sigma_resolve,
        "sigma_resolve_monotone": monotone,
        "sigma_resolve_spread": float(spread),
        "delta": args.delta,
        "base_tokens": int(base_tokens),
        "schedules": schedules,
        "half_a_pass": half_a_pass,
        "half_b_pass": half_b_pass,
        "verdict": verdict,
    }
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        artifacts=artifacts,
        device=device,
    )

    log.info("\n" + "=" * 70)
    log.info(
        f"  σ_resolve per band (k low→high, R≥{thr}): {[round(s, 3) for s in sigma_resolve]}"
    )
    log.info(f"  monotone={monotone}  spread={spread:.3f}")
    for nm, sch in schedules.items():
        log.info(
            f"  {nm}: t*={[round(t, 3) for t in sch['t_star']]}  "
            f"steps={sch['steps_per_stage']}  attn ×{sch['attn_speedup']:.2f}"
        )
    log.info(f"  {verdict}")
    log.info(f"  → {run_dir}  (open autoregression.png)")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
