"""Phase-0 offline target sanity for the SPD trajectory-adapter "teacher_lowres"
target (proposal.md Idea 1).

`scripts/distill_spd.py` (Case B) trains the SPD low-res prefix on the *analytic*
segment target: a straight line from the lowpassed clean latent ``x0_si`` toward a
noise field, ``v = ε_si − x0_si``. Idea 1 proposes instead supervising the prefix
toward the **DCT-lowpass projection of the frozen teacher's own dynamics** — so
the coarse prefix learns the full model's low-frequency reverse *decisions*
(``E[x0 | x_t]`` in the LL band) rather than the segment line to ground truth.

That is only worth wiring into the trainer if the two targets actually differ in
a way that carries information. This probe is the gate. It does **no training**
and changes **no pipeline code** — it loads the bare DiT, pulls cached
(latent, text) pairs, and at each σ in the stage-0 band compares, *at a shared
low-res input state*:

    analytic  : v_a = T_Φ(ε_full − x0_full)              → implied x0 = T_Φ(x0_full)   (true clean LL)
    teacher   : v_t = T_Φ( v_θ(x_full_t, σ, c) )         → implied x0 = T_Φ(x_full_t − σ·v_θ)

where ``T_Φ`` is the SPD DCT low-pass (``networks.spd.dct_lowpass_init``) to the
stage-0 grid. Both share the input ``x_low_t = T_Φ(x_full_t)`` (the orthonormal
DCT makes the lowpass of full-res white noise white unit-variance noise on the
low grid, so this input also matches the inference stage-0 entry).

The diagnostics, per σ, averaged over samples:

  * ``rel_x0``  = ‖x0_teacher_LL − x0_true_LL‖ / ‖x0_true_LL‖
                  — how far the teacher's denoised LL estimate sits from the true
                  clean LL the analytic target points at. Large at high σ
                  (teacher can't recover detail), should shrink toward 0 as σ→0.
  * ``rel_v``   = ‖v_t − v_a‖ / ‖v_a‖            — velocity-target magnitude gap.
  * ``cos_v``   = cos(v_t, v_a)                   — **direction** gap. This is the
                  load-bearing number: if cos_v ≈ 1 across the operative band the
                  teacher adds no new guidance and Idea 1 is a redundant auxiliary
                  → keep analytic, skip to Idea 2 (on-policy tail).

Gate (auto, but the call is ultimately a judgment on the curves):
  * NaN/Inf-free across all teacher forwards + projections.
  * Targets smooth in σ (small second-difference on the cos_v / rel curves).
  * Differ *meaningfully in direction* somewhere in the operative band
    (``cos_v`` drops below ~0.95 near the transition σ, not just blows up in
    magnitude at σ→1 where every denoiser is near the mean).

CFG note: the teacher velocity here is **conditional-only** (no negative branch),
matching the analytic target's pure-geometry framing and keeping the probe
self-contained (no text-encoder load for an uncond embedding). Whether the
prefix should be supervised toward the CFG-guided velocity is a Phase-1 knob;
see proposal.md "decisions before coding".

Usage:
  uv run python -m bench.spd.probe_teacher_lowres                       # 16 samples, s0=0.5, transition σ=0.5
  uv run python -m bench.spd.probe_teacher_lowres --num_samples 32 --s0 0.5 --transition_sigma 0.4
  uv run python -m bench.spd.probe_teacher_lowres --n_sigma 13 --seed 0
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bench._common import make_run_dir, write_result  # noqa: E402
from library.anima import weights as anima_utils  # noqa: E402
from library.datasets.distill import CachedDataset  # noqa: E402
from networks.spd import dct_lowpass_init  # noqa: E402

log = logging.getLogger("bench.spd.probe_teacher")
logging.basicConfig(level=logging.INFO, format="%(message)s")

DEFAULT_DIT = "models/diffusion_models/anima-base-v1.0.safetensors"
DEFAULT_DATA = "post_image_dataset/lora"


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative L2 gap ‖a − b‖ / ‖b‖ over a whole tensor (float64 accum)."""
    a = a.double()
    b = b.double()
    denom = b.norm().item()
    return float((a - b).norm().item() / (denom + 1e-12))


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two tensors flattened to a single vector."""
    a = a.double().flatten()
    b = b.double().flatten()
    return float(torch.dot(a, b).item() / (a.norm().item() * b.norm().item() + 1e-12))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dit", default=DEFAULT_DIT)
    ap.add_argument("--data_dir", default=DEFAULT_DATA)
    ap.add_argument("--attn_mode", default="flash")
    ap.add_argument("--num_samples", type=int, default=16)
    ap.add_argument(
        "--s0",
        type=float,
        default=0.5,
        help="Stage-0 resolution scale (the prefix grid).",
    )
    ap.add_argument(
        "--transition_sigma",
        type=float,
        default=0.5,
        help="Lower bound of the stage-0 band (handoff σ to full res).",
    )
    ap.add_argument(
        "--n_sigma",
        type=int,
        default=9,
        help="σ grid points across [transition_sigma, 1.0].",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--lora_weight",
        default=None,
        help="Optional analytic-trained SPD LoRA. When given, additionally measures "
        "what the trained adapter actually does at the low-res state vs the teacher "
        "target — the residual cos gap is the ceiling on what teacher_lowres could add.",
    )
    ap.add_argument("--label", default="teacher-lowres-phase0")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if not (0.0 < args.s0 < 1.0):
        raise SystemExit(f"--s0 must be in (0,1), got {args.s0}")
    if not (0.0 < args.transition_sigma < 1.0):
        raise SystemExit(
            f"--transition_sigma must be in (0,1), got {args.transition_sigma}"
        )

    device = torch.device(args.device)
    dtype = torch.bfloat16
    torch.manual_seed(args.seed)

    # σ grid: from the handoff σ up to (near) pure noise. Endpoint slightly below
    # 1.0 — at σ=1 the FM state is pure noise and the velocity-direction comparison
    # is dominated by the noise both targets share.
    sigmas = np.linspace(args.transition_sigma, 0.995, args.n_sigma).tolist()

    # --- Cached (latent, text) pairs, batch_size=1 (resolutions vary per sample) ---
    dataset = CachedDataset(args.data_dir, batch_size=1)
    if len(dataset.samples) == 0:
        raise SystemExit(f"No cached pairs under {args.data_dir}")
    n = min(args.num_samples, len(dataset.samples))
    log.info(
        "Probing %d cached samples (of %d) over σ=%s at s0=%.3f",
        n,
        len(dataset.samples),
        [round(s, 3) for s in sigmas],
        args.s0,
    )

    # --- Bare DiT (frozen teacher, no LoRA) ---
    log.info("Loading bare DiT (frozen teacher, no adapter) ...")
    model = anima_utils.load_anima_model(
        device,
        args.dit,
        attn_mode=args.attn_mode,
        split_attn=False,
        loading_device=device,
        dit_weight_dtype=dtype,
    )
    model.to(device)
    model.eval()
    patch = model.patch_spatial
    for p in model.parameters():
        p.requires_grad_(False)

    # --- Optional: toggleable analytic-trained SPD LoRA (the user's arm) ---
    # set_enabled(False) gives the bare teacher; set_enabled(True) gives the
    # trained adapter — both from one model, so all velocities live at the
    # same low-res input.
    network = None
    if args.lora_weight is not None:
        from safetensors.torch import load_file
        from networks import lora_anima

        lora_sd = load_file(args.lora_weight)
        lora_sd = {k: v for k, v in lora_sd.items() if k.startswith("lora_unet_")}
        network, weights_sd = lora_anima.create_network_from_weights(
            multiplier=1.0,
            file=args.lora_weight,
            ae=None,
            text_encoders=[],
            unet=model,
            weights_sd=lora_sd,
            for_inference=True,
        )
        network.apply_to([], model, apply_text_encoder=False, apply_unet=True)
        network.load_state_dict(weights_sd, strict=False)
        network.to(device, dtype=dtype)
        network.eval()
        log.info(
            "Loaded SPD LoRA arm: %s (%d modules)",
            args.lora_weight,
            len(network.unet_loras),
        )

    def _fwd(x5, sig, c, enabled=False):
        """forward_mini_train_dit at x5's own resolution, LoRA on/off."""
        if network is not None:
            network.set_enabled(enabled)
        t = x5.new_full((1,), float(sig), dtype=dtype)
        pad = torch.zeros(1, 1, x5.shape[-2], x5.shape[-1], dtype=dtype, device=device)
        ac = (
            torch.autocast("cuda", dtype=dtype)
            if device.type == "cuda"
            else torch.autocast("cpu", dtype=dtype, enabled=False)
        )
        with torch.no_grad(), ac:
            return model.forward_mini_train_dit(
                x5, t, c, padding_mask=pad, skip_pooled_text_proj=True
            ).float()

    # Independent generator for the per-(sample,σ) full-res noise draw.
    gen = torch.Generator(device=device).manual_seed(args.seed + 7919)

    # rows[sigma_idx] accumulates per-sample metric dicts.
    rows: list[list[dict]] = [[] for _ in sigmas]
    any_nan = False

    for si in range(n):
        _idx, latents, crossattn_emb, _pooled = dataset[si]
        # latents: (16, H, W) → (1, 16, 1, H, W)
        x0_full = latents.to(device, dtype).unsqueeze(0).unsqueeze(2)
        c = crossattn_emb.to(device, dtype).unsqueeze(0)

        # True clean LL — the analytic target's implied x0 (σ-independent).
        x0_true_low = dct_lowpass_init(x0_full, args.s0, patch).float()

        eps_full = torch.randn(
            x0_full.shape, generator=gen, device=device, dtype=torch.float32
        )

        for k, sig in enumerate(sigmas):
            x_full_t = ((1.0 - sig) * x0_full.float() + sig * eps_full).to(dtype)

            # Bare full-res teacher (LoRA off), then project to the stage-0 grid.
            v_teacher = _fwd(x_full_t, sig, c, enabled=False)
            x0_teacher_full = x_full_t.float() - sig * v_teacher
            # v_teacher is (B,C,1,H,W), matching dct_lowpass_init's expected layout.
            v_teacher_low = dct_lowpass_init(v_teacher, args.s0, patch).float()
            x0_teacher_low = dct_lowpass_init(x0_teacher_full, args.s0, patch).float()
            # analytic velocity at the shared low-res input = T_Φ(ε_full − x0_full)
            v_analytic_low = dct_lowpass_init(
                (eps_full - x0_full.float()), args.s0, patch
            ).float()

            bad = bool(
                torch.isnan(v_teacher).any()
                or torch.isinf(v_teacher).any()
                or torch.isnan(x0_teacher_low).any()
            )
            any_nan = any_nan or bad

            row = {
                "rel_x0": _rel(x0_teacher_low, x0_true_low),
                "rel_v": _rel(v_teacher_low, v_analytic_low),
                "cos_v": _cos(v_teacher_low, v_analytic_low),
                "nan": bad,
            }

            # --- LoRA arm: what the trained adapter actually does at low res ---
            if network is not None:
                x_low_t = dct_lowpass_init(
                    x_full_t, args.s0, patch
                )  # shared low-res input
                v_bare_low = _fwd(x_low_t, sig, c, enabled=False).float()
                v_lora_low = _fwd(x_low_t, sig, c, enabled=True).float()
                x0_bare_low = x_low_t.float() - sig * v_bare_low
                x0_lora_low = x_low_t.float() - sig * v_lora_low
                row.update(
                    {
                        # how much the adapter moves the low-res prediction at all
                        "adapter_shift_cos": _cos(v_lora_low, v_bare_low),
                        # adapter vs its analytic training target
                        "cos_lora_analytic": _cos(v_lora_low, v_analytic_low),
                        # adapter vs the teacher target (residual = 1 − this = ceiling for teacher_lowres)
                        "cos_lora_teacher": _cos(v_lora_low, v_teacher_low),
                        # does bare-DiT-at-low-res already match the teacher target?
                        "cos_bare_teacher": _cos(v_bare_low, v_teacher_low),
                        # x0 tracking of the true clean LL
                        "rel_x0_lora_true": _rel(x0_lora_low, x0_true_low),
                        "rel_x0_bare_true": _rel(x0_bare_low, x0_true_low),
                    }
                )

            rows[k].append(row)
        if (si + 1) % 4 == 0 or si == n - 1:
            log.info("  ... %d/%d samples", si + 1, n)

    # --- Aggregate per-σ ---
    def _mean(k: int, key: str) -> float:
        return float(np.mean([r[key] for r in rows[k]]))

    lora_keys = [
        "adapter_shift_cos",
        "cos_lora_analytic",
        "cos_lora_teacher",
        "cos_bare_teacher",
        "rel_x0_lora_true",
        "rel_x0_bare_true",
    ]
    has_lora = network is not None
    per_sigma = []
    for k, sig in enumerate(sigmas):
        entry = {
            "sigma": float(sig),
            "rel_x0": _mean(k, "rel_x0"),
            "rel_v": _mean(k, "rel_v"),
            "cos_v": _mean(k, "cos_v"),
        }
        if has_lora:
            for key in lora_keys:
                entry[key] = _mean(k, key)
            # the ceiling on what a teacher target could add over analytic, here
            entry["teacher_residual"] = 1.0 - entry["cos_lora_teacher"]
        per_sigma.append(entry)

    cos_curve = np.array([p["cos_v"] for p in per_sigma])
    relx0_curve = np.array([p["rel_x0"] for p in per_sigma])
    # Smoothness: mean abs second difference of the cos_v curve (jumps → unstable).
    cos_secdiff = (
        float(np.mean(np.abs(np.diff(cos_curve, 2)))) if len(cos_curve) >= 3 else 0.0
    )
    relx0_secdiff = (
        float(np.mean(np.abs(np.diff(relx0_curve, 2))))
        if len(relx0_curve) >= 3
        else 0.0
    )

    # Operative band = the lower third of σ (nearest the handoff), where the
    # prefix's final decision actually matters. Direction divergence there is
    # what makes the teacher target non-redundant.
    n_band = max(1, len(per_sigma) // 3)
    band = per_sigma[:n_band]
    band_min_cos = min(p["cos_v"] for p in band)
    band_mean_cos = float(np.mean([p["cos_v"] for p in band]))

    SMOOTH_TOL = 0.03  # cos second-diff above this = bumpy/unstable curve
    COS_MEANINGFUL = 0.95  # cos below this in the band = genuinely different direction

    smooth = cos_secdiff <= SMOOTH_TOL and relx0_secdiff <= 0.10
    meaningful = band_min_cos < COS_MEANINGFUL

    if any_nan:
        verdict = "FAIL_NAN: teacher forward or projection produced NaN/Inf — target unusable."
    elif not smooth:
        verdict = (
            f"CAUTION_BUMPY: cos_v curve is not smooth in σ (2nd-diff {cos_secdiff:.4f} "
            f"> {SMOOTH_TOL}); the projected teacher target is unstable across the band. "
            "Inspect curves before training."
        )
    elif meaningful:
        verdict = (
            f"PASS: projected teacher target is NaN-free, smooth, and differs in "
            f"DIRECTION from analytic in the operative band (min cos_v {band_min_cos:.3f} "
            f"< {COS_MEANINGFUL} near σ={args.transition_sigma:.2f}). Idea 1 is not "
            "redundant — proceed to Phase 1 (low-res prefix overfit, judge on LL-x0 tracking)."
        )
    else:
        verdict = (
            f"WEAK_REDUNDANT: target is clean+smooth but cos_v stays high in the operative "
            f"band (min {band_min_cos:.3f} ≥ {COS_MEANINGFUL}); the teacher gives ~the same "
            "direction as the analytic line where the prefix operates. Keep the cheaper "
            "analytic target and skip to Idea 2 (on-policy handoff tail)."
        )

    # --- LoRA-arm verdict: the trained adapter's residual to the teacher target ---
    lora_verdict = None
    lora_summary = {}
    if has_lora:
        band_l = per_sigma[:n_band]
        max_residual = max(p["teacher_residual"] for p in band_l)
        mean_shift = float(np.mean([p["adapter_shift_cos"] for p in band_l]))
        # did training toward analytic move the adapter *toward* the teacher?
        d_to_teacher = float(
            np.mean([p["cos_lora_teacher"] - p["cos_bare_teacher"] for p in band_l])
        )
        lora_summary = {
            "band_max_teacher_residual": max_residual,
            "band_mean_adapter_shift_cos": mean_shift,
            "band_mean_lora_minus_bare_to_teacher": d_to_teacher,
        }
        RESIDUAL_TOL = 0.05  # < this in the band = no room for a teacher target to help
        if max_residual < RESIDUAL_TOL:
            lora_verdict = (
                f"LORA_CONFIRMS_REDUNDANT: the analytic-trained adapter already lands on the "
                f"teacher's LL decision in the operative band (max residual 1−cos {max_residual:.3f} "
                f"< {RESIDUAL_TOL}). teacher_lowres has no headroom to add — dead twice over."
            )
        else:
            lora_verdict = (
                f"LORA_RESIDUAL_PRESENT: analytic-trained adapter leaves a teacher-shaped gap "
                f"(max residual {max_residual:.3f} ≥ {RESIDUAL_TOL}) in the operative band. "
                f"BUT note the teacher target is lowpass(E_full[·]) ≠ true low-res dynamics — a "
                f"gap may be teacher BIAS, not signal. Δ(lora−bare → teacher)={d_to_teacher:+.3f} "
                f"(adapter moved toward teacher if >0). Judge before trusting teacher_lowres."
            )

    metrics = {
        "s0": args.s0,
        "transition_sigma": args.transition_sigma,
        "n_samples": n,
        "sigmas": sigmas,
        "per_sigma": per_sigma,
        "cos_v_secdiff": cos_secdiff,
        "rel_x0_secdiff": relx0_secdiff,
        "operative_band_n": n_band,
        "operative_band_min_cos_v": band_min_cos,
        "operative_band_mean_cos_v": band_mean_cos,
        "any_nan_inf": any_nan,
        "smooth": smooth,
        "meaningful_direction_gap": meaningful,
        "cfg": "conditional-only (no negative branch — Phase-1 knob)",
        "verdict": verdict,
        "lora_weight": args.lora_weight,
        "lora_arm": lora_summary or None,
        "lora_verdict": lora_verdict,
    }

    run_dir = make_run_dir("spd", label=args.label)

    # Optional curve plot (guarded — probe stays useful headless).
    artifacts: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sig_arr = [p["sigma"] for p in per_sigma]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(sig_arr, cos_curve, "o-", label="cos(v_teacher, v_analytic) [LL]")
        ax[0].axhline(
            COS_MEANINGFUL,
            ls="--",
            c="r",
            lw=0.8,
            label=f"meaningful < {COS_MEANINGFUL}",
        )
        ax[0].axvspan(
            args.transition_sigma,
            band[-1]["sigma"],
            alpha=0.1,
            color="green",
            label="operative band",
        )
        ax[0].set_xlabel("σ")
        ax[0].set_ylabel("cosine")
        ax[0].set_title("velocity direction gap")
        ax[0].legend(fontsize=7)
        ax[0].grid(alpha=0.3)
        ax[1].plot(
            sig_arr,
            relx0_curve,
            "o-",
            c="purple",
            label="rel ‖x0_teacher−x0_true‖ [LL]",
        )
        ax[1].plot(
            sig_arr,
            [p["rel_v"] for p in per_sigma],
            "s-",
            c="orange",
            label="rel ‖v gap‖",
        )
        ax[1].set_xlabel("σ")
        ax[1].set_ylabel("relative L2")
        ax[1].set_title("magnitude gaps")
        ax[1].legend(fontsize=7)
        ax[1].grid(alpha=0.3)
        fig.suptitle(f"SPD teacher_lowres Phase-0  (s0={args.s0}, n={n})")
        fig.tight_layout()
        fig.savefig(run_dir / "target_gap_curves.png", dpi=110)
        plt.close(fig)
        artifacts.append("target_gap_curves.png")
    except Exception as e:  # noqa: BLE001
        log.warning("plot skipped (%s)", e)

    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        artifacts=artifacts,
        device=device,
    )

    log.info("\n" + "=" * 72)
    if has_lora:
        log.info("  σ      cos_v   |  lora→teach  bare→teach  resid  | adpt_shift")
        for p in per_sigma:
            log.info(
                "  %.3f  %6.3f  |   %6.3f      %6.3f    %6.3f |   %6.3f",
                p["sigma"],
                p["cos_v"],
                p["cos_lora_teacher"],
                p["cos_bare_teacher"],
                p["teacher_residual"],
                p["adapter_shift_cos"],
            )
    else:
        log.info("  σ      cos_v    rel_x0   rel_v")
        for p in per_sigma:
            log.info(
                "  %.3f  %6.3f   %6.3f   %6.3f",
                p["sigma"],
                p["cos_v"],
                p["rel_x0"],
                p["rel_v"],
            )
    log.info("-" * 72)
    log.info("  cos_v 2nd-diff (smoothness): %.4f  (tol %.2f)", cos_secdiff, SMOOTH_TOL)
    log.info(
        "  operative-band min cos_v:    %.3f  (meaningful < %.2f)",
        band_min_cos,
        COS_MEANINGFUL,
    )
    log.info("  %s", verdict)
    if lora_verdict is not None:
        log.info("  %s", lora_verdict)
    log.info("  → %s", run_dir)
    log.info("=" * 72)


if __name__ == "__main__":
    main()
