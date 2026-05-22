"""Phase-0 gate for the SPD trajectory-adapter "on-policy handoff tail"
(proposal.md Idea 2 — the main lever after Idea 1 died as redundant).

`scripts/distill_spd.py` (Case B) trains the full-res tail on the **analytic**
stage entry that `networks.spd.spd_stage_target` builds: it spectrally expands
``(1−t_trans)·x0_lo + t_trans·ε`` — a straight line from the *true* clean LL to
noise. At inference the tail instead sees the state the *trained (or bare)
prefix actually rolls to* from pure noise, then expanded. Idea 2 is to train the
tail on those on-policy states (DAgger-style exposure-bias correction).

That is only worth wiring into the trainer if it clears two *separate* bars:
(1) the analytic-trained tail is meaningfully worse at the on-policy states it
actually sees than at the analytic states it trained on (HEADROOM), and (2) the
prefix's handoff state drifts *further* from the analytic one as the transition
σ drops (EXPOSURE BIAS GROWS — the distinctive Idea-2 "aggressive schedule"
prediction, the inverse of the Idea-1 failure where the teacher got *more*
redundant as σ dropped). This probe is the gate. It does **no training** and
changes **no pipeline code** — it loads the bare DiT (optionally + an
analytic-trained SPD LoRA), rolls the prefix with the shipped
``spd_rollout_to_stage`` (bit-for-bit the sampler's Euler+expand), and compares,
at the *same* aligned σ̃, the tail's behaviour at the on-policy vs analytic entry.

Per transition σ, averaged over samples (tail forward = deployment setting:
LoRA-on if ``--lora_weight`` given, else bare):

  * ``rel_xtilde`` / ``cos_xtilde`` — how far the on-policy expanded state x̃_on
    sits from the analytic x̃_an the trainer assumes (built at the *same* σ_cross
    + paired HF fill, so the only difference is rolled-state vs idealized-state).
    **This IS the exposure bias**; the gate's "grows as σ↓" test reads it, not
    the residual gap.
  * ``resid_an`` = ‖v_θ(x̃_an) − (x̃_an−x0)/σ̃‖ / ‖·‖  — tail velocity residual at
    the *training* (analytic) state. With the LoRA arm this is what the analytic
    objective drives toward 0.
  * ``resid_on`` = same at the *inference* (on-policy) state. **The headroom
    number is ``ratio = resid_on/resid_an``**: ≈1 ⇒ tail already generalizes →
    redundant; ≥1.15 across knees ⇒ the analytic objective leaves the visited
    states under-served. (The *gap* resid_on−resid_an is reported but NOT used to
    test growth — it is confounded by the analytic baseline's own off-knee
    collapse, so it can shrink even as absolute on-policy error rises.)
  * ``rel_x0_on`` / ``rel_x0_an`` — implied-clean recovery ‖(x̃−σ̃v)−x0‖/‖x0‖,
    the interpretable read of the same headroom.

CFG note: conditional-only (no negative branch), matching the analytic target's
pure-geometry framing and the trainer's ``skip_pooled_text_proj=True`` forward.
CFG-on rollout is a follow-up only if the seam proves CFG-dependent (the
compose-report finding is that the seam reorientation is the grid-change
attention re-mix, ~CFG-independent).

Usage:
  uv run python -m bench.spd.probe_onpolicy_handoff
  uv run python -m bench.spd.probe_onpolicy_handoff --transition_sigmas 0.5 0.4 0.35
  uv run python -m bench.spd.probe_onpolicy_handoff --lora_weight output/ckpt/anima_spd.safetensors
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
from networks.spd import (  # noqa: E402
    dct_lowpass_init,
    spd_rollout_to_stage,
    spectral_expand,
)

log = logging.getLogger("bench.spd.probe_onpolicy")
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
        "--s0", type=float, default=0.5, help="Stage-0 resolution scale (prefix grid)."
    )
    ap.add_argument(
        "--transition_sigmas",
        type=float,
        nargs="+",
        default=[0.5, 0.4, 0.35],
        help="Handoff σ values to sweep (descending = more aggressive). The gate "
        "checks the exposure bias (state divergence) grows as these drop.",
    )
    ap.add_argument(
        "--infer_steps",
        type=int,
        default=24,
        help="Euler steps for the prefix rollout (matches the deployed sampler).",
    )
    ap.add_argument("--flow_shift", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--lora_weight",
        default=None,
        help="Optional analytic-trained SPD LoRA. When given, both the rollout and "
        "the tail forward use the adapter (deployment setting) — the sharpest test "
        "of whether analytic training already covers the on-policy states.",
    )
    ap.add_argument("--label", default="onpolicy-handoff-phase0")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if not (0.0 < args.s0 < 1.0):
        raise SystemExit(f"--s0 must be in (0,1), got {args.s0}")
    for ts in args.transition_sigmas:
        if not (0.0 < ts < 1.0):
            raise SystemExit(f"--transition_sigmas must be in (0,1), got {ts}")
    # Descending order so "bias grows as σ drops" reads left→right; sort to be safe.
    trans_list = sorted(args.transition_sigmas, reverse=True)

    device = torch.device(args.device)
    dtype = torch.bfloat16
    torch.manual_seed(args.seed)

    # --- Cached (latent, text) pairs, batch_size=1 (resolutions vary per sample) ---
    dataset = CachedDataset(args.data_dir, batch_size=1)
    if len(dataset.samples) == 0:
        raise SystemExit(f"No cached pairs under {args.data_dir}")
    n = min(args.num_samples, len(dataset.samples))
    log.info(
        "Probing %d cached samples (of %d) over transition σ=%s at s0=%.3f, %d steps",
        n,
        len(dataset.samples),
        trans_list,
        args.s0,
        args.infer_steps,
    )

    # --- Bare DiT (frozen) ---
    log.info("Loading DiT (frozen) ...")
    model = anima_utils.load_anima_model(
        device,
        args.dit,
        attn_mode=args.attn_mode,
        loading_device=device,
        dit_weight_dtype=dtype,
    )
    model.to(device)
    model.eval()
    patch = model.patch_spatial
    for p in model.parameters():
        p.requires_grad_(False)

    # --- Optional analytic-trained SPD LoRA (deployment adapter) ---
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
            "Loaded SPD LoRA: %s (%d modules) — rollout + tail run with adapter on",
            args.lora_weight,
            len(network.unet_loras),
        )

    deploy_enabled = network is not None  # rollout + tail use the adapter if present

    def _fwd(x5, sig, c, enabled):
        """forward_mini_train_dit at x5's own resolution, LoRA on/off."""
        if network is not None:
            network.set_enabled(enabled)
        t = x5.new_full((x5.shape[0],), float(sig), dtype=dtype)
        pad = torch.zeros(
            x5.shape[0], 1, x5.shape[-2], x5.shape[-1], dtype=dtype, device=device
        )
        ac = (
            torch.autocast("cuda", dtype=dtype)
            if device.type == "cuda"
            else torch.autocast("cpu", dtype=dtype, enabled=False)
        )
        with torch.no_grad(), ac:
            return model.forward_mini_train_dit(
                x5, t, c, padding_mask=pad, skip_pooled_text_proj=True
            ).float()

    stages = [args.s0, 1.0]
    # Noise sharing is the load-bearing control. The on-policy rollout and the
    # analytic entry start from the *same* low-res field ``eps_lo`` (the rollout
    # init's lowpass), and the spectral-expand HF fill is paired via gen-state
    # save/restore. A *perfect* analytic prefix then rolls eps_lo exactly onto
    # ``x_entry_an``, so the entire on-policy↔analytic gap is the prefix's
    # accumulated drift — the exposure bias — not a noise-draw difference. With
    # the bare model (no --lora_weight) the prefix is *not* anchored to x0 (its
    # training never pointed it at this latent), so the bare arm is a NaN/smoke
    # check only; the real gate needs the analytic adapter.
    gen_roll = torch.Generator(device=device).manual_seed(args.seed + 7919)
    gen_hf = torch.Generator(device=device).manual_seed(args.seed + 104729)

    rows: dict[float, list[dict]] = {ts: [] for ts in trans_list}
    any_nan = False

    for si in range(n):
        _idx, latents, crossattn_emb, _pooled = dataset[si]
        x0_full = latents.to(device, dtype).unsqueeze(0).unsqueeze(2)  # (1,16,1,H,W)
        c = crossattn_emb.to(device, dtype).unsqueeze(0)
        H_full, W_full = int(x0_full.shape[-2]), int(x0_full.shape[-1])
        x0_lo = dct_lowpass_init(x0_full, args.s0, patch).float()  # true clean LL

        for ts in trans_list:
            transition_sigmas = [ts]

            # Shared init field: full-res white noise, lowpassed to the prefix
            # grid. The rollout starts from eps_full (lowpassed internally); the
            # analytic entry uses that same lowpass eps_lo (see noise-sharing note).
            eps_full = torch.randn(
                x0_full.shape, generator=gen_roll, device=device, dtype=torch.float32
            ).to(dtype)
            eps_lo = dct_lowpass_init(eps_full, args.s0, patch).float()

            # --- On-policy prefix rollout (deployment setting) → x̃_on at σ̃ ---
            x_entry_on, sig_cross, scale_lo = spd_rollout_to_stage(
                lambda x5, sig: _fwd(x5, sig, c, enabled=deploy_enabled),
                eps_full,
                stages,
                transition_sigmas,
                infer_steps=args.infer_steps,
                flow_shift=args.flow_shift,
                patch=patch,
                gen=gen_roll,
                stop_stage=1,
            )
            hf_state = gen_hf.get_state()  # pair the HF fill across both expansions
            x_tilde_on, sig_tilde = spectral_expand(
                x_entry_on, sig_cross, scale_lo, 1.0, H_full, W_full, patch, gen_hf
            )

            # --- Analytic entry at the *same* σ_cross + same HF fill → x̃_an ---
            x_entry_an = ((1.0 - sig_cross) * x0_lo + sig_cross * eps_lo).to(dtype)
            gen_hf.set_state(hf_state)
            x_tilde_an, sig_tilde_an = spectral_expand(
                x_entry_an, sig_cross, scale_lo, 1.0, H_full, W_full, patch, gen_hf
            )
            assert abs(sig_tilde - sig_tilde_an) < 1e-5, (sig_tilde, sig_tilde_an)

            # --- Tail forwards + velocity residuals toward the true clean latent ---
            x0f = x0_full.float()
            v_tgt_on = (x_tilde_on.float() - x0f) / sig_tilde
            v_tgt_an = (x_tilde_an.float() - x0f) / sig_tilde
            v_on = _fwd(x_tilde_on, sig_tilde, c, enabled=deploy_enabled)
            v_an = _fwd(x_tilde_an, sig_tilde, c, enabled=deploy_enabled)

            x0hat_on = x_tilde_on.float() - sig_tilde * v_on
            x0hat_an = x_tilde_an.float() - sig_tilde * v_an

            bad = bool(
                torch.isnan(x_tilde_on).any()
                or torch.isnan(v_on).any()
                or torch.isinf(v_on).any()
            )
            any_nan = any_nan or bad

            resid_on = _rel(v_on, v_tgt_on)
            resid_an = _rel(v_an, v_tgt_an)
            row = {
                "sigma_tilde": float(sig_tilde),
                "sigma_cross": float(sig_cross),
                "rel_xtilde": _rel(x_tilde_on, x_tilde_an),
                "cos_xtilde": _cos(x_tilde_on, x_tilde_an),
                "resid_on": resid_on,
                "resid_an": resid_an,
                "gap": resid_on - resid_an,
                "ratio": resid_on / (resid_an + 1e-9),
                "rel_x0_on": _rel(x0hat_on, x0f),
                "rel_x0_an": _rel(x0hat_an, x0f),
                "nan": bad,
            }

            # Bare-tail residual at the on-policy state: how much the analytic
            # adapter helped on-policy (only meaningful when a LoRA is loaded).
            if network is not None:
                v_on_bare = _fwd(x_tilde_on, sig_tilde, c, enabled=False)
                row["resid_on_bare"] = _rel(v_on_bare, v_tgt_on)

            rows[ts].append(row)
        if (si + 1) % 4 == 0 or si == n - 1:
            log.info("  ... %d/%d samples", si + 1, n)

    # --- Aggregate per transition σ ---
    has_lora = network is not None

    def _mean(ts: float, key: str) -> float:
        return float(np.mean([r[key] for r in rows[ts]]))

    keys = [
        "sigma_tilde",
        "rel_xtilde",
        "cos_xtilde",
        "resid_on",
        "resid_an",
        "gap",
        "ratio",
        "rel_x0_on",
        "rel_x0_an",
    ]
    if has_lora:
        keys.append("resid_on_bare")
    per_sigma = []
    for ts in trans_list:
        entry = {"transition_sigma": float(ts)}
        for k in keys:
            entry[k] = _mean(ts, k)
        per_sigma.append(entry)

    # Gate. trans_list is descending, so per_sigma[0] is the *safest* knee and
    # per_sigma[-1] the *most aggressive* — Idea 2's payoff regime.
    #
    # Two orthogonal questions, deliberately *not* conflated:
    #
    #   (1) HEADROOM — is the analytic-trained tail meaningfully worse at the
    #       on-policy states it actually sees than at the analytic states it
    #       trained on? Measured by the velocity-residual ratio resid_on/resid_an.
    #       If ratio ≈ 1 everywhere the tail already generalizes → redundant (Idea
    #       1's fate). Note the *gap* resid_on−resid_an is NOT used for the "grows"
    #       test: it is confounded by the analytic baseline's own off-knee collapse
    #       (resid_an balloons away from the trained knee), so a shrinking gap can
    #       coexist with worsening absolute on-policy error.
    #
    #   (2) EXPOSURE BIAS GROWS — does the prefix's handoff state drift *further*
    #       from the analytic state as the knee drops? This is the proposal's
    #       actual "aggressive schedule" prediction, and it lives in the STATE
    #       divergence (rel_xtilde ↑ / cos_xtilde ↓), the literal exposure bias —
    #       not in the residual gap.
    safe = per_sigma[0]
    aggr = per_sigma[-1]
    ratio_aggr = aggr["ratio"]
    ratio_min = min(p["ratio"] for p in per_sigma)
    # State divergence worsens monotonically toward the aggressive knee.
    bias_grows = (
        aggr["rel_xtilde"] > safe["rel_xtilde"] + 1e-3
        and aggr["cos_xtilde"] < safe["cos_xtilde"] - 1e-3
    )

    RATIO_TOL = 1.15  # on-policy residual ≥15% above analytic = real headroom
    headroom = ratio_min >= RATIO_TOL  # consistent across the swept knees

    if any_nan:
        verdict = "FAIL_NAN: rollout or tail forward produced NaN/Inf — unusable."
    elif not has_lora:
        verdict = (
            "SMOKE_ONLY: ran bare (no --lora_weight). The bare prefix is not anchored to "
            "x0 (its training never pointed it at this latent), so the on-policy↔analytic "
            f"gap here (aggressive ratio {ratio_aggr:.2f}×) is dominated by free-rollout "
            "drift, not the exposure bias of a deployed adapter. Path is NaN-free and runs; "
            "re-run with --lora_weight output/ckpt/anima_spd.safetensors for the real gate."
        )
    elif headroom and bias_grows:
        verdict = (
            "PASS: real, consistent on-policy headroom AND growing exposure bias. The "
            f"analytic-trained tail is {ratio_min:.2f}–{max(p['ratio'] for p in per_sigma):.2f}× "
            "worse at on-policy states than at the analytic states it trained on (every knee), "
            f"and the handoff-state drift grows as the knee drops (rel_x̃ {safe['rel_xtilde']:.2f}"
            f"→{aggr['rel_xtilde']:.2f}, cos_x̃ {safe['cos_xtilde']:.2f}→{aggr['cos_xtilde']:.2f}). "
            "Inverse of Idea-1's redundancy. Wire --on_policy_ratio (two-pass tail) in "
            "distill_spd.py; judge Phase-2 on perceptual quality, not resid (FM-MSE doesn't "
            "track quality on Anima)."
        )
    elif headroom and not bias_grows:
        verdict = (
            f"WEAK_FLAT: real on-policy headroom (ratio ≥ {ratio_min:.2f}× across knees) but the "
            f"handoff-state drift does NOT grow as σ drops (rel_x̃ {safe['rel_xtilde']:.2f}"
            f"→{aggr['rel_xtilde']:.2f}). On-policy tail should help at a fixed knee, but the "
            "'push the schedule lower' story is unproven — judge Phase-2 at the trained knee."
        )
    else:
        verdict = (
            f"WEAK_REDUNDANT: tail residual at on-policy states ≈ analytic states (min ratio "
            f"{ratio_min:.2f}× < {RATIO_TOL}); the tail already generalizes from analytic to "
            "on-policy entries. On-policy training is redundant like Idea 1 — keep the cheaper "
            "analytic loss."
        )

    metrics = {
        "s0": args.s0,
        "stages": stages,
        "transition_sigmas_swept": trans_list,
        "infer_steps": args.infer_steps,
        "flow_shift": args.flow_shift,
        "n_samples": n,
        "per_sigma": per_sigma,
        "safe_knee": safe,
        "aggressive_knee": aggr,
        "headroom": headroom,
        "ratio_min": ratio_min,
        "exposure_bias_grows": bias_grows,
        "any_nan_inf": any_nan,
        "deployment_setting": "lora-on" if has_lora else "bare",
        "cfg": "conditional-only (no negative branch)",
        "lora_weight": args.lora_weight,
        "verdict": verdict,
    }

    run_dir = make_run_dir("spd", label=args.label)

    artifacts: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ts_arr = [p["transition_sigma"] for p in per_sigma]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(
            ts_arr,
            [p["resid_on"] for p in per_sigma],
            "o-",
            label="resid_on (inference state)",
        )
        ax[0].plot(
            ts_arr,
            [p["resid_an"] for p in per_sigma],
            "s-",
            label="resid_an (training state)",
        )
        if has_lora:
            ax[0].plot(
                ts_arr,
                [p["resid_on_bare"] for p in per_sigma],
                "^--",
                c="grey",
                label="resid_on (bare tail)",
            )
        ax[0].set_xlabel("transition σ (← more aggressive)")
        ax[0].set_ylabel("rel. velocity residual to x0")
        ax[0].set_title("tail residual: on-policy vs analytic entry")
        ax[0].invert_xaxis()
        ax[0].legend(fontsize=7)
        ax[0].grid(alpha=0.3)
        ax[1].plot(
            ts_arr,
            [p["rel_xtilde"] for p in per_sigma],
            "d-",
            c="teal",
            label="rel ‖x̃_on − x̃_an‖ (exposure bias)",
        )
        ax[1].plot(
            ts_arr,
            [p["ratio"] for p in per_sigma],
            "o-",
            c="crimson",
            label="resid_on / resid_an (headroom)",
        )
        ax[1].axhline(
            RATIO_TOL, ls="--", c="r", lw=0.8, label=f"headroom ≥ {RATIO_TOL}"
        )
        ax[1].set_xlabel("transition σ (← more aggressive)")
        ax[1].set_ylabel("bias distance / residual ratio")
        ax[1].set_title("exposure bias grows as σ↓; headroom at every knee")
        ax[1].invert_xaxis()
        ax[1].legend(fontsize=7)
        ax[1].grid(alpha=0.3)
        fig.suptitle(
            f"SPD on-policy handoff Phase-0  (s0={args.s0}, n={n}, "
            f"{'LoRA' if has_lora else 'bare'})"
        )
        fig.tight_layout()
        fig.savefig(run_dir / "handoff_gap_curves.png", dpi=110)
        plt.close(fig)
        artifacts.append("handoff_gap_curves.png")
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

    log.info("\n" + "=" * 78)
    hdr = (
        "  trans_σ  σ̃     rel_x̃  cos_x̃  resid_on resid_an   gap   ratio  rx0_on rx0_an"
    )
    log.info(hdr)
    for p in per_sigma:
        log.info(
            "  %6.3f  %5.3f  %5.3f  %5.3f   %6.3f   %6.3f  %6.3f %5.2f  %5.3f %5.3f",
            p["transition_sigma"],
            p["sigma_tilde"],
            p["rel_xtilde"],
            p["cos_xtilde"],
            p["resid_on"],
            p["resid_an"],
            p["gap"],
            p["ratio"],
            p["rel_x0_on"],
            p["rel_x0_an"],
        )
    log.info("-" * 78)
    log.info("  headroom (min ratio %.2f× ≥ %.2f): %s", ratio_min, RATIO_TOL, headroom)
    log.info(
        "  exposure bias grows as σ↓: %s  (rel_x̃ %.3f→%.3f, cos_x̃ %.3f→%.3f)",
        bias_grows,
        safe["rel_xtilde"],
        aggr["rel_xtilde"],
        safe["cos_xtilde"],
        aggr["cos_xtilde"],
    )
    log.info("  %s", verdict)
    log.info("  → %s", run_dir)
    log.info("=" * 78)


if __name__ == "__main__":
    main()
