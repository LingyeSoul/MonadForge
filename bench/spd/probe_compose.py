"""Phase-0(0) arithmetic gate for the SPD ∘ Spectrum composition.

See ``docs/proposal/spd_spectrum_compose.md`` — *Open Question 1*: before any
GPU probe, a back-of-envelope FLOP estimate of risk **R2** decides whether the
composition can multiply at all. *"If the arithmetic says no, this proposal is a
paper exercise and we stop here."* This script is that gate. No model, no GPU —
it consumes the already-measured per-band ``σ_resolve`` curve (``bench/spd/``
Phase 1) plus the σ schedule and tallies a block-FLOP proxy for five denoisers.

The load-bearing R2 fact (sharper than the proposal states it): to refresh the
**HF coefficients of the block-output feature** you must *run the blocks* — the
feature is the blocks' output (``spectrum.py`` ``_capture_pre_hook``). There is
no partial forward that yields only HF. So a band-aligned step can skip blocks
**only when every band is settled**, i.e. σ < min(σ_resolve) (the top band). For
σ above that, some band is still evolving → full block forward. Meanwhile
Spectrum-alone forecasts the *whole* feature (LL+HF together) through the smooth
middle and tolerates the HF approximation — which is exactly how it earns its
block-skips. So the band-aligned variant is *strictly more conservative* than
Spectrum-alone, and the question this script answers numerically is whether the
SPD token-saving on the early stage buys back more than that gives up.

Cost proxy: per step, blocks cost ∝ tokens, with tokens(stage) = scale²·N_tok.
attention ∝ tokens², FFN ∝ tokens. A *cached* (block-skipped) step costs ~0
blocks (Spectrum's fast path runs only t_embedder+final_layer+unpatchify). The
proxy counts block compute only — it is a relative-speedup estimate, not a wall
clock.

Five denoisers tallied (vs the full-res all-forward baseline):
  * baseline          — N steps, scale 1.0, all forward.
  * spectrum_alone     — scale 1.0, growing-window skip (faithful to spectrum.py).
  * spd_alone          — per-stage resolution, all forward.
  * naive_compose      — SPD resolutions + Spectrum skip, forecaster RESET at the
                         handoff (re-warm = warmup_steps forced forwards after it).
  * banded_compose     — SPD resolutions + skip only where σ < σ_resolve(top band)
                         (and outside the forced tail): the principled variant.

Verdict: PROCEED to Phase-0(a)/(b) iff banded_compose speedup > spectrum_alone;
else R2 is confirmed (paper exercise on the *speed* axis) — stop or pivot to the
quality reframing.

This file also hosts the two GPU probes the arith gate green-lights:
  * ``--mode gpu``        — Phase-0(a) naive-compose eyeball (renders the four
                            denoisers per prompt×seed, montage for the seam call).
  * ``--mode continuity`` — Phase-0(b) feature LL-DCT continuity, the *real*
                            precondition: does the captured ``final_layer`` feature's
                            LL band flow through the SPD handoff (so the band-aligned
                            forecaster has signal), or does it jump at the seam?
  * ``--mode frontier``   — seam-continuity vs detail sweep on the frozen DiT. Phase-0(b)
                            found the seam reorients at ×4–5.6 for the paper γ=1 fill;
                            this asks whether *any* (transition σ, HF-injection γ) cell
                            already clears the seam gate with detail retained — i.e.
                            whether a trained "opposite" LoRA (a loss_seam regularizer)
                            has a frozen operating point to anchor to, or must shift the
                            whole Pareto frontier. Step 1 of that investigation.
The GPU modes default to sampling 5 real captions from ``image_dataset/`` so the seam is
probed on the production prompt distribution, not one hand-written caption.

Usage:
  uv run python -m bench.spd.probe_compose
  uv run python -m bench.spd.probe_compose --from_result bench/spd/results/<run>/result.json
  uv run python -m bench.spd.probe_compose --spd_stages 0.5 1.0 --spd_transition_sigmas 0.7
  uv run python -m bench.spd.probe_compose --infer_steps 28 --flow_shift 1.0
  uv run python -m bench.spd.probe_compose --mode gpu --n_prompts 5
  uv run python -m bench.spd.probe_compose --mode continuity --n_prompts 5
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
from pathlib import Path

from bench._common import make_run_dir, write_result

log = logging.getLogger("bench.spd.probe_compose")
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Phase-1 measured σ_resolve (bench/spd/plan.md, 2026-05-21): the σ below which a
# radial band's power stays ≥80% of its final value. 6 bands, centers (b+0.5)/6.
DEFAULT_SIGMA_RESOLVE = [1.00, 0.75, 0.54, 0.39, 0.32, 0.29]


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001 — arith mode runs without torch
        return False


def build_sigmas(infer_steps: int, flow_shift: float) -> list[float]:
    """σ schedule, identical to library.inference.sampling.get_timesteps_sigmas
    (shift-warped linspace 1→0). Falls back to inline math if torch is absent."""
    try:
        import torch

        from library.inference.sampling import get_timesteps_sigmas

        _, sigmas = get_timesteps_sigmas(infer_steps, flow_shift, torch.device("cpu"))
        return [float(s) for s in sigmas]
    except Exception:  # noqa: BLE001 — keep the gate runnable with no torch
        out = []
        for k in range(infer_steps + 1):
            s = 1.0 - k / infer_steps
            out.append((flow_shift * s) / (1.0 + (flow_shift - 1.0) * s))
        return out


def stage_of(sigma: float, transition_sigmas: list[float]) -> int:
    """Index of the active resolution stage at this σ (advances once σ drops to
    or below each transition threshold). transition_sigmas is descending."""
    st = 0
    for thr in transition_sigmas:
        if sigma <= thr:
            st += 1
    return st


def step_cost(scale: float, base_tokens: float) -> tuple[float, float]:
    """Block-FLOP proxy for one forwarded step at this resolution scale."""
    toks = (scale**2) * base_tokens
    return toks**2, toks  # (attention O(N²), FFN O(N))


def spectrum_skip_mask(
    num_steps: int,
    warmup_steps: int,
    tail: int,
    window_size: float,
    flex_window: float,
    reset_at: set[int] | None = None,
) -> list[bool]:
    """Faithful replica of spectrum.py's growing-window cache decision. Returns
    is_cached[i] (True = block-skipped). ``reset_at`` forces a fresh warmup of
    ``warmup_steps`` actual forwards starting at each given step index (models
    the forecaster reset at an SPD handoff for the naive compose)."""
    stop_at = num_steps - tail
    reset_at = reset_at or set()
    is_cached = [False] * num_steps
    consec_cached = 0
    curr_ws = window_size
    warm_until = warmup_steps  # forced-actual region [0, warm_until)
    for i in range(num_steps):
        if i in reset_at:
            warm_until = i + warmup_steps  # re-warm window after the handoff
            curr_ws = window_size
            consec_cached = 0
        if i < warm_until or i >= stop_at:
            actual = True
        else:
            actual = (consec_cached + 1) % max(1, math.floor(curr_ws)) == 0
        if actual:
            consec_cached = 0
            if i >= warmup_steps:
                curr_ws = round(curr_ws + flex_window, 3)
        else:
            consec_cached += 1
        is_cached[i] = not actual
    return is_cached


def tally(
    sigmas: list[float],
    scale_per_step: list[float],
    is_cached: list[bool],
    base_tokens: float,
) -> dict:
    """Sum the block-FLOP proxy over a trajectory. Cached steps cost ~0 blocks."""
    attn = mlp = 0.0
    n_fwd = 0
    for i, sc in enumerate(scale_per_step):
        if is_cached[i]:
            continue
        a, m = step_cost(sc, base_tokens)
        attn += a
        mlp += m
        n_fwd += 1
    return {
        "attn": attn,
        "mlp": mlp,
        "n_forward": n_fwd,
        "n_steps": len(scale_per_step),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase-0(a) GPU eyeball: render baseline / spectrum-alone / spd-alone /
# naive-compose so coherence across the SPD handoff can be judged visually. The
# "naive compose" runs Spectrum's forecaster through SPD resolution stages and
# RESETS it at the handoff (feature shape changes) — the floor the proposal asks
# us to measure before building the band-aligned forecaster. No CMMD; the call
# is visual (open compare_seed*.png).
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_DIT = "models/diffusion_models/anima-base-v1.0.safetensors"
DEFAULT_TE = "models/text_encoders/qwen_3_06b_base.safetensors"
DEFAULT_VAE = "models/vae/qwen_image_vae.safetensors"
DEFAULT_PROMPT = (
    "masterpiece, best quality, score_7, safe. An anime girl with long silver hair"
    " in a red kimono standing under a maple tree, autumn leaves, detailed face."
)
DEFAULT_NEG = "worst quality, low quality, blurry, jpeg artifacts, watermark."


# ─────────────────────────────────────────────────────────────────────────────
# Real-prompt sampling. Phase-0(a)/(b) want the *production* prompt distribution,
# not one hand-written caption — the LL-continuity question is conditioning-
# dependent (the feature is a function of the cross-attn text). Sample N real
# captions from the training set (the `.txt` sidecars in `image_dataset/`, the
# caption master per the root CLAUDE.md) so the seam is probed across varied
# content. Deterministic given --prompt_sample_seed.
# ─────────────────────────────────────────────────────────────────────────────


def sample_real_prompts(prompts_dir: str, n: int, seed: int) -> list[tuple[str, str]]:
    """Pick ``n`` caption sidecars at random; return [(stem, caption), ...]."""
    root = Path(prompts_dir)
    txts = sorted(root.rglob("*.txt"))
    if not txts:
        return []
    rng = random.Random(seed)
    picks = rng.sample(txts, min(n, len(txts)))
    out: list[tuple[str, str]] = []
    for p in picks:
        try:
            cap = " ".join(p.read_text(encoding="utf-8", errors="ignore").split())
        except Exception:  # noqa: BLE001 — skip unreadable sidecars
            continue
        if cap:
            out.append((p.stem, cap))
    return out


def resolve_prompts(args) -> list[tuple[str, str]]:
    """(label, prompt) list for the GPU modes. n_prompts>0 → sample real captions;
    else fall back to the single literal --prompt."""
    if args.n_prompts and args.n_prompts > 0:
        pr = sample_real_prompts(
            args.prompts_dir, args.n_prompts, args.prompt_sample_seed
        )
        if pr:
            log.info(
                "Sampled %d real prompt(s) from %s (seed %d):",
                len(pr),
                args.prompts_dir,
                args.prompt_sample_seed,
            )
            for stem, cap in pr:
                log.info("  [%s] %s", stem, (cap[:90] + "…") if len(cap) > 90 else cap)
            prefix = args.prompt_prefix or ""
            return [(stem, prefix + cap) for stem, cap in pr]
        log.warning(
            "No .txt captions under %s — falling back to --prompt.", args.prompts_dir
        )
    return [("default", args.prompt)]


def _safe_tag(label: str) -> str:
    """Filesystem-safe short tag from a prompt label."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", label)[:40] or "p"


# ─────────────────────────────────────────────────────────────────────────────
# Phase-0(b) feature LL-DCT continuity primitives. The captured `final_layer`
# input is x_B_T_H_W_D (B, T, H_patch, W_patch, D) — block-output feature on the
# patch grid. We extract its LL band onto the *common* stage-0 patch grid via SPD's
# own `dct_lowpass_init` (the operator `spectral_expand` inverts), so the LL vector
# has the same shape AND the same basis at every step — that common band is exactly
# what the band-aligned forecaster would carry across the handoff.
# ─────────────────────────────────────────────────────────────────────────────


def feat_ll_vector(feat, h0_p: int, w0_p: int, *, standardize: bool):
    """Bring the feature's LL band onto the *common* (h0_p × w0_p) stage-0 grid and
    flatten it. Returns a flat float tensor whose length is identical at every step.

    ``feat``: (B, T, H_p, W_p, D). The LL band is extracted with SPD's own
    ``dct_lowpass_init`` (DCT, keep the top-left (h0_p, w0_p) coefficients, iDCT at
    that size) — i.e. *exactly* the operator SPD uses to define the low-res content,
    so the stage-0→stage-1 correspondence is the one ``spectral_expand`` carries.
    Crucially the low-pass to the common grid happens **before** any standardization,
    so the per-step normalization is computed over the *same* grid at every step.

    NB: an earlier version standardized over each step's *native* grid (32² vs 64²)
    *before* the DCT — the grid doubling made the std differ by ~2×, injecting a
    spurious scale step at the seam that read as a (false) discontinuity. Verified on
    a by-construction-continuous synthetic: native-grid standardize → seam jump ≈1.0;
    this common-grid form → ≈0. (See the Phase-0(b) erratum.)

    With ``standardize`` each (T·D) channel is then centered/unit-scaled over the
    common grid, so the residual measures LL *shape* continuity (κ scale removed);
    without it the raw vector still carries SPD's κ magnitude step.
    """
    import torch  # noqa: F401 — local so the arith gate imports without torch

    from networks.spd import dct_lowpass_init

    B, T, H_p, W_p, D = feat.shape
    x = feat.permute(0, 1, 4, 2, 3).reshape(B, T * D, 1, H_p, W_p).float()
    scale = h0_p / H_p
    if scale < 1.0:  # stage-1+ feature → DCT low-pass down to the stage-0 grid
        x = dct_lowpass_init(x, scale, patch=1)
    x = x.squeeze(2)  # (B, T·D, h0_p, w0_p) on the common grid
    if standardize:
        m = x.mean(dim=(-2, -1), keepdim=True)
        s = x.std(dim=(-2, -1), keepdim=True) + 1e-6
        x = (x - m) / s
    return x.reshape(-1).contiguous()


def _cheb_design(taus, M: int):
    """Chebyshev design matrix [T0..TM] (mirrors spectrum.ChebyshevForecaster)."""
    import torch

    taus = taus.reshape(-1, 1)
    K = taus.shape[0]
    T0 = torch.ones((K, 1), dtype=taus.dtype)
    if M == 0:
        return T0
    cols = [T0, taus]
    for _ in range(2, M + 1):
        cols.append(2 * taus * cols[-1] - cols[-2])
    return torch.cat(cols[: M + 1], dim=1)


def _extrap_resid(taus, H, fit_idx: list[int], target: int, M: int) -> float:
    """Relative L2 error of a degree-M Chebyshev fit on ``fit_idx`` predicting the
    LL vector at step ``target`` (keyed to σ̃ via ``taus``). This is the one-step
    forecast error the band-aligned forecaster would incur."""
    import torch

    X = _cheb_design(taus[fit_idx], M)
    coef = torch.linalg.lstsq(X, H[fit_idx]).solution
    pred = _cheb_design(taus[target : target + 1], M) @ coef
    err = (pred - H[target : target + 1]).norm()
    return float(err / (H[target : target + 1].norm() + 1e-8))


def continuity_metrics(records: list[dict], M: int, window: int) -> dict:
    """Compare the seam (stage-0 fit → first stage-1 step) against the natural
    within-stage one-step forecast error, for both raw and channel-standardized
    LL trajectories. ratio = seam / within: ≈1 ⇒ the LL band flows through the
    handoff like any other step; ≫1 ⇒ it jumps at the seam.

    σ̃ keying (proposal seam-handling note 1): the basis is fit on the *re-spaced*
    σ the sampler actually queries, so the polynomial doesn't straddle the reshape
    discontinuity in step→σ.
    """
    import torch

    n = len(records)
    sig = torch.tensor([r["sigma"] for r in records], dtype=torch.float32)
    smax, smin = float(sig.max()), float(sig.min())
    taus = 2.0 * (sig - smin) / (smax - smin + 1e-8) - 1.0  # σ̃ → [-1, 1]

    trans = [k for k, r in enumerate(records) if r["transitioned"]]
    t_idx = trans[0] if trans else None
    out: dict = {
        "transition_step": t_idx,
        "kappa": records[t_idx]["kappa"] if t_idx else None,
    }
    if t_idx is None or t_idx < 2 or (n - t_idx) < window + 1:
        out["error"] = (
            "no usable transition (need ≥2 stage-0 and >window stage-1 steps)"
        )
        return out

    s0 = list(range(0, t_idx))
    s1 = list(range(t_idx, n))

    def _for(H):
        win0 = min(window, len(s0))
        seam = _extrap_resid(taus, H, s0[-win0:], t_idx, M)
        withins = [
            _extrap_resid(taus, H, s1[pos - window : pos], s1[pos], M)
            for pos in range(window, len(s1))
        ]
        within = float(sum(withins) / len(withins)) if withins else float("nan")
        jump = float((H[t_idx] - H[t_idx - 1]).norm() / (H[t_idx - 1].norm() + 1e-8))
        return {
            "seam_resid": seam,
            "within_resid": within,
            "ratio": seam / within
            if within and math.isfinite(within)
            else float("nan"),
            "seam_jump_rel": jump,
        }

    H_raw = torch.stack([r["ll_raw"] for r in records])
    H_std = torch.stack([r["ll_std"] for r in records])
    out["raw"] = _for(H_raw)
    out["standardized"] = _for(H_std)
    out["ll_norm_raw"] = [float(H_raw[i].norm()) for i in range(n)]
    out["ll_norm_std"] = [float(H_std[i].norm()) for i in range(n)]
    return out


def denoise_compose(
    anima,
    x5_init,
    embed,
    neg_embed,
    sigmas,
    guidance,
    device,
    patch,
    stages,
    transition_sigmas,
    gen,
    *,
    use_spectrum,
    warmup_steps,
    tail,
    window_size,
    flex_window,
    m,
    lam,
    w,
):
    """Euler denoise with optional SPD resolution stages and optional Spectrum
    forecasting. When both are on (naive compose), the forecaster is reset at the
    handoff because the captured feature changes token grid. Returns (latent,
    stats). Mirrors probe_lowres_denoise.denoise + networks.spectrum loop bodies."""
    import torch
    import torch.nn.functional as F

    from networks.spd import dct_lowpass_init, spectral_expand
    from networks.spectrum import SpectrumPredictor, _spectrum_fast_forward

    try:
        from library.inference.adapters import set_hydra_sigma
    except Exception:  # noqa: BLE001
        set_hydra_sigma = lambda *_a, **_k: None  # noqa: E731 — bare DiT has no hydra

    H_full, W_full = x5_init.shape[-2], x5_init.shape[-1]
    sigmas = sigmas.clone().float()
    cur_scale = stages[0]
    x5 = x5_init
    if cur_scale < 1.0:
        x5 = dct_lowpass_init(x5, cur_scale, patch)
    stage_idx = 0
    do_cfg = guidance != 1.0
    n = len(sigmas) - 1
    stop_at = n - tail

    captured: dict = {}
    hook = None
    if use_spectrum:
        hook = anima.final_layer.register_forward_pre_hook(
            lambda module, args: captured.__setitem__("feat", args[0].detach().clone())
        )
    cond_fc = uncond_fc = None
    cur_feat_shape = None
    curr_ws = window_size
    consec_cached = 0
    warm_until = warmup_steps
    stats = {"n_forward": 0, "n_cached": 0, "n_resets": 0}

    def full_velocity(x, sigma):
        t = x.new_full((x.shape[0],), float(sigma))
        pad = torch.zeros(
            x.shape[0], 1, x.shape[-2], x.shape[-1], dtype=x.dtype, device=device
        )
        set_hydra_sigma(anima, t)
        v_c = anima(x, t, embed, padding_mask=pad)
        feat_c = captured.get("feat")
        v_u = feat_u = None
        if do_cfg:
            v_u = anima(x, t, neg_embed, padding_mask=pad)
            feat_u = captured.get("feat")
        return v_c, v_u, feat_c, feat_u

    try:
        with torch.no_grad():  # no graph retention — else activations accumulate → OOM
            for i in range(n):
                sigma = float(sigmas[i])
                while (
                    stage_idx < len(transition_sigmas)
                    and sigma <= transition_sigmas[stage_idx]
                ):
                    nxt = stages[stage_idx + 1]
                    if nxt > cur_scale:
                        orig = float(sigmas[i])
                        x5, sigma_new = spectral_expand(
                            x5, sigma, cur_scale, nxt, H_full, W_full, patch, gen
                        )
                        cur_scale = nxt
                        if orig > 0 and sigma_new != orig:
                            sigmas[i + 1 :] = sigma_new * (sigmas[i + 1 :] / orig)
                        sigma = sigma_new
                        if use_spectrum:  # naive reset: feature grid changed
                            cond_fc = uncond_fc = None
                            cur_feat_shape = None
                            curr_ws = window_size
                            consec_cached = 0
                            warm_until = i + warmup_steps
                            stats["n_resets"] += 1
                    stage_idx += 1

                if not use_spectrum:
                    v_c, v_u, _, _ = full_velocity(x5, sigma)
                    v = (v_u + guidance * (v_c - v_u)) if do_cfg else v_c
                    stats["n_forward"] += 1
                else:
                    actual = (
                        i < warm_until
                        or i >= stop_at
                        or cond_fc is None
                        or (consec_cached + 1) % max(1, math.floor(curr_ws)) == 0
                    )
                    if actual:
                        v_c, v_u, feat_c, feat_u = full_velocity(x5, sigma)
                        if cond_fc is None or tuple(feat_c.shape[1:]) != cur_feat_shape:
                            cur_feat_shape = tuple(feat_c.shape[1:])
                            cond_fc = SpectrumPredictor(
                                m, lam, w, device, feat_c.shape[1:], n
                            )
                            if do_cfg:
                                uncond_fc = SpectrumPredictor(
                                    m, lam, w, device, feat_u.shape[1:], n
                                )
                        cond_fc.update(float(i), feat_c)
                        if do_cfg:
                            uncond_fc.update(float(i), feat_u)
                        v = (v_u + guidance * (v_c - v_u)) if do_cfg else v_c
                        if i >= warmup_steps:
                            curr_ws = round(curr_ws + flex_window, 3)
                        consec_cached = 0
                        stats["n_forward"] += 1
                    else:
                        t = x5.new_full((x5.shape[0],), float(sigma))
                        set_hydra_sigma(anima, t)
                        v_c = _spectrum_fast_forward(
                            anima, t, cond_fc.predict(float(i))
                        )
                        if do_cfg:
                            v_u = _spectrum_fast_forward(
                                anima, t, uncond_fc.predict(float(i))
                            )
                            v = v_u + guidance * (v_c - v_u)
                        else:
                            v = v_c
                        consec_cached += 1
                        stats["n_cached"] += 1

                dt = float(sigmas[i + 1]) - sigma
                x5 = (x5.float() + v.float() * dt).to(torch.bfloat16)
    finally:
        if hook is not None:
            hook.remove()

    if cur_scale < 1.0:  # never reached full res — bicubic rescue so decode works
        x5 = (
            F.interpolate(
                x5.squeeze(2).float(),
                size=(H_full, W_full),
                mode="bicubic",
                align_corners=False,
            )
            .unsqueeze(2)
            .to(torch.bfloat16)
        )
    return x5, stats


def _setup_models(args, device, prompts: list[tuple[str, str]], *, need_vae: bool):
    """Shared GPU setup for the eyeball + continuity modes: strategies, bare DiT,
    per-prompt cond embeds (+ one shared neg embed), σ schedule, optional VAE.

    Returns (anima, patch, embeds, neg_embed, vae, sigmas) where ``embeds`` is a
    list of (label, embed) aligned with ``prompts``.
    """
    import sys

    import torch

    import inference as inference_mod
    from library.anima import strategy as strategy_anima, text_strategies
    from library.inference import sampling as inference_utils
    from library.inference.models import load_dit_model
    from library.inference.text import MAX_CROSSATTN_TOKENS, prepare_text_inputs
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
        prompts[0][1],
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
    # --lora statically merges the SPD trajectory adapter (Case B,
    # output/ckpt/anima_spd.safetensors) into the DiT so the continuity / eyeball
    # probes see the *trained* trajectory, not the bare base model. The adapter is
    # schedule-coupled (configs/methods/spd.toml) — run at its trained knee.
    iargs.lora_weight = [args.lora] if getattr(args, "lora", None) else None
    iargs.sampler = "euler"

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

    log.info(
        "Loading DiT (eager)%s ...",
        f" + LoRA {iargs.lora_weight[0]}" if iargs.lora_weight else " (no LoRA)",
    )
    anima = load_dit_model(iargs, device, torch.bfloat16)
    patch = anima.patch_spatial

    embeds: list[tuple[str, "torch.Tensor"]] = []
    neg_embed = None
    for label, prompt in prompts:
        iargs.prompt = prompt
        ctx, ctx_null = prepare_text_inputs(iargs, device, anima, shared_models=None)
        embeds.append((label, ctx["embed"][0].to(device, torch.bfloat16)))
        if neg_embed is None:
            neg_embed = ctx_null["embed"][0].to(device, torch.bfloat16)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    vae = None
    if need_vae:
        vae = qwen_vae.load_vae(args.vae, device="cpu", spatial_chunk_size=64)
        vae.to(torch.bfloat16).eval()

    _, sigmas = inference_utils.get_timesteps_sigmas(
        args.infer_steps, args.flow_shift, device
    )
    return anima, patch, embeds, neg_embed, vae, sigmas.to(device)


def run_gpu(args) -> None:
    """Phase-0(a) eyeball: render the four denoisers per prompt×seed, save montages."""
    import time

    import torch
    from PIL import Image
    from diffusers.utils.torch_utils import randn_tensor

    from bench.spd.probe_lowres_denoise import _laplacian_var, _to_pil
    from library.inference.output import decode_latent

    device = torch.device(args.device)
    stages = list(args.spd_stages)
    transitions = list(args.spd_transition_sigmas)
    assert len(transitions) == len(stages) - 1, "transitions must be len(stages)-1"
    log.info("SPD schedule: stages=%s @ σ=%s", stages, transitions)

    prompts = resolve_prompts(args)
    anima, patch, embeds, neg_embed, vae, sigmas = _setup_models(
        args, device, prompts, need_vae=True
    )

    # Four denoisers: (label, stages, transitions, use_spectrum).
    configs = [
        ("baseline", [1.0], [], False),
        ("spectrum", [1.0], [], True),
        ("spd", stages, transitions, False),
        ("naive_compose", stages, transitions, True),
    ]
    sp = dict(
        warmup_steps=args.warmup_steps,
        tail=args.tail,
        window_size=args.window_size,
        flex_window=args.flex_window,
        m=args.m,
        lam=args.lam,
        w=args.w,
    )
    run_dir = make_run_dir("spd", label=args.label)
    h_lat, w_lat = args.height // 8, args.width // 8
    per_run = []

    for p_label, embed in embeds:
        tag = _safe_tag(p_label)
        for seed in args.seeds:
            log.info("\n=== prompt [%s] seed %d ===", p_label, seed)
            init = randn_tensor(
                (1, anima.LATENT_CHANNELS, 1, h_lat, w_lat),
                generator=torch.Generator(device="cpu").manual_seed(seed),
                device=device,
                dtype=torch.bfloat16,
            )
            imgs, row = {}, {"prompt": p_label, "seed": seed}
            for label, st, tr, use_sp in configs:
                spd_gen = torch.Generator(device=device).manual_seed(seed + 10_000)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                lat, stats = denoise_compose(
                    anima,
                    init,
                    embed,
                    neg_embed,
                    sigmas,
                    args.guidance_scale,
                    device,
                    patch,
                    st,
                    tr,
                    spd_gen,
                    use_spectrum=use_sp,
                    **sp,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                nan = bool(torch.isnan(lat).any() or torch.isinf(lat).any())
                img = _to_pil(decode_latent(vae, lat, device))
                img.save(run_dir / f"{label}_{tag}_seed{seed}.png")
                imgs[label] = img
                row[label] = {
                    "time_s": round(dt, 2),
                    "nan_inf": nan,
                    "sharpness": round(_laplacian_var(img), 1),
                    **stats,
                }
                log.info(
                    "  %-14s %5.1fs  fwd %2d cached %2d resets %d  sharp %.0f  nan=%s",
                    label,
                    dt,
                    stats["n_forward"],
                    stats["n_cached"],
                    stats["n_resets"],
                    _laplacian_var(img),
                    nan,
                )
                del lat
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            # 1×4 montage in config order for eyeballing the handoff.
            order = [c[0] for c in configs]
            w_each, h_each = imgs[order[0]].width, imgs[order[0]].height
            montage = Image.new("RGB", (w_each * len(order), h_each), "white")
            for j, label in enumerate(order):
                montage.paste(imgs[label], (j * w_each, 0))
            montage.save(run_dir / f"compare_{tag}_seed{seed}.png")
            for label in order:
                base_t = row["baseline"]["time_s"]
                row[label]["speedup_vs_baseline"] = round(
                    base_t / row[label]["time_s"], 2
                )
            per_run.append(row)

    verdict = (
        "EYEBALL: open compare_*.png — panels are [baseline | spectrum | spd | "
        "naive_compose]. Judge naive_compose for handoff smear / re-warm artifacts vs "
        "spectrum-alone. If naive_compose is coherent AND faster than spectrum, the "
        "Phase-0(a) kill-up may fire (ship naive, skip the band-aligned forecaster); if "
        "it smears at the seam, the band-aligned variant is the only path → Phase 0(b)."
    )
    metrics = {
        "stages": stages,
        "transition_sigmas": transitions,
        "infer_steps": args.infer_steps,
        "guidance_scale": args.guidance_scale,
        "resolution_hw": [args.height, args.width],
        "prompts": [{"label": lbl, "prompt": pr} for lbl, pr in prompts],
        "per_run": per_run,
        "verdict": verdict,
    }
    artifacts = [p.name for p in sorted(run_dir.glob("*.png"))]
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        artifacts=artifacts,
        device=device,
    )
    log.info("\n%s\nwrote %s", verdict, run_dir / "result.json")


def _capture_spd_features(
    anima,
    x5_init,
    embed,
    neg_embed,
    sigmas,
    guidance,
    device,
    patch,
    stages,
    transitions,
    gen,
    hf_scale: float = 1.0,
):
    """Run an SPD trajectory (no forecasting) and record, per Euler step, the
    re-spaced σ̃ and the LL-DCT vector of the captured `final_layer` feature
    (cond branch). ``hf_scale`` attenuates the fresh-HF noise fill at the handoff
    (the frontier γ knob; 1.0 = paper default). Returns ``(records, x5_final)``:
    the per-step record dicts for ``continuity_metrics`` plus the final latent
    (bicubic-rescued to full res if a stage never handed off) for a detail proxy.
    """
    import torch

    from networks.spd import dct_lowpass_init, spectral_expand

    try:
        from library.inference.adapters import set_hydra_sigma
    except Exception:  # noqa: BLE001
        set_hydra_sigma = lambda *_a, **_k: None  # noqa: E731

    H_full, W_full = x5_init.shape[-2], x5_init.shape[-1]
    sigmas = sigmas.clone().float()
    cur_scale = stages[0]
    x5 = x5_init
    if cur_scale < 1.0:
        x5 = dct_lowpass_init(x5, cur_scale, patch)
    stage_idx = 0
    do_cfg = guidance != 1.0
    n = len(sigmas) - 1

    captured: dict = {}
    hook = anima.final_layer.register_forward_pre_hook(
        lambda module, args: captured.__setitem__("feat", args[0].detach())
    )
    records: list[dict] = []
    ll_hw: tuple[int, int] | None = None  # stage-0 patch grid = LL block size
    try:
        with torch.no_grad():
            for i in range(n):
                sigma = float(sigmas[i])
                transitioned, kappa = False, None
                while stage_idx < len(transitions) and sigma <= transitions[stage_idx]:
                    nxt = stages[stage_idx + 1]
                    if nxt > cur_scale:
                        orig = float(sigmas[i])
                        r = nxt / cur_scale
                        kappa = r / (1.0 + (r - 1.0) * sigma)
                        x5, sigma_new = spectral_expand(
                            x5, sigma, cur_scale, nxt, H_full, W_full, patch, gen,
                            hf_scale=hf_scale,
                        )
                        cur_scale = nxt
                        if orig > 0 and sigma_new != orig:
                            sigmas[i + 1 :] = sigma_new * (sigmas[i + 1 :] / orig)
                        sigma = sigma_new
                        transitioned = True
                    stage_idx += 1

                t = x5.new_full((x5.shape[0],), float(sigma))
                pad = torch.zeros(
                    x5.shape[0],
                    1,
                    x5.shape[-2],
                    x5.shape[-1],
                    dtype=x5.dtype,
                    device=device,
                )
                set_hydra_sigma(anima, t)
                v_c = anima(x5, t, embed, padding_mask=pad)
                feat_c = captured["feat"]  # (B, T, H_p, W_p, D), cond branch
                if ll_hw is None:
                    ll_hw = (feat_c.shape[2], feat_c.shape[3])
                records.append(
                    {
                        "i": i,
                        "sigma": sigma,
                        "scale": cur_scale,
                        "transitioned": transitioned,
                        "kappa": kappa,
                        "ll_raw": feat_ll_vector(
                            feat_c, *ll_hw, standardize=False
                        ).cpu(),
                        "ll_std": feat_ll_vector(
                            feat_c, *ll_hw, standardize=True
                        ).cpu(),
                    }
                )
                if do_cfg:
                    v_u = anima(x5, t, neg_embed, padding_mask=pad)
                    v = v_u + guidance * (v_c - v_u)
                else:
                    v = v_c
                dt = float(sigmas[i + 1]) - sigma
                x5 = (x5.float() + v.float() * dt).to(torch.bfloat16)
    finally:
        hook.remove()

    if cur_scale < 1.0:  # never handed off to full res — bicubic rescue
        import torch.nn.functional as F

        x5 = (
            F.interpolate(
                x5.squeeze(2).float(),
                size=(H_full, W_full),
                mode="bicubic",
                align_corners=False,
            )
            .unsqueeze(2)
            .to(torch.bfloat16)
        )
    return records, x5


def run_continuity(args) -> None:
    """Phase-0(b): feature LL-DCT continuity across the SPD handoff.

    For each sampled prompt, capture the cond `final_layer` feature over a single
    SPD trajectory, 2D-DCT it on the patch grid, and ask whether the stage-0 LL
    band's Chebyshev trajectory carries cleanly into stage-1 (seam residual ≈
    within-stage one-step forecast residual). Standardized = shape continuity
    (κ scale removed); raw = with the magnitude step SPD's κ injects. The PASS on
    *standardized* ratio is the real precondition for the band-aligned forecaster.
    """
    import torch
    from diffusers.utils.torch_utils import randn_tensor

    device = torch.device(args.device)
    stages = list(args.spd_stages)
    transitions = list(args.spd_transition_sigmas)
    assert len(transitions) == len(stages) - 1, "transitions must be len(stages)-1"
    assert any(s < 1.0 for s in stages), "continuity probe needs a real low-res stage"
    log.info("SPD schedule: stages=%s @ σ=%s", stages, transitions)

    prompts = resolve_prompts(args)
    anima, patch, embeds, neg_embed, _vae, sigmas = _setup_models(
        args, device, prompts, need_vae=False
    )
    h_lat, w_lat = args.height // 8, args.width // 8
    seed = args.seeds[0]

    run_dir = make_run_dir("spd", label=args.label)
    per_prompt = []
    for p_label, embed in embeds:
        log.info("\n=== continuity: prompt [%s] (seed %d) ===", p_label, seed)
        init = randn_tensor(
            (1, anima.LATENT_CHANNELS, 1, h_lat, w_lat),
            generator=torch.Generator(device="cpu").manual_seed(seed),
            device=device,
            dtype=torch.bfloat16,
        )
        spd_gen = torch.Generator(device=device).manual_seed(seed + 10_000)
        records, _ = _capture_spd_features(
            anima,
            init,
            embed,
            neg_embed,
            sigmas,
            args.guidance_scale,
            device,
            patch,
            stages,
            transitions,
            spd_gen,
        )
        m = continuity_metrics(records, args.cheb_degree, args.fit_window)
        m["prompt"] = p_label
        if "error" in m:
            log.warning("  [%s] %s", p_label, m["error"])
        else:
            log.info(
                "  [%s] transition@step %d  κ=%.3f  | std: seam %.4f within %.4f ratio %.2f"
                "  | raw: ratio %.2f jump %.3f",
                p_label,
                m["transition_step"],
                m["kappa"],
                m["standardized"]["seam_resid"],
                m["standardized"]["within_resid"],
                m["standardized"]["ratio"],
                m["raw"]["ratio"],
                m["raw"]["seam_jump_rel"],
            )
        per_prompt.append(m)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    ok = [m for m in per_prompt if "error" not in m]
    import statistics as _st

    def _agg(key, sub):
        vals = [m[sub][key] for m in ok if math.isfinite(m[sub][key])]
        return (_st.mean(vals), _st.pstdev(vals) if len(vals) > 1 else 0.0, len(vals))

    artifacts = []
    if ok:
        std_ratio_mean, std_ratio_std, n_ok = _agg("ratio", "standardized")
        raw_ratio_mean, _, _ = _agg("ratio", "raw")
        std_jump_mean, _, _ = _agg("seam_jump_rel", "standardized")
        raw_jump_mean, _, _ = _agg("seam_jump_rel", "raw")
        passed = std_ratio_mean <= args.pass_ratio
        verdict = (
            (
                f"PASS — standardized LL ratio ×{std_ratio_mean:.2f}±{std_ratio_std:.2f} "
                f"≤ {args.pass_ratio} ({n_ok} prompts): the feature's LL band flows through "
                f"the handoff like a within-stage step (shape jump {std_jump_mean:.3f}). The "
                f"band-aligned forecast has signal → Phase 1. (raw ratio ×{raw_ratio_mean:.2f}, "
                f"raw jump {raw_jump_mean:.3f} = the expected κ magnitude step.)"
            )
            if passed
            else (
                f"FAIL — standardized LL ratio ×{std_ratio_mean:.2f}±{std_ratio_std:.2f} "
                f"> {args.pass_ratio} ({n_ok} prompts): the LL band jumps in *shape* at the "
                f"seam (jump {std_jump_mean:.3f}), not just scale — the nonlinear feature does "
                f"not inherit the latent's spectral continuity. Band-aligned forecasting is "
                f"dead; ship Case-A SPD or Spectrum standalone (the proposal's most-likely killer)."
            )
        )
        agg = {
            "std_ratio_mean": std_ratio_mean,
            "std_ratio_std": std_ratio_std,
            "raw_ratio_mean": raw_ratio_mean,
            "std_jump_mean": std_jump_mean,
            "raw_jump_mean": raw_jump_mean,
            "n_prompts_ok": n_ok,
            "pass": passed,
        }
        # Per-prompt LL-norm plot (eyeball the seam): raw vs standardized.
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
            for m in ok:
                steps = list(range(len(m["ll_norm_std"])))
                axes[0].plot(
                    steps, m["ll_norm_raw"], "o-", ms=3, label=m["prompt"][:18]
                )
                axes[1].plot(
                    steps, m["ll_norm_std"], "o-", ms=3, label=m["prompt"][:18]
                )
                axes[0].axvline(m["transition_step"], ls="--", lw=1, color="0.5")
                axes[1].axvline(m["transition_step"], ls="--", lw=1, color="0.5")
            axes[0].set_title("LL-block ‖·‖ (raw — carries κ scale step)")
            axes[1].set_title("LL-block ‖·‖ (standardized — shape only)")
            for ax in axes:
                ax.set_xlabel("Euler step")
                ax.grid(alpha=0.3)
                ax.legend(fontsize=7)
            fig.suptitle(
                f"Phase-0(b) feature LL-DCT continuity  ({verdict.split(' —')[0]})"
            )
            fig.tight_layout()
            fig.savefig(run_dir / "ll_continuity.png", dpi=130)
            artifacts.append("ll_continuity.png")
        except Exception as e:  # noqa: BLE001
            log.warning("plot skipped: %s", e)
    else:
        agg = {"pass": False, "n_prompts_ok": 0}
        verdict = (
            "INVALID — no prompt produced a usable transition; check the SPD schedule."
        )

    # Drop bulky per-step norm arrays from the JSON record (kept only for the plot).
    for m in per_prompt:
        m.pop("ll_norm_raw", None)
        m.pop("ll_norm_std", None)

    log.info("\n=== SPD ∘ Spectrum — Phase-0(b) feature LL-DCT continuity ===")
    log.info("VERDICT: %s", verdict)
    metrics = {
        "stages": stages,
        "transition_sigmas": transitions,
        "infer_steps": args.infer_steps,
        "guidance_scale": args.guidance_scale,
        "resolution_hw": [args.height, args.width],
        "seed": seed,
        "cheb_degree": args.cheb_degree,
        "fit_window": args.fit_window,
        "pass_ratio": args.pass_ratio,
        "prompts": [{"label": lbl, "prompt": pr} for lbl, pr in prompts],
        "per_prompt": per_prompt,
        "aggregate": agg,
        "verdict": verdict,
        "pass": agg["pass"],
    }
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        artifacts=artifacts,
        device=device,
    )
    log.info("wrote %s", run_dir / "result.json")


def _hf_energy_frac(x5, stage0_scale: float, patch: int) -> float:
    """Fraction of the final latent's 2D-DCT energy *above* the stage-0 LL cutoff
    — the band SPD defers and the full-res tail must develop. ≈ baseline ⇒ full
    detail; ≪ baseline ⇒ the run degenerated toward an upsampled-LL blur (the
    failure mode a continuity regularizer would chase). Latent-space proxy, no VAE."""
    import torch  # noqa: F401

    from networks.spd import _snap, dct2

    B, C, T, H, W = x5.shape
    xi = dct2(x5.squeeze(2).float())
    h0 = min(_snap(H * stage0_scale, patch), H)
    w0 = min(_snap(W * stage0_scale, patch), W)
    total = float(xi.pow(2).sum())
    ll = float(xi[:, :, :h0, :w0].pow(2).sum())
    return (total - ll) / (total + 1e-12)


def run_frontier(args) -> None:
    """Step 1 of the 'opposite LoRA' (seam-continuity) investigation — the offline
    frozen-DiT frontier sweep that decides whether training is worth attempting.

    For each (transition σ, HF-injection γ) cell on the *frozen* base DiT, measure
    two axes per the discussion in proposal.md:
      * **continuity** — the standardized seam/within LL-residual ratio (the same
        metric the Phase-0(b) report found at ×4–5.6 for γ=1; the band-aligned
        forecaster needs this ≤ pass_ratio);
      * **detail** — HF latent energy of the final image relative to the full-res
        baseline (detail_retention; ≪1 ⇒ continuity was bought with blur).

    The verdict reads the frozen Pareto frontier: if some cell already clears the
    seam gate *with detail retained*, a trained delta only has to push the γ=1
    operating point onto that frontier (plausible → run step 2, the loss_seam
    overfit). If the gate is cleared only where detail collapses, the tradeoff is
    monotone and a LoRA must *shift* the frontier — a much bigger bet. If no cell
    clears it at all, the reorientation is intrinsic and the idea is likely dead.

    The (σ0.5, γ1.0) cell reproduces the report's ×4.49 anchor — a built-in
    cross-check that this sweep is consistent with the continuity probe.
    """
    import statistics as _st

    import torch
    from diffusers.utils.torch_utils import randn_tensor

    device = torch.device(args.device)
    s0 = float(args.spd_stages[0])
    assert s0 < 1.0, "frontier needs a real low-res stage-0 (spd_stages[0] < 1.0)"
    sig_grid = list(args.frontier_transition_sigmas)
    gam_grid = list(args.frontier_hf_scales)
    log.info(
        "Frontier sweep: stage0=%.3f  σ∈%s × γ∈%s  (%d cells × %d prompts)",
        s0,
        sig_grid,
        gam_grid,
        len(sig_grid) * len(gam_grid),
        args.n_prompts if args.n_prompts > 0 else 1,
    )

    prompts = resolve_prompts(args)
    anima, patch, embeds, neg_embed, _vae, sigmas = _setup_models(
        args, device, prompts, need_vae=False
    )
    h_lat, w_lat = args.height // 8, args.width // 8
    seed = args.seeds[0]

    # Per-prompt full-res baseline → reference HF energy for detail_retention.
    base_hf: dict[str, float] = {}
    inits: dict[str, "torch.Tensor"] = {}
    for p_label, embed in embeds:
        init = randn_tensor(
            (1, anima.LATENT_CHANNELS, 1, h_lat, w_lat),
            generator=torch.Generator(device="cpu").manual_seed(seed),
            device=device,
            dtype=torch.bfloat16,
        )
        inits[p_label] = init
        _rec, x5_base = _capture_spd_features(
            anima, init, embed, neg_embed, sigmas, args.guidance_scale,
            device, patch, [1.0], [], torch.Generator(device=device).manual_seed(seed),
        )
        base_hf[p_label] = _hf_energy_frac(x5_base, s0, patch)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    log.info("baseline HF-energy frac: %s", {k: round(v, 4) for k, v in base_hf.items()})

    cells: list[dict] = []
    for sig in sig_grid:
        for gam in gam_grid:
            ratios, dets = [], []
            for p_label, embed in embeds:
                spd_gen = torch.Generator(device=device).manual_seed(seed + 10_000)
                records, x5 = _capture_spd_features(
                    anima, inits[p_label], embed, neg_embed, sigmas,
                    args.guidance_scale, device, patch, [s0, 1.0], [sig], spd_gen,
                    hf_scale=gam,
                )
                m = continuity_metrics(records, args.cheb_degree, args.fit_window)
                if "error" not in m and math.isfinite(m["standardized"]["ratio"]):
                    ratios.append(m["standardized"]["ratio"])
                det = _hf_energy_frac(x5, s0, patch) / (base_hf[p_label] + 1e-12)
                dets.append(det)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            cell = {
                "transition_sigma": sig,
                "hf_scale": gam,
                "seam_ratio_mean": _st.mean(ratios) if ratios else float("nan"),
                "seam_ratio_std": _st.pstdev(ratios) if len(ratios) > 1 else 0.0,
                "detail_retention_mean": _st.mean(dets) if dets else float("nan"),
                "n_prompts_ok": len(ratios),
            }
            cells.append(cell)
            log.info(
                "  σ=%.2f γ=%.2f  seam ×%.2f±%.2f  detail %.2f  (n=%d)",
                sig, gam, cell["seam_ratio_mean"], cell["seam_ratio_std"],
                cell["detail_retention_mean"], cell["n_prompts_ok"],
            )

    # ── Verdict: read the frozen Pareto frontier ──
    valid = [c for c in cells if math.isfinite(c["seam_ratio_mean"])]
    below = [c for c in valid if c["seam_ratio_mean"] <= args.pass_ratio]
    anchor = next(
        (c for c in cells if abs(c["transition_sigma"] - 0.5) < 1e-6
         and abs(c["hf_scale"] - 1.0) < 1e-6),
        None,
    )
    anchor_note = (
        f" (σ0.5,γ1.0 anchor ×{anchor['seam_ratio_mean']:.2f} vs report ×4.49)"
        if anchor else ""
    )
    if not valid:
        verdict = "INVALID — no cell produced a usable transition; check the grid."
        decision = "invalid"
    elif not below:
        gentlest = min(valid, key=lambda c: c["seam_ratio_mean"])
        verdict = (
            f"INTRINSIC FAIL — no cell clears the seam gate (≤{args.pass_ratio}); the "
            f"gentlest (σ={gentlest['transition_sigma']:.2f}, γ={gentlest['hf_scale']:.2f}) "
            f"still ×{gentlest['seam_ratio_mean']:.2f}. The LL reorientation is not a "
            f"continuity↔detail tradeoff knob — even minimal HF injection scrambles the "
            f"feature pattern. A LoRA loss_seam has no frozen operating point to anchor to; "
            f"strongly reconsider the whole 'opposite LoRA' bet.{anchor_note}"
        )
        decision = "intrinsic_fail"
    else:
        best = max(below, key=lambda c: c["detail_retention_mean"])
        if best["detail_retention_mean"] >= args.detail_pass:
            verdict = (
                f"MOVABLE — cell (σ={best['transition_sigma']:.2f}, γ={best['hf_scale']:.2f}) "
                f"clears the seam gate (×{best['seam_ratio_mean']:.2f} ≤ {args.pass_ratio}) "
                f"with detail retained ({best['detail_retention_mean']:.2f} ≥ "
                f"{args.detail_pass}). A frozen operating point already sits on the right "
                f"side; training only has to drag the realistic γ=1 point onto this frontier "
                f"→ PROCEED to step 2 (loss_seam overfit on 8–16 prompts).{anchor_note}"
            )
            decision = "movable"
        else:
            verdict = (
                f"FRONTIER DEGENERATE — the seam gate is cleared only where detail collapses "
                f"(best retention {best['detail_retention_mean']:.2f} < {args.detail_pass}, at "
                f"σ={best['transition_sigma']:.2f}, γ={best['hf_scale']:.2f}). Continuity and "
                f"detail trade off monotonically on the frozen model, so a LoRA must *shift* "
                f"the Pareto frontier (give γ=1 detail at γ→0 continuity), not slide along it — "
                f"a much bigger bet. Proceed to step 2 only if willing to gamble on a frontier "
                f"shift; otherwise shelve.{anchor_note}"
            )
            decision = "frontier_degenerate"

    log.info("\n=== SPD seam-continuity frontier (frozen DiT) ===")
    log.info("VERDICT: %s", verdict)

    run_dir = make_run_dir("spd", label=args.label)
    artifacts = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 5.5))
        for sig in sig_grid:
            row = [c for c in cells if c["transition_sigma"] == sig
                   and math.isfinite(c["seam_ratio_mean"])]
            row.sort(key=lambda c: c["hf_scale"])
            ax.plot(
                [c["seam_ratio_mean"] for c in row],
                [c["detail_retention_mean"] for c in row],
                "o-", label=f"σ={sig:.2f}",
            )
            for c in row:
                ax.annotate(f"γ{c['hf_scale']:.2f}",
                            (c["seam_ratio_mean"], c["detail_retention_mean"]),
                            fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.axvline(args.pass_ratio, ls="--", color="r", lw=1,
                   label=f"seam gate ×{args.pass_ratio}")
        ax.axhline(args.detail_pass, ls=":", color="g", lw=1,
                   label=f"detail floor {args.detail_pass}")
        ax.set_xlabel("seam LL-residual ratio (standardized) — lower = more continuous")
        ax.set_ylabel("detail retention (HF energy vs baseline) — higher = sharper")
        ax.set_title(f"SPD seam-continuity frontier  [{decision.upper()}]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(run_dir / "frontier.png", dpi=130)
        artifacts.append("frontier.png")
    except Exception as e:  # noqa: BLE001
        log.warning("plot skipped: %s", e)

    metrics = {
        "stage0_scale": s0,
        "frontier_transition_sigmas": sig_grid,
        "frontier_hf_scales": gam_grid,
        "infer_steps": args.infer_steps,
        "guidance_scale": args.guidance_scale,
        "resolution_hw": [args.height, args.width],
        "seed": seed,
        "pass_ratio": args.pass_ratio,
        "detail_pass": args.detail_pass,
        "baseline_hf_frac": base_hf,
        "prompts": [{"label": lbl, "prompt": pr} for lbl, pr in prompts],
        "cells": cells,
        "decision": decision,
        "verdict": verdict,
    }
    write_result(
        run_dir, script=__file__, args=args, metrics=metrics,
        artifacts=artifacts, device=device,
    )
    log.info("wrote %s", run_dir / "result.json")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--mode",
        choices=["arith", "gpu", "continuity", "frontier"],
        default="arith",
        help="arith = FLOP gate (no GPU, default); gpu = Phase-0(a) eyeball render; "
        "continuity = Phase-0(b) feature LL-DCT continuity probe; frontier = "
        "seam-continuity vs detail sweep on the frozen DiT (step 1 of the "
        "'opposite LoRA' bet — decides if loss_seam training is worth attempting).",
    )
    ap.add_argument("--infer_steps", type=int, default=28)
    ap.add_argument("--flow_shift", type=float, default=1.0)
    ap.add_argument(
        "--spd_stages",
        type=float,
        nargs="+",
        default=[0.5, 1.0],
        help="Ascending resolution scales (default single-late knee 0.5→1.0).",
    )
    ap.add_argument(
        "--spd_transition_sigmas",
        type=float,
        nargs="+",
        default=[0.7],
        help="σ thresholds, descending; len = len(stages)-1 (default 0.7 knee).",
    )
    ap.add_argument(
        "--sigma_resolve",
        type=float,
        nargs="+",
        default=None,
        help="Per-band σ_resolve, low→high freq. Default = Phase-1 measured curve.",
    )
    ap.add_argument(
        "--from_result",
        type=str,
        default=None,
        help="measure_autoregression result.json to pull sigma_resolve_per_band from.",
    )
    # Spectrum schedule knobs (faithful to spectrum.py defaults).
    ap.add_argument("--warmup_steps", type=int, default=6)
    ap.add_argument("--tail", type=int, default=3)
    ap.add_argument("--window_size", type=float, default=2.0)
    ap.add_argument("--flex_window", type=float, default=0.25)
    ap.add_argument("--base_tokens", type=float, default=4096.0)
    ap.add_argument(
        "--spectrum_node_speedup",
        type=float,
        default=1.75,
        help="The real shipped-node Spectrum speedup (~×1.75 on Anima — matches the "
        "spectrum.py library-default schedule). The verdict anchors here as the true "
        "competitor.",
    )
    ap.add_argument("--m", type=int, default=3, help="Chebyshev basis size (Spectrum).")
    ap.add_argument("--lam", type=float, default=0.1, help="Ridge λ (Spectrum).")
    ap.add_argument(
        "--w", type=float, default=0.3, help="Chebyshev/Taylor blend (Spectrum)."
    )
    # GPU eyeball (--mode gpu) only:
    ap.add_argument("--dit", default=DEFAULT_DIT)
    ap.add_argument("--text_encoder", default=DEFAULT_TE)
    ap.add_argument("--vae", default=DEFAULT_VAE)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--negative_prompt", default=DEFAULT_NEG)
    ap.add_argument(
        "--lora",
        default=None,
        help="Optional LoRA to statically merge (e.g. output/ckpt/anima_spd.safetensors, "
        "the Case-B SPD trajectory adapter). Run at the adapter's trained SPD knee.",
    )
    # Real-prompt sampling (gpu / continuity modes). n_prompts>0 samples that many
    # caption sidecars from the training set so the seam is probed on the
    # production prompt distribution; 0 falls back to the single literal --prompt.
    ap.add_argument("--n_prompts", type=int, default=5)
    ap.add_argument("--prompts_dir", default="image_dataset")
    ap.add_argument("--prompt_sample_seed", type=int, default=0)
    ap.add_argument(
        "--prompt_prefix",
        default="",
        help="Optional quality prefix prepended to each sampled caption (default none).",
    )
    # Phase-0(b) continuity probe knobs.
    ap.add_argument("--cheb_degree", type=int, default=3, help="Chebyshev fit degree.")
    ap.add_argument(
        "--fit_window",
        type=int,
        default=6,
        help="Steps per Chebyshev fit window (seam + within-stage forecasts).",
    )
    ap.add_argument(
        "--pass_ratio",
        type=float,
        default=2.5,
        help="PASS iff mean standardized seam/within LL residual ratio ≤ this.",
    )
    # Frontier sweep (--mode frontier) only. The grid is a 2-stage [s0,1.0]
    # schedule (s0 = spd_stages[0]); σ and γ (HF-injection scale) are swept.
    ap.add_argument(
        "--frontier_transition_sigmas",
        type=float,
        nargs="+",
        default=[0.7, 0.5],
        help="Transition-σ grid for the frontier sweep (gentle→aggressive).",
    )
    ap.add_argument(
        "--frontier_hf_scales",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 1.0],
        help="HF-injection γ grid (0 = no fresh HF = max continuity/min detail; "
        "1.0 = paper default). Each cell = (σ × γ).",
    )
    ap.add_argument(
        "--detail_pass",
        type=float,
        default=0.8,
        help="Detail-retention floor (final HF energy vs baseline) a seam-passing "
        "cell must clear to count as a non-degenerate (movable) operating point.",
    )
    # 768² default fits the bare DiT + CFG on a 16GB card without block-swap;
    # the handoff-coherence call is resolution-independent. Bump for bigger GPUs.
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--guidance_scale", type=float, default=4.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[40, 41])
    ap.add_argument(
        "--device",
        default="cuda" if _cuda_available() else "cpu",
    )
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    if args.mode == "gpu":
        if args.label is None:
            args.label = "phase0a-eyeball"
        return run_gpu(args)
    if args.mode == "continuity":
        if args.label is None:
            args.label = "phase0b-ll-continuity"
        return run_continuity(args)
    if args.mode == "frontier":
        if args.label is None:
            args.label = "seam-frontier"
        return run_frontier(args)
    if args.label is None:
        args.label = "phase0-arith"

    sigma_resolve = args.sigma_resolve or list(DEFAULT_SIGMA_RESOLVE)
    if args.from_result:
        rec = json.loads(Path(args.from_result).read_text())
        pulled = rec.get("metrics", {}).get("sigma_resolve_per_band")
        if pulled:
            sigma_resolve = [float(x) for x in pulled]
            log.info("σ_resolve pulled from %s", args.from_result)
    sigma_top = min(sigma_resolve)  # top band locks last → strict skip threshold

    stages = list(args.spd_stages)
    transitions = list(args.spd_transition_sigmas)
    assert len(transitions) == len(stages) - 1, "transitions must be len(stages)-1"

    sigmas = build_sigmas(args.infer_steps, args.flow_shift)
    n = args.infer_steps
    sig_at_step = sigmas[:n]  # σ before each Euler step
    stop_at = n - args.tail

    # Per-step resolution scale under the SPD schedule.
    scale_full = [1.0] * n
    scale_spd = [stages[stage_of(s, transitions)] for s in sig_at_step]

    # Transition step indices (first step of each new stage) for the naive reset.
    transition_steps = set()
    prev = 0
    for i, s in enumerate(sig_at_step):
        st = stage_of(s, transitions)
        if st != prev:
            transition_steps.add(i)
            prev = st

    # --- five denoisers ---
    no_skip = [False] * n
    spectrum_cached = spectrum_skip_mask(
        n, args.warmup_steps, args.tail, args.window_size, args.flex_window
    )
    naive_cached = spectrum_skip_mask(
        n,
        args.warmup_steps,
        args.tail,
        args.window_size,
        args.flex_window,
        reset_at=transition_steps,
    )
    # Banded compose: skip iff every band settled (σ < σ_top) AND outside warmup/tail.
    banded_cached = [
        (sig_at_step[i] < sigma_top) and (args.warmup_steps <= i < stop_at)
        for i in range(n)
    ]

    methods = {
        "baseline": tally(sigmas, scale_full, no_skip, args.base_tokens),
        "spectrum_alone": tally(sigmas, scale_full, spectrum_cached, args.base_tokens),
        "spd_alone": tally(sigmas, scale_spd, no_skip, args.base_tokens),
        "naive_compose": tally(sigmas, scale_spd, naive_cached, args.base_tokens),
        "banded_compose": tally(sigmas, scale_spd, banded_cached, args.base_tokens),
    }
    base_attn = methods["baseline"]["attn"]
    base_mlp = methods["baseline"]["mlp"]
    for m in methods.values():
        m["attn_speedup"] = base_attn / m["attn"] if m["attn"] > 0 else float("inf")
        m["mlp_speedup"] = base_mlp / m["mlp"] if m["mlp"] > 0 else float("inf")

    # Sensitivity: how many full-res steps would be skippable if the threshold were
    # each band's σ_resolve instead of the strict top band (shows the R2 cliff).
    n_skippable_by_band = {}
    for b, sr in enumerate(sorted(sigma_resolve, reverse=True)):
        cnt = sum(
            1
            for i in range(n)
            if sig_at_step[i] < sr and args.warmup_steps <= i < stop_at
        )
        n_skippable_by_band[f"band_le_{sr:.2f}"] = cnt

    # The shipped Spectrum node is the real competitor (~×3.75), not spectrum.py's
    # conservative library defaults. Under this full-res, cached-cost≈0 proxy a
    # speedup S implies N/S forwards, so anchor the comparison there.
    node_su = args.spectrum_node_speedup
    methods["spectrum_node"] = {
        "attn_speedup": node_su,
        "mlp_speedup": node_su,
        "n_forward": round(n / node_su),
        "n_steps": n,
        "note": "synthetic: cited shipped-node operating point, the true competitor",
    }
    sa_proxy = methods["spectrum_alone"]["attn_speedup"]
    ba = methods["banded_compose"]["attn_speedup"]
    na = methods["naive_compose"]["attn_speedup"]
    n_skip_band = (
        methods["banded_compose"]["n_steps"] - methods["banded_compose"]["n_forward"]
    )
    margin = ba - node_su  # the binding comparison
    if margin > 0.05:
        verdict = (
            f"PROCEED — banded_compose attn ×{ba:.2f} > Spectrum-node ×{node_su:.2f} "
            f"(+{margin:.2f}). The SPD token-saving survives R2 even against the "
            f"aggressive node; run Phase-0(a) naive floor then 0(b) LL-DCT continuity."
        )
    else:
        verdict = (
            f"STOP/REFRAME — banded_compose attn ×{ba:.2f} < Spectrum-node ×{node_su:.2f} "
            f"({margin:+.2f}). R2 confirmed: HF refresh forces full block forwards, so the "
            f"principled compose can skip blocks only on σ<{sigma_top:.2f} ({n_skip_band} of "
            f"{n} steps) and forfeits the middle-trajectory skips that earn Spectrum its "
            f"×{node_su:.2f}. SPD's resolution saving on the early stage (proxy banded ×{ba:.2f} "
            f"vs spd_alone ×{methods['spd_alone']['attn_speedup']:.2f}) does not buy that back. "
            f"Speed-stacking is a paper exercise; the only live angle is the quality reframing "
            f"(HF forwards buying detail), not throughput. (Proxy spectrum_alone at library "
            f"defaults ×{sa_proxy:.2f}, naive_compose ×{na:.2f} — both also < node.)"
        )
    log.info("\n=== SPD ∘ Spectrum — Phase-0 arithmetic gate ===")
    log.info("σ_resolve(top band) = %.2f  (skip threshold)", sigma_top)
    for name, m in methods.items():
        n_skip = m["n_steps"] - m["n_forward"]
        log.info(
            "  %-16s attn ×%.2f  mlp ×%.2f  forwards %2d/%d  skipped %d",
            name,
            m["attn_speedup"],
            m["mlp_speedup"],
            m["n_forward"],
            m["n_steps"],
            n_skip,
        )
    log.info("skippable-by-threshold: %s", n_skippable_by_band)
    log.info("\nVERDICT: %s", verdict)

    run_dir = make_run_dir("spd", label=args.label)
    metrics = {
        "sigma_resolve_per_band": sigma_resolve,
        "sigma_resolve_top": sigma_top,
        "spd_stages": stages,
        "spd_transition_sigmas": transitions,
        "transition_steps": sorted(transition_steps),
        "methods": methods,
        "n_skippable_by_band_threshold": n_skippable_by_band,
        "banded_vs_spectrum_attn_margin": margin,
        "verdict": verdict,
        "pass": margin > 0.05,
    }
    write_result(run_dir, script=__file__, args=args, metrics=metrics, label=args.label)
    log.info("\nwrote %s", run_dir / "result.json")


if __name__ == "__main__":
    main()
