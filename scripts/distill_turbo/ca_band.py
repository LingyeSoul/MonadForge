"""CA-branch FEI band-deficit reweighting (item 2; see ``item2_plan.md``).

Reweights δ_cfg by where the student's x0 FEI lags the teacher's. Per Phase 0
(turbo_C, n=90) ``w_high`` is the active arm and ``w_low`` stays ≈ 1, but both
branches are wired because a future student's failure mode may flip.

The FEI gap is measured at the **student's sampler time t** — see
``item2_diagnosis.md``. The first wiring measured it at τ_ca on the
teacher's denoise of the renoised x_pred, which introduced a structural
~0.08 LP shift (the teacher's denoise smooths its input toward the data
mean, which is LP-biased) — that floor masked Phase 0's ≈ 0.012 lever and
inverted the active-arm sign. The fix is a single extra no-grad teacher
forward at ``(x_t, t, c)`` so both x0 estimates live at the same σ:

* ``x0_T = x_t − t · v_teacher_at_t``
* ``x0_S = x_pred = x_t − t · v_student``

When the student LoRA is near zero (early steps), ``v_student ≈
v_teacher_at_t`` so both deficits should collapse to ≈ 0 — the on-init
sanity check the first wiring failed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from library.runtime.fei import (
    compute_fei_2band,
    fei_sigma_low,
    gaussian_blur_2d,
)


@dataclass
class BandDeficitDiag:
    """Per-step in-window means + the in-window batch count.

    All tensors are 0-dim GPU scalars; safe to accumulate without syncs.
    """

    w_high: torch.Tensor  # mean(w_high) over in-window samples
    w_low: torch.Tensor  # mean(w_low)
    dh_pos: torch.Tensor  # mean relu(e_high_T − e_high_S) — raw deficit before β-gain
    dl_pos: torch.Tensor  # mean relu(e_low_T  − e_low_S)
    in_window_count: torch.Tensor  # # of in-window samples this step


def apply_ca_band_deficit(
    delta_cfg: torch.Tensor,
    *,
    x_t: torch.Tensor,
    v_teacher_at_t: torch.Tensor,
    t: torch.Tensor,
    x_pred: torch.Tensor,
    tau_ca: torch.Tensor,
    beta: float,
    divisor: float,
    window_lo: float,
    window_hi: float,
) -> tuple[torch.Tensor, BandDeficitDiag]:
    """Return ``(reweighted_delta_cfg, diagnostics)``.

    The branch's gate (compile-stable bool) lives in the caller; this function
    assumes the feature is enabled.

    Inputs:

    * ``delta_cfg`` — original CA difference ``v_real_cond_ca − v_real_uncond``
      in the student's working dtype. Lives at τ_ca.
    * ``x_t`` — student input ``(B, 16, H, W)`` at sampler time ``t``.
    * ``v_teacher_at_t`` — no-grad teacher velocity at ``(x_t, t, c)``; the
      caller pays one extra forward to produce this.
    * ``t`` — student sampler time per batch (``[B]``).
    * ``x_pred`` — student's x0 estimate at ``t`` (already computed pre-CA).
    * ``tau_ca`` — per-batch τ_ca in [0, 1]; gates *reweighting* application.

    The FEI gap is computed at ``t`` (not τ_ca) so both x0 estimates live at
    the same sampler time — see module docstring. Outside
    ``[window_lo, window_hi)`` on **τ_ca** the per-sample weights collapse to
    1.0, so the recomposition rounds back to ``delta_cfg`` up to fp32 LP+HP
    roundoff.
    """
    B = delta_cfg.shape[0]
    h_lat, w_lat = x_pred.shape[-2], x_pred.shape[-1]
    sigma_low_val = fei_sigma_low(h_lat, w_lat, divisor)
    t_e_band = t.view(B, 1, 1, 1).float()

    with torch.no_grad():
        # Teacher x0 at student's t:  x_t − t · v_teacher_at_t.
        # Same sampler time as the student's x_pred — no operator mismatch.
        x0_T = x_t.float() - t_e_band * v_teacher_at_t.float()
        e_T = compute_fei_2band(x0_T, sigma_low_val)  # [B, 2]
        # Student x0 at t is just x_pred (already computed in step 1).
        e_S = compute_fei_2band(x_pred.detach().float(), sigma_low_val)
        # Per-sample relu deficits. Because e_low + e_high ≡ 1, at most one of
        # these is positive per sample — the other is 0.
        dlow_pos = (e_T[:, 0] - e_S[:, 0]).clamp_min(0.0)  # [B]
        dhigh_pos = (e_T[:, 1] - e_S[:, 1]).clamp_min(0.0)
        # τ_ca window mask: 1 if τ_ca ∈ [window_lo, window_hi), else 0. Gate
        # stays inside no_grad so the mask is purely numerical (no autograd
        # branches). The measurement above is unconditional in t; only the
        # *application* to δ_cfg is τ_ca-gated.
        in_window = (
            (tau_ca.float() >= window_lo) & (tau_ca.float() < window_hi)
        ).float()  # [B]
        iw4 = in_window.view(B, 1, 1, 1)
        # Effective weights — outside the window they collapse to 1.0, giving an
        # identity recomposition (LP+HP = δ_cfg) up to fp32 LP+HP roundoff.
        w_low = 1.0 + beta * iw4 * dlow_pos.view(B, 1, 1, 1)
        w_high = 1.0 + beta * iw4 * dhigh_pos.view(B, 1, 1, 1)

    # LP/HP split done in fp32 so the recomposition rounds back to δ_cfg cleanly
    # when w_low = w_high = 1.
    delta_cfg_f32 = delta_cfg.float()
    delta_cfg_lp = gaussian_blur_2d(delta_cfg_f32, sigma_low_val)
    delta_cfg_hp = delta_cfg_f32 - delta_cfg_lp
    delta_cfg_out = (w_low * delta_cfg_lp + w_high * delta_cfg_hp).to(delta_cfg.dtype)

    # Diagnostic reductions (cheap; reused by the logging accumulator).
    with torch.no_grad():
        in_window_count = in_window.sum()
        denom = in_window_count.clamp_min(1.0)
        diag = BandDeficitDiag(
            w_high=(in_window * w_high.view(B)).sum() / denom,
            w_low=(in_window * w_low.view(B)).sum() / denom,
            dh_pos=(in_window * dhigh_pos).sum() / denom,
            dl_pos=(in_window * dlow_pos).sum() / denom,
            in_window_count=in_window_count,
        )
    return delta_cfg_out, diag
