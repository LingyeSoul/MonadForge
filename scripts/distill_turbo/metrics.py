"""GPU-side accumulators + single-sync flush for the turbo training loop.

All accumulators live on-device; they're flushed in one stacked ``.tolist()``
at every ``log_interval`` so per-step CUDA syncs go to zero.

Health-scalar semantics (see proposal log for context):

* ``grad``      — overall DMD2 gradient magnitude into x_pred
* ``dm``        — DM regularizer strength (v_real − v_fake)
* ``cfg``       — CA branch strength (CFG bake direction)
* ``xpred``     — x_pred dispersion: → 0 means collapse to mean, drifting upward
  means student is exploding.
* ``v_student`` — direct student velocity magnitude; runaway student manifests
  here before x_pred_std catches up (x_pred = x_t − t·v_student).

Fake-tracking ratios (real DMD2 health signals — ``loss_student`` is a
sign-random gradient vehicle, not a real loss):

* ``rel_gap``  — rms(τ·Δ_dm) / rms(τ·v_real_dm): fraction of teacher score the DM
  gap still represents. ↑ = fake lagging → bump fake.
* ``mag_ratio`` — rms(v_fake_dm) / rms(v_real_dm): ≈1 healthy; collapse/blow-up bad.
* ``cos``       — cosine(v_fake_dm, v_real_dm): ↓ = fake pointing the wrong way.
* ``dm_to_ca``  — effective DM vs CA magnitude. Decoupled DMD wants CA as the
  engine and DM as the shield, so DM ≳ CA for long stretches is a red flag.
  Accumulated only on do_ca steps (own denominator).

CA band-deficit diagnostics (item 2; only logged when ``band_steps > 0``):

* ``band_w_high`` — mean(w_high) over in-window samples (active arm on turbo_C)
* ``band_w_low``  — mean(w_low) (≈ 1.0 expected under turbo_C)
* ``band_dh_pos`` — mean relu(e_high_T − e_high_S) — raw deficit before β-gain
* ``band_dl_pos`` — mean relu(e_low_T − e_low_S)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class FlushedMetrics:
    fake: float
    grad: float
    dm: float
    cfg: float
    xpred: float
    v_student: float
    rel_gap: float
    mag_ratio: float
    cos: float
    dm_to_ca: float
    ca_steps: float
    band_w_high: float
    band_w_low: float
    band_dh_pos: float
    band_dl_pos: float
    band_steps: float
    alpha: float


class TurboMetrics:
    """GPU-resident accumulators with a single-sync stacked flush."""

    def __init__(self, device: torch.device):
        z = lambda: torch.zeros((), device=device)  # noqa: E731
        # Always-on rms scalars.
        self.fake = z()
        self.grad = z()
        self.dm = z()
        self.cfg = z()
        self.xpred = z()
        self.v_student = z()
        # Fake-tracking.
        self.rel_gap = z()
        self.mag_ratio = z()
        self.cos = z()
        # CA-conditional (own denom).
        self.dm_to_ca = z()
        self.ca_steps = z()
        # CA-band-conditional (own denom).
        self.band_w_high = z()
        self.band_w_low = z()
        self.band_dh_pos = z()
        self.band_dl_pos = z()
        self.band_steps = z()
        # Pure-Python (no GPU work).
        self.alpha = 0.0

    @torch.no_grad()
    def accumulate_per_step(
        self,
        *,
        fake_loss_mean_t: torch.Tensor,
        grad_signal: torch.Tensor,
        delta_dm: torch.Tensor,
        delta_cfg: torch.Tensor,
        x_pred: torch.Tensor,
        v_student: torch.Tensor,
        tau_dm_e: torch.Tensor,
        v_real_cond_dm: torch.Tensor,
        v_fake_cond_dm: torch.Tensor,
    ) -> None:
        eps_r = 1e-8
        self.fake.add_(fake_loss_mean_t.float())
        self.grad.add_(grad_signal.float().pow(2).mean().sqrt())
        self.dm.add_(delta_dm.float().pow(2).mean().sqrt())
        self.cfg.add_(delta_cfg.float().pow(2).mean().sqrt())
        self.xpred.add_(x_pred.detach().float().std())
        self.v_student.add_(v_student.detach().float().pow(2).mean().sqrt())
        # Fake-tracking diagnostics at the DM eval point.
        vr = v_real_cond_dm.float()
        vf = v_fake_cond_dm.float()
        dm_w = (tau_dm_e * delta_dm.float()).pow(2).mean().sqrt()
        self.rel_gap.add_(dm_w / ((tau_dm_e * vr).pow(2).mean().sqrt() + eps_r))
        self.mag_ratio.add_(
            vf.pow(2).mean().sqrt() / (vr.pow(2).mean().sqrt() + eps_r)
        )
        self.cos.add_((vf * vr).sum() / (vf.norm() * vr.norm() + eps_r))

    @torch.no_grad()
    def accumulate_dm_to_ca(
        self,
        *,
        tau_ca_e: torch.Tensor,
        alpha_eff: float,
        delta_cfg: torch.Tensor,
        delta_dm: torch.Tensor,
        tau_dm_e: torch.Tensor,
    ) -> None:
        eps_r = 1e-8
        dm_w = (tau_dm_e * delta_dm.float()).pow(2).mean().sqrt()
        ca_w = (tau_ca_e * (alpha_eff - 1.0) * delta_cfg.float()).pow(2).mean().sqrt()
        self.dm_to_ca.add_(dm_w / (ca_w + eps_r))
        self.ca_steps.add_(1.0)

    @torch.no_grad()
    def accumulate_band(self, diag) -> None:
        """Only count this step if any sample was actually weighted.

        Otherwise the in-window-mean is undefined (we clamped denom to 1.0 in
        ca_band.apply_ca_band_deficit, but the per-step mean would be 0/1 = noise).
        """
        step_active = (diag.in_window_count > 0).float()
        self.band_w_high.add_(diag.w_high * step_active)
        self.band_w_low.add_(diag.w_low * step_active)
        self.band_dh_pos.add_(diag.dh_pos * step_active)
        self.band_dl_pos.add_(diag.dl_pos * step_active)
        self.band_steps.add_(step_active)

    def add_alpha(self, alpha_eff: float) -> None:
        self.alpha += alpha_eff

    def flush(self, log_interval: int) -> FlushedMetrics:
        """One CUDA sync per log boundary: stack everything, read once."""
        stacked = (
            torch.stack(
                [
                    self.fake,
                    self.grad,
                    self.dm,
                    self.cfg,
                    self.xpred,
                    self.v_student,
                    self.rel_gap,
                    self.mag_ratio,
                    self.cos,
                ]
            )
            / log_interval
        )
        # dm_to_ca has its own denominator (only do_ca steps contribute).
        dm_to_ca = self.dm_to_ca / self.ca_steps.clamp(min=1.0)
        # Band diagnostics have their own denominator (in-window-active steps).
        band_denom = self.band_steps.clamp(min=1.0)
        band_w_high_avg = self.band_w_high / band_denom
        band_w_low_avg = self.band_w_low / band_denom
        band_dh_pos_avg = self.band_dh_pos / band_denom
        band_dl_pos_avg = self.band_dl_pos / band_denom
        packed = torch.cat(
            [
                stacked,
                dm_to_ca.reshape(1),
                self.ca_steps.reshape(1),
                band_w_high_avg.reshape(1),
                band_w_low_avg.reshape(1),
                band_dh_pos_avg.reshape(1),
                band_dl_pos_avg.reshape(1),
                self.band_steps.reshape(1),
            ]
        ).tolist()
        return FlushedMetrics(
            fake=packed[0],
            grad=packed[1],
            dm=packed[2],
            cfg=packed[3],
            xpred=packed[4],
            v_student=packed[5],
            rel_gap=packed[6],
            mag_ratio=packed[7],
            cos=packed[8],
            dm_to_ca=packed[9],
            ca_steps=packed[10],
            band_w_high=packed[11],
            band_w_low=packed[12],
            band_dh_pos=packed[13],
            band_dl_pos=packed[14],
            band_steps=packed[15],
            alpha=self.alpha / log_interval,
        )

    def reset(self) -> None:
        for t in (
            self.fake, self.grad, self.dm, self.cfg, self.xpred, self.v_student,
            self.rel_gap, self.mag_ratio, self.cos,
            self.dm_to_ca, self.ca_steps,
            self.band_w_high, self.band_w_low, self.band_dh_pos, self.band_dl_pos,
            self.band_steps,
        ):
            t.zero_()
        self.alpha = 0.0


def write_scalars(writer, m: FlushedMetrics, step: int) -> None:
    """Push every available scalar to TensorBoard at the canonical key names."""
    writer.add_scalar("train/fake_loss", m.fake, step)
    writer.add_scalar("train/alpha_eff", m.alpha, step)
    writer.add_scalar("train/grad_signal_rms", m.grad, step)
    writer.add_scalar("train/delta_dm_rms", m.dm, step)
    writer.add_scalar("train/delta_cfg_rms", m.cfg, step)
    writer.add_scalar("train/x_pred_std", m.xpred, step)
    writer.add_scalar("train/v_student_rms", m.v_student, step)
    writer.add_scalar("train/dm_rel_gap", m.rel_gap, step)
    writer.add_scalar("train/dm_mag_ratio", m.mag_ratio, step)
    writer.add_scalar("train/dm_cos", m.cos, step)
    if m.ca_steps > 0:
        writer.add_scalar("train/dm_to_ca", m.dm_to_ca, step)
    if m.band_steps > 0:
        # Active arm under turbo_C; both logged so we catch a post-finetune
        # sign flip (student_under_low) automatically.
        writer.add_scalar("train/band_w_high", m.band_w_high, step)
        writer.add_scalar("train/band_w_low", m.band_w_low, step)
        writer.add_scalar("train/band_dh_pos", m.band_dh_pos, step)
        writer.add_scalar("train/band_dl_pos", m.band_dl_pos, step)


def tqdm_postfix(m: FlushedMetrics) -> dict:
    """tqdm postfix dict — short keys for the live progress line."""
    postfix = {
        "g": f"{m.grad:.2e}",
        "relg": f"{m.rel_gap:.3f}",
        "cos": f"{m.cos:.3f}",
        "dmca": f"{m.dm_to_ca:.2f}",
        "xp": f"{m.xpred:.3f}",
        "fake": f"{m.fake:.2e}",
    }
    if m.band_steps > 0:
        postfix["wh"] = f"{m.band_w_high:.3f}"
    return postfix
