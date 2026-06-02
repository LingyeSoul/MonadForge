"""GPU-side accumulators + single-sync flush for the turbo training loop.

All accumulators live on-device; they're flushed in one stacked ``.tolist()``
at every ``log_interval`` so per-step CUDA syncs go to zero.

Health-scalar semantics (see proposal log for context):

* ``grad``      — overall DMD gradient magnitude into x_pred
* ``dm``        — DM regularizer strength (v_real − v_fake)
* ``xpred``     — x_pred dispersion: → 0 means collapse to mean, drifting upward
  means student is exploding.
* ``v_student`` — direct student velocity magnitude; runaway student manifests
  here before x_pred_std catches up (x_pred = x_t − t·v_student).

Fake-tracking ratios (real DMD health signals — ``loss_student`` is a
sign-random gradient vehicle, not a real loss):

* ``rel_gap``  — rms(τ·Δ_dm) / rms(τ·v_real_dm): fraction of teacher score the DM
  gap still represents. ↑ = fake lagging → bump fake.
* ``mag_ratio`` — rms(v_fake_dm) / rms(v_real_dm): ≈1 healthy; collapse/blow-up bad.
* ``cos``       — cosine(v_fake_dm, v_real_dm): ↓ = fake pointing the wrong way.

Mean-variance reg (lever B / paper Eq. 7; 0 when disabled):

* ``mv`` — the per-step Eq.7 KL value (pre-weight). Higher = the student's
  per-image stats are further from the real-latent target.

DP-DMD diversity loss:

* ``div`` — the first-step diversity MSE ‖v_first − v_target‖² (pre-weight).
  Falling = the student's step-1 velocity is converging on the teacher's
  K-step anchor (the diverse landing point).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class FlushedMetrics:
    fake: float
    grad: float
    dm: float
    xpred: float
    v_student: float
    rel_gap: float
    mag_ratio: float
    cos: float
    mv: float
    div: float


class TurboMetrics:
    """GPU-resident accumulators with a single-sync stacked flush."""

    def __init__(self, device: torch.device):
        z = lambda: torch.zeros((), device=device)  # noqa: E731
        # Always-on rms scalars.
        self.fake = z()
        self.grad = z()
        self.dm = z()
        self.xpred = z()
        self.v_student = z()
        # Fake-tracking.
        self.rel_gap = z()
        self.mag_ratio = z()
        self.cos = z()
        # Mean-variance reg (lever B / Eq.7); 0 when disabled.
        self.mv = z()
        # DP-DMD first-step diversity loss.
        self.div = z()

    @torch.no_grad()
    def accumulate_per_step(
        self,
        *,
        fake_loss_mean_t: torch.Tensor,
        grad_signal: torch.Tensor,
        delta_dm: torch.Tensor,
        x_pred: torch.Tensor,
        v_student: torch.Tensor,
        tau_dm_e: torch.Tensor,
        v_real_cond_dm: torch.Tensor,
        v_fake_cond_dm: torch.Tensor,
        mv_loss: torch.Tensor,
    ) -> None:
        eps_r = 1e-8
        self.fake.add_(fake_loss_mean_t.float())
        self.mv.add_(mv_loss.detach().float())
        self.grad.add_(grad_signal.float().pow(2).mean().sqrt())
        self.dm.add_(delta_dm.float().pow(2).mean().sqrt())
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
    def add_div(self, div_loss_t: torch.Tensor) -> None:
        """Accumulate the DP-DMD first-step diversity loss (pre-weight)."""
        self.div.add_(div_loss_t.detach().float())

    def flush(self, log_interval: int) -> FlushedMetrics:
        """One CUDA sync per log boundary: stack everything, read once."""
        packed = (
            torch.stack(
                [
                    self.fake,
                    self.grad,
                    self.dm,
                    self.xpred,
                    self.v_student,
                    self.rel_gap,
                    self.mag_ratio,
                    self.cos,
                    self.mv,
                    self.div,
                ]
            )
            / log_interval
        ).tolist()
        return FlushedMetrics(
            fake=packed[0],
            grad=packed[1],
            dm=packed[2],
            xpred=packed[3],
            v_student=packed[4],
            rel_gap=packed[5],
            mag_ratio=packed[6],
            cos=packed[7],
            mv=packed[8],
            div=packed[9],
        )

    def reset(self) -> None:
        for t in (
            self.fake, self.grad, self.dm, self.xpred, self.v_student,
            self.rel_gap, self.mag_ratio, self.cos, self.mv, self.div,
        ):
            t.zero_()


def write_scalars(writer, m: FlushedMetrics, step: int) -> None:
    """Push every available scalar to TensorBoard at the canonical key names."""
    writer.add_scalar("train/fake_loss", m.fake, step)
    writer.add_scalar("train/grad_signal_rms", m.grad, step)
    writer.add_scalar("train/delta_dm_rms", m.dm, step)
    writer.add_scalar("train/x_pred_std", m.xpred, step)
    writer.add_scalar("train/v_student_rms", m.v_student, step)
    writer.add_scalar("train/dm_rel_gap", m.rel_gap, step)
    writer.add_scalar("train/dm_mag_ratio", m.mag_ratio, step)
    writer.add_scalar("train/dm_cos", m.cos, step)
    writer.add_scalar("train/mean_var_kl", m.mv, step)
    writer.add_scalar("train/div_loss", m.div, step)


def tqdm_postfix(m: FlushedMetrics) -> dict:
    """tqdm postfix dict — short keys for the live progress line."""
    postfix = {
        "g": f"{m.grad:.2e}",
        "relg": f"{m.rel_gap:.3f}",
        "cos": f"{m.cos:.3f}",
        "xp": f"{m.xpred:.3f}",
        "fake": f"{m.fake:.2e}",
    }
    if m.mv > 0:
        postfix["mv"] = f"{m.mv:.3f}"
    if m.div > 0:
        postfix["div"] = f"{m.div:.3f}"
    return postfix
