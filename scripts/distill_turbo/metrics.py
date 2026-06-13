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

from dataclasses import dataclass, fields

import torch

from library.training.accumulator import ScalarAccumulator


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
    gan_gen: float
    gan_disc: float
    repa: float
    repa_active: float


# Single source of truth for the accumulator key set — adding a logged scalar is
# one field on FlushedMetrics (+ its accumulate call + a write_scalars line),
# never a fourth edit to a parallel stack/reset list.
_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(FlushedMetrics))


class TurboMetrics:
    """GPU-resident accumulators with a single-sync flush.

    Thin wrapper over :class:`library.training.accumulator.ScalarAccumulator`
    that keeps the typed :class:`FlushedMetrics` public surface (consumed by
    :func:`write_scalars` / :func:`tqdm_postfix`). Every field is pre-touched so
    a disabled path (``div`` without DP-DMD, ``gan_*`` without the GAN) still
    flushes a complete record.
    """

    def __init__(self, device: torch.device):
        self._acc = ScalarAccumulator(device)
        for name in _FIELDS:
            self._acc.add(name, 0.0)

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
        self._acc.add("fake", fake_loss_mean_t.float())
        self._acc.add("mv", mv_loss.detach().float())
        self._acc.add("grad", grad_signal.float().pow(2).mean().sqrt())
        self._acc.add("dm", delta_dm.float().pow(2).mean().sqrt())
        self._acc.add("xpred", x_pred.detach().float().std())
        self._acc.add("v_student", v_student.detach().float().pow(2).mean().sqrt())
        # Fake-tracking diagnostics at the DM eval point.
        vr = v_real_cond_dm.float()
        vf = v_fake_cond_dm.float()
        dm_w = (tau_dm_e * delta_dm.float()).pow(2).mean().sqrt()
        self._acc.add("rel_gap", dm_w / ((tau_dm_e * vr).pow(2).mean().sqrt() + eps_r))
        self._acc.add(
            "mag_ratio", vf.pow(2).mean().sqrt() / (vr.pow(2).mean().sqrt() + eps_r)
        )
        self._acc.add("cos", (vf * vr).sum() / (vf.norm() * vr.norm() + eps_r))

    @torch.no_grad()
    def add_div(self, div_loss_t: torch.Tensor) -> None:
        """Accumulate the DP-DMD first-step diversity loss (pre-weight)."""
        self._acc.add("div", div_loss_t.detach().float())

    @torch.no_grad()
    def add_gan(
        self, gan_gen_loss: torch.Tensor, gan_disc_mean_t: torch.Tensor
    ) -> None:
        """Accumulate the GAN generator/discriminator losses (pre-weight)."""
        self._acc.add("gan_gen", gan_gen_loss.detach().float())
        self._acc.add("gan_disc", gan_disc_mean_t.detach().float())

    @torch.no_grad()
    def add_repa(self, repa_loss: torch.Tensor, *, active: bool) -> None:
        """Accumulate the REPA align loss (pre-weight) + its run indicator.

        The term is amortized (``repa.every_n``), so the flushed ``repa`` mean
        is diluted by skipped steps; consumers normalize by ``repa_active``
        (the active fraction) to recover the per-run align loss.
        """
        self._acc.add("repa", repa_loss.detach().float())
        self._acc.add("repa_active", 1.0 if active else 0.0)

    def flush(self, log_interval: int) -> FlushedMetrics:
        """One CUDA sync per log boundary: read every accumulator, mean it."""
        m = self._acc.flush()
        return FlushedMetrics(**{k: m[k] / log_interval for k in _FIELDS})

    def reset(self) -> None:
        self._acc.reset()


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
    writer.add_scalar("train/gan_gen_loss", m.gan_gen, step)
    writer.add_scalar("train/gan_disc_loss", m.gan_disc, step)
    writer.add_scalar("train/repa_active", m.repa_active, step)
    if m.repa_active > 0:
        # Normalize the interval mean to active steps so the curve reads as the
        # per-run align loss regardless of the every_n amortization.
        writer.add_scalar("train/repa_align_loss", m.repa / m.repa_active, step)


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
    if m.gan_gen != 0 or m.gan_disc != 0:
        postfix["gen"] = f"{m.gan_gen:.3f}"
        postfix["dsc"] = f"{m.gan_disc:.3f}"
    if m.repa_active > 0:
        postfix["repa"] = f"{m.repa / m.repa_active:.4f}"
    return postfix
