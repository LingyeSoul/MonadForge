"""TurboMetrics — ScalarAccumulator-backed flush keeps the typed contract.

Guards the migration off the hand-rolled 12-field stack/reset triplication:
the field set must stay complete (disabled div/gan paths included) and the
accumulate→flush arithmetic must match the documented per-step formulas.
"""

from dataclasses import fields

import pytest
import torch

from scripts.distill_turbo.metrics import FlushedMetrics, TurboMetrics

DEVICE = torch.device("cpu")


def _step_inputs(scale: float):
    g = torch.manual_seed(int(scale * 1000))
    rand = lambda: torch.rand(2, 4, generator=g) * scale  # noqa: E731
    return dict(
        fake_loss_mean_t=torch.tensor(scale),
        grad_signal=rand(),
        delta_dm=rand(),
        x_pred=rand(),
        v_student=rand(),
        tau_dm_e=rand(),
        v_real_cond_dm=rand(),
        v_fake_cond_dm=rand(),
        mv_loss=torch.tensor(scale * 0.5),
    )


def test_flush_returns_complete_record_without_div_or_gan():
    m = TurboMetrics(DEVICE)
    m.accumulate_per_step(**_step_inputs(1.0))
    out = m.flush(log_interval=1)
    assert isinstance(out, FlushedMetrics)
    # div / gan paths never fired → still present, exactly 0.
    assert out.div == 0.0
    assert out.gan_gen == 0.0 and out.gan_disc == 0.0
    # Every declared field is populated.
    for f in fields(FlushedMetrics):
        assert getattr(out, f.name) is not None


def test_accumulate_flush_matches_reference_formulas():
    log_interval = 3
    eps_r = 1e-8
    m = TurboMetrics(DEVICE)
    ref = {k: 0.0 for k in (f.name for f in fields(FlushedMetrics))}

    for i in range(log_interval):
        inp = _step_inputs(1.0 + i)
        m.accumulate_per_step(**inp)
        # Reference: replicate the documented per-step contributions.
        vr = inp["v_real_cond_dm"].float()
        vf = inp["v_fake_cond_dm"].float()
        dm_w = (inp["tau_dm_e"] * inp["delta_dm"].float()).pow(2).mean().sqrt()
        ref["fake"] += inp["fake_loss_mean_t"].item()
        ref["mv"] += inp["mv_loss"].item()
        ref["grad"] += inp["grad_signal"].float().pow(2).mean().sqrt().item()
        ref["dm"] += inp["delta_dm"].float().pow(2).mean().sqrt().item()
        ref["xpred"] += inp["x_pred"].float().std().item()
        ref["v_student"] += inp["v_student"].float().pow(2).mean().sqrt().item()
        ref["rel_gap"] += (
            dm_w / ((inp["tau_dm_e"] * vr).pow(2).mean().sqrt() + eps_r)
        ).item()
        ref["mag_ratio"] += (
            vf.pow(2).mean().sqrt() / (vr.pow(2).mean().sqrt() + eps_r)
        ).item()
        ref["cos"] += ((vf * vr).sum() / (vf.norm() * vr.norm() + eps_r)).item()
        # div + gan also exercised this step.
        m.add_div(torch.tensor(0.2 * (i + 1)))
        ref["div"] += 0.2 * (i + 1)
        m.add_gan(torch.tensor(0.3 * (i + 1)), torch.tensor(0.4 * (i + 1)))
        ref["gan_gen"] += 0.3 * (i + 1)
        ref["gan_disc"] += 0.4 * (i + 1)

    out = m.flush(log_interval)
    for name, total in ref.items():
        assert getattr(out, name) == pytest.approx(total / log_interval, rel=1e-5), name


def test_reset_zeroes_then_reaccumulates():
    m = TurboMetrics(DEVICE)
    m.accumulate_per_step(**_step_inputs(2.0))
    m.flush(log_interval=1)
    m.reset()
    out = m.flush(log_interval=1)
    for f in fields(FlushedMetrics):
        assert getattr(out, f.name) == pytest.approx(0.0)
    # Accumulators are reusable after reset.
    m.accumulate_per_step(**_step_inputs(2.0))
    assert m.flush(log_interval=1).fake == pytest.approx(2.0)
