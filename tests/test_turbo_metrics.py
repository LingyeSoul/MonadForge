"""TurboMetrics — ScalarAccumulator-backed flush keeps the typed contract.

Guards the migration off the hand-rolled 12-field stack/reset triplication:
the field set must stay complete (disabled div/gan paths included) and the
accumulate→flush arithmetic must match the documented per-step formulas.
"""

from dataclasses import fields

import pytest
import torch

from scripts.distill_turbo.metrics import (
    TAU_PROFILE_BINS,
    FlushedMetrics,
    TauBinCriticLoss,
    TurboMetrics,
    console_step_line,
    tqdm_postfix,
    tqdm_rate,
)

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
        m.add_gan(
            torch.tensor(0.3 * (i + 1)),
            torch.tensor(0.4 * (i + 1)),
            torch.tensor(0.5 * (i + 1)),
            torch.tensor(0.6 * (i + 1)),
        )
        ref["gan_gen"] += 0.3 * (i + 1)
        ref["gan_disc"] += 0.4 * (i + 1)
        ref["gan_margin"] += 0.5 * (i + 1)
        ref["gan_spread"] += 0.6 * (i + 1)

    out = m.flush(log_interval)
    for name, total in ref.items():
        assert getattr(out, name) == pytest.approx(total / log_interval, rel=1e-5), name


class _SpyWriter:
    def __init__(self):
        self.scalars = {}

    def add_scalar(self, key, value, step):
        self.scalars[key] = (value, step)


def test_tau_bins_route_by_tau():
    prof = TauBinCriticLoss(DEVICE)
    # One loss per bin, τ at each bin's center → per-bin mean == that loss.
    for i in range(TAU_PROFILE_BINS):
        tau = torch.tensor([(i + 0.5) / TAU_PROFILE_BINS])
        prof.add(torch.tensor(float(i + 1)), tau)
    sums, cnts = prof.flush()
    assert cnts == [1.0] * TAU_PROFILE_BINS
    assert sums == pytest.approx([float(i + 1) for i in range(TAU_PROFILE_BINS)])


def test_tau_bins_edge_values():
    prof = TauBinCriticLoss(DEVICE)
    prof.add(torch.tensor(1.0), torch.tensor([0.0]))  # → bin 0
    prof.add(torch.tensor(2.0), torch.tensor([1.0]))  # τ=1.0 → LAST bin, not nbins
    sums, cnts = prof.flush()
    assert cnts[0] == 1.0 and sums[0] == pytest.approx(1.0)
    assert cnts[-1] == 1.0 and sums[-1] == pytest.approx(2.0)
    assert sum(cnts) == 2.0


def test_tau_bins_batch_attribution_and_mean():
    prof = TauBinCriticLoss(DEVICE, nbins=4)
    # B=2: the batch-mean scalar loss is attributed to BOTH samples' bins.
    prof.add(torch.tensor(3.0), torch.tensor([0.1, 0.9]))
    # Second update into bin 0 only → bin-0 mean = (3 + 5) / 2.
    prof.add(torch.tensor(5.0), torch.tensor([0.2]))
    w = _SpyWriter()
    prof.write(w, step=7)
    assert w.scalars["train/fake_loss_tau0"] == (pytest.approx(4.0), 7)
    assert w.scalars["train/fake_loss_tau3"] == (pytest.approx(3.0), 7)
    # Empty bins are skipped, not written as 0.
    assert "train/fake_loss_tau1" not in w.scalars
    assert "train/fake_loss_tau2" not in w.scalars


def test_tau_bins_write_resets_even_without_writer():
    prof = TauBinCriticLoss(DEVICE, prefix="warmup/fake_loss_tau")
    prof.add(torch.tensor(2.0), torch.tensor([0.5]))
    prof.write(None, step=1)  # no writer (--no_log) must still reset
    sums, cnts = prof.flush()
    assert sums == [0.0] * TAU_PROFILE_BINS
    assert cnts == [0.0] * TAU_PROFILE_BINS
    # Reusable after the flush, with the custom prefix.
    prof.add(torch.tensor(2.0), torch.tensor([0.5]))
    w = _SpyWriter()
    prof.write(w, step=2)
    assert w.scalars["warmup/fake_loss_tau4"] == (pytest.approx(2.0), 2)


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


# --- console step line (issue #2: tqdm's \r line is invisible in a redirected
# log, so the same metrics are mirrored as a greppable logger.info line) ---


def _metrics(**over) -> FlushedMetrics:
    base = dict.fromkeys((f.name for f in fields(FlushedMetrics)), 0.0)
    base.update(fake=0.034, grad=1.2e-3, rel_gap=0.142, cos=0.988, xpred=0.98)
    base.update(over)
    return FlushedMetrics(**base)


def test_console_step_line_is_greppable_and_carries_postfix_stats():
    line = console_step_line(_metrics(), step=3500, total=8000, rate=2.0)
    # `grep "step "` on a redirected log must find the step + total.
    assert line.startswith("step 3500/8000 (43.8%)")
    # every tqdm postfix stat is mirrored, so the file loses nothing vs the TTY
    for key, val in tqdm_postfix(_metrics()).items():
        assert f"{key}={val}" in line
    assert "2.00 it/s" in line
    assert "ETA 37:30" in line  # (8000-3500)/2 s


def test_console_step_line_without_a_rate_yet():
    line = console_step_line(_metrics(), step=10, total=8000, rate=None)
    assert "? it/s | ETA ?" in line


def test_tqdm_rate_falls_back_to_run_average_before_the_ema_warms():
    class _Bar:
        format_dict = {"rate": None, "elapsed": 4.0, "n": 120, "initial": 100}

    assert tqdm_rate(_Bar()) == pytest.approx(5.0)  # 20 steps / 4s

    class _Ema(_Bar):
        format_dict = {"rate": 3.0, "elapsed": 4.0, "n": 120, "initial": 100}

    assert tqdm_rate(_Ema()) == pytest.approx(3.0)  # EMA wins when present

    class _Cold(_Bar):
        format_dict = {"rate": None, "elapsed": 0.0, "n": 100, "initial": 100}

    assert tqdm_rate(_Cold()) is None  # no division by zero on the first line
