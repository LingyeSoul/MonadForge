"""ScalarAccumulator — single-sync named flush + parity with the old pack."""

import pytest
import torch

from library.training.accumulator import ScalarAccumulator

DEVICE = torch.device("cpu")


def test_scalar_add_and_flush():
    acc = ScalarAccumulator(DEVICE)
    for _ in range(3):
        acc.add("loss", 2.0)
        acc.add("other", torch.tensor(0.5))
    m = acc.flush()
    assert m["loss"] == pytest.approx(6.0)
    assert m["other"] == pytest.approx(1.5)
    # flush does not reset.
    assert acc.flush()["loss"] == pytest.approx(6.0)


def test_vector_entry_add_at():
    acc = ScalarAccumulator(DEVICE)
    acc.add_at("stage_loss", 0, 1.0, width=3)
    acc.add_at("stage_loss", 0, 2.0, width=3)
    acc.add_at("stage_loss", 2, 5.0, width=3)
    acc.add_at("stage_cnt", 0, 1.0, width=3)
    acc.add_at("stage_cnt", 0, 1.0, width=3)
    acc.add_at("stage_cnt", 2, 1.0, width=3)
    m = acc.flush()
    assert m["stage_loss"] == [3.0, 0.0, 5.0]
    assert m["stage_cnt"] == [2.0, 0.0, 1.0]


def test_flush_reset_zeroes():
    acc = ScalarAccumulator(DEVICE)
    acc.add("x", 4.0)
    assert acc.flush_reset()["x"] == pytest.approx(4.0)
    # After reset the entry persists but reads zero.
    assert acc.flush()["x"] == pytest.approx(0.0)
    acc.add("x", 1.0)
    assert acc.flush()["x"] == pytest.approx(1.0)


def test_empty_flush():
    assert ScalarAccumulator(DEVICE).flush() == {}


def test_missing_key_is_absent_not_error():
    # Mirrors the gate-off path: gate_w/ungated are never added, so callers
    # must use .get(...) — the keys simply don't appear.
    acc = ScalarAccumulator(DEVICE)
    acc.add("loss", 1.0)
    m = acc.flush()
    assert "gate_w" not in m
    assert m.get("gate_w", 0.0) == 0.0


def test_shape_conflict_raises():
    acc = ScalarAccumulator(DEVICE)
    acc.add("v", 1.0)
    with pytest.raises(ValueError):
        acc.add_at("v", 0, 1.0, width=4)  # scalar reused as vector
    acc2 = ScalarAccumulator(DEVICE)
    acc2.add_at("w", 0, 1.0, width=3)
    with pytest.raises(ValueError):
        acc2.add_at("w", 0, 1.0, width=4)  # width mismatch


def test_parity_with_old_packed_layout():
    """The new named flush must reproduce the retired torch.cat([...]) pack.

    Old distill_spd layout (the magic-index version this replaces):
        packed = cat([loss/log_interval, sqrt(up_sq), sqrt(down_sq),
                      gate_w/n_micro, ungated/n_micro, stage_means, stage_cnt])
    """
    log_interval, grad_accum, n_stages = 4, 2, 3
    n_micro = log_interval * grad_accum

    torch.manual_seed(0)
    step_losses = [torch.rand(()) for _ in range(log_interval)]
    gate_ws = [torch.rand(()) for _ in range(n_micro)]
    ungated = [torch.rand(()) for _ in range(n_micro)]
    stage_cnt = torch.tensor([3.0, 0.0, 5.0])
    # Real invariant: loss-sum and count increment in lockstep, so a count-0
    # stage always has loss-sum 0 (nothing was ever added to it).
    stage_loss = torch.rand(n_stages) * (stage_cnt > 0)
    up_sq = torch.rand(()) * 10
    down_sq = torch.rand(()) * 10

    # --- old path ---
    stage_means = stage_loss / stage_cnt.clamp(min=1)
    old = torch.cat(
        [
            (sum(step_losses) / log_interval).reshape(1),
            up_sq.sqrt().reshape(1),
            down_sq.sqrt().reshape(1),
            (sum(gate_ws) / n_micro).reshape(1),
            (sum(ungated) / n_micro).reshape(1),
            stage_means,
            stage_cnt,
        ]
    ).tolist()
    old_avg, old_up, old_down = old[0], old[1], old[2]
    old_gate, old_ung = old[3], old[4]
    old_stage_vals = old[5 : 5 + n_stages]
    old_stage_cnts = old[5 + n_stages : 5 + 2 * n_stages]

    # --- new path ---
    acc = ScalarAccumulator(DEVICE)
    for sl in step_losses:
        acc.add("loss", sl)
    for gw, ug in zip(gate_ws, ungated):
        acc.add("gate_w", gw)
        acc.add("ungated", ug)
    for si in range(n_stages):
        if stage_cnt[si] > 0:
            acc.add_at("stage_loss", si, stage_loss[si], width=n_stages)
        for _ in range(int(stage_cnt[si])):
            acc.add_at("stage_cnt", si, 1.0, width=n_stages)
    acc.add("up_sq", up_sq)
    acc.add("down_sq", down_sq)
    m = acc.flush_reset()

    new_avg = m["loss"] / log_interval
    new_up = m["up_sq"] ** 0.5
    new_down = m["down_sq"] ** 0.5
    new_gate = m["gate_w"] / n_micro
    new_ung = m["ungated"] / n_micro
    new_stage_cnts = m["stage_cnt"]
    new_stage_vals = [
        ls / c if c > 0 else 0.0 for ls, c in zip(m["stage_loss"], new_stage_cnts)
    ]

    assert new_avg == pytest.approx(old_avg)
    assert new_up == pytest.approx(old_up)
    assert new_down == pytest.approx(old_down)
    assert new_gate == pytest.approx(old_gate)
    assert new_ung == pytest.approx(old_ung)
    assert new_stage_cnts == pytest.approx(old_stage_cnts)
    # Old code put stage_means for *all* stages (untouched → 0 via /clamp(min=1)
    # with 0 numerator); new code yields 0.0 for count==0. Both are 0 there.
    assert new_stage_vals == pytest.approx(old_stage_vals)
