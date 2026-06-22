from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from library.training.loop import _check_loss_finite, _should_check_loss_finite


def test_check_loss_finite_accepts_finite_loss():
    _check_loss_finite(torch.tensor(1.0))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_check_loss_finite_raises_on_nonfinite_loss(value):
    with pytest.raises(FloatingPointError, match="non-finite training loss"):
        _check_loss_finite(torch.tensor(value))


def test_check_loss_finite_fp16_message_adds_fp16_hint():
    with pytest.raises(FloatingPointError, match="fp16 autocast is active"):
        _check_loss_finite(torch.tensor(float("nan")), mixed_precision="fp16")


def test_check_loss_finite_non_fp16_message_is_generic():
    with pytest.raises(FloatingPointError) as exc:
        _check_loss_finite(torch.tensor(float("nan")), mixed_precision="bf16")
    assert "fp16 autocast is active" not in str(exc.value)


def test_should_check_loss_finite_matches_log_cadence():
    state = SimpleNamespace(
        accelerator=SimpleNamespace(sync_gradients=True),
        args=SimpleNamespace(log_every_n_steps=2, max_train_steps=5),
        global_step=0,
    )
    assert not _should_check_loss_finite(state)

    state.global_step = 1
    assert _should_check_loss_finite(state)

    state.global_step = 4
    assert _should_check_loss_finite(state)


def test_should_check_loss_finite_skips_accumulation_microsteps():
    state = SimpleNamespace(
        accelerator=SimpleNamespace(sync_gradients=False),
        args=SimpleNamespace(log_every_n_steps=1, max_train_steps=1),
        global_step=0,
    )
    assert not _should_check_loss_finite(state)
