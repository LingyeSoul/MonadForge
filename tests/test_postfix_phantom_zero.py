"""Regression tests for the phantom-zero in the tqdm postfix.

Root cause (RCA): ``_log_step`` cached ``avr_loss`` / ``lr`` default to ``0.0``,
and ``state.progress_bar.update(1)`` — the refresh that flushes the postfix to
stderr — runs at the TOP of the ``sync_gradients`` block, i.e. BEFORE
``_log_step`` stages the postfix via ``set_postfix``. So the first logged
step's postfix is whatever ``set_postfix`` staged on the PREVIOUS step. On the
step before the first real log, that staged value was the uninitialized
``0.0`` default, which the WebUI's stdout parser read as a real
``avr_loss=0, lr=0`` data point and plotted as a zero on the loss/LR curves.

Data evidence: across 37 historical jobs every ``avr_loss=0`` tqdm line (that
is NOT the final step, where the cosine scheduler legitimately drives lr→0)
appeared at exactly step 2 — the first ``should_log_step=True`` after the
uninitialized default was staged.

Fix: track whether a real loss/lr has been observed (``_postfix_has_real``)
and omit the fields from the postfix until one has, so tqdm renders a
postfix-less bar instead of a phantom-zero one.

These tests exercise ``_log_step`` in isolation against a stub trainer / state
so no model or GPU is needed. They lock in:
  1. pre-first-log steps never stage ``avr_loss``/``lr`` (phantom zero blocked)
  2. the first logged step stages the REAL values (not the 0.0 default)
  3. subsequent steps stage the cached real value
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from library.training.loop import _log_step
from library.training.loss_recorder import LossRecorder


def _make_state(*, global_step: int, log_every: int = 2, max_train_steps: int = 66):
    """Minimal LoopState stub with the attrs _log_step touches."""
    lr_scheduler = MagicMock()
    lr_scheduler.get_last_lr.return_value = [1e-4]
    progress_bar = MagicMock()

    loss_recorder = LossRecorder()

    state = SimpleNamespace(
        args=SimpleNamespace(
            log_every_n_steps=log_every,
            max_train_steps=max_train_steps,
            scale_weight_norms=None,
            vr_loss_weight=0.0,
            optimizer_type="AdamW",
            network_train_unet_only=True,
        ),
        accelerator=SimpleNamespace(
            sync_gradients=True,
            unwrap_model=lambda net: SimpleNamespace(_use_hydra=False),
            is_main_process=True,
        ),
        network=object(),
        loss_recorder=loss_recorder,
        lr_scheduler=lr_scheduler,
        lr_descriptions=["unet"],
        optimizer=MagicMock(param_groups=[{"lr": 1e-4, "d": 1.0}]),
        progress_bar=progress_bar,
        is_tracking=False,
        global_step=global_step,
    )
    return state


def _fresh_trainer():
    """Trainer stub with NO cached postfix attrs — mirrors first-run state."""
    t = SimpleNamespace(_adapters=[], progress_sink=None)
    # Deliberately do NOT set _last_postfix_avr / _lr / _postfix_has_real —
    # _log_step must treat their absence as "never seen a real value".
    return t


def _loss_tensor(value: float):
    """Stub loss: ``_log_step`` only calls ``loss.detach().item()`` on log
    steps. Mimic that contract without pulling in torch (the loop module
    imports torch at the top, so this test still needs the project venv — but
    the stub keeps the test logic torch-free and fast)."""

    class _StubLoss:
        def detach(self):
            return self

        def item(self):
            return value

    return _StubLoss()


def _last_set_postfix_kwargs(progress_bar: MagicMock) -> dict:
    """Pull the kwargs dict from the last ``set_postfix`` call."""
    progress_bar.set_postfix.assert_called()
    # set_postfix(refresh=False, **{**max_mean_logs, **logs}) — kwargs only.
    return {
        k: v
        for k, v in progress_bar.set_postfix.call_args.kwargs.items()
        if k != "refresh"
    }


def test_pre_first_log_step_omits_postfix_fields():
    """Step 1 (before the first log at step 2) must NOT stage avr_loss/lr.

    This is the regression: previously it staged the 0.0 default, which the
    next ``update(1)`` flushed to stderr as ``avr_loss=0, lr=0``.
    """
    state = _make_state(global_step=1)  # 1 % 2 != 0 → not a log step
    trainer = _fresh_trainer()

    _log_step(
        trainer,
        state,
        loss=_loss_tensor(0.05),
        step=1,
        epoch=0,
        keys_scaled=None,
        mean_norm=None,
        maximum_norm=None,
        max_mean_logs={},
    )

    kwargs = _last_set_postfix_kwargs(state.progress_bar)
    assert "avr_loss" not in kwargs, (
        "phantom-zero regression: pre-first-log step staged avr_loss — the "
        "0.0 default would be flushed to stderr as a fake zero on the chart"
    )
    assert "lr" not in kwargs


def test_first_logged_step_stages_real_values():
    """Step 2 (first log step) must stage the REAL loss/lr, not the 0.0 default."""
    state = _make_state(global_step=2)  # 2 % 2 == 0 → log step
    trainer = _fresh_trainer()

    _log_step(
        trainer,
        state,
        loss=_loss_tensor(0.0722),
        step=2,
        epoch=0,
        keys_scaled=None,
        mean_norm=None,
        maximum_norm=None,
        max_mean_logs={},
    )

    kwargs = _last_set_postfix_kwargs(state.progress_bar)
    assert "avr_loss" in kwargs
    assert "lr" in kwargs
    # The real loss is 0.0722 — never the 0.0 default.
    assert kwargs["avr_loss"] != 0.0
    assert kwargs["lr"] != 0.0
    assert abs(kwargs["avr_loss"] - 0.0722) < 1e-6


def test_subsequent_non_log_step_stages_cached_real_values():
    """After the first log, a non-log step stages the cached real value.

    Ensures the fix didn't accidentally blank the postfix for all non-log
    steps after the first real value lands.
    """
    # First: run a log step to populate the cache.
    state = _make_state(global_step=2)
    trainer = _fresh_trainer()
    _log_step(
        trainer,
        state,
        loss=_loss_tensor(0.0722),
        step=2,
        epoch=0,
        keys_scaled=None,
        mean_norm=None,
        maximum_norm=None,
        max_mean_logs={},
    )

    # Now a non-log step 3 reuses the cache.
    state.global_step = 3
    state.progress_bar.reset_mock()
    _log_step(
        trainer,
        state,
        loss=_loss_tensor(0.09),
        step=3,
        epoch=0,
        keys_scaled=None,
        mean_norm=None,
        maximum_norm=None,
        max_mean_logs={},
    )

    kwargs = _last_set_postfix_kwargs(state.progress_bar)
    assert "avr_loss" in kwargs
    assert "lr" in kwargs
    # Cached real value from step 2, not the step-3 loss (not a log step).
    assert abs(kwargs["avr_loss"] - 0.0722) < 1e-6


def test_has_real_flag_persists_across_steps():
    """``_postfix_has_real`` flips True on first log and stays True."""
    state = _make_state(global_step=2)
    trainer = _fresh_trainer()
    assert not getattr(trainer, "_postfix_has_real", False)

    _log_step(
        trainer,
        state,
        loss=_loss_tensor(0.0722),
        step=2,
        epoch=0,
        keys_scaled=None,
        mean_norm=None,
        maximum_norm=None,
        max_mean_logs={},
    )
    assert trainer._postfix_has_real is True

    # A later non-log step must not reset it.
    state.global_step = 3
    _log_step(
        trainer,
        state,
        loss=_loss_tensor(0.09),
        step=3,
        epoch=0,
        keys_scaled=None,
        mean_norm=None,
        maximum_norm=None,
        max_mean_logs={},
    )
    assert trainer._postfix_has_real is True


# ---------------------------------------------------------------------------
# Prodigy / D-Adaptation: the postfix must show the *effective* lr (d * lr),
# not the base multiplier the user set (e.g. 1.0). Regression for the
# "WebUI always shows lr=1.0 with Prodigy" bug.
# ---------------------------------------------------------------------------


def test_prodigy_postfix_reports_effective_lr_not_base():
    """Prodigy: ``lr`` in the postfix must be ``d * lr``, not the base lr.

    Prodigy's user-set ``lr`` (e.g. 1.0) is only a multiplier on the
    optimizer's internal distance estimate ``d``; the real lr applied to the
    params is ``d * lr`` (``d`` grows from ~1e-6 upward). Reporting the base
    lr made the dashboard show a flat 1.0 for the whole run. Here the base lr
    is 1.0 and ``d`` is 0.01, so the postfix must report 0.01.
    """
    state = _make_state(global_step=2)  # log step
    state.args.optimizer_type = "Prodigy"
    state.lr_scheduler.get_last_lr.return_value = [1.0]
    # d=0.01 → effective lr 0.01, NOT the base 1.0.
    state.optimizer = MagicMock(param_groups=[{"lr": 1.0, "d": 0.01}])
    trainer = _fresh_trainer()

    _log_step(
        trainer,
        state,
        loss=_loss_tensor(0.05),
        step=2,
        epoch=0,
        keys_scaled=None,
        mean_norm=None,
        maximum_norm=None,
        max_mean_logs={},
    )

    kwargs = _last_set_postfix_kwargs(state.progress_bar)
    assert kwargs["lr"] == pytest.approx(0.01), (
        "Prodigy postfix must show the effective lr (d * lr = 0.01), not the "
        "base multiplier (1.0) — the dashboard would otherwise render a flat 1.0"
    )


@pytest.mark.parametrize("optimizer_type", ["DAdaptAdam", "DAdaptation"])
def test_dadapt_postfix_reports_effective_lr(optimizer_type):
    """All D-Adaptation variants apply ``d * lr`` — same fix must cover them."""
    state = _make_state(global_step=2)
    state.args.optimizer_type = optimizer_type
    state.lr_scheduler.get_last_lr.return_value = [1.0]
    state.optimizer = MagicMock(param_groups=[{"lr": 1.0, "d": 0.005}])
    trainer = _fresh_trainer()

    _log_step(
        trainer,
        state,
        loss=_loss_tensor(0.05),
        step=2,
        epoch=0,
        keys_scaled=None,
        mean_norm=None,
        maximum_norm=None,
        max_mean_logs={},
    )

    kwargs = _last_set_postfix_kwargs(state.progress_bar)
    assert kwargs["lr"] == pytest.approx(0.005)


def test_non_adaptive_optimizer_postfix_unchanged():
    """AdamW must keep reporting the plain base lr — no ``d`` multiplier.

    Guards against the Prodigy fix accidentally bleeding into ordinary
    optimizers: AdamW param_groups have no ``d`` field, so the effective-lr
    helper must return the base lr untouched.
    """
    state = _make_state(global_step=2)
    state.args.optimizer_type = "AdamW"
    state.lr_scheduler.get_last_lr.return_value = [1e-4]
    # No "d" key — a real AdamW group.
    state.optimizer = MagicMock(param_groups=[{"lr": 1e-4}])
    trainer = _fresh_trainer()

    _log_step(
        trainer,
        state,
        loss=_loss_tensor(0.05),
        step=2,
        epoch=0,
        keys_scaled=None,
        mean_norm=None,
        maximum_norm=None,
        max_mean_logs={},
    )

    kwargs = _last_set_postfix_kwargs(state.progress_bar)
    assert kwargs["lr"] == pytest.approx(1e-4)
