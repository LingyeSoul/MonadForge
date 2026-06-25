"""Tests for ``generate_step_logs`` lr keying, esp. Prodigy / D-Adaptation.

Regression for the "WebUI always shows lr=1.0 with Prodigy" bug. The trainer
assembles the per-step ``logs`` dict here; the dashboard reads the plain
``lr/<desc>`` key. For adaptive optimizers (Prodigy, D-Adaptation) the base
``lr`` (e.g. 1.0) is only a multiplier on the optimizer's internal distance
estimate ``d`` — the real lr applied to params is ``d * lr``. The dict must
emit that effective value under ``lr/<desc>`` (and keep the base lr under
``lr/base/<desc>``) so the dashboard tracks the real, rising lr instead of a
flat 1.0.

These exercise ``generate_step_logs`` directly with stub scheduler / args, so
no model or GPU is needed.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from library.training.log_dispatch import generate_step_logs


def _make_lr_scheduler(*, lr=1.0, d=0.01, n_groups=1):
    """Stub scheduler exposing ``get_last_lr`` + ``optimizers[-1].param_groups``."""
    sched = MagicMock()
    sched.get_last_lr.return_value = [lr] * n_groups
    group = {"lr": lr, "d": d}
    sched.optimizers = [MagicMock(param_groups=[group for _ in range(n_groups)])]
    return sched


def _make_args(optimizer_type="Prodigy"):
    return SimpleNamespace(
        optimizer_type=optimizer_type,
        network_train_unet_only=True,
        vr_loss_weight=0.0,
    )


def test_prodigy_emits_effective_lr_under_plain_key():
    """``lr/unet`` must carry the effective ``d * lr``, not the base multiplier.

    Base lr 1.0, d=0.01 → effective lr 0.01. Without the fix this would be 1.0
    (the dashboard's "always shows lr=1.0 with Prodigy" symptom).
    """
    sched = _make_lr_scheduler(lr=1.0, d=0.01)
    args = _make_args("Prodigy")

    logs = generate_step_logs(
        args,
        current_loss=0.1,
        avr_loss=0.12,
        lr_scheduler=sched,
        lr_descriptions=["unet"],
    )

    assert logs["lr/unet"] == pytest.approx(0.01), (
        "lr/unet must be the effective lr (d * lr = 0.01), not the base 1.0"
    )


def test_prodigy_keeps_base_lr_under_separate_key():
    """The raw base lr must still be available under ``lr/base/<desc>``."""
    sched = _make_lr_scheduler(lr=1.0, d=0.01)
    args = _make_args("Prodigy")

    logs = generate_step_logs(
        args,
        current_loss=0.1,
        avr_loss=0.12,
        lr_scheduler=sched,
        lr_descriptions=["unet"],
    )

    assert logs["lr/base/unet"] == pytest.approx(1.0)
    # Backward-compat alias preserved for existing tensorboard curves.
    assert logs["lr/d*lr/unet"] == pytest.approx(0.01)


@pytest.mark.parametrize("optimizer_type", ["DAdaptAdam", "DAdaptation", "DAdaptSGD"])
def test_dadapt_variants_all_emit_effective_lr(optimizer_type):
    """Every D-Adaptation variant applies ``d * lr`` — the fix must cover all."""
    sched = _make_lr_scheduler(lr=1.0, d=0.005)
    args = _make_args(optimizer_type)

    logs = generate_step_logs(
        args,
        current_loss=0.1,
        avr_loss=0.12,
        lr_scheduler=sched,
        lr_descriptions=["unet"],
    )

    assert logs["lr/unet"] == pytest.approx(0.005)


def test_non_adaptive_optimizer_keeps_base_lr():
    """AdamW must keep the plain base lr under ``lr/<desc>`` — no d multiplier.

    Guards against the Prodigy fix bleeding into ordinary optimizers.
    """
    sched = _make_lr_scheduler(lr=1e-4, d=1.0)
    args = _make_args("AdamW")

    logs = generate_step_logs(
        args,
        current_loss=0.1,
        avr_loss=0.12,
        lr_scheduler=sched,
        lr_descriptions=["unet"],
    )

    assert logs["lr/unet"] == pytest.approx(1e-4)
    # No d-adaptation keys emitted for a plain optimizer.
    assert "lr/base/unet" not in logs
    assert "lr/d*lr/unet" not in logs


def test_prodigy_plus_schedulefree_emits_effective_lr():
    """ProdigyPlusScheduleFree must also report d*lr, not base lr.

    ``is_d_adaptation_optimizer`` uses ``startswith("prodigy")`` so it matches
    both ``"Prodigy"`` and ``"ProdigyPlusScheduleFree"``. Without the fix the
    dashboard would show a flat 1.0 for this optimizer variant too.
    """
    sched = _make_lr_scheduler(lr=1.0, d=0.01)
    args = _make_args("ProdigyPlusScheduleFree")

    logs = generate_step_logs(
        args,
        current_loss=0.1,
        avr_loss=0.12,
        lr_scheduler=sched,
        lr_descriptions=["unet"],
    )

    assert logs["lr/unet"] == pytest.approx(0.01), (
        "ProdigyPlusScheduleFree must emit effective lr (d * lr = 0.01), "
        "not the base multiplier (1.0)"
    )
    assert logs["lr/base/unet"] == pytest.approx(1.0)
    assert logs["lr/d*lr/unet"] == pytest.approx(0.01)
