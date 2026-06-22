"""Drift guards for the WebUI's curated optimizer / scheduler option lists.

The WebUI ships a hand-curated ``_SELECT_OPTIONS`` for ``optimizer_type`` and
``lr_scheduler`` (``webui/services/config_service.py``). It must mirror exactly
what the trainer accepts: ``library/training/optimizers.py::get_optimizer`` and
``library/training/schedulers.py::get_scheduler_fix``. A prior drift after the
training-knobs prune (commit 772dda7) left the WebUI offering ``Adafactor`` and
five LR-scheduler names that the trainer had just dropped — selecting them
either crashed (Adafactor → AttributeError; inverse_sqrt → TypeError;
cosine_with_min_lr → ValueError) or silently dropped their parameters
(cosine_with_restarts / polynomial / warmup_stable_decay).

These tests lock the contract: every name the WebUI offers is accepted by the
trainer, and the names the WebUI must not offer stay off the list.
"""

from __future__ import annotations

from webui.services.config_service import _SELECT_OPTIONS

# Names pruned from the trainer in commit 772dda7. They must never reappear in
# the WebUI's curated lists — each one either crashes the trainer or silently
# drops its parameters (see module docstring).
_PRUNED_OPTIMIZERS = {"Adafactor"}
_PRUNED_SCHEDULERS = {
    "cosine_with_restarts",
    "polynomial",
    "inverse_sqrt",
    "cosine_with_min_lr",
    "warmup_stable_decay",
}


def test_no_pruned_optimizers_offered():
    offered = set(_SELECT_OPTIONS["optimizer_type"])
    leaked = offered & _PRUNED_OPTIMIZERS
    assert not leaked, (
        f"WebUI offers optimizer(s) the trainer no longer supports: {leaked}. "
        f"Remove them from _SELECT_OPTIONS['optimizer_type'] in "
        f"webui/services/config_service.py."
    )


def test_no_pruned_schedulers_offered():
    offered = set(_SELECT_OPTIONS["lr_scheduler"])
    leaked = offered & _PRUNED_SCHEDULERS
    assert not leaked, (
        f"WebUI offers lr_scheduler(s) the trainer no longer supports: {leaked}. "
        f"Remove them from _SELECT_OPTIONS['lr_scheduler'] in "
        f"webui/services/config_service.py."
    )


def test_every_offered_optimizer_is_accepted_by_trainer():
    """Every optimizer in the WebUI list must resolve in ``get_optimizer``
    without raising. Guards against both pruned names and typos."""
    import argparse

    from library.training.optimizers import get_optimizer

    for name in _SELECT_OPTIONS["optimizer_type"]:
        args = argparse.Namespace(
            optimizer_type=name,
            optimizer_args=None,
            learning_rate=1e-4,
            max_grad_norm=0.0,
        )
        # get_optimizer only needs a single trivial trainable param; it builds
        # the optimizer class. We don't step it, so the heavy imports (bnb /
        # lion / schedulefree) only fire for the names that need them.
        param = __import__("torch").nn.Parameter(__import__("torch").zeros(1))
        try:
            get_optimizer(args, [{"params": [param]}])
        except ImportError:
            # An optional dependency (bitsandbytes / lion_pytorch / schedulefree)
            # not installed in the test env is fine — the *name* is still valid,
            # the trainer would just ask the user to install it. The drift bug
            # raised AttributeError/KeyError, not ImportError.
            continue


def test_every_offered_scheduler_is_accepted_by_trainer():
    """Every scheduler in the WebUI list must resolve in ``get_scheduler_fix``
    without raising. Guards against both pruned names and typos."""
    import argparse

    import torch
    from library.training.schedulers import get_scheduler_fix

    for name in _SELECT_OPTIONS["lr_scheduler"]:
        # ``constant`` rejects num_warmup_steps; the rest require it. Match
        # each name's contract so the test exercises the real build path.
        # ``optimizer_type`` is read by ``is_schedulefree_optimizer``; keep a
        # non-schedulefree type so we test the scheduler side directly.
        needs_warmup = name != "constant"
        # ``piecewise_constant`` is parametric — it requires a ``step_rules``
        # value passed through ``--lr_scheduler_args value="..."`` (the trainer
        # surfaces this as a required knob, not a free name). Exercising it
        # here would test the step_rules contract, not the name-acceptance
        # contract this guard is about, so skip it (it's still in the offered
        # list and is exercised by the round-trip tests in test_config.py).
        if name == "piecewise_constant":
            continue
        args = argparse.Namespace(
            lr_scheduler=name,
            lr_scheduler_type="",
            lr_scheduler_args=None,
            lr_warmup_steps=10 if needs_warmup else None,
            max_train_steps=100,
            optimizer_type="AdamW",
        )
        param = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD([param], lr=1e-3)
        # Should not raise. (The pruned names raised TypeError / ValueError
        # here — that's exactly what this test catches.)
        get_scheduler_fix(args, optimizer, num_processes=1)


def test_piecewise_constant_still_offered():
    """``piecewise_constant`` is parametric (``step_rules`` is fed via
    ``--lr_scheduler_args step_rules="..."``), so its parameter contract is
    exercised by the round-trip tests in test_config.py rather than the
    name-acceptance loop above. This just pins that it remains offered (so
    the WebUI dropdown stays in sync with what the trainer accepts)."""
    assert "piecewise_constant" in _SELECT_OPTIONS["lr_scheduler"]
