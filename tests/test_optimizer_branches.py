"""Tests for the new optimizer branches and routing helpers in optimizers.py.

Covers the branches added in the Automagic support change:
  - Adafactor ``relative_step`` branch (incl. the cross-function handshake that
    mutates ``args`` + pops per-group ``lr``).
  - CAME branch (smoke).
  - ProdigyPlusScheduleFree branch (skipped if the package is absent).
  - the ``is_self_managed_lr_optimizer`` / ``is_prodigy_plus_schedulefree_*``
    / ``is_schedulefree_optimizer`` routing helpers.

Conventions mirror ``test_scheduler_args_parsing.py`` / ``test_config_service.py``:
lazy imports inside the test body, ``pytest.importorskip`` for heavy deps, and a
module-level ``_make_args`` fabricator.
"""

from __future__ import annotations

import argparse
import logging

import pytest


def _make_args(
    optimizer_type="AdamW",
    optimizer_args=None,
    learning_rate=1e-3,
    max_grad_norm=0.0,
    lr_scheduler="constant",
):
    return argparse.Namespace(
        optimizer_type=optimizer_type,
        optimizer_args=optimizer_args,
        learning_rate=learning_rate,
        max_grad_norm=max_grad_norm,
        lr_scheduler=lr_scheduler,
    )


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------


def test_is_self_managed_lr_optimizer_routes_automagic():
    pytest.importorskip("torch")
    import torch

    from library.training.automagic import Automagic
    from library.training.optimizers import is_self_managed_lr_optimizer

    p = torch.nn.Parameter(torch.zeros(2))
    auto = Automagic([p], lr=1e-3)
    assert is_self_managed_lr_optimizer(auto, _make_args(optimizer_type="Automagic"))
    # plain AdamW is not self-managed
    adamw = torch.optim.AdamW([p], lr=1e-3)
    assert not is_self_managed_lr_optimizer(
        adamw, _make_args(optimizer_type="AdamW")
    )
    # wrong optimizer_type string => False even if the method exists
    assert not is_self_managed_lr_optimizer(
        auto, _make_args(optimizer_type="AdamW")
    )


def test_is_prodigy_plus_schedulefree_helpers_flag_consistency():
    pytest.importorskip("torch")
    from library.training.optimizers import (
        is_prodigy_plus_schedulefree_enabled,
        is_prodigy_plus_schedulefree_type,
    )

    def args(optimizer_args=None):
        return _make_args(
            optimizer_type="ProdigyPlusScheduleFree", optimizer_args=optimizer_args
        )

    # type detection
    assert is_prodigy_plus_schedulefree_type(args())
    assert not is_prodigy_plus_schedulefree_type(_make_args(optimizer_type="AdamW"))

    # enabled flag: default True, explicit True/False honored
    assert is_prodigy_plus_schedulefree_enabled(args())                      # default
    assert is_prodigy_plus_schedulefree_enabled(args(["use_schedulefree=True"]))
    assert not is_prodigy_plus_schedulefree_enabled(args(["use_schedulefree=False"]))
    # non-prodigy-plus optimizer => always False regardless of the kwarg
    assert not is_prodigy_plus_schedulefree_enabled(
        _make_args(optimizer_type="AdamW", optimizer_args=["use_schedulefree=True"])
    )


def test_is_schedulefree_optimizer_routes_prodigy_plus_correctly():
    pytest.importorskip("torch")
    import torch

    from library.training.optimizers import is_schedulefree_optimizer

    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=1e-3)  # the optimizer instance is irrelevant here
    # ProdigyPlus with schedule-free disabled => the generic endswith() check
    # would wrongly return True; the dedicated branch must override to False.
    args_on = _make_args(
        optimizer_type="ProdigyPlusScheduleFree", optimizer_args=["use_schedulefree=True"]
    )
    args_off = _make_args(
        optimizer_type="ProdigyPlusScheduleFree", optimizer_args=["use_schedulefree=False"]
    )
    assert is_schedulefree_optimizer(opt, args_on) is True
    assert is_schedulefree_optimizer(opt, args_off) is False
    # a ScheduleFree-named optimizer still routes via the generic endswith()
    assert is_schedulefree_optimizer(
        opt, _make_args(optimizer_type="AdamWScheduleFree")
    )


# ---------------------------------------------------------------------------
# get_optimizer — Adafactor branch
# ---------------------------------------------------------------------------


def test_get_optimizer_adafactor_relative_step_handshake(caplog):
    """relative_step=True (default) mutates args + pops per-group lr.

    This is a *deliberate* cross-function handshake: get_optimizer rewrites
    ``args.lr_scheduler = 'adafactor:<lr>'`` and nulls ``args.learning_rate`` so
    the subsequent ``get_scheduler_fix(args, ...)`` call (train.py:1914) builds
    ``AdafactorSchedule``. The test pins that contract.
    """
    pytest.importorskip("transformers")
    import torch

    from library.training.optimizers import get_optimizer
    import transformers

    p = torch.nn.Parameter(torch.zeros(4))
    groups = [
        {"params": [p], "lr": 1e-4},
        {"params": [torch.nn.Parameter(torch.zeros(4))], "lr": 2e-4},
    ]
    args = _make_args(
        optimizer_type="Adafactor", learning_rate=1e-3, lr_scheduler="constant"
    )
    with caplog.at_level(logging.WARNING, logger="library.training.optimizers"):
        _name, _opt_args, optimizer = get_optimizer(args, groups)

    assert isinstance(optimizer, transformers.optimization.Adafactor)
    # handshake mutations
    assert args.learning_rate is None
    assert args.lr_scheduler == "adafactor:0.001"
    # per-group lr keys were present and popped before construction — proven by
    # the "ignored" warning (only fires when at least one group carried an lr).
    # (Adafactor's own constructor re-injects a per-group lr from defaults, so
    # we assert on the warning rather than on key absence post-construction.)
    assert any("ignored" in r.getMessage() for r in caplog.records)


def test_get_optimizer_adafactor_warmup_init_forces_relative_step():
    """relative_step=False + warmup_init=True is silently flipped to True."""
    pytest.importorskip("transformers")
    import torch

    from library.training.optimizers import get_optimizer

    p = torch.nn.Parameter(torch.zeros(4))
    args = _make_args(
        optimizer_type="Adafactor",
        optimizer_args=["relative_step=False", "warmup_init=True"],
        learning_rate=1e-3,
    )
    _n, _a, opt = get_optimizer(args, [{"params": [p]}])
    # because relative_step got flipped to True, args.learning_rate is nulled
    assert args.learning_rate is None


# ---------------------------------------------------------------------------
# get_optimizer — CAME branch
# ---------------------------------------------------------------------------


def test_get_optimizer_came_branch_normalizes_betas_and_steps():
    pytest.importorskip("pytorch_optimizer")
    import torch

    from library.training.optimizers import get_optimizer
    import pytorch_optimizer

    p = torch.nn.Parameter(torch.ones(2, 2))
    args = _make_args(
        optimizer_type="CAME",
        optimizer_args=["weight_decay=0.01", "betas=0.8,0.95"],
        learning_rate=1e-4,
    )
    name, optimizer_args, opt = get_optimizer(args, [{"params": [p]}])

    assert isinstance(opt, pytorch_optimizer.CAME)
    assert name == "pytorch_optimizer.optimizer.came.CAME"
    assert optimizer_args == "weight_decay=0.01,betas=(0.8, 0.95, 0.9999)"
    assert opt.param_groups[0]["betas"] == (0.8, 0.95, 0.9999)

    before = p.detach().clone()
    p.square().sum().backward()
    opt.step()
    assert not torch.equal(p.detach(), before)


def test_get_optimizer_came_fully_qualified_name():
    pytest.importorskip("pytorch_optimizer")
    import pytorch_optimizer
    import torch

    from library.training.optimizers import get_optimizer

    p = torch.nn.Parameter(torch.ones(2, 2))
    args = _make_args(
        optimizer_type="pytorch_optimizer.CAME",
        optimizer_args=["betas=0.9,0.999"],
    )
    _, _, opt = get_optimizer(args, [p])

    assert isinstance(opt, pytorch_optimizer.CAME)
    assert opt.param_groups[0]["betas"] == (0.9, 0.999, 0.9999)


def test_get_optimizer_custom_came_preserves_two_betas(monkeypatch):
    import sys
    import types

    import torch

    from library.training.optimizers import get_optimizer

    class CAME(torch.optim.Optimizer):
        def __init__(self, params, lr, betas):
            super().__init__(params, {"lr": lr, "betas": betas})

    module = types.ModuleType("custom_optimizer")
    CAME.__module__ = module.__name__
    module.CAME = CAME
    monkeypatch.setitem(sys.modules, module.__name__, module)

    args = _make_args(
        optimizer_type="custom_optimizer.CAME",
        optimizer_args=["betas=0.8,0.9"],
    )
    _, optimizer_args, opt = get_optimizer(args, [torch.nn.Parameter(torch.ones(1))])

    assert opt.param_groups[0]["betas"] == (0.8, 0.9)
    assert optimizer_args == "betas=(0.8, 0.9)"


def test_get_optimizer_came_preserves_explicit_beta3():
    pytest.importorskip("pytorch_optimizer")
    import torch

    from library.training.optimizers import get_optimizer

    args = _make_args(
        optimizer_type="CAME",
        optimizer_args=["betas=0.8,0.95,0.97"],
    )
    _, optimizer_args, opt = get_optimizer(args, [torch.nn.Parameter(torch.ones(1))])

    assert opt.param_groups[0]["betas"] == (0.8, 0.95, 0.97)
    assert optimizer_args == "betas=(0.8, 0.95, 0.97)"


def test_came_steps_across_multiple_input_resolutions():
    pytest.importorskip("pytorch_optimizer")
    import torch

    from library.training.optimizers import get_optimizer

    torch.manual_seed(7)
    model = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1)
    args = _make_args(optimizer_type="CAME", learning_rate=2e-4)
    _, _, opt = get_optimizer(args, model.parameters())

    state_shapes = None
    for height, width in ((8, 8), (12, 16), (16, 12)):
        opt.zero_grad()
        before = model.weight.detach().clone()
        x = torch.randn(2, 3, height, width)
        loss = model(x).square().mean()
        loss.backward()
        opt.step()

        assert torch.isfinite(loss)
        assert not torch.equal(model.weight.detach(), before)
        current_shapes = {
            key: tuple(value.shape)
            for key, value in opt.state[model.weight].items()
            if torch.is_tensor(value)
        }
        if state_shapes is None:
            state_shapes = current_shapes
        else:
            assert current_shapes == state_shapes


# ---------------------------------------------------------------------------
# get_optimizer — ProdigyPlusScheduleFree branch (skipped if pkg missing)
# ---------------------------------------------------------------------------


def test_get_optimizer_prodigy_plus_warnings(caplog):
    pytest.importorskip("prodigyplus.prodigy_plus_schedulefree")
    import torch

    from library.training.optimizers import get_optimizer

    p = torch.nn.Parameter(torch.zeros(4))
    # misconfigure lr / scheduler / grad-norm to trigger all three advisory warns
    args = _make_args(
        optimizer_type="ProdigyPlusScheduleFree",
        learning_rate=0.5,                 # != 1.0
        lr_scheduler="cosine",             # != constant
        max_grad_norm=1.0,                 # != 0
    )
    with caplog.at_level(logging.WARNING, logger="library.training.optimizers"):
        get_optimizer(args, [{"params": [p]}])
    msgs = [r.getMessage() for r in caplog.records]
    assert any("learning_rate=1.0" in m for m in msgs)
    assert any("lr_scheduler=constant" in m for m in msgs)
    assert any("max_grad_norm=0" in m for m in msgs)
