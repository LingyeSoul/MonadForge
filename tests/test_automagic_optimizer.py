"""Invariant tests for the Automagic optimizer.

This is the first optimizer-``.step()`` test in the repo. Automagic is a
numerics-heavy custom optimizer (Adafactor-style factored second moment +
element-wise LR-mask polarity adaptation), so these tests lock down the
arithmetic that a silent sign error would otherwise diverge silently.

All tests are CPU-only (guaranteed by ``conftest.py``'s ``CUDA_VISIBLE_DEVICES``
guard). Lazy-import torch + the module under test so collection is cheap.

Conventions mirror ``test_scheduler_args_parsing.py`` / ``test_config_service.py``.
"""

from __future__ import annotations

import logging

import pytest


def _make_param(value: float = 0.0, *, shape=(1,)):
    """Tiny trainable leaf parameter (torch imported by caller)."""
    import torch

    return torch.nn.Parameter(torch.full(shape, float(value)))


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def _make_optimizer(**overrides):
    import torch

    from library.training.automagic import Automagic

    p = torch.nn.Parameter(torch.zeros(2))
    kwargs = dict(params=[p], lr=1e-2, min_lr=1e-7, max_lr=1.0, lr_bump=1e-4)
    kwargs.update(overrides)
    return Automagic(**kwargs)


def test_init_rejects_nonpositive_min_lr():
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="min_lr must be positive"):
        _make_optimizer(min_lr=0.0)
    with pytest.raises(ValueError, match="min_lr must be positive"):
        _make_optimizer(min_lr=-1e-7)


def test_init_rejects_max_lr_below_min_lr():
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="max_lr must be >= min_lr"):
        _make_optimizer(min_lr=1e-3, max_lr=1e-4)


def test_init_accepts_max_lr_equal_to_min_lr():
    """``max_lr == min_lr`` is allowed (the predicate is strict ``<``)."""
    pytest.importorskip("torch")
    opt = _make_optimizer(min_lr=1e-3, max_lr=1e-3)
    assert opt.param_groups[0]["max_lr"] == 1e-3


def test_init_rejects_nonpositive_lr_bump():
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="lr_bump must be positive"):
        _make_optimizer(lr_bump=0.0)
    with pytest.raises(ValueError, match="lr_bump must be positive"):
        _make_optimizer(lr_bump=-1e-6)


def test_init_rejects_beta2_outside_half_open_interval():
    """``beta2`` must be in ``[0, 1)``: 0.0 accepted, 1.0 / negative rejected."""
    pytest.importorskip("torch")
    # boundary accepted
    _make_optimizer(beta2=0.0)
    with pytest.raises(ValueError, match="beta2 must be in"):
        _make_optimizer(beta2=1.0)
    with pytest.raises(ValueError, match="beta2 must be in"):
        _make_optimizer(beta2=-0.1)


def test_init_rejects_nonpositive_clip_threshold():
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="clip_threshold must be positive"):
        _make_optimizer(clip_threshold=0.0)


def test_init_clamps_lr_above_max_and_warns(caplog):
    pytest.importorskip("torch")
    with caplog.at_level(logging.WARNING, logger="library.training.automagic"):
        opt = _make_optimizer(lr=5.0, max_lr=1.0)
    assert opt.param_groups[0]["lr"] == 1.0
    assert any("clamping" in r.getMessage() and ">" in r.getMessage() for r in caplog.records)


def test_init_clamps_lr_below_min_and_warns(caplog):
    pytest.importorskip("torch")
    with caplog.at_level(logging.WARNING, logger="library.training.automagic"):
        # min_lr must be > 0, so pick a lr below it
        opt = _make_optimizer(lr=1e-9, min_lr=1e-7)
    assert opt.param_groups[0]["lr"] == 1e-7
    assert any("clamping" in r.getMessage() and "<" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# step() — 1-D grad lr_mask arithmetic (deterministic)
# ---------------------------------------------------------------------------


def _step_with_grad(p, grad_value, opt):
    """Assign a constant grad to a 1-D param and step once. Returns the param state."""
    import torch

    p.grad = torch.full_like(p, float(grad_value))
    opt.step()
    return opt.state[p]


def test_lr_mask_dips_on_step1_then_increases_monotonically():
    """Constant-sign grad: step 1 dips (init polarity False), then +lr_bump/step.

    With lr=1e-2, min_lr=1e-7, max_lr=1.0, lr_bump=1e-4 and a positive grad:
      step 1: polarity False->True => disagreement => lr_mask - lr_bump = 9.9e-3
      step 2: +lr_bump => 1.0e-2
      step 3: +lr_bump => 1.01e-2
    """
    pytest.importorskip("torch")
    p = _make_param(0.0, shape=(1,))
    opt = _make_optimizer(params=[p], lr=1e-2, min_lr=1e-7, max_lr=1.0, lr_bump=1e-4)

    s1 = _step_with_grad(p, 1.0, opt)
    assert s1["lr_mask"].item() == pytest.approx(1e-2 - 1e-4)
    assert bool(s1["last_polarity"].item()) is True

    s2 = _step_with_grad(p, 1.0, opt)
    assert s2["lr_mask"].item() == pytest.approx(1e-2)

    s3 = _step_with_grad(p, 1.0, opt)
    assert s3["lr_mask"].item() == pytest.approx(1e-2 + 1e-4)


def test_lr_mask_saturates_at_max_lr():
    """Many consecutive same-direction steps clamp at max_lr."""
    pytest.importorskip("torch")
    p = _make_param(0.0, shape=(1,))
    opt = _make_optimizer(params=[p], lr=1e-2, min_lr=1e-7, max_lr=1e-2, lr_bump=1e-3)
    for _ in range(50):
        _step_with_grad(p, 1.0, opt)
    assert opt.state[p]["lr_mask"].item() == pytest.approx(1e-2)


def test_lr_mask_decreases_on_sign_flip_and_clamps_at_min():
    """Flipping grad sign => disagreement => lr_mask - lr_bump, clamped at min_lr."""
    pytest.importorskip("torch")
    p = _make_param(0.0, shape=(1,))
    opt = _make_optimizer(params=[p], lr=1e-2, min_lr=1e-3, max_lr=1.0, lr_bump=1e-3)
    # establish a positive polarity first
    _step_with_grad(p, 1.0, opt)
    before = opt.state[p]["lr_mask"].item()
    # flip sign => disagreement => lr_mask - lr_bump
    _step_with_grad(p, -1.0, opt)
    after = opt.state[p]["lr_mask"].item()
    assert after == pytest.approx(before - 1e-3)
    # keep flipping every step so polarity never agrees => clamp at min_lr
    for _ in range(50):
        sign = opt.state[p]["last_polarity"].item()  # bool of last update sign
        # step with the OPPOSITE sign to force disagreement each time
        _step_with_grad(p, -1.0 if sign else 1.0, opt)
    assert opt.state[p]["lr_mask"].item() == pytest.approx(1e-3)


# ---------------------------------------------------------------------------
# step() — descent invariant (convex toy task)
# ---------------------------------------------------------------------------


def test_step_reduces_loss_on_linear_regression():
    """One Automagic step on convex MSE must descend (update is preconditioned grad)."""
    pytest.importorskip("torch")
    import torch

    torch.manual_seed(0)
    model = torch.nn.Linear(4, 1, bias=True)
    x = torch.randn(8, 4)
    target = torch.randn(8, 1)
    opt = _make_optimizer(params=model.parameters(), lr=1e-2, max_lr=1e-1, lr_bump=1e-4)

    loss_before = torch.nn.functional.mse_loss(model(x), target).item()
    opt.zero_grad()
    torch.nn.functional.mse_loss(model(x), target).backward()
    opt.step()
    loss_after = torch.nn.functional.mse_loss(model(x), target).item()
    assert loss_after < loss_before


# ---------------------------------------------------------------------------
# step() — sparse grad rejection
# ---------------------------------------------------------------------------


def test_step_rejects_sparse_gradient():
    pytest.importorskip("torch")
    import torch

    p = _make_param(0.0, shape=(4,))
    opt = _make_optimizer(params=[p])
    p.grad = torch.sparse_coo_tensor(
        indices=torch.zeros((1, 1), dtype=torch.long),
        values=torch.ones(1),
        size=p.shape,
    ).coalesce()
    with pytest.raises(RuntimeError, match="sparse gradients"):
        opt.step()


# ---------------------------------------------------------------------------
# State keys per grad ndim
# ---------------------------------------------------------------------------


def test_state_keys_for_1d_grad():
    pytest.importorskip("torch")
    p = _make_param(0.0, shape=(3,))  # 1-D like a bias
    opt = _make_optimizer(params=[p])
    _step_with_grad(p, 0.5, opt)
    keys = set(opt.state[p].keys())
    assert "exp_avg_sq" in keys
    assert "exp_avg_sq_row" not in keys
    assert "exp_avg_sq_col" not in keys
    assert {"step", "lr_mask", "avg_lr", "last_polarity", "RMS"} <= keys


def test_state_keys_for_2d_grad():
    pytest.importorskip("torch")
    import torch

    p = torch.nn.Parameter(torch.zeros(3, 5))  # 2-D like a Linear weight
    opt = _make_optimizer(params=[p])
    p.grad = torch.full_like(p, 0.5)
    opt.step()
    keys = set(opt.state[p].keys())
    assert "exp_avg_sq_row" in keys
    assert "exp_avg_sq_col" in keys
    assert "exp_avg_sq" not in keys


# ---------------------------------------------------------------------------
# get_learning_rates / get_avg_learning_rate (contract for get_dummy_scheduler)
# ---------------------------------------------------------------------------


def test_get_learning_rates_before_step_returns_group_lr():
    """No state yet => falls back to group['lr'] (the constructor-clamped lr)."""
    pytest.importorskip("torch")
    opt = _make_optimizer(lr=1e-2)
    assert opt.get_learning_rates() == [pytest.approx(1e-2)]
    assert opt.get_avg_learning_rate() == pytest.approx(1e-2)


def test_get_learning_rates_after_step_returns_avg_lr_mean():
    pytest.importorskip("torch")
    p = _make_param(0.0, shape=(2,))
    opt = _make_optimizer(params=[p], lr=1e-2, lr_bump=1e-4)
    _step_with_grad(p, 1.0, opt)
    # avg_lr is stored as new_lr.mean() after the step
    expected = opt.state[p]["avg_lr"]
    assert opt.get_learning_rates() == [pytest.approx(expected)]
    assert opt.get_avg_learning_rate() == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Weight decay (decoupled, shrinks magnitude)
# ---------------------------------------------------------------------------


def test_weight_decay_shrinks_param_magnitude():
    """With grad zeroed, the update term is ~0 so only WD acts; wd>0 shrinks |p|."""
    pytest.importorskip("torch")
    import torch

    p = torch.nn.Parameter(torch.tensor([1.0, -2.0, 0.5]))
    opt = _make_optimizer(
        params=[p], lr=1e-2, lr_bump=1e-5, weight_decay=1.0
    )
    before = p.detach().clone()
    p.grad = torch.zeros_like(p)
    opt.step()
    # |p_after| < |p_before| element-wise (WD factor 1 - wd*new_lr < 1)
    assert (p.abs() < before.abs()).all()


# ---------------------------------------------------------------------------
# Param swapping (deterministic only with stdlib random.seed)
# ---------------------------------------------------------------------------


def test_swap_paramiters_factor_zero_activates_nothing():
    pytest.importorskip("torch")
    import torch

    params = [torch.nn.Parameter(torch.zeros(10)) for _ in range(5)]
    opt = _make_optimizer(params=params, do_paramiter_swapping=False)
    opt.swap_paramiters_factor = 0.0
    opt.do_paramiter_swapping_factor = 0.0
    # call swap with factor 0 via the explicit method
    opt.paramiter_swapping_factor = 0.0
    opt.swap_paramiters()
    active = sum(1 for p in params if p.requires_grad)
    assert active == 0


def test_swap_paramiters_factor_one_activates_everything():
    pytest.importorskip("torch")
    import torch

    params = [torch.nn.Parameter(torch.zeros(10)) for _ in range(5)]
    opt = _make_optimizer(params=params, do_paramiter_swapping=False)
    opt.paramiter_swapping_factor = 1.0
    opt.swap_paramiters()
    assert all(p.requires_grad for p in params)


def test_swap_paramiters_mid_factor_overshoots_within_one_param():
    """Active numel must reach >= target and be within one param of target."""
    pytest.importorskip("torch")
    import random

    import torch

    params = [torch.nn.Parameter(torch.zeros(20)) for _ in range(10)]
    total = sum(p.numel() for p in params)  # 200
    factor = 0.25  # target = 50
    opt = _make_optimizer(params=params, do_paramiter_swapping=False)
    opt.paramiter_swapping_factor = factor
    random.seed(42)
    opt.swap_paramiters()
    active_numel = sum(p.numel() for p in params if p.requires_grad)
    target = int(total * factor)
    assert active_numel >= target
    # within one param (20) of target
    assert active_numel - target < 20
