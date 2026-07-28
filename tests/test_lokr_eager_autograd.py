"""Eager FP32 LoKr rematerialization regression tests."""

from __future__ import annotations

import pytest
import torch
from lycoris.functional.lokr import bypass_forward_diff

from networks.lora_modules.custom_autograd import eager_lokr_residual
from networks.lora_modules.lokr import LoKRModule


def _factors(layout: str, dtype: torch.dtype):
    torch.manual_seed(30)
    up, uq, vp, vq, rank = 3, 4, 5, 6, 2
    if layout == "full":
        values = (
            torch.randn(up, uq, dtype=dtype),
            None,
            None,
            torch.randn(vp, vq, dtype=dtype),
            None,
            None,
        )
    elif layout == "w2_decomposed":
        values = (
            torch.randn(up, uq, dtype=dtype),
            None,
            None,
            None,
            torch.randn(vp, rank, dtype=dtype),
            torch.randn(rank, vq, dtype=dtype),
        )
    elif layout == "w1_decomposed":
        values = (
            None,
            torch.randn(up, rank, dtype=dtype),
            torch.randn(rank, uq, dtype=dtype),
            torch.randn(vp, vq, dtype=dtype),
            None,
            None,
        )
    else:
        values = (
            None,
            torch.randn(up, rank, dtype=dtype),
            torch.randn(rank, uq, dtype=dtype),
            None,
            torch.randn(vp, rank, dtype=dtype),
            torch.randn(rank, vq, dtype=dtype),
        )
    return values, uq * vq, up * vp


def _run_function(layout: str, custom: bool):
    factors0, in_features, out_features = _factors(layout, torch.float16)
    source0 = torch.randn(2, 11, in_features, dtype=torch.float16)
    grad_out = torch.randn(11, 2, out_features, dtype=torch.float16)

    source = source0.clone().requires_grad_()
    x = source.transpose(0, 1)
    base_leaf = torch.randn_like(grad_out).requires_grad_()
    base = base_leaf + 0.0
    factors = tuple(
        factor.clone().requires_grad_() if factor is not None else None
        for factor in factors0
    )
    scalar = torch.tensor(0.625, requires_grad=True)
    residual_scale = 0.75

    if custom:
        output = eager_lokr_residual(
            base,
            x,
            *factors,
            scalar,
            residual_scale,
            chunk_size=4,
        )
    else:
        rank = (
            next(
                factor.shape[1]
                for factor in (factors[1], factors[4])
                if factor is not None
            )
            if layout != "full"
            else 1
        )
        delta = bypass_forward_diff(
            x.float(),
            None,
            *(
                factor.float() if factor is not None else None
                for factor in factors
            ),
            None,
            gamma=rank,
        )
        output = base + (delta * residual_scale * scalar).to(base.dtype)

    requested = (
        source,
        base_leaf,
        *(factor for factor in factors if factor is not None),
        scalar,
    )
    grads = torch.autograd.grad(output, requested, grad_out)
    return output.detach(), grads


@pytest.mark.parametrize(
    "layout",
    ["full", "w2_decomposed", "w1_decomposed", "decompose_both"],
)
def test_eager_lokr_forward_and_grads_match_official_bypass(layout):
    expected_output, expected_grads = _run_function(layout, custom=False)
    actual_output, actual_grads = _run_function(layout, custom=True)

    assert torch.equal(actual_output, expected_output)
    for index, (actual, expected) in enumerate(
        zip(actual_grads, expected_grads)
    ):
        if index == len(actual_grads) - 1:
            # The scalar gradient reduces all rows, so chunked accumulation
            # changes its FP32 summation order.
            assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-6)
        else:
            assert torch.equal(actual, expected)


def test_eager_lokr_saves_original_storage_not_fp32_activations():
    factors, in_features, out_features = _factors("full", torch.float16)
    x = torch.randn(
        2,
        11,
        in_features,
        dtype=torch.float16,
        requires_grad=True,
    ).transpose(0, 1)
    base = torch.zeros(11, 2, out_features, dtype=torch.float16)
    factors = tuple(
        factor.requires_grad_() if factor is not None else None for factor in factors
    )
    saved = []

    def pack(tensor):
        saved.append((tuple(tensor.shape), tensor.dtype))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        eager_lokr_residual(
            base,
            x,
            *factors,
            torch.tensor(1.0),
            1.0,
            chunk_size=4,
        )

    assert ((11, 2, in_features), torch.float16) in saved
    assert ((11, 2, in_features), torch.float32) not in saved
    assert ((22, 4, 5), torch.float32) not in saved


@pytest.mark.parametrize(("layout", "seed"), [("full", 3), ("decompose_both", 95)])
def test_eager_lokr_fp32_factor_grads_stay_within_chunked_reduction_tolerance(
    layout,
    seed,
):
    """Production row counts keep bounded memory with a documented FP32 drift."""
    factors0, in_features, out_features = _factors(layout, torch.float32)
    torch.manual_seed(seed)
    x0 = torch.randn(4200, in_features, dtype=torch.float16) * 0.1
    grad_out = torch.randn(4200, out_features, dtype=torch.float16) * 0.1

    def run(custom: bool):
        x = x0.clone().requires_grad_()
        factors = tuple(
            factor.clone().requires_grad_() if factor is not None else None
            for factor in factors0
        )
        scalar = torch.tensor(0.625, dtype=torch.float32, requires_grad=True)
        base = torch.zeros_like(grad_out)
        if custom:
            output = eager_lokr_residual(
                base,
                x,
                *factors,
                scalar,
                0.75,
                chunk_size=1024,
            )
        else:
            rank = 1 if layout == "full" else 2
            delta = bypass_forward_diff(
                x.float(),
                None,
                *factors,
                None,
                gamma=rank,
            )
            output = base + (delta * 0.75 * scalar).to(base.dtype)
        requested = (
            x,
            *(factor for factor in factors if factor is not None),
            scalar,
        )
        return output.detach(), torch.autograd.grad(output, requested, grad_out)

    expected_output, expected_grads = run(custom=False)
    actual_output, actual_grads = run(custom=True)

    assert torch.allclose(actual_output, expected_output, rtol=1e-3, atol=2e-3)
    assert torch.equal(actual_grads[0], expected_grads[0])
    # The bounded path cannot use a single full-width factor-gradient GEMM
    # without recreating the activation workspace it exists to avoid. FP32
    # chunk accumulation changes only the reduction order, so lock the measured
    # 4200-row drift instead of requiring bit identity.
    for actual, expected in zip(actual_grads[1:], expected_grads[1:]):
        assert torch.allclose(actual, expected, rtol=2e-4, atol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_eager_lokr_cuda_stays_within_public_fp16_tolerance():
    factors0, in_features, out_features = _factors("full", torch.float32)
    factors0 = tuple(
        factor.cuda() if factor is not None else None for factor in factors0
    )
    torch.manual_seed(99)
    x0 = torch.randn(2100, in_features, device="cuda", dtype=torch.float16) * 0.1
    grad_out = (
        torch.randn(2100, out_features, device="cuda", dtype=torch.float16) * 0.1
    )

    def run(custom: bool):
        x = x0.clone().requires_grad_()
        factors = tuple(
            factor.clone().requires_grad_() if factor is not None else None
            for factor in factors0
        )
        scalar = torch.tensor(0.625, device="cuda", requires_grad=True)
        base = torch.zeros_like(grad_out)
        if custom:
            output = eager_lokr_residual(
                base,
                x,
                *factors,
                scalar,
                0.75,
                chunk_size=1024,
            )
        else:
            delta = bypass_forward_diff(
                x.float(),
                None,
                *factors,
                None,
                gamma=1,
            )
            output = base + (delta * 0.75 * scalar).to(base.dtype)
        requested = (
            x,
            *(factor for factor in factors if factor is not None),
            scalar,
        )
        return output.detach(), torch.autograd.grad(output, requested, grad_out)

    expected_output, expected_grads = run(custom=False)
    actual_output, actual_grads = run(custom=True)

    assert torch.allclose(actual_output, expected_output, rtol=1e-3, atol=2e-3)
    assert torch.allclose(
        actual_grads[0], expected_grads[0], rtol=1e-3, atol=2e-3
    )
    for actual, expected in zip(actual_grads[1:], expected_grads[1:]):
        assert torch.allclose(actual, expected, rtol=2e-4, atol=2e-3)


def test_eager_lokr_backward_handles_frozen_residual_inputs():
    factors, in_features, out_features = _factors("full", torch.float16)
    base_leaf = torch.zeros(
        3,
        out_features,
        dtype=torch.float16,
        requires_grad=True,
    )
    base = base_leaf + 0.0
    output = eager_lokr_residual(
        base,
        torch.randn(3, in_features, dtype=torch.float16),
        *factors,
        torch.tensor(1.0),
        1.0,
        chunk_size=2,
    )

    output.sum().backward()

    assert torch.equal(base_leaf.grad, torch.ones_like(base_leaf))


def _run_full_factor_module(custom: bool):
    torch.manual_seed(31)
    base = torch.nn.Linear(64, 96, bias=False).half()
    base.weight.requires_grad_(False)
    module = LoKRModule(
        "m",
        base,
        multiplier=0.375,
        lora_dim=4,
        alpha=32,
        lokr_factor=8,
        full_factor=True,
    ).half()
    with torch.no_grad():
        module.lokr_w1.normal_(0, 0.1)
        module.lokr_w2.normal_(0, 0.1)
    module.fp32_compute = True
    module.use_custom_down_autograd = custom
    module.train()
    module.apply_to()

    x = torch.randn(2, 7, 64, dtype=torch.float16, requires_grad=True)
    grad_out = torch.randn(2, 7, 96, dtype=torch.float16)
    output = base(x)
    output.backward(grad_out)
    grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in module.named_parameters()
    }
    return output.detach(), x.grad.detach(), grads, module.scale


def test_full_factor_module_custom_path_preserves_unit_scale_and_multiplier():
    expected_output, expected_x_grad, expected_grads, expected_scale = (
        _run_full_factor_module(False)
    )
    actual_output, actual_x_grad, actual_grads, actual_scale = (
        _run_full_factor_module(True)
    )

    assert expected_scale == actual_scale == 1.0
    assert torch.equal(actual_output, expected_output)
    assert torch.equal(actual_x_grad, expected_x_grad)
    assert actual_grads.keys() == expected_grads.keys()
    for name in actual_grads:
        assert torch.equal(actual_grads[name], expected_grads[name]), name


def test_compile_trace_bypasses_eager_lokr_function(monkeypatch):
    base = torch.nn.Linear(16, 16, bias=False).half()
    module = LoKRModule(
        "m",
        base,
        lora_dim=2,
        alpha=2,
        lokr_factor=4,
        full_factor=True,
    ).half()
    module.fp32_compute = True
    module.use_custom_down_autograd = True
    module.train()

    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("custom LoKr Function entered during compile trace")

    monkeypatch.setattr(
        "networks.lora_modules.lokr.eager_lokr_residual",
        fail_if_called,
    )
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    output = module(torch.randn(2, 3, 16, dtype=torch.float16))
    assert output.dtype == torch.float16
    assert called is False
