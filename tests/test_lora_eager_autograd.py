"""Eager FP32 LoRA down-projection saved-activation regression tests."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from networks.lora_modules.custom_autograd import (
    eager_lora_down_project,
    eager_lora_up_residual,
)
from networks.lora_modules.lora import LoRAModule


def _reference(x, weight, inv_scale):
    x_work = x.float()
    if inv_scale is not None:
        x_work = x_work * inv_scale.float()
    return F.linear(x_work, weight.float())


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("scaled", [False, True])
def test_eager_down_forward_and_grads_match_plain_autograd(dtype, scaled):
    torch.manual_seed(0)
    x0 = torch.randn(2, 9, 16, dtype=dtype)
    w0 = torch.randn(4, 16, dtype=torch.float16)
    inv = torch.rand(16, dtype=torch.float32) + 0.5 if scaled else None
    grad_out = torch.randn(2, 9, 4, dtype=torch.float32)

    x_ref = x0.clone().requires_grad_()
    w_ref = w0.clone().requires_grad_()
    out_ref = _reference(x_ref, w_ref, inv)
    out_ref.backward(grad_out)

    x_got = x0.clone().requires_grad_()
    w_got = w0.clone().requires_grad_()
    out_got = eager_lora_down_project(x_got, w_got, inv)
    out_got.backward(grad_out)

    assert torch.equal(out_ref, out_got)
    assert torch.equal(x_ref.grad, x_got.grad)
    assert torch.equal(w_ref.grad, w_got.grad)


@pytest.mark.parametrize(("x_grad", "weight_grad"), [(True, False), (False, True)])
def test_eager_down_handles_frozen_inputs(x_grad, weight_grad):
    torch.manual_seed(93)
    x0 = torch.randn(9, 16, dtype=torch.float16)
    weight0 = torch.randn(4, 16, dtype=torch.float16)
    grad_out = torch.randn(9, 4, dtype=torch.float32)

    def run(custom):
        x = x0.clone().requires_grad_(x_grad)
        weight = weight0.clone().requires_grad_(weight_grad)
        output = (
            eager_lora_down_project(x, weight, None)
            if custom
            else _reference(x, weight, None)
        )
        requested = tuple(
            tensor for tensor in (x, weight) if tensor.requires_grad
        )
        return torch.autograd.grad(output, requested, grad_out)

    expected = run(False)
    actual = run(True)

    assert len(actual) == len(expected)
    for actual_grad, expected_grad in zip(actual, expected):
        assert torch.equal(actual_grad, expected_grad)


@pytest.mark.parametrize("scaled", [False, True])
def test_eager_down_fp32_weight_grad_uses_reference_reduction(scaled):
    torch.manual_seed(94)
    x0 = torch.randn(4200, 257, dtype=torch.float16)
    weight0 = torch.randn(17, 257, dtype=torch.float32)
    inv = torch.rand(257, dtype=torch.float32) + 0.5 if scaled else None
    grad_out = torch.randn(4200, 17, dtype=torch.float32)

    x_ref = x0.clone().requires_grad_()
    weight_ref = weight0.clone().requires_grad_()
    _reference(x_ref, weight_ref, inv).backward(grad_out)

    x_got = x0.clone().requires_grad_()
    weight_got = weight0.clone().requires_grad_()
    eager_lora_down_project(x_got, weight_got, inv).backward(grad_out)

    assert torch.equal(weight_got.grad, weight_ref.grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("scaled", [False, True])
def test_eager_down_cuda_stays_within_fp16_tolerance(scaled):
    torch.manual_seed(98)
    rows, in_features, out_features = 4200, 257, 17
    x0 = (
        torch.randn(rows, in_features, device="cuda", dtype=torch.float16) * 0.1
    )
    weight0 = (
        torch.randn(
            out_features,
            in_features,
            device="cuda",
            dtype=torch.float16,
        )
        * 0.1
    )
    inv = (
        torch.rand(in_features, device="cuda", dtype=torch.float32) + 0.5
        if scaled
        else None
    )
    grad_out = torch.randn(
        rows,
        out_features,
        device="cuda",
        dtype=torch.float32,
    )

    def run(custom):
        x = x0.clone().requires_grad_()
        weight = weight0.clone().requires_grad_()
        output = (
            eager_lora_down_project(x, weight, inv)
            if custom
            else _reference(x, weight, inv)
        )
        return output.detach(), torch.autograd.grad(output, (x, weight), grad_out)

    expected_output, expected_grads = run(False)
    actual_output, actual_grads = run(True)

    assert torch.allclose(actual_output, expected_output, rtol=2e-4, atol=2e-4)
    assert torch.allclose(
        actual_grads[0], expected_grads[0], rtol=1e-3, atol=1e-2
    )
    assert torch.equal(actual_grads[1], expected_grads[1])


@pytest.mark.parametrize("scaled", [False, True])
def test_eager_down_saves_original_fp16_input_not_fp32_copy(scaled):
    torch.manual_seed(1)
    x = torch.randn(2, 9, 16, dtype=torch.float16, requires_grad=True)
    weight = torch.randn(4, 16, dtype=torch.float16, requires_grad=True)
    inv = torch.rand(16, dtype=torch.float32) + 0.5 if scaled else None
    saved = []

    def pack(tensor):
        saved.append((tuple(tensor.shape), tensor.dtype))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        _out = eager_lora_down_project(x, weight, inv)

    assert ((2, 9, 16), torch.float16) in saved
    assert ((2, 9, 16), torch.float32) not in saved
    assert ((4, 16), torch.float16) in saved


@pytest.mark.parametrize("channel_scaling", [False, True])
def test_lora_module_custom_eager_path_matches_plain_path(channel_scaling):
    channel_scale = None
    if channel_scaling:
        channel_scale = torch.rand(16, dtype=torch.float32) + 0.5

    def run(use_custom):
        torch.manual_seed(2)
        base = torch.nn.Linear(16, 12, bias=False).to(torch.float16)
        base.weight.requires_grad_(False)
        module = LoRAModule(
            "m",
            base,
            multiplier=0.75,
            lora_dim=4,
            alpha=4,
            channel_scale=channel_scale,
        )
        with torch.no_grad():
            module.lora_up.weight.copy_(torch.randn_like(module.lora_up.weight) * 0.1)
        module.apply_to()
        module.train()
        module.fp32_compute = True
        module.use_custom_down_autograd = use_custom
        module.org_forward = lambda x: torch.zeros(
            (*x.shape[:-1], 12), dtype=torch.float16, device=x.device
        )

        x = torch.randn(2, 9, 16, dtype=torch.float16, requires_grad=True)
        grad_out = torch.randn(2, 9, 12, dtype=torch.float16)
        out = module(x)
        out.backward(grad_out)
        grads = {
            name: param.grad.detach().clone()
            for name, param in module.named_parameters()
            if param.grad is not None
        }
        return out.detach(), x.grad.detach(), grads

    ref_out, ref_x_grad, ref_grads = run(False)
    got_out, got_x_grad, got_grads = run(True)

    assert torch.equal(ref_out, got_out)
    assert torch.equal(ref_x_grad, got_x_grad)
    assert ref_grads.keys() == got_grads.keys()
    for name in ref_grads:
        assert torch.equal(ref_grads[name], got_grads[name]), name


def test_compile_trace_bypasses_custom_eager_function(monkeypatch):
    torch.manual_seed(3)
    base = torch.nn.Linear(16, 12, bias=False).to(torch.float16)
    module = LoRAModule("m", base, lora_dim=4, alpha=4)
    module.fp32_compute = True
    module.use_custom_down_autograd = True
    module.train()

    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("custom eager Function entered during compile trace")

    monkeypatch.setattr(
        "networks.lora_modules.lora.eager_lora_down_project", fail_if_called
    )
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    x = torch.randn(2, 9, 16, dtype=torch.float16)
    out = module._project_down(x, torch.float32)
    assert out.dtype == torch.float32
    assert called is False


@pytest.mark.parametrize("org_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("scale", [0.375, 1.0])
def test_eager_up_residual_chunks_forward_and_backward(org_dtype, scale):
    torch.manual_seed(4)
    base = torch.nn.Linear(7, 11, bias=False).to(org_dtype)
    base.weight.requires_grad_(False)
    x0 = torch.randn(2, 5, 7, dtype=org_dtype)
    rank0 = torch.randn(2, 5, 3, dtype=torch.float32)
    weight0 = torch.randn(11, 3, dtype=torch.float16)
    grad_out = torch.randn(2, 5, 11, dtype=org_dtype)
    x_ref = x0.clone().requires_grad_()
    rank_ref = rank0.clone().requires_grad_()
    weight_ref = weight0.clone().requires_grad_()
    org_ref = base(x_ref)
    delta_ref = F.linear(rank_ref.float(), weight_ref.float()) * scale
    out_ref = (
        org_ref.float() + delta_ref
        if org_dtype == torch.float16
        else org_ref + delta_ref.to(org_ref.dtype)
    )
    out_ref.backward(grad_out)

    x_got = x0.clone().requires_grad_()
    rank_got = rank0.clone().requires_grad_()
    weight_got = weight0.clone().requires_grad_()
    org_got = base(x_got)
    org_storage = org_got.data_ptr()
    out_got = eager_lora_up_residual(
        org_got,
        rank_got,
        weight_got,
        scale,
        chunk_size=3,
    )
    if org_dtype == torch.float16:
        assert out_got.dtype == torch.float32
        assert out_got.data_ptr() != org_storage
    else:
        assert out_got.data_ptr() == org_storage
    out_got.backward(grad_out)

    assert torch.allclose(out_ref, out_got, rtol=1e-6, atol=3e-7)
    assert torch.equal(x_ref.grad, x_got.grad)
    assert torch.allclose(rank_ref.grad, rank_got.grad, rtol=1e-6, atol=3e-7)
    assert torch.equal(weight_ref.grad, weight_got.grad)


@pytest.mark.parametrize(
    ("base_grad", "rank_grad", "weight_grad"),
    [
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (True, True, False),
        (True, False, True),
        (False, True, True),
    ],
)
def test_eager_up_handles_frozen_input_combinations(
    base_grad,
    rank_grad,
    weight_grad,
):
    torch.manual_seed(95)
    rank0 = torch.randn(9, 3, dtype=torch.float32)
    weight0 = torch.randn(11, 3, dtype=torch.float16)
    grad_out = torch.randn(9, 11, dtype=torch.float16)

    def run(custom):
        base_leaf = torch.zeros(9, 11, dtype=torch.float16).requires_grad_(base_grad)
        base = base_leaf + 0.0 if base_grad else base_leaf
        rank_input = rank0.clone().requires_grad_(rank_grad)
        weight = weight0.clone().requires_grad_(weight_grad)
        output = (
            eager_lora_up_residual(base, rank_input, weight, 0.375, chunk_size=4)
            if custom
            else base + (F.linear(rank_input, weight.float()) * 0.375).half()
        )
        requested = tuple(
            tensor
            for tensor, needs_grad in zip(
                (base_leaf, rank_input, weight),
                (base_grad, rank_grad, weight_grad),
            )
            if needs_grad
        )
        return torch.autograd.grad(output, requested, grad_out)

    expected = run(False)
    actual = run(True)

    assert len(actual) == len(expected)
    for actual_grad, expected_grad in zip(actual, expected):
        assert torch.allclose(actual_grad, expected_grad, rtol=1e-6, atol=3e-7)


def test_eager_up_backward_handles_frozen_weight():
    torch.manual_seed(96)
    rank0 = torch.randn(9, 3, dtype=torch.float32)
    weight = torch.randn(11, 3, dtype=torch.float32)
    grad_out = torch.randn(9, 11, dtype=torch.float16)

    rank_ref = rank0.clone().requires_grad_()
    expected = F.linear(rank_ref, weight) * 0.375
    expected.backward(grad_out)

    rank_got = rank0.clone().requires_grad_()
    base = torch.zeros(9, 11, dtype=torch.float16, requires_grad=True) + 0.0
    actual = eager_lora_up_residual(
        base,
        rank_got,
        weight,
        0.375,
        chunk_size=4,
    )
    actual.backward(grad_out)

    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected)
    assert torch.allclose(rank_got.grad, rank_ref.grad, rtol=1e-6, atol=3e-7)


def test_eager_up_residual_keeps_large_finite_merge_in_fp32():
    base = torch.full((1, 1), 40_000.0, dtype=torch.float16, requires_grad=True)
    rank_input = torch.ones((1, 1), dtype=torch.float32, requires_grad=True)
    weight = torch.full((1, 1), 30_000.0, dtype=torch.float32, requires_grad=True)

    out = eager_lora_up_residual(base + 0.0, rank_input, weight, 1.0)
    out.sum().backward()

    assert out.dtype == torch.float32
    assert torch.equal(out, torch.tensor([[70_000.0]], dtype=torch.float32))
    assert torch.isfinite(out).all()
    assert torch.isfinite(base.grad).all()
    assert torch.isfinite(rank_input.grad).all()
    assert torch.isfinite(weight.grad).all()


def test_eager_up_frozen_weight_does_not_scale_fp32_residual_gradient():
    torch.manual_seed(97)
    rank0 = torch.randn(9, 3, dtype=torch.float32)
    weight = torch.randn(11, 3, dtype=torch.float32)
    grad_out = torch.randn(9, 11, dtype=torch.float32)
    original_grad_out = grad_out.clone()

    rank_ref = rank0.clone().requires_grad_()
    base_ref = torch.zeros(9, 11, dtype=torch.float32, requires_grad=True)
    expected = base_ref + F.linear(rank_ref, weight) * 0.375
    expected.backward(grad_out)

    rank_got = rank0.clone().requires_grad_()
    base_leaf = torch.zeros(9, 11, dtype=torch.float32, requires_grad=True)
    actual = eager_lora_up_residual(
        base_leaf + 0.0,
        rank_got,
        weight,
        0.375,
        chunk_size=4,
    )
    actual.backward(grad_out)

    assert torch.equal(grad_out, original_grad_out)
    assert torch.equal(base_leaf.grad, original_grad_out)
    assert torch.equal(base_ref.grad, original_grad_out)
    assert torch.allclose(rank_got.grad, rank_ref.grad, rtol=1e-6, atol=3e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_eager_up_cuda_stays_within_fp16_tolerance():
    torch.manual_seed(99)
    rows, rank, out_features = 4200, 17, 257
    rank0 = torch.randn(rows, rank, device="cuda", dtype=torch.float32) * 0.1
    weight0 = (
        torch.randn(out_features, rank, device="cuda", dtype=torch.float16) * 0.1
    )
    grad_out = (
        torch.randn(rows, out_features, device="cuda", dtype=torch.float16) * 0.1
    )
    scale = 0.375

    def run(custom):
        base_leaf = torch.zeros(
            rows,
            out_features,
            device="cuda",
            dtype=torch.float16,
            requires_grad=True,
        )
        base = base_leaf + 0.0
        rank_input = rank0.clone().requires_grad_()
        weight = weight0.clone().requires_grad_()
        output = (
            eager_lora_up_residual(
                base,
                rank_input,
                weight,
                scale,
                chunk_size=3072,
            )
            if custom
            else base.float() + F.linear(rank_input, weight.float()) * scale
        )
        grads = torch.autograd.grad(output, (base_leaf, rank_input, weight), grad_out)
        return output.detach(), grads

    expected_output, expected_grads = run(False)
    actual_output, actual_grads = run(True)

    assert torch.allclose(actual_output, expected_output, rtol=1e-3, atol=2e-3)
    assert torch.equal(actual_grads[0], expected_grads[0])
    assert torch.allclose(
        actual_grads[1], expected_grads[1], rtol=2e-4, atol=2e-5
    )
    assert torch.equal(actual_grads[2], expected_grads[2])
