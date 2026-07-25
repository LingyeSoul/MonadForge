"""Eager FP32 LoRA down-projection saved-activation regression tests."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from networks.lora_modules.custom_autograd import eager_lora_down_project
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
            module.lora_up.weight.copy_(
                torch.randn_like(module.lora_up.weight) * 0.1
            )
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
