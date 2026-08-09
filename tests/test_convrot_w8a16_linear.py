"""Unit tests for ConvRot W8A16 linear path."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from library.runtime.convrot.linear_w8a16 import (
    ConvRotW8A16Linear,
    w8a16_forward_from_buffers,
)
from library.runtime.convrot.linear_w8a8 import ConvRotW8A8Linear
from library.runtime.convrot.rht import group_rht, normalized_hadamard


def test_w8a16_matches_rotated_dequant_linear() -> None:
    torch.manual_seed(0)
    linear = nn.Linear(32, 16, bias=False)
    linear.weight.requires_grad_(False)
    group = 8
    wrapped = ConvRotW8A16Linear.from_linear(linear, group_size=group)
    x = torch.randn(5, 32)

    h = normalized_hadamard(group, dtype=torch.float32)
    w_hat = wrapped.dequantized_weight(dtype=torch.float32)
    x_rot = group_rht(x.float(), group, hadamard=h)
    expected = F.linear(x_rot, w_hat)
    actual = w8a16_forward_from_buffers(
        x,
        wrapped.quantized_weight,
        wrapped.scale,
        group_size=group,
        hadamard=h,
    )

    assert actual.shape == expected.shape
    assert torch.allclose(actual.float(), expected, atol=1e-4, rtol=1e-4)
    bf16 = linear(x)
    rel = (actual - bf16).norm() / bf16.norm().clamp_min(1e-8)
    assert rel.item() < 0.05


def test_w8a16_functional_matches_module() -> None:
    torch.manual_seed(1)
    linear = nn.Linear(64, 32, bias=False)
    linear.weight.requires_grad_(False)
    group = 16
    wrapped = ConvRotW8A16Linear.from_linear(linear, group_size=group)
    x = torch.randn(2, 7, 64)
    y_mod = wrapped(x)
    y_fn = w8a16_forward_from_buffers(
        x,
        wrapped.quantized_weight,
        wrapped.scale,
        group_size=group,
    )
    assert torch.allclose(y_mod, y_fn, atol=1e-5, rtol=1e-5)


def test_w8a16_stop_grad_on_base_weight() -> None:
    torch.manual_seed(2)
    linear = nn.Linear(16, 8, bias=False)
    linear.weight.requires_grad_(False)
    wrapped = ConvRotW8A16Linear.from_linear(linear, group_size=8)
    x = torch.randn(3, 16, requires_grad=True)
    y = wrapped(x)
    y.sum().backward()
    assert x.grad is not None
    assert not any(p.requires_grad for p in wrapped.parameters())


def test_w8a8_forward_runs_and_is_finite() -> None:
    torch.manual_seed(3)
    linear = nn.Linear(32, 16, bias=False)
    linear.weight.requires_grad_(False)
    wrapped = ConvRotW8A8Linear.from_linear(linear, group_size=8)
    x = torch.randn(4, 32)
    y = wrapped(x)
    assert y.shape == (4, 16)
    assert torch.isfinite(y).all()
    bf16 = linear(x)
    rel = (y - bf16).norm() / bf16.norm().clamp_min(1e-8)
    assert rel.item() < 0.15
