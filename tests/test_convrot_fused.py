"""Tests for fused RHT + quant + GEMM paths."""

from __future__ import annotations

import os

import pytest
import torch
from torch import nn

from library.runtime.convrot.fused import fused_w8a16_forward, fused_w8a8_forward
from library.runtime.convrot.linear_w8a16 import w8a16_forward_from_buffers
from library.runtime.convrot.linear_w8a8 import w8a8_forward_from_buffers
from library.runtime.convrot.quant import rotate_and_quantize_weight
from library.runtime.convrot.rht import group_fwht, group_rht, normalized_hadamard, rht_backend


def test_group_fwht_matches_dense_sylvester() -> None:
    torch.manual_seed(0)
    for g in (8, 16, 64, 256):
        x = torch.randn(2, 3, g * 4, dtype=torch.float64)
        h = normalized_hadamard(g, dtype=torch.float64)
        dense = group_rht(x, g, hadamard=h)
        fast = group_fwht(x, g)
        rel = (fast - dense).norm() / dense.norm().clamp_min(1e-12)
        assert rel.item() < 1e-6, f"group={g} rel={rel.item()}"


def test_group_fwht_is_involutory() -> None:
    torch.manual_seed(1)
    x = torch.randn(4, 128, dtype=torch.float64)
    y = group_fwht(group_fwht(x, 32), 32)
    assert torch.allclose(y, x, atol=1e-6, rtol=1e-6)


def test_rht_backend_defaults_to_dense(monkeypatch) -> None:
    monkeypatch.delenv("ANIMA_CONVROT_RHT", raising=False)
    assert rht_backend() == "dense"
    monkeypatch.setenv("ANIMA_CONVROT_RHT", "fwht")
    assert rht_backend() == "fwht"


def test_group_rht_none_matches_dense_by_default(monkeypatch) -> None:
    monkeypatch.delenv("ANIMA_CONVROT_RHT", raising=False)
    torch.manual_seed(9)
    x = torch.randn(3, 64, dtype=torch.float64)
    h = normalized_hadamard(16, dtype=torch.float64)
    y_none = group_rht(x, 16, hadamard=None)
    y_h = group_rht(x, 16, hadamard=h)
    assert torch.allclose(y_none, y_h, atol=1e-10)


def test_fused_w8a16_matches_legacy_dense_path() -> None:
    torch.manual_seed(2)
    w = torch.randn(64, 128)
    q, scale = rotate_and_quantize_weight(w, 32)
    x = torch.randn(5, 128)
    h = normalized_hadamard(32, dtype=torch.float32)
    # Force non-fused dense for explicit legacy reference.
    os.environ["ANIMA_CONVROT_FUSED"] = "0"
    try:
        legacy = w8a16_forward_from_buffers(
            x, q, scale, group_size=32, hadamard=h
        )
    finally:
        os.environ.pop("ANIMA_CONVROT_FUSED", None)
    os.environ["ANIMA_CONVROT_W8A16_KERNEL"] = "dequant"
    try:
        fused = fused_w8a16_forward(x, q, scale, group_size=32, hadamard=h)
    finally:
        os.environ.pop("ANIMA_CONVROT_W8A16_KERNEL", None)
    rel = (fused - legacy).norm() / legacy.norm().clamp_min(1e-8)
    assert rel.item() < 1e-4


def test_fused_w8a16_default_path_finite_and_grad() -> None:
    torch.manual_seed(3)
    linear = nn.Linear(128, 64, bias=False)
    linear.weight.requires_grad_(False)
    q, scale = rotate_and_quantize_weight(linear.weight.detach(), 32)
    x = torch.randn(7, 128, requires_grad=True)
    y = w8a16_forward_from_buffers(x, q, scale, group_size=32)
    y.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(y).all()
    bf = linear(x.detach())
    rel = (y.detach() - bf).norm() / bf.norm().clamp_min(1e-8)
    assert rel.item() < 0.08


def test_fused_w8a8_matches_legacy_and_has_grad() -> None:
    torch.manual_seed(4)
    w = torch.randn(64, 128)
    q, scale = rotate_and_quantize_weight(w, 32)
    x = torch.randn(6, 128, requires_grad=True)
    h = normalized_hadamard(32)
    os.environ["ANIMA_CONVROT_FUSED"] = "0"
    try:
        legacy = w8a8_forward_from_buffers(
            x.detach(), q, scale, group_size=32, hadamard=h
        )
    finally:
        os.environ.pop("ANIMA_CONVROT_FUSED", None)
    fused = fused_w8a8_forward(x.detach(), q, scale, group_size=32, hadamard=h)
    rel = (fused - legacy).norm() / legacy.norm().clamp_min(1e-8)
    assert rel.item() < 1e-4

    y = fused_w8a8_forward(x, q, scale, group_size=32, hadamard=h)
    y.sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA for int8pack")
def test_w8a16_int8pack_kernel_close_to_dequant() -> None:
    torch.manual_seed(5)
    device = torch.device("cuda")
    w = torch.randn(128, 256, device=device)
    q, scale = rotate_and_quantize_weight(w, 64)
    x = torch.randn(32, 256, device=device, dtype=torch.bfloat16)
    os.environ["ANIMA_CONVROT_W8A16_KERNEL"] = "dequant"
    try:
        y_ref = fused_w8a16_forward(x.float(), q, scale, group_size=64)
    finally:
        os.environ.pop("ANIMA_CONVROT_W8A16_KERNEL", None)
    os.environ["ANIMA_CONVROT_W8A16_KERNEL"] = "int8pack"
    try:
        y_pack = fused_w8a16_forward(x, q, scale, group_size=64)
    finally:
        os.environ.pop("ANIMA_CONVROT_W8A16_KERNEL", None)
    rel = (y_pack.float() - y_ref.float()).norm() / y_ref.float().norm().clamp_min(1e-8)
    # Packed kernel is approximate vs fp32 dequant; keep a generous bound.
    assert rel.item() < 0.05


def test_fused_w8a16_returns_input_half_dtype() -> None:
    """E: org_forward must not promote half acts to float32 for QKV path."""
    torch.manual_seed(11)
    w = torch.randn(64, 128)
    q, scale = rotate_and_quantize_weight(w, 32)
    for dtype in (torch.bfloat16, torch.float16):
        x = torch.randn(4, 128, dtype=dtype)
        y = fused_w8a16_forward(x, q, scale, group_size=32)
        assert y.dtype == dtype, f"expected {dtype}, got {y.dtype}"


def test_fused_w8a8_returns_input_half_dtype() -> None:
    torch.manual_seed(12)
    w = torch.randn(64, 128)
    q, scale = rotate_and_quantize_weight(w, 32)
    x = torch.randn(4, 128, dtype=torch.bfloat16)
    y = fused_w8a8_forward(x, q, scale, group_size=32)
    assert y.dtype == torch.bfloat16
