"""Tests for ConvRot int8 GEMM path."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from library.runtime.convrot.gemm import (
    can_use_torch_int_mm,
    int8_mm_scaled,
    quantize_activation_absmax_int8,
    w8a8_int_linear,
)
from library.runtime.convrot.linear_w8a8 import ConvRotW8A8Linear
from library.runtime.convrot.quant import rotate_and_quantize_weight


def test_int8_mm_scaled_float_matches_dequant_matmul() -> None:
    torch.manual_seed(0)
    m, k, n = 7, 64, 32
    x = torch.randn(m, k)
    w = torch.randn(n, k)
    x_q, x_s = quantize_activation_absmax_int8(x)
    w_q, w_s = rotate_and_quantize_weight(w, 32)  # also tests shape; use plain quant path
    # Use per-channel quant without rotation for exact dequant reference:
    from library.runtime.convrot.quant import quantize_weight_per_output_channel

    w_q, w_s = quantize_weight_per_output_channel(w)
    y = int8_mm_scaled(x_q, x_s, w_q, w_s, prefer="float")
    ref = (x_q.float() * x_s) @ (w_q.float() * w_s[:, None]).t()
    assert torch.allclose(y, ref, atol=1e-4, rtol=1e-4)


def test_quantize_activation_half_matches_fp32_codes() -> None:
    """Half inputs still produce fp32-stable int8 codes (half-mul path rejected)."""
    torch.manual_seed(11)
    x = torch.randn(8, 64, dtype=torch.bfloat16)
    q_h, s_h = quantize_activation_absmax_int8(x)
    work_f = x.float()
    amax = work_f.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = amax / 127.0
    q_ref = (work_f / scale).round().clamp(-127, 127).to(torch.int8)
    assert torch.equal(q_h, q_ref)
    assert torch.allclose(s_h, scale, rtol=0, atol=0)
    assert q_h.dtype == torch.int8
    assert s_h.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for _int_mm")
def test_int8_mm_scaled_cuda_int_mm_matches_float_when_supported() -> None:
    torch.manual_seed(1)
    m, k, n = 32, 256, 128
    device = torch.device("cuda")
    x = torch.randn(m, k, device=device)
    w = torch.randn(n, k, device=device)
    from library.runtime.convrot.quant import quantize_weight_per_output_channel

    x_q, x_s = quantize_activation_absmax_int8(x)
    w_q, w_s = quantize_weight_per_output_channel(w)
    assert can_use_torch_int_mm(m, k, n, device=device)
    y_int = int8_mm_scaled(x_q, x_s, w_q, w_s, prefer="int_mm")
    y_fp = int8_mm_scaled(x_q, x_s, w_q, w_s, prefer="float")
    # Scales make results float; integer path should match float ref closely.
    rel = (y_int - y_fp).norm() / y_fp.norm().clamp_min(1e-8)
    assert rel.item() < 1e-5


def test_w8a8_int_linear_ste_grad() -> None:
    torch.manual_seed(2)
    x = torch.randn(5, 64, requires_grad=True)
    w = torch.randn(32, 64)
    from library.runtime.convrot.quant import quantize_weight_per_output_channel

    w_q, w_s = quantize_weight_per_output_channel(w)
    y = w8a8_int_linear(x, w_q, w_s)
    y.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert x.grad.abs().sum().item() > 0


def test_convrot_w8a8_module_finite_and_close_to_bf16() -> None:
    torch.manual_seed(3)
    linear = nn.Linear(64, 32, bias=False)
    linear.weight.requires_grad_(False)
    wrapped = ConvRotW8A8Linear.from_linear(linear, group_size=16)
    x = torch.randn(4, 64)
    y = wrapped(x)
    bf = linear(x)
    assert torch.isfinite(y).all()
    rel = (y - bf).norm() / bf.norm().clamp_min(1e-8)
    assert rel.item() < 0.15


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA preferred for backend env")
def test_backend_env_float_forces_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANIMA_CONVROT_INT8_GEMM", "float")
    torch.manual_seed(4)
    m, k, n = 32, 128, 64
    device = torch.device("cuda")
    x = torch.randn(m, k, device=device)
    from library.runtime.convrot.quant import quantize_weight_per_output_channel

    w = torch.randn(n, k, device=device)
    x_q, x_s = quantize_activation_absmax_int8(x)
    w_q, w_s = quantize_weight_per_output_channel(w)
    y = int8_mm_scaled(x_q, x_s, w_q, w_s)  # prefer auto but env=float
    assert torch.isfinite(y).all()


def test_int8_mm_scaled_kn_layout_matches_nk() -> None:
    """P1.6: pre-transposed [K,N] must match classic [N,K] float path."""
    torch.manual_seed(6)
    m, k, n = 24, 64, 48
    from library.runtime.convrot.quant import quantize_weight_per_output_channel

    x = torch.randn(m, k)
    w = torch.randn(n, k)
    x_q, x_s = quantize_activation_absmax_int8(x)
    w_q, w_s = quantize_weight_per_output_channel(w)
    y_nk = int8_mm_scaled(x_q, x_s, w_q, w_s, prefer="float", weight_layout="nk")
    y_kn = int8_mm_scaled(
        x_q, x_s, w_q.t().contiguous(), w_s, prefer="float", weight_layout="kn"
    )
    assert torch.allclose(y_nk, y_kn, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA for _int_mm kn")
def test_int8_mm_scaled_kn_cuda_int_mm_matches_nk() -> None:
    torch.manual_seed(7)
    m, k, n = 32, 256, 128
    device = torch.device("cuda")
    from library.runtime.convrot.quant import quantize_weight_per_output_channel

    x = torch.randn(m, k, device=device)
    w = torch.randn(n, k, device=device)
    x_q, x_s = quantize_activation_absmax_int8(x)
    w_q, w_s = quantize_weight_per_output_channel(w)
    y_nk = int8_mm_scaled(x_q, x_s, w_q, w_s, prefer="int_mm", weight_layout="nk")
    y_kn = int8_mm_scaled(
        x_q, x_s, w_q.t().contiguous(), w_s, prefer="int_mm", weight_layout="kn"
    )
    rel = (y_kn - y_nk).norm() / y_nk.norm().clamp_min(1e-8)
    assert rel.item() < 1e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA for SM86 row padding")
def test_int8_mm_scaled_cuda_pads_4200_token_family() -> None:
    """4200 rows must stay on the int8 path on Ampere instead of failing cuBLASLt."""
    torch.manual_seed(8)
    m, k, n = 4200, 64, 64
    device = torch.device("cuda")
    x_q = torch.randint(-127, 128, (m, k), device=device, dtype=torch.int8)
    w_q = torch.randint(-127, 128, (k, n), device=device, dtype=torch.int8)
    x_scale = torch.rand(m, 1, device=device, dtype=torch.float32)
    w_scale = torch.rand(n, device=device, dtype=torch.float32)

    y_int = int8_mm_scaled(
        x_q,
        x_scale,
        w_q,
        w_scale,
        prefer="int_mm",
        weight_layout="kn",
    )
    y_ref = int8_mm_scaled(
        x_q,
        x_scale,
        w_q,
        w_scale,
        prefer="float",
        weight_layout="kn",
    )

    assert y_int.shape == (m, n)
    assert torch.allclose(y_int, y_ref, atol=1e-3, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA for compiled padding")
def test_int8_mm_scaled_compiled_4200_token_family() -> None:
    torch.manual_seed(9)
    m, k, n = 4200, 64, 64
    device = torch.device("cuda")
    x_q = torch.randint(-127, 128, (m, k), device=device, dtype=torch.int8)
    w_q = torch.randint(-127, 128, (k, n), device=device, dtype=torch.int8)
    x_scale = torch.rand(m, 1, device=device, dtype=torch.float32)
    w_scale = torch.rand(n, device=device, dtype=torch.float32)

    def run(xq, xs, wq, ws):
        return int8_mm_scaled(
            xq,
            xs,
            wq,
            ws,
            prefer="int_mm",
            weight_layout="kn",
        )

    compiled = torch.compile(run, backend="inductor")
    y = compiled(x_q, x_scale, w_q, w_scale)
    y_ref = int8_mm_scaled(
        x_q,
        x_scale,
        w_q,
        w_scale,
        prefer="float",
        weight_layout="kn",
    )

    assert y.shape == (m, n)
    assert torch.allclose(y, y_ref, atol=1e-3, rtol=1e-5)
