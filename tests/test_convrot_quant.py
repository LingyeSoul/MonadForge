"""Unit tests for ConvRot rotated-domain quant helpers."""

from __future__ import annotations

import torch

from library.runtime.convrot.quant import (
    dequantize_weight,
    fake_quantize_activation_int8,
    quantize_weight_per_output_channel,
    rotate_and_quantize_weight,
)
from library.runtime.convrot.rht import group_rht_weight


def test_quantize_weight_per_output_channel_shapes_and_roundtrip() -> None:
    torch.manual_seed(0)
    w = torch.randn(11, 32)
    q, scale = quantize_weight_per_output_channel(w)
    assert q.dtype is torch.int8
    assert q.shape == w.shape
    assert scale.shape == (11,)
    restored = dequantize_weight(q, scale, dtype=w.dtype)
    rel = (restored - w).norm() / w.norm().clamp_min(1e-8)
    assert rel.item() < 0.05


def test_dequantize_weight_half_dtype_close_to_fp32_path() -> None:
    """P1.5: target-dtype dequant must not drift far from fp32-then-cast."""
    torch.manual_seed(11)
    w = torch.randn(17, 64)
    q, scale = quantize_weight_per_output_channel(w)
    ref = (q.to(torch.float32) * scale.to(torch.float32)[:, None]).to(torch.bfloat16)
    got = dequantize_weight(q, scale, dtype=torch.bfloat16)
    assert got.dtype is torch.bfloat16
    # bf16 rounding of (i8 * scale) can differ slightly by order of cast;
    # keep a generous bound for absmax scales in typical weight range.
    rel = (got.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-8)
    # i8 * scale in bf16 vs fp32-then-cast: ~0.3% typical from cast order.
    assert rel.item() < 1e-2


def test_dequantize_weight_out_scratch_matches_alloc(monkeypatch) -> None:
    """P1.8: writing into a shared scratch buffer must match fresh allocate."""
    from library.runtime.convrot.quant import clear_dequant_scratch, get_dequant_scratch

    monkeypatch.setenv("ANIMA_CONVROT_DEQUANT_SCRATCH", "1")
    torch.manual_seed(12)
    w = torch.randn(11, 48)
    q, scale = quantize_weight_per_output_channel(w)
    ref = dequantize_weight(q, scale, dtype=torch.float32)
    clear_dequant_scratch()
    scratch = get_dequant_scratch(11, 48, device=q.device, dtype=torch.float32)
    assert scratch is not None and tuple(scratch.shape) == (11, 48)
    got = dequantize_weight(q, scale, dtype=torch.float32, out=scratch)
    assert got.data_ptr() == scratch.data_ptr()
    assert torch.allclose(got, ref, atol=0, rtol=0)
    # Larger N*K grows flat capacity; smaller view still exact.
    big = get_dequant_scratch(20, 64, device=q.device, dtype=torch.float32)
    assert big is not None and tuple(big.shape) == (20, 64)
    got2 = dequantize_weight(
        q,
        scale,
        dtype=torch.float32,
        out=get_dequant_scratch(11, 48, device=q.device, dtype=torch.float32),
    )
    assert torch.allclose(got2, ref, atol=0, rtol=0)
    # max(N)×max(K) trap must NOT apply: capacity is max(N*K), not 20*64 after 11*48.
    clear_dequant_scratch()
    a = get_dequant_scratch(8192, 2048, device=q.device, dtype=torch.bfloat16)
    b = get_dequant_scratch(2048, 8192, device=q.device, dtype=torch.bfloat16)
    assert a is not None and b is not None
    # Same flat storage; both views have numel 8192*2048.
    assert a.numel() == 8192 * 2048
    assert b.numel() == 8192 * 2048
    # Underlying flat buffer capacity equals max product, not 8192*8192.
    from library.runtime.convrot.quant import _DEQUANT_SCRATCH

    flat = next(iter(_DEQUANT_SCRATCH.values()))
    assert flat.numel() == 8192 * 2048
    assert flat.numel() < 8192 * 8192


def test_dequant_scratch_default_off(monkeypatch) -> None:
    from library.runtime.convrot.quant import clear_dequant_scratch, get_dequant_scratch

    monkeypatch.delenv("ANIMA_CONVROT_DEQUANT_SCRATCH", raising=False)
    clear_dequant_scratch()
    assert get_dequant_scratch(8, 8, device=torch.device("cpu"), dtype=torch.float32) is None


def test_rotate_and_quantize_weight_uses_rotated_domain() -> None:
    torch.manual_seed(1)
    w = torch.randn(5, 64)
    group = 16
    q, scale = rotate_and_quantize_weight(w, group)
    rotated = group_rht_weight(w.to(torch.float32), group)
    q_ref, scale_ref = quantize_weight_per_output_channel(rotated)
    assert torch.equal(q, q_ref)
    assert torch.allclose(scale, scale_ref)


def test_fake_quantize_activation_int8_ste_passes_grad() -> None:
    torch.manual_seed(2)
    x = torch.randn(4, 32, requires_grad=True)
    y = fake_quantize_activation_int8(x)
    loss = y.pow(2).mean()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    # STE: grad should be non-zero almost everywhere for random input.
    assert x.grad.abs().sum().item() > 0


def test_fake_quantize_stays_in_float_scale() -> None:
    torch.manual_seed(3)
    x = torch.randn(2, 16) * 3
    y = fake_quantize_activation_int8(x)
    # Fake path should remain near original magnitude, not int8 code range.
    assert y.abs().max().item() > 1.0
    assert (y - x).abs().max().item() < 0.5
