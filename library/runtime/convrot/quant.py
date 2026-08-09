"""Weight / activation int8 quant helpers for ConvRot (rotated domain)."""

from __future__ import annotations

import os

import torch

from library.runtime.convrot.rht import (
    assert_group_divides,
    group_rht_weight,
    hadamard_kind,
    normalized_hadamard,
)

INT8_MAX = 127.0
SCALE_EPS = 1e-12


def quantize_weight_per_output_channel(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-channel absmax int8 quant on a 2D weight ``[out, in]``."""
    if weight.dim() != 2:
        raise ValueError(f"expected a 2D Linear weight, got shape={tuple(weight.shape)}")
    rows = weight.detach().to(torch.float32)
    amax = rows.abs().amax(dim=1)
    scale = (amax / INT8_MAX).clamp_min(SCALE_EPS)
    quantized = (rows / scale[:, None]).round().clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
    return quantized, scale


def dequantize_weight(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dequant int8 weights.

    When ``dtype`` is fp16/bf16, cast then multiply in that dtype (no full fp32
    materialize). Numerically within rounding of the fp32-then-cast path for
    per-channel absmax scales; hot path measured cheaper on Ampere.

    ``out`` (optional): preallocated ``[out_features, in_features]`` buffer in
    the target dtype/device. When provided, write into a ``[N, K]`` view and
    return that view (P1.8 shared scratch).
    """
    if quantized.dim() != 2:
        raise ValueError("quantized weight must be 2D")
    if scale.shape != (quantized.shape[0],):
        raise ValueError("scale must have one value per output channel")
    n, k = quantized.shape
    if out is not None:
        if out.shape != (n, k):
            raise ValueError(
                f"out buffer shape {tuple(out.shape)} must equal weight {(n, k)}"
            )
        if out.device != quantized.device:
            raise ValueError("out buffer device must match quantized weight")
        out.copy_(quantized.to(dtype=out.dtype))
        out.mul_(scale.to(device=out.device, dtype=out.dtype)[:, None])
        return out
    if dtype is None or dtype == torch.float32:
        weight = quantized.to(torch.float32) * scale.to(
            device=quantized.device, dtype=torch.float32
        )[:, None]
        return weight
    # Half / bf16 / other: avoid allocating a full fp32 [out, in] temporary.
    return quantized.to(dtype) * scale.to(device=quantized.device, dtype=dtype)[:, None]


# Process-level dequant scratch for sequential DiT forwards (P1.8).
# Keyed by (device_type, device_index, dtype). Stores a **flat** buffer sized
# to max(N*K) seen so far, then viewed as [N, K] — avoids the
# max(N)×max(K) trap (Anima mlp would otherwise allocate 8192×8192).
# Safe under single-stream sequential layer execution.
_DEQUANT_SCRATCH: dict[tuple[str, int, torch.dtype], torch.Tensor] = {}
_SCRATCH_ENV = "ANIMA_CONVROT_DEQUANT_SCRATCH"


def dequant_scratch_enabled() -> bool:
    # Default OFF: resident max(N*K) buffer adds ~30–40MB peak on Anima mlp
    # with no measured step-time win on 3080 (P1.8 hot test). Opt-in via env.
    raw = str(os.environ.get(_SCRATCH_ENV, "0") or "0").strip().lower()
    return raw in {"1", "true", "on", "yes"}


def clear_dequant_scratch() -> None:
    _DEQUANT_SCRATCH.clear()


def get_dequant_scratch(
    n: int,
    k: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Return a reusable ``[n, k]`` view into a flat process buffer, or None."""
    if not dequant_scratch_enabled():
        return None
    if device.type == "meta":
        return None
    n = int(n)
    k = int(k)
    if n <= 0 or k <= 0:
        return None
    idx = device.index if device.index is not None else -1
    key = (device.type, int(idx), dtype)
    needed = n * k
    buf = _DEQUANT_SCRATCH.get(key)
    if (
        buf is None
        or buf.device != device
        or buf.dtype != dtype
        or int(buf.numel()) < needed
    ):
        prev = 0 if buf is None else int(buf.numel())
        # Grow geometrically a bit to absorb shape jitter without thrashing.
        capacity = max(prev, needed)
        buf = torch.empty(capacity, device=device, dtype=dtype)
        _DEQUANT_SCRATCH[key] = buf
    return buf[:needed].view(n, k)


def rotate_and_quantize_weight(
    weight: torch.Tensor,
    group_size: int,
    *,
    hadamard: torch.Tensor | None = None,
    kind: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RHT on in-features then per-output-channel int8 quant.

    Returns ``(W_q int8 [out, in], scale float32 [out])``.
    ``kind`` defaults to ``ANIMA_CONVROT_HADAMARD`` (sylvester|regular).
    """
    kind_resolved = kind or hadamard_kind()
    assert_group_divides(int(weight.shape[-1]), group_size, kind=kind_resolved)  # type: ignore[arg-type]
    if hadamard is None:
        hadamard = normalized_hadamard(
            group_size,
            device=weight.device,
            dtype=torch.float32,
            kind=kind_resolved,  # type: ignore[arg-type]
        )
    rotated = group_rht_weight(
        weight.detach().to(torch.float32),
        group_size,
        hadamard=hadamard,
        kind=kind_resolved,  # type: ignore[arg-type]
    )
    return quantize_weight_per_output_channel(rotated)


def dynamic_absmax_quantize_activation(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic per-token absmax int8 quant on last dim.

    Returns ``(x_q int8, scale float32)`` where scale broadcasts over the last
    feature dim: shape is ``x.shape[:-1] + (1,)``.
    """
    if x.shape[-1] == 0:
        empty_q = x.to(torch.int8)
        empty_s = torch.ones(
            *x.shape[:-1],
            1,
            device=x.device,
            dtype=torch.float32,
        )
        return empty_q, empty_s
    work = x.to(torch.float32)
    # STE-friendly path: quantize in forward, straight-through via identity on backward
    # for the activation tensor when training W8A8 fake path.
    amax = work.detach().abs().amax(dim=-1, keepdim=True).clamp_min(SCALE_EPS)
    scale = amax / INT8_MAX
    quantized = (work / scale).round().clamp(-INT8_MAX, INT8_MAX)
    dequant = quantized * scale
    # Fake-quant in original float scale; STE treats round as identity.
    fake = work + (dequant - work).detach()
    return fake, scale


def fake_quantize_activation_int8(x: torch.Tensor) -> torch.Tensor:
    """Return STE fake-quantized activation in the same dtype as ``x``."""
    fake, _scale = dynamic_absmax_quantize_activation(x)
    return fake.to(dtype=x.dtype)
