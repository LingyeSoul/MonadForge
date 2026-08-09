"""W8A16 ConvRot Linear: rotated int8 weights, bf16/fp16 activations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from library.runtime.convrot.fused import fused_enabled, fused_w8a16_forward
from library.runtime.convrot.quant import (
    dequantize_weight,
    get_dequant_scratch,
    rotate_and_quantize_weight,
)
from library.runtime.convrot.rht import assert_group_divides, group_rht


@dataclass(frozen=True)
class ConvRotWeightPayload:
    quantized_weight: torch.Tensor  # int8 [out, in]
    scale: torch.Tensor  # float32 [out]
    group_size: int
    in_features: int
    out_features: int


class ConvRotW8A16Linear(nn.Module):
    """Frozen Linear with group-RHT + int8 weight storage and W8A16 forward.

    Default path uses fused dense-RHT + dequant ``F.linear`` (see ``fused.py``).
    Output lives in the original feature space because Hadamard is orthogonal.
    """

    def __init__(
        self,
        quantized_weight: torch.Tensor,
        scale: torch.Tensor,
        *,
        group_size: int,
    ) -> None:
        super().__init__()
        if quantized_weight.dtype is not torch.int8:
            raise TypeError("quantized_weight must be torch.int8")
        if quantized_weight.dim() != 2:
            raise ValueError("quantized_weight must be 2D")
        if scale.shape != (quantized_weight.shape[0],):
            raise ValueError("scale must have one value per output channel")
        assert_group_divides(int(quantized_weight.shape[1]), group_size)
        self.group_size = int(group_size)
        self.register_buffer("quantized_weight", quantized_weight.contiguous())
        self.register_buffer("scale", scale.to(torch.float32).contiguous())

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        group_size: int = 256,
    ) -> "ConvRotW8A16Linear":
        if linear.bias is not None:
            raise ValueError("ConvRotW8A16Linear only supports bias=False Linear modules")
        if linear.weight.requires_grad:
            raise ValueError("ConvRotW8A16Linear only supports frozen Linear weights")
        assert_group_divides(int(linear.in_features), group_size)
        q, scale = rotate_and_quantize_weight(linear.weight.detach(), group_size)
        return cls(q, scale, group_size=group_size)

    @property
    def in_features(self) -> int:
        return int(self.quantized_weight.shape[1])

    @property
    def out_features(self) -> int:
        return int(self.quantized_weight.shape[0])

    def dequantized_weight(self, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        return dequantize_weight(self.quantized_weight, self.scale, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return w8a16_forward_from_buffers(
            x,
            self.quantized_weight,
            self.scale,
            group_size=self.group_size,
        )


def build_w8a16_payload_from_linear(
    linear: nn.Linear,
    *,
    group_size: int,
) -> ConvRotWeightPayload:
    if linear.bias is not None:
        raise ValueError("ConvRot only supports bias=False Linear modules")
    if linear.weight.requires_grad:
        raise ValueError("ConvRot only supports frozen Linear weights")
    assert_group_divides(int(linear.in_features), group_size)
    q, scale = rotate_and_quantize_weight(linear.weight.detach(), group_size)
    return ConvRotWeightPayload(
        quantized_weight=q.contiguous(),
        scale=scale.to(torch.float32).contiguous(),
        group_size=int(group_size),
        in_features=int(linear.in_features),
        out_features=int(linear.out_features),
    )


def w8a16_forward_from_buffers(
    x: torch.Tensor,
    quantized_weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    group_size: int,
    hadamard: torch.Tensor | None = None,
) -> torch.Tensor:
    """Functional W8A16 path used by LoRA ``org_forward`` closures.

    Default uses the fused autograd path (dense RHT + dequant linear). An
    optional precomputed ``hadamard`` is forwarded into the fused path so the
    hot path avoids env/cache lookups (P1-H). Set ``ANIMA_CONVROT_FUSED=0`` to
    force the legacy non-fused dense path (still honors ``hadamard``).
    """
    assert_group_divides(int(quantized_weight.shape[1]), group_size)
    if fused_enabled():
        return fused_w8a16_forward(
            x,
            quantized_weight,
            scale,
            group_size=group_size,
            hadamard=hadamard,
        )
    # RHT + dequant linear prefer bf16/fp16 TC (matches fused path / P1.5).
    compute_dtype = (
        x.dtype
        if x.is_floating_point() and x.dtype in (torch.float16, torch.bfloat16)
        else torch.bfloat16
        if x.device.type == "cuda"
        else torch.float32
    )
    x_work = x.to(dtype=compute_dtype) if x.dtype != compute_dtype else x
    x_rot = group_rht(x_work, group_size, hadamard=hadamard)
    n, k = int(quantized_weight.shape[0]), int(quantized_weight.shape[1])
    scratch = get_dequant_scratch(n, k, device=x.device, dtype=compute_dtype)
    weight = dequantize_weight(
        quantized_weight, scale, dtype=compute_dtype, out=scratch
    )
    if weight.device != x.device:
        weight = weight.to(device=x.device)
    y = F.linear(x_rot, weight, None)
    return y.to(dtype=x.dtype) if x.is_floating_point() and y.dtype != x.dtype else y
