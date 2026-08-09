"""W8A8 ConvRot Linear: rotated int8 weights + dynamic int8 activations.

Default path uses fused dense RHT + act quant + int8 GEMM (``torch._int_mm``
when shapes allow). Autograd uses STE through the act quant (base weights frozen).
"""

from __future__ import annotations

import torch
from torch import nn

from library.runtime.convrot.fused import fused_enabled, fused_w8a8_forward
from library.runtime.convrot.gemm import w8a8_int_linear
from library.runtime.convrot.quant import rotate_and_quantize_weight
from library.runtime.convrot.rht import assert_group_divides, group_rht


class ConvRotW8A8Linear(nn.Module):
    """Frozen Linear with group-RHT weights and online act RHT + int8 GEMM."""

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
    ) -> "ConvRotW8A8Linear":
        if linear.bias is not None:
            raise ValueError("ConvRotW8A8Linear only supports bias=False Linear modules")
        if linear.weight.requires_grad:
            raise ValueError("ConvRotW8A8Linear only supports frozen Linear weights")
        assert_group_divides(int(linear.in_features), group_size)
        q, scale = rotate_and_quantize_weight(linear.weight.detach(), group_size)
        return cls(q, scale, group_size=group_size)

    @property
    def in_features(self) -> int:
        return int(self.quantized_weight.shape[1])

    @property
    def out_features(self) -> int:
        return int(self.quantized_weight.shape[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return w8a8_forward_from_buffers(
            x,
            self.quantized_weight,
            self.scale,
            group_size=self.group_size,
        )


def w8a8_forward_from_buffers(
    x: torch.Tensor,
    quantized_weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    group_size: int,
    hadamard: torch.Tensor | None = None,
    weight_layout: str = "nk",
) -> torch.Tensor:
    """Functional W8A8 path for LoRA ``org_forward`` closures.

    Optional precomputed ``hadamard`` is forwarded into the fused path (P1-H).
    ``weight_layout="kn"`` expects pre-transposed int8 ``[in, out]`` (P1.6).
    """
    layout = "kn" if weight_layout == "kn" else "nk"
    if layout == "kn":
        assert_group_divides(int(quantized_weight.shape[0]), group_size)
    else:
        assert_group_divides(int(quantized_weight.shape[1]), group_size)
    if fused_enabled():
        return fused_w8a8_forward(
            x,
            quantized_weight,
            scale,
            group_size=group_size,
            hadamard=hadamard,
            weight_layout=layout,
        )
    # RHT in act TC dtype; absmax quant still promotes to fp32 internally (P1.5).
    compute_dtype = (
        x.dtype
        if x.is_floating_point() and x.dtype in (torch.float16, torch.bfloat16)
        else torch.bfloat16
        if x.device.type == "cuda"
        else torch.float32
    )
    x_work = x.to(dtype=compute_dtype) if x.dtype != compute_dtype else x
    x_rot = group_rht(x_work, group_size, hadamard=hadamard)
    y = w8a8_int_linear(
        x_rot,
        quantized_weight.to(device=x.device),
        scale.to(device=x.device),
        weight_layout=layout,  # type: ignore[arg-type]
    )
    return y.to(dtype=x.dtype) if x.is_floating_point() and y.dtype != x.dtype else y
