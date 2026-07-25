"""Eager-only memory-saving autograd for FP32 LoRA down projections.

The V100/fp16 training path intentionally runs the small LoRA rank GEMMs in
FP32.  Plain eager autograd then saves the converted/scaled FP32 activation for
``lora_down.weight``'s backward, even when the layer input was already stored
as FP16.  Across all adapted DiT linears that retained copy costs several GiB.

These Functions keep the forward and gradient arithmetic in FP32 but save the
*original* input/weight storage.  Casts and channel scaling are reconstructed
in backward.  Unlike activation compression, an FP16 input is not quantized a
second time; an FP32 input remains FP32 and therefore keeps exact semantics.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _flatten_last(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1, x.shape[-1])


class EagerLoRADownProjectFn(torch.autograd.Function):
    """Unscaled ``F.linear(x.float(), weight.float())`` with recomputed casts."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        out = F.linear(x.float(), weight.float())
        ctx.save_for_backward(x, weight)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, weight = ctx.saved_tensors
        go = grad_out.float()

        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = go.matmul(weight.float()).to(x.dtype)

        grad_weight = None
        if ctx.needs_input_grad[1]:
            grad_weight = _flatten_last(go).transpose(0, 1).matmul(
                _flatten_last(x.float())
            )
            grad_weight = grad_weight.to(weight.dtype)

        return grad_x, grad_weight


class EagerScaledLoRADownProjectFn(torch.autograd.Function):
    """Exact eager path for ``F.linear(x.float() * inv_scale, weight.float())``.

    Scaling is deliberately applied to the activation, in the same FP32 order
    as ``BaseLoRAModule._rebalance(x.to(torch.float32))``.  The older retired
    Function folded the scale into the weight; that saves a forward temporary
    but changes FP32 rounding order.  This experiment prioritizes matching the
    current mixed-precision path exactly.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        inv_scale: torch.Tensor,
    ) -> torch.Tensor:
        x_scaled = x.float() * inv_scale.float()
        out = F.linear(x_scaled, weight.float())
        ctx.save_for_backward(x, weight, inv_scale)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, weight, inv_scale = ctx.saved_tensors
        go = grad_out.float()
        inv = inv_scale.float()

        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = (go.matmul(weight.float()) * inv).to(x.dtype)

        grad_weight = None
        if ctx.needs_input_grad[1]:
            x_scaled = x.float() * inv
            grad_weight = _flatten_last(go).transpose(0, 1).matmul(
                _flatten_last(x_scaled)
            )
            grad_weight = grad_weight.to(weight.dtype)

        return grad_x, grad_weight, None


def eager_lora_down_project(
    x: torch.Tensor,
    weight: torch.Tensor,
    inv_scale: torch.Tensor | None,
) -> torch.Tensor:
    if inv_scale is None:
        return EagerLoRADownProjectFn.apply(x, weight)
    return EagerScaledLoRADownProjectFn.apply(x, weight, inv_scale)
