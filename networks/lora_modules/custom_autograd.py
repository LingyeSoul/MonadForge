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


# Avoid fragmented rank GEMMs without retaining full FP32 layer activations.
EAGER_LORA_CHUNK_ROWS = 3072


def _flatten_last(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1, x.shape[-1])


def _empty_linear_output(x: torch.Tensor, out_features: int) -> torch.Tensor:
    return torch.empty(
        (*x.shape[:-1], out_features),
        dtype=torch.float32,
        device=x.device,
    )


class EagerLoRADownProjectFn(torch.autograd.Function):
    """Chunked ``F.linear(x.float(), weight.float())`` with recomputed casts."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        x_flat = _flatten_last(x)
        out = _empty_linear_output(x, weight.shape[0])
        out_flat = _flatten_last(out)
        weight_fp32 = weight.float()
        for start in range(0, x_flat.shape[0], EAGER_LORA_CHUNK_ROWS):
            end = min(start + EAGER_LORA_CHUNK_ROWS, x_flat.shape[0])
            out_flat[start:end] = F.linear(x_flat[start:end].float(), weight_fp32)
        ctx.save_for_backward(x, weight)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, weight = ctx.saved_tensors
        x_flat = _flatten_last(x)
        grad_flat = _flatten_last(grad_out)
        weight_fp32 = weight.float()

        grad_x = None
        grad_x_flat = torch.empty_like(x_flat) if ctx.needs_input_grad[0] else None
        grad_weight_fp32 = (
            torch.zeros_like(weight, dtype=torch.float32)
            if ctx.needs_input_grad[1]
            else None
        )

        for start in range(0, x_flat.shape[0], EAGER_LORA_CHUNK_ROWS):
            end = min(start + EAGER_LORA_CHUNK_ROWS, x_flat.shape[0])
            grad_chunk = grad_flat[start:end].float()
            if grad_x_flat is not None:
                grad_x_flat[start:end] = grad_chunk.matmul(weight_fp32).to(x.dtype)
            if grad_weight_fp32 is not None:
                grad_weight_fp32.add_(
                    grad_chunk.transpose(0, 1).matmul(x_flat[start:end].float())
                )

        if grad_x_flat is not None:
            grad_x = grad_x_flat.reshape_as(x)
        grad_weight = (
            grad_weight_fp32.to(weight.dtype) if grad_weight_fp32 is not None else None
        )
        return grad_x, grad_weight


class EagerScaledLoRADownProjectFn(torch.autograd.Function):
    """Chunked eager path for FP32 activation scaling plus down projection."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        inv_scale: torch.Tensor,
    ) -> torch.Tensor:
        x_flat = _flatten_last(x)
        out = _empty_linear_output(x, weight.shape[0])
        out_flat = _flatten_last(out)
        weight_fp32 = weight.float()
        inv = inv_scale.float()
        for start in range(0, x_flat.shape[0], EAGER_LORA_CHUNK_ROWS):
            end = min(start + EAGER_LORA_CHUNK_ROWS, x_flat.shape[0])
            x_chunk = x_flat[start:end].to(torch.float32, copy=True)
            x_chunk.mul_(inv)
            out_flat[start:end] = F.linear(x_chunk, weight_fp32)
        ctx.save_for_backward(x, weight, inv_scale)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, weight, inv_scale = ctx.saved_tensors
        x_flat = _flatten_last(x)
        grad_flat = _flatten_last(grad_out)
        weight_fp32 = weight.float()
        inv = inv_scale.float()

        grad_x = None
        grad_x_flat = torch.empty_like(x_flat) if ctx.needs_input_grad[0] else None
        grad_weight_fp32 = (
            torch.zeros_like(weight, dtype=torch.float32)
            if ctx.needs_input_grad[1]
            else None
        )

        for start in range(0, x_flat.shape[0], EAGER_LORA_CHUNK_ROWS):
            end = min(start + EAGER_LORA_CHUNK_ROWS, x_flat.shape[0])
            grad_chunk = grad_flat[start:end].float()
            if grad_x_flat is not None:
                grad_x_chunk = grad_chunk.matmul(weight_fp32)
                grad_x_chunk.mul_(inv)
                grad_x_flat[start:end] = grad_x_chunk.to(x.dtype)
            if grad_weight_fp32 is not None:
                x_chunk = x_flat[start:end].float()
                x_chunk.mul_(inv)
                grad_weight_fp32.add_(grad_chunk.transpose(0, 1).matmul(x_chunk))

        if grad_x_flat is not None:
            grad_x = grad_x_flat.reshape_as(x)
        grad_weight = (
            grad_weight_fp32.to(weight.dtype) if grad_weight_fp32 is not None else None
        )
        return grad_x, grad_weight, None


class EagerLoRAUpResidualFn(torch.autograd.Function):
    """Chunked FP32 LoRA up projection fused into an FP16 residual.

    Plain eager evaluation materializes the complete FP32 ``lora_up`` output
    and then another equally large tensor for scalar multiplication before it
    can cast back to the model dtype.  At ~4k image tokens, the first qkv LoRA
    alone therefore needs another ~96 MiB allocation.  Compiled graphs fuse
    that elementwise tail; this eager Function obtains the same memory shape by
    updating the freshly-created frozen-base output in place and limiting the
    FP32 projection temporary to ``chunk_size`` rows.

    Backward is chunked as well so converting ``grad_out`` to FP32 cannot
    recreate the full-width temporary.  The LoRA rank activation is small and
    remains saved in its original dtype.
    """

    @staticmethod
    def forward(
        ctx,
        org_forwarded: torch.Tensor,
        rank_input: torch.Tensor,
        weight: torch.Tensor,
        residual_scale: float,
        chunk_size: int,
    ) -> torch.Tensor:
        scale = float(residual_scale)
        chunk = max(1, int(chunk_size))
        org_flat = _flatten_last(org_forwarded)
        rank_flat = _flatten_last(rank_input)
        weight_fp32 = weight.float()

        # ``org_forwarded`` is the fresh output of the frozen base Linear.  Its
        # backward does not consume the output value, so it is safe to reuse its
        # storage as the final residual rather than allocating another FP16
        # tensor of the same full output shape.
        ctx.mark_dirty(org_forwarded)
        for start in range(0, org_flat.shape[0], chunk):
            end = min(start + chunk, org_flat.shape[0])
            delta = F.linear(rank_flat[start:end].float(), weight_fp32)
            if scale != 1.0:
                delta.mul_(scale)
            org_flat[start:end].add_(delta.to(org_forwarded.dtype))

        ctx.save_for_backward(rank_input, weight)
        ctx.residual_scale = scale
        ctx.chunk_size = chunk
        return org_forwarded

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        rank_input, weight = ctx.saved_tensors
        grad_flat = _flatten_last(grad_out)
        rank_flat = _flatten_last(rank_input)
        weight_fp32 = weight.float()
        scale = ctx.residual_scale
        chunk = ctx.chunk_size

        grad_rank = None
        if ctx.needs_input_grad[1]:
            grad_rank_flat = torch.empty_like(rank_flat)
        else:
            grad_rank_flat = None

        grad_weight = None
        if ctx.needs_input_grad[2]:
            grad_weight_fp32 = torch.zeros_like(weight, dtype=torch.float32)
        else:
            grad_weight_fp32 = None

        for start in range(0, grad_flat.shape[0], chunk):
            end = min(start + chunk, grad_flat.shape[0])
            grad_chunk = grad_flat[start:end].float()
            if scale != 1.0:
                grad_chunk = grad_chunk * scale

            if grad_rank_flat is not None:
                grad_rank_flat[start:end] = grad_chunk.matmul(weight_fp32).to(
                    rank_input.dtype
                )
            if grad_weight_fp32 is not None:
                grad_weight_fp32.add_(
                    grad_chunk.transpose(0, 1).matmul(rank_flat[start:end].float())
                )

        if grad_rank_flat is not None:
            grad_rank = grad_rank_flat.reshape_as(rank_input)
        if grad_weight_fp32 is not None:
            grad_weight = grad_weight_fp32.to(weight.dtype)

        # The residual branch is an identity with respect to the frozen base
        # output.  Scalar/chunk arguments are non-Tensor controls.
        return grad_out, grad_rank, grad_weight, None, None


def eager_lora_up_residual(
    org_forwarded: torch.Tensor,
    rank_input: torch.Tensor,
    weight: torch.Tensor,
    residual_scale: float,
    chunk_size: int = EAGER_LORA_CHUNK_ROWS,
) -> torch.Tensor:
    return EagerLoRAUpResidualFn.apply(
        org_forwarded,
        rank_input,
        weight,
        residual_scale,
        chunk_size,
    )


def eager_lora_down_project(
    x: torch.Tensor,
    weight: torch.Tensor,
    inv_scale: torch.Tensor | None,
) -> torch.Tensor:
    if inv_scale is None:
        return EagerLoRADownProjectFn.apply(x, weight)
    return EagerScaledLoRADownProjectFn.apply(x, weight, inv_scale)
