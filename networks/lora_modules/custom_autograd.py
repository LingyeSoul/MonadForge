"""Eager-only memory-saving autograd for FP32 LoRA/LoKr projections.

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
# Full-factor LoKr projects close to the base layer's full width. A smaller
# chunk keeps its transient FP32 workspace bounded on 16 GiB V100 cards.
EAGER_LOKR_CHUNK_ROWS = 1024


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
        grad_weight = None
        if ctx.needs_input_grad[1]:
            # Preserve the reference Linear backward's reduction order. The
            # full cast is transient in backward instead of being retained by
            # every adapted layer from forward until its backward executes.
            grad_weight = grad_flat.float().transpose(0, 1).matmul(
                x_flat.float()
            ).to(weight.dtype)

        if grad_x_flat is None:
            return None, grad_weight

        for start in range(0, x_flat.shape[0], EAGER_LORA_CHUNK_ROWS):
            end = min(start + EAGER_LORA_CHUNK_ROWS, x_flat.shape[0])
            grad_chunk = grad_flat[start:end].float()
            if grad_x_flat is not None:
                grad_x_flat[start:end] = grad_chunk.matmul(weight_fp32).to(x.dtype)

        if grad_x_flat is not None:
            grad_x = grad_x_flat.reshape_as(x)
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
        grad_weight = None
        if ctx.needs_input_grad[1]:
            weight_input = x_flat.to(torch.float32, copy=True)
            weight_input.mul_(inv)
            grad_weight = grad_flat.float().transpose(0, 1).matmul(
                weight_input
            ).to(weight.dtype)
            del weight_input

        if grad_x_flat is None:
            return None, grad_weight, None

        for start in range(0, x_flat.shape[0], EAGER_LORA_CHUNK_ROWS):
            end = min(start + EAGER_LORA_CHUNK_ROWS, x_flat.shape[0])
            grad_chunk = grad_flat[start:end].float()
            if grad_x_flat is not None:
                grad_x_chunk = grad_chunk.matmul(weight_fp32)
                grad_x_chunk.mul_(inv)
                grad_x_flat[start:end] = grad_x_chunk.to(x.dtype)

        if grad_x_flat is not None:
            grad_x = grad_x_flat.reshape_as(x)
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
        if not (ctx.needs_input_grad[1] or ctx.needs_input_grad[2]):
            return grad_out, None, None, None, None

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

        # Match the reference ``F.linear`` backward's single GEMM for the
        # parameter gradient. Accumulating per-token chunks changes the FP32
        # reduction order and can round to a different FP16 update, even
        # though the two expressions are mathematically identical. The
        # adapter weight is tiny compared with the activation. Scale the one
        # required FP32 grad-output cast in place so the full GEMM does not
        # retain a second output-sized FP32 temporary.
        grad_weight = None
        grad_weight_input = None
        if ctx.needs_input_grad[2]:
            grad_weight_input = grad_flat.to(
                torch.float32,
                copy=scale != 1.0,
            )
            if scale != 1.0:
                grad_weight_input.mul_(scale)
            grad_weight = grad_weight_input.transpose(0, 1).matmul(
                rank_flat.float()
            ).to(weight.dtype)

        if grad_rank_flat is None:
            return grad_out, None, grad_weight, None, None

        for start in range(0, grad_flat.shape[0], chunk):
            end = min(start + chunk, grad_flat.shape[0])
            if grad_weight_input is not None:
                # Reuse the full FP32 cast required by the exact weight-gradient
                # GEMM instead of overlapping it with another output-width cast.
                grad_chunk = grad_weight_input[start:end]
            else:
                grad_chunk = grad_flat[start:end].to(
                    torch.float32,
                    copy=scale != 1.0,
                )
                if scale != 1.0:
                    # The forced copy keeps an FP32 upstream gradient untouched
                    # and lets FP16 use one cast buffer instead of cast + multiply.
                    grad_chunk.mul_(scale)

            if grad_rank_flat is not None:
                grad_rank_flat[start:end] = grad_chunk.matmul(weight_fp32).to(
                    rank_input.dtype
                )

        if grad_rank_flat is not None:
            grad_rank = grad_rank_flat.reshape_as(rank_input)
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


def _lokr_linear_chunk(
    x: torch.Tensor,
    w1: torch.Tensor | None,
    w1a: torch.Tensor | None,
    w1b: torch.Tensor | None,
    w2: torch.Tensor | None,
    w2a: torch.Tensor | None,
    w2b: torch.Tensor | None,
) -> torch.Tensor:
    """Evaluate LyCORIS' linear LoKr bypass formula for one row chunk."""
    c = w1.float() if w1 is not None else w1a.float().matmul(w1b.float())
    grouped = x.float().reshape(x.shape[0], c.shape[1], -1)
    if w2 is not None:
        projected = F.linear(grouped, w2.float())
    else:
        projected = F.linear(
            F.linear(grouped, w2b.float()),
            w2a.float(),
        )
    crossed = projected.transpose(-1, -2)
    output = F.linear(crossed, c).transpose(-1, -2)
    return output.reshape(x.shape[0], -1)


class EagerLoKrResidualFn(torch.autograd.Function):
    """Chunk and rematerialize the FP32 linear LoKr bypass.

    LyCORIS' eager bypass keeps the full FP32 input plus its near-full-width
    Kronecker-group intermediate for factor backward. This Function saves only
    the original input and factor storage. Each row chunk is reconstructed in
    backward, with factor gradients accumulated in FP32 before one final cast
    to the parameter dtype.
    """

    @staticmethod
    def forward(
        ctx,
        org_forwarded: torch.Tensor,
        x: torch.Tensor,
        w1: torch.Tensor | None,
        w1a: torch.Tensor | None,
        w1b: torch.Tensor | None,
        w2: torch.Tensor | None,
        w2a: torch.Tensor | None,
        w2b: torch.Tensor | None,
        scalar: torch.Tensor,
        residual_scale: float,
        chunk_size: int,
    ) -> torch.Tensor:
        scale = float(residual_scale)
        chunk = max(1, int(chunk_size))
        x_flat = _flatten_last(x)
        out_flat = _flatten_last(org_forwarded)
        factors = (w1, w1a, w1b, w2, w2a, w2b)

        ctx.mark_dirty(org_forwarded)
        for start in range(0, x_flat.shape[0], chunk):
            end = min(start + chunk, x_flat.shape[0])
            delta = _lokr_linear_chunk(x_flat[start:end], *factors)
            if scale != 1.0:
                delta.mul_(scale)
            delta.mul_(scalar.float())
            out_flat[start:end].add_(delta.to(org_forwarded.dtype))

        present = tuple(factor is not None for factor in factors)
        ctx.save_for_backward(
            x,
            scalar,
            *(factor for factor in factors if factor is not None),
        )
        ctx.factor_present = present
        ctx.residual_scale = scale
        ctx.chunk_size = chunk
        return org_forwarded

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_out: torch.Tensor):
        x, scalar, *saved_factors = ctx.saved_tensors
        factor_iter = iter(saved_factors)
        factors = tuple(
            next(factor_iter) if present else None
            for present in ctx.factor_present
        )
        needs_scalar_grad = ctx.needs_input_grad[8]
        if not (
            ctx.needs_input_grad[1]
            or needs_scalar_grad
            or any(ctx.needs_input_grad[2:8])
        ):
            return grad_out, None, None, None, None, None, None, None, None, None, None

        x_flat = _flatten_last(x)
        grad_flat = _flatten_last(grad_out)
        chunk = ctx.chunk_size

        grad_x_flat = torch.empty_like(x_flat) if ctx.needs_input_grad[1] else None
        grad_factors_fp32 = [
            (
                torch.zeros_like(factor, dtype=torch.float32)
                if factor is not None and ctx.needs_input_grad[2 + index]
                else None
            )
            for index, factor in enumerate(factors)
        ]
        grad_scalar_fp32 = (
            torch.zeros_like(scalar, dtype=torch.float32)
            if needs_scalar_grad
            else None
        )

        for start in range(0, x_flat.shape[0], chunk):
            end = min(start + chunk, x_flat.shape[0])
            with torch.enable_grad():
                local_x = (
                    x_flat[start:end]
                    .detach()
                    .float()
                    .requires_grad_(ctx.needs_input_grad[1])
                )
                local_factors = tuple(
                    (
                        factor.detach()
                        .float()
                        .requires_grad_(grad_factor is not None)
                        if factor is not None
                        else None
                    )
                    for factor, grad_factor in zip(factors, grad_factors_fp32)
                )
                local_scalar = (
                    scalar.detach()
                    .float()
                    .requires_grad_(needs_scalar_grad)
                )
                local_out = _lokr_linear_chunk(local_x, *local_factors)
                if ctx.residual_scale != 1.0:
                    local_out = local_out * ctx.residual_scale
                local_out = local_out * local_scalar

                requested = []
                requested_slots = []
                if ctx.needs_input_grad[1]:
                    requested.append(local_x)
                    requested_slots.append(("x", -1))
                for index, (factor, grad_factor) in enumerate(
                    zip(local_factors, grad_factors_fp32)
                ):
                    if factor is not None and grad_factor is not None:
                        requested.append(factor)
                        requested_slots.append(("factor", index))
                if needs_scalar_grad:
                    requested.append(local_scalar)
                    requested_slots.append(("scalar", -1))

                local_grads = torch.autograd.grad(
                    local_out,
                    requested,
                    grad_outputs=grad_flat[start:end].float(),
                    allow_unused=False,
                )

            for (kind, index), local_grad in zip(requested_slots, local_grads):
                if kind == "x":
                    grad_x_flat[start:end] = local_grad.to(x.dtype)
                elif kind == "scalar":
                    grad_scalar_fp32.add_(local_grad)
                else:
                    grad_factors_fp32[index].add_(local_grad)

        grad_x = grad_x_flat.reshape_as(x) if grad_x_flat is not None else None
        grad_factors = tuple(
            (
                grad_factor.to(factor.dtype)
                if factor is not None and grad_factor is not None
                else None
            )
            for factor, grad_factor in zip(factors, grad_factors_fp32)
        )
        grad_scalar = (
            grad_scalar_fp32.to(scalar.dtype)
            if grad_scalar_fp32 is not None
            else None
        )
        return grad_out, grad_x, *grad_factors, grad_scalar, None, None


def eager_lokr_residual(
    org_forwarded: torch.Tensor,
    x: torch.Tensor,
    w1: torch.Tensor | None,
    w1a: torch.Tensor | None,
    w1b: torch.Tensor | None,
    w2: torch.Tensor | None,
    w2a: torch.Tensor | None,
    w2b: torch.Tensor | None,
    scalar: torch.Tensor,
    residual_scale: float,
    chunk_size: int = EAGER_LOKR_CHUNK_ROWS,
) -> torch.Tensor:
    return EagerLoKrResidualFn.apply(
        org_forwarded,
        x,
        w1,
        w1a,
        w1b,
        w2,
        w2a,
        w2b,
        scalar,
        residual_scale,
        chunk_size,
    )
