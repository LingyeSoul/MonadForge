"""Eager-only fused autograd helpers for memory-constrained Volta training."""

from __future__ import annotations

import torch


# A 1024px Anima sample is only ~4200 tokens. Keeping it in one chunk removes
# dozens of tiny eager pointwise launches on Volta, while still bounding custom
# larger buckets.
_EAGER_ROPE_SEQ_CHUNK = 8192


def _seq_slice(
    tensor: torch.Tensor,
    axis: int,
    start: int,
    length: int,
) -> torch.Tensor:
    if tensor.shape[axis] == 1:
        return tensor
    return tensor.narrow(axis, start, length)


def _rotate_in_place(
    tensor: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    seq_axis: int,
    rot_dim: int,
    chunk_size: int,
) -> None:
    half = rot_dim // 2
    seq_len = tensor.shape[seq_axis]
    for start in range(0, seq_len, chunk_size):
        length = min(chunk_size, seq_len - start)
        target = tensor.narrow(seq_axis, start, length)
        original = target[..., :rot_dim].clone()
        cos_chunk = _seq_slice(cos, seq_axis, start, length)
        sin_chunk = _seq_slice(sin, seq_axis, start, length)
        x1 = original[..., :half]
        x2 = original[..., half:rot_dim]
        target[..., :half].copy_(
            x1 * cos_chunk[..., :half] - x2 * sin_chunk[..., :half]
        )
        target[..., half:rot_dim].copy_(
            x2 * cos_chunk[..., half:rot_dim] + x1 * sin_chunk[..., half:rot_dim]
        )


def _rotate_grad(
    grad_out: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    seq_axis: int,
    rot_dim: int,
    chunk_size: int,
) -> torch.Tensor:
    half = rot_dim // 2
    grad_input = torch.empty_like(grad_out)
    seq_len = grad_out.shape[seq_axis]
    for start in range(0, seq_len, chunk_size):
        length = min(chunk_size, seq_len - start)
        source = grad_out.narrow(seq_axis, start, length)
        target = grad_input.narrow(seq_axis, start, length)
        cos_chunk = _seq_slice(cos, seq_axis, start, length)
        sin_chunk = _seq_slice(sin, seq_axis, start, length)
        g1 = source[..., :half]
        g2 = source[..., half:rot_dim]
        target[..., :half] = (
            g1 * cos_chunk[..., :half] + g2 * sin_chunk[..., half:rot_dim]
        )
        target[..., half:rot_dim] = (
            -g1 * sin_chunk[..., :half] + g2 * cos_chunk[..., half:rot_dim]
        )
        if rot_dim < source.shape[-1]:
            target[..., rot_dim:] = source[..., rot_dim:]
    return grad_input


class EagerRotaryQKFn(torch.autograd.Function):
    """Apply RoPE to q/k in place with chunked forward/backward temporaries.

    The q/k tensors are fresh RMSNorm outputs. Reusing their storage avoids the
    four full-width pointwise/cat temporaries created by the eager expression.
    RoPE is linear, so backward only needs the small precomputed cos/sin tables;
    neither original q nor k must be retained.
    """

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        seq_axis: int,
        rot_dim: int,
        chunk_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_axis = int(seq_axis)
        rot_dim = int(rot_dim)
        chunk_size = max(1, int(chunk_size))
        ctx.mark_dirty(q, k)
        _rotate_in_place(
            q,
            cos,
            sin,
            seq_axis=seq_axis,
            rot_dim=rot_dim,
            chunk_size=chunk_size,
        )
        _rotate_in_place(
            k,
            cos,
            sin,
            seq_axis=seq_axis,
            rot_dim=rot_dim,
            chunk_size=chunk_size,
        )
        ctx.save_for_backward(cos, sin)
        ctx.seq_axis = seq_axis
        ctx.rot_dim = rot_dim
        ctx.chunk_size = chunk_size
        return q, k

    @staticmethod
    def backward(ctx, grad_q: torch.Tensor, grad_k: torch.Tensor):
        cos, sin = ctx.saved_tensors
        kwargs = {
            "seq_axis": ctx.seq_axis,
            "rot_dim": ctx.rot_dim,
            "chunk_size": ctx.chunk_size,
        }
        input_grad_q = _rotate_grad(grad_q, cos, sin, **kwargs)
        input_grad_k = _rotate_grad(grad_k, cos, sin, **kwargs)
        return input_grad_q, input_grad_k, None, None, None, None, None


def eager_rotary_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    seq_axis: int,
    rot_dim: int,
    chunk_size: int = _EAGER_ROPE_SEQ_CHUNK,
) -> tuple[torch.Tensor, torch.Tensor]:
    return EagerRotaryQKFn.apply(
        q,
        k,
        cos,
        sin,
        seq_axis,
        rot_dim,
        chunk_size,
    )


# V100-tuned balance between eager launch overhead and rematerialization memory.
_EAGER_MLP_ROW_CHUNK = 3072


def _flatten_last(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(-1, tensor.shape[-1])


def _lora_linear_chunk(
    x: torch.Tensor,
    base_weight: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    inv_scale: torch.Tensor,
    rank_mask: torch.Tensor,
    residual_scale: float,
) -> torch.Tensor:
    """Evaluate one frozen Linear plus FP32 LoRA rank path for a small row chunk."""
    base = torch.nn.functional.linear(x.to(base_weight.dtype), base_weight)
    rank_x = x.float()
    if inv_scale.numel() != 0:
        rank_x = rank_x * inv_scale.float()
    rank = torch.nn.functional.linear(rank_x, down_weight.float())
    rank = rank * rank_mask.float()
    delta = torch.nn.functional.linear(rank, up_weight.float())
    if residual_scale != 1.0:
        delta.mul_(residual_scale)
    base.add_(delta.to(base.dtype))
    return base


class EagerFusedLoRAMLPFn(torch.autograd.Function):
    """Chunk and rematerialize a two-Linear GELU MLP adapted by plain LoRA.

    Eager autograd normally keeps both the full ``d_ff`` pre-activation needed
    by GELU backward and the equally large post-GELU tensor needed by the
    second LoRA down projection. At Anima's ~4k image-token buckets those two
    FP16 tensors are roughly 132 MiB per block, and eager GELU first needs a
    second full allocation before either can be released.

    The compiled path fuses/partitions this region. This Function gives the
    non-compiled V100 path the same bounded-memory shape: forward evaluates the
    complete MLP in row chunks and saves only its original ``d_model`` input;
    backward rematerializes each chunk and asks autograd for the exact local
    gradients. This is operator-local fusion/rematerialization, not block-level
    gradient checkpointing or block swapping.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        base_weight1: torch.Tensor,
        down_weight1: torch.Tensor,
        up_weight1: torch.Tensor,
        inv_scale1: torch.Tensor,
        rank_mask1: torch.Tensor,
        residual_scale1: float,
        base_weight2: torch.Tensor,
        down_weight2: torch.Tensor,
        up_weight2: torch.Tensor,
        inv_scale2: torch.Tensor,
        rank_mask2: torch.Tensor,
        residual_scale2: float,
        chunk_size: int,
    ) -> torch.Tensor:
        chunk = max(1, int(chunk_size))
        x_flat = _flatten_last(x)
        out = torch.empty(
            (*x.shape[:-1], base_weight2.shape[0]),
            dtype=base_weight2.dtype,
            device=x.device,
        )
        out_flat = _flatten_last(out)
        scale1 = float(residual_scale1)
        scale2 = float(residual_scale2)

        for start in range(0, x_flat.shape[0], chunk):
            end = min(start + chunk, x_flat.shape[0])
            pre_activation = _lora_linear_chunk(
                x_flat[start:end],
                base_weight1,
                down_weight1,
                up_weight1,
                inv_scale1,
                rank_mask1,
                scale1,
            )
            hidden = torch.nn.functional.gelu(pre_activation)
            out_flat[start:end] = _lora_linear_chunk(
                hidden,
                base_weight2,
                down_weight2,
                up_weight2,
                inv_scale2,
                rank_mask2,
                scale2,
            )

        # T-LoRA's shared mask is mutated before each forward. Keep the tiny
        # rank-sized value used by this invocation so gradient accumulation
        # cannot observe a later microbatch's mask.
        ctx.save_for_backward(
            x,
            base_weight1,
            down_weight1,
            up_weight1,
            inv_scale1,
            rank_mask1.detach().clone(),
            base_weight2,
            down_weight2,
            up_weight2,
            inv_scale2,
            rank_mask2.detach().clone(),
        )
        ctx.residual_scale1 = scale1
        ctx.residual_scale2 = scale2
        ctx.chunk_size = chunk
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_out: torch.Tensor):
        (
            x,
            base_weight1,
            down_weight1,
            up_weight1,
            inv_scale1,
            rank_mask1,
            base_weight2,
            down_weight2,
            up_weight2,
            inv_scale2,
            rank_mask2,
        ) = ctx.saved_tensors
        x_flat = _flatten_last(x)
        grad_flat = _flatten_last(grad_out)
        chunk = ctx.chunk_size

        grad_x_flat = torch.empty_like(x_flat) if ctx.needs_input_grad[0] else None
        trainable = (down_weight1, up_weight1, down_weight2, up_weight2)
        trainable_indices = (2, 3, 8, 9)
        grad_params = [
            torch.zeros_like(param) if ctx.needs_input_grad[index] else None
            for param, index in zip(trainable, trainable_indices)
        ]

        for start in range(0, x_flat.shape[0], chunk):
            end = min(start + chunk, x_flat.shape[0])
            with torch.enable_grad():
                x_chunk = x_flat[start:end].detach().requires_grad_(
                    ctx.needs_input_grad[0]
                )
                local_params = [
                    param.detach().requires_grad_(ctx.needs_input_grad[index])
                    for param, index in zip(trainable, trainable_indices)
                ]
                local_down1, local_up1, local_down2, local_up2 = local_params
                pre_activation = _lora_linear_chunk(
                    x_chunk,
                    base_weight1,
                    local_down1,
                    local_up1,
                    inv_scale1,
                    rank_mask1,
                    ctx.residual_scale1,
                )
                hidden = torch.nn.functional.gelu(pre_activation)
                local_out = _lora_linear_chunk(
                    hidden,
                    base_weight2,
                    local_down2,
                    local_up2,
                    inv_scale2,
                    rank_mask2,
                    ctx.residual_scale2,
                )

                requested = []
                requested_slots = []
                if ctx.needs_input_grad[0]:
                    requested.append(x_chunk)
                    requested_slots.append(("x", None))
                for param_index, (param, grad_param) in enumerate(
                    zip(local_params, grad_params)
                ):
                    if grad_param is not None:
                        requested.append(param)
                        requested_slots.append(("param", param_index))

                local_grads = torch.autograd.grad(
                    local_out,
                    requested,
                    grad_outputs=grad_flat[start:end],
                    allow_unused=False,
                )

            for slot, local_grad in zip(requested_slots, local_grads):
                kind, param_index = slot
                if kind == "x":
                    grad_x_flat[start:end] = local_grad.to(x.dtype)
                else:
                    grad_params[param_index].add_(
                        local_grad.to(grad_params[param_index].dtype)
                    )

        grad_x = grad_x_flat.reshape_as(x) if grad_x_flat is not None else None
        grad_down1, grad_up1, grad_down2, grad_up2 = grad_params
        return (
            grad_x,
            None,
            grad_down1,
            grad_up1,
            None,
            None,
            None,
            None,
            grad_down2,
            grad_up2,
            None,
            None,
            None,
            None,
        )


def eager_fused_lora_mlp_tensors(
    x: torch.Tensor,
    base_weight1: torch.Tensor,
    down_weight1: torch.Tensor,
    up_weight1: torch.Tensor,
    inv_scale1: torch.Tensor,
    rank_mask1: torch.Tensor,
    residual_scale1: float,
    base_weight2: torch.Tensor,
    down_weight2: torch.Tensor,
    up_weight2: torch.Tensor,
    inv_scale2: torch.Tensor,
    rank_mask2: torch.Tensor,
    residual_scale2: float,
    chunk_size: int = _EAGER_MLP_ROW_CHUNK,
) -> torch.Tensor:
    return EagerFusedLoRAMLPFn.apply(
        x,
        base_weight1,
        down_weight1,
        up_weight1,
        inv_scale1,
        rank_mask1,
        residual_scale1,
        base_weight2,
        down_weight2,
        up_weight2,
        inv_scale2,
        rank_mask2,
        residual_scale2,
        chunk_size,
    )


def _lokr_linear_chunk(
    x: torch.Tensor,
    base_weight: torch.Tensor,
    w1: torch.Tensor | None,
    w1a: torch.Tensor | None,
    w1b: torch.Tensor | None,
    w2: torch.Tensor | None,
    w2a: torch.Tensor | None,
    w2b: torch.Tensor | None,
    scalar: torch.Tensor,
    residual_scale: float,
) -> torch.Tensor:
    """Evaluate one frozen Linear plus FP32 LoKr bypass for a row chunk."""
    base = torch.nn.functional.linear(x.to(base_weight.dtype), base_weight)
    c = w1.float() if w1 is not None else w1a.float().matmul(w1b.float())
    grouped = x.float().reshape(x.shape[0], c.shape[1], -1)
    if w2 is not None:
        projected = torch.nn.functional.linear(grouped, w2.float())
    else:
        projected = torch.nn.functional.linear(
            torch.nn.functional.linear(grouped, w2b.float()),
            w2a.float(),
        )
    crossed = projected.transpose(-1, -2)
    delta = torch.nn.functional.linear(crossed, c).transpose(-1, -2)
    delta = delta.reshape(x.shape[0], -1)
    if residual_scale != 1.0:
        delta.mul_(residual_scale)
    delta.mul_(scalar.float())
    base.add_(delta.to(base.dtype))
    return base


class EagerFusedLoKrMLPFn(torch.autograd.Function):
    """Chunk and rematerialize a two-Linear GELU MLP adapted by LoKr."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        base_weight1: torch.Tensor,
        w1_1: torch.Tensor | None,
        w1a_1: torch.Tensor | None,
        w1b_1: torch.Tensor | None,
        w2_1: torch.Tensor | None,
        w2a_1: torch.Tensor | None,
        w2b_1: torch.Tensor | None,
        scalar1: torch.Tensor,
        residual_scale1: float,
        base_weight2: torch.Tensor,
        w1_2: torch.Tensor | None,
        w1a_2: torch.Tensor | None,
        w1b_2: torch.Tensor | None,
        w2_2: torch.Tensor | None,
        w2a_2: torch.Tensor | None,
        w2b_2: torch.Tensor | None,
        scalar2: torch.Tensor,
        residual_scale2: float,
        chunk_size: int,
    ) -> torch.Tensor:
        chunk = max(1, int(chunk_size))
        x_flat = _flatten_last(x)
        out = torch.empty(
            (*x.shape[:-1], base_weight2.shape[0]),
            dtype=base_weight2.dtype,
            device=x.device,
        )
        out_flat = _flatten_last(out)
        factors = (
            w1_1,
            w1a_1,
            w1b_1,
            w2_1,
            w2a_1,
            w2b_1,
            w1_2,
            w1a_2,
            w1b_2,
            w2_2,
            w2a_2,
            w2b_2,
        )
        scale1 = float(residual_scale1)
        scale2 = float(residual_scale2)

        for start in range(0, x_flat.shape[0], chunk):
            end = min(start + chunk, x_flat.shape[0])
            pre_activation = _lokr_linear_chunk(
                x_flat[start:end],
                base_weight1,
                *factors[:6],
                scalar1,
                scale1,
            )
            hidden = torch.nn.functional.gelu(pre_activation)
            out_flat[start:end] = _lokr_linear_chunk(
                hidden,
                base_weight2,
                *factors[6:],
                scalar2,
                scale2,
            )

        ctx.save_for_backward(
            x,
            base_weight1,
            scalar1,
            base_weight2,
            scalar2,
            *(factor for factor in factors if factor is not None),
        )
        ctx.factor_present = tuple(factor is not None for factor in factors)
        ctx.residual_scale1 = scale1
        ctx.residual_scale2 = scale2
        ctx.chunk_size = chunk
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_out: torch.Tensor):
        (
            x,
            base_weight1,
            scalar1,
            base_weight2,
            scalar2,
            *saved_factors,
        ) = ctx.saved_tensors
        factor_iter = iter(saved_factors)
        factors = tuple(
            next(factor_iter) if present else None
            for present in ctx.factor_present
        )
        factor_input_indices = (*range(2, 8), *range(11, 17))
        scalar_input_indices = (8, 17)
        x_flat = _flatten_last(x)
        grad_flat = _flatten_last(grad_out)
        grad_x_flat = torch.empty_like(x_flat) if ctx.needs_input_grad[0] else None
        grad_factors_fp32 = [
            (
                torch.zeros_like(factor, dtype=torch.float32)
                if factor is not None and ctx.needs_input_grad[input_index]
                else None
            )
            for factor, input_index in zip(factors, factor_input_indices)
        ]
        scalars = (scalar1, scalar2)
        grad_scalars_fp32 = [
            (
                torch.zeros_like(scalar, dtype=torch.float32)
                if ctx.needs_input_grad[input_index]
                else None
            )
            for scalar, input_index in zip(scalars, scalar_input_indices)
        ]

        for start in range(0, x_flat.shape[0], ctx.chunk_size):
            end = min(start + ctx.chunk_size, x_flat.shape[0])
            with torch.enable_grad():
                local_x = x_flat[start:end].detach().requires_grad_(
                    ctx.needs_input_grad[0]
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
                local_scalars = tuple(
                    scalar.detach()
                    .float()
                    .requires_grad_(grad_scalar is not None)
                    for scalar, grad_scalar in zip(scalars, grad_scalars_fp32)
                )
                pre_activation = _lokr_linear_chunk(
                    local_x,
                    base_weight1,
                    *local_factors[:6],
                    local_scalars[0],
                    ctx.residual_scale1,
                )
                hidden = torch.nn.functional.gelu(pre_activation)
                local_out = _lokr_linear_chunk(
                    hidden,
                    base_weight2,
                    *local_factors[6:],
                    local_scalars[1],
                    ctx.residual_scale2,
                )

                requested = []
                requested_slots = []
                if ctx.needs_input_grad[0]:
                    requested.append(local_x)
                    requested_slots.append(("x", -1))
                for index, (factor, grad_factor) in enumerate(
                    zip(local_factors, grad_factors_fp32)
                ):
                    if factor is not None and grad_factor is not None:
                        requested.append(factor)
                        requested_slots.append(("factor", index))
                for index, (scalar, grad_scalar) in enumerate(
                    zip(local_scalars, grad_scalars_fp32)
                ):
                    if grad_scalar is not None:
                        requested.append(scalar)
                        requested_slots.append(("scalar", index))

                local_grads = torch.autograd.grad(
                    local_out,
                    requested,
                    grad_outputs=grad_flat[start:end],
                    allow_unused=False,
                )

            for (kind, index), local_grad in zip(requested_slots, local_grads):
                if kind == "x":
                    grad_x_flat[start:end] = local_grad.to(x.dtype)
                elif kind == "factor":
                    grad_factors_fp32[index].add_(local_grad)
                else:
                    grad_scalars_fp32[index].add_(local_grad)

        grad_x = grad_x_flat.reshape_as(x) if grad_x_flat is not None else None
        grad_factors = tuple(
            (
                grad_factor.to(factor.dtype)
                if factor is not None and grad_factor is not None
                else None
            )
            for factor, grad_factor in zip(factors, grad_factors_fp32)
        )
        grad_scalars = tuple(
            (
                grad_scalar.to(scalar.dtype)
                if grad_scalar is not None
                else None
            )
            for scalar, grad_scalar in zip(scalars, grad_scalars_fp32)
        )
        return (
            grad_x,
            None,
            *grad_factors[:6],
            grad_scalars[0],
            None,
            None,
            *grad_factors[6:],
            grad_scalars[1],
            None,
            None,
        )


def eager_fused_lokr_mlp_tensors(
    x: torch.Tensor,
    base_weight1: torch.Tensor,
    w1_1: torch.Tensor | None,
    w1a_1: torch.Tensor | None,
    w1b_1: torch.Tensor | None,
    w2_1: torch.Tensor | None,
    w2a_1: torch.Tensor | None,
    w2b_1: torch.Tensor | None,
    scalar1: torch.Tensor,
    residual_scale1: float,
    base_weight2: torch.Tensor,
    w1_2: torch.Tensor | None,
    w1a_2: torch.Tensor | None,
    w1b_2: torch.Tensor | None,
    w2_2: torch.Tensor | None,
    w2a_2: torch.Tensor | None,
    w2b_2: torch.Tensor | None,
    scalar2: torch.Tensor,
    residual_scale2: float,
    chunk_size: int = 1024,
) -> torch.Tensor:
    return EagerFusedLoKrMLPFn.apply(
        x,
        base_weight1,
        w1_1,
        w1a_1,
        w1b_1,
        w2_1,
        w2a_1,
        w2b_1,
        scalar1,
        residual_scale1,
        base_weight2,
        w1_2,
        w1a_2,
        w1b_2,
        w2_2,
        w2a_2,
        w2b_2,
        scalar2,
        residual_scale2,
        chunk_size,
    )


def _plain_lora_owner(linear: torch.nn.Linear):
    owner = getattr(linear.forward, "__self__", None)
    required = (
        "lora_down",
        "lora_up",
        "org_module_ref",
        "_timestep_mask",
        "use_custom_down_autograd",
        "fp32_compute",
    )
    if owner is None or not all(hasattr(owner, name) for name in required):
        return None
    if type(owner).__name__ != "LoRAModule":
        return None
    if owner.org_module_ref[0] is not linear:
        return None
    if not owner.training or not owner.enabled or getattr(owner, "_fused", False):
        return None
    if not owner.use_custom_down_autograd or not owner.fp32_compute:
        return None
    if any(
        value is not None
        for value in (owner.dropout, owner.rank_dropout, owner.module_dropout)
    ):
        return None
    if not isinstance(owner.lora_down, torch.nn.Linear) or not isinstance(
        owner.lora_up, torch.nn.Linear
    ):
        return None
    return owner


def _lokr_owner(linear: torch.nn.Linear):
    owner = getattr(linear.forward, "__self__", None)
    required = (
        "use_w1",
        "use_w2",
        "scalar",
        "org_module_ref",
        "use_custom_down_autograd",
        "fp32_compute",
    )
    if owner is None or not all(hasattr(owner, name) for name in required):
        return None
    if type(owner).__name__ != "LoKRModule":
        return None
    if owner.org_module_ref[0] is not linear:
        return None
    if not owner.training or not owner.enabled or getattr(owner, "_fused", False):
        return None
    if not owner.use_custom_down_autograd or not owner.fp32_compute:
        return None
    if any(
        bool(value)
        for value in (owner.dropout, owner.rank_dropout, owner.module_dropout)
    ):
        return None
    return owner


def eager_fused_lora_mlp(
    x: torch.Tensor,
    layer1: torch.nn.Linear,
    layer2: torch.nn.Linear,
) -> torch.Tensor | None:
    """Return the fused V100 eager MLP result, or ``None`` when inapplicable."""
    if (
        torch.compiler.is_compiling()
        or not torch.is_grad_enabled()
        or not x.is_cuda
        or layer1.weight.dtype != torch.float16
        or layer2.weight.dtype != torch.float16
        or torch.cuda.get_device_capability(x.device) != (7, 0)
    ):
        return None

    lora1 = _plain_lora_owner(layer1)
    lora2 = _plain_lora_owner(layer2)
    if lora1 is None or lora2 is None:
        lokr1 = _lokr_owner(layer1)
        lokr2 = _lokr_owner(layer2)
        if lokr1 is None or lokr2 is None:
            return None
        return eager_fused_lokr_mlp_tensors(
            x,
            layer1.weight,
            lokr1.lokr_w1 if lokr1.use_w1 else None,
            None if lokr1.use_w1 else lokr1.lokr_w1_a,
            None if lokr1.use_w1 else lokr1.lokr_w1_b,
            lokr1.lokr_w2 if lokr1.use_w2 else None,
            None if lokr1.use_w2 else lokr1.lokr_w2_a,
            None if lokr1.use_w2 else lokr1.lokr_w2_b,
            lokr1.scalar,
            lokr1.multiplier * lokr1.scale,
            layer2.weight,
            lokr2.lokr_w1 if lokr2.use_w1 else None,
            None if lokr2.use_w1 else lokr2.lokr_w1_a,
            None if lokr2.use_w1 else lokr2.lokr_w1_b,
            lokr2.lokr_w2 if lokr2.use_w2 else None,
            None if lokr2.use_w2 else lokr2.lokr_w2_a,
            None if lokr2.use_w2 else lokr2.lokr_w2_b,
            lokr2.scalar,
            lokr2.multiplier * lokr2.scale,
        )

    inv1 = (
        lora1.inv_scale
        if lora1._has_channel_scale
        else lora1._timestep_mask.reshape(-1)[:0]
    )
    inv2 = (
        lora2.inv_scale
        if lora2._has_channel_scale
        else lora2._timestep_mask.reshape(-1)[:0]
    )
    return eager_fused_lora_mlp_tensors(
        x,
        layer1.weight,
        lora1.lora_down.weight,
        lora1.lora_up.weight,
        inv1,
        lora1._timestep_mask,
        lora1.multiplier * lora1.scale,
        layer2.weight,
        lora2.lora_down.weight,
        lora2.lora_up.weight,
        inv2,
        lora2._timestep_mask,
        lora2.multiplier * lora2.scale,
    )
