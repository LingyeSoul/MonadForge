"""Dynamo-compatible wrappers for the Volta FlashAttention extension.

The V100 fork calls raw pybind CUDA functions from a custom autograd Function.
Dynamo graph-breaks on those calls, which also strips MonadForge's dynamic
sequence marks. Opaque custom ops keep the CUDA kernels in the graph while fake
implementations describe their output shapes to Dynamo and AOTAutograd.

The wrapper signatures and tensor layout follow ``flash-attention-v100``'s
``flash_attn_v100/flash_attn_interface.py``. That project is BSD-3-Clause,
Copyright (c) 2025 D. Skryabin and its contributors. No kernel source is copied
or modified here.
"""

from __future__ import annotations

import warnings

import flash_attn_v100_cuda as _v100_cuda
import torch
import torch.nn.functional as F

_cuda_dense_fwd = _v100_cuda.fwd
_cuda_dense_bwd = _v100_cuda.bwd
_cuda_varlen_fwd = _v100_cuda.varlen_fwd
_cuda_varlen_bwd = _v100_cuda.varlen_bwd


@torch.library.custom_op(
    "monadforge::v100_flash_dense_fwd",
    mutates_args=(),
    tags=(torch.Tag.nondeterministic_seeded,),
)
def _dense_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alibi_slopes: torch.Tensor | None,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_left: int,
    window_right: int,
    softcap: float,
    return_softmax: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _cuda_dense_fwd(
        q,
        k,
        v,
        None,
        alibi_slopes,
        dropout_p,
        softmax_scale,
        causal,
        window_left,
        window_right,
        softcap,
        return_softmax,
        None,
    )


@_dense_fwd.register_fake
def _dense_fwd_fake(
    q,
    k,
    v,
    alibi_slopes,
    dropout_p,
    softmax_scale,
    causal,
    window_left,
    window_right,
    softcap,
    return_softmax,
):
    batch, heads, query_len, _ = q.shape
    softmax = (
        q.new_empty((batch, heads, query_len, k.shape[2]))
        if return_softmax
        else q.new_empty((0,))
    )
    return (
        torch.empty_like(q),
        q.new_empty((batch, heads, query_len), dtype=torch.float32),
        softmax,
        q.new_empty((2,), dtype=torch.int64),
    )


@torch.library.custom_op(
    "monadforge::v100_flash_dense_bwd",
    mutates_args=(),
    tags=(torch.Tag.nondeterministic_bitwise,),
)
def _dense_bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    alibi_slopes: torch.Tensor | None,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_left: int,
    window_right: int,
    softcap: float,
    deterministic: bool,
    rng_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    grads = _cuda_dense_bwd(
        dout,
        q,
        k,
        v,
        out,
        lse,
        dq,
        dk,
        dv,
        alibi_slopes,
        dropout_p,
        softmax_scale,
        causal,
        window_left,
        window_right,
        softcap,
        deterministic,
        None,
        rng_state,
    )
    return grads[0], grads[1], grads[2]


@_dense_bwd.register_fake
def _dense_bwd_fake(
    dout,
    q,
    k,
    v,
    out,
    lse,
    alibi_slopes,
    dropout_p,
    softmax_scale,
    causal,
    window_left,
    window_right,
    softcap,
    deterministic,
    rng_state,
):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


@torch.library.custom_op(
    "monadforge::v100_flash_varlen_fwd",
    mutates_args=(),
    tags=(torch.Tag.nondeterministic_seeded,),
)
def _varlen_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    block_table: torch.Tensor | None,
    alibi_slopes: torch.Tensor | None,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_left: int,
    window_right: int,
    softcap: float,
    return_softmax: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _cuda_varlen_fwd(
        q,
        k,
        v,
        None,
        cu_seqlens_q,
        cu_seqlens_k,
        None,
        None,
        block_table,
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        False,
        causal,
        window_left,
        window_right,
        softcap,
        return_softmax,
        None,
        0,
    )


@_varlen_fwd.register_fake
def _varlen_fwd_fake(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    block_table,
    alibi_slopes,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p,
    softmax_scale,
    causal,
    window_left,
    window_right,
    softcap,
    return_softmax,
):
    softmax = (
        q.new_empty((q.shape[0], q.shape[1], max_seqlen_k))
        if return_softmax
        else q.new_empty((0,))
    )
    return (
        torch.empty_like(q),
        q.new_empty((q.shape[1], q.shape[0]), dtype=torch.float32),
        softmax,
        q.new_empty((2,), dtype=torch.int64),
    )


@torch.library.custom_op(
    "monadforge::v100_flash_varlen_bwd",
    mutates_args=(),
    tags=(torch.Tag.nondeterministic_bitwise,),
)
def _varlen_bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    alibi_slopes: torch.Tensor | None,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_left: int,
    window_right: int,
    softcap: float,
    deterministic: bool,
    rng_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    grads = _cuda_varlen_bwd(
        dout,
        q,
        k,
        v,
        out,
        lse,
        dq,
        dk,
        dv,
        cu_seqlens_q,
        cu_seqlens_k,
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        False,
        causal,
        window_left,
        window_right,
        softcap,
        deterministic,
        None,
        rng_state,
    )
    return grads[0], grads[1], grads[2]


@_varlen_bwd.register_fake
def _varlen_bwd_fake(
    dout,
    q,
    k,
    v,
    out,
    lse,
    cu_seqlens_q,
    cu_seqlens_k,
    alibi_slopes,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p,
    softmax_scale,
    causal,
    window_left,
    window_right,
    softcap,
    deterministic,
    rng_state,
):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


class _V100FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
        is_grad_enabled,
    ):
        is_grad = is_grad_enabled and any(x.requires_grad for x in (q, k, v))
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        head_size = q.shape[-1]
        pad_size = (-head_size) % 8
        if pad_size:
            q = F.pad(q, (0, pad_size))
            k = F.pad(k, (0, pad_size))
            v = F.pad(v, (0, pad_size))
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

        scale = head_size**-0.5 if softmax_scale is None else softmax_scale
        window_left, window_right = window_size
        out, lse, softmax, rng_state = _dense_fwd(
            q,
            k,
            v,
            alibi_slopes,
            dropout_p,
            scale,
            causal,
            window_left,
            window_right,
            softcap,
            return_softmax and dropout_p > 0.0,
        )
        result = out[..., :head_size].permute(0, 2, 1, 3).contiguous()

        if is_grad:
            ctx.save_for_backward(q, k, v, out, lse, rng_state)
            ctx.dropout_p = dropout_p
            ctx.softmax_scale = scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.softcap = softcap
            ctx.alibi_slopes = alibi_slopes
            ctx.deterministic = deterministic
            ctx.head_size = head_size
            ctx.pad_size = pad_size

        return (result, lse, softmax) if return_softmax else result

    @staticmethod
    def backward(ctx, dout, *unused_grads):
        q, k, v, out, lse, rng_state = ctx.saved_tensors
        if ctx.pad_size:
            dout = F.pad(dout, (0, ctx.pad_size))
        dout = dout.permute(0, 2, 1, 3).contiguous()
        dq, dk, dv = _dense_bwd(
            dout,
            q,
            k,
            v,
            out,
            lse,
            ctx.alibi_slopes,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
            ctx.softcap,
            ctx.deterministic,
            rng_state,
        )
        return (
            dq[..., : ctx.head_size].permute(0, 2, 1, 3).contiguous(),
            dk[..., : ctx.head_size].permute(0, 2, 1, 3).contiguous(),
            dv[..., : ctx.head_size].permute(0, 2, 1, 3).contiguous(),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _V100FlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        block_table,
        is_grad_enabled,
    ):
        is_grad = is_grad_enabled and any(x.requires_grad for x in (q, k, v))
        cu_seqlens_q = cu_seqlens_q.to(torch.int32)
        cu_seqlens_k = cu_seqlens_k.to(torch.int32)

        head_size = q.shape[-1]
        pad_size = (-head_size) % 8
        if pad_size:
            q = F.pad(q, (0, pad_size))
            k = F.pad(k, (0, pad_size))
            v = F.pad(v, (0, pad_size))
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

        scale = head_size**-0.5 if softmax_scale is None else softmax_scale
        window_left, window_right = window_size
        out, lse, softmax, rng_state = _varlen_fwd(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            block_table,
            alibi_slopes,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            scale,
            causal,
            window_left,
            window_right,
            softcap,
            return_attn_probs and dropout_p > 0.0,
        )
        result = out[..., :head_size].contiguous()

        if is_grad:
            ctx.save_for_backward(
                q,
                k,
                v,
                out,
                lse,
                cu_seqlens_q,
                cu_seqlens_k,
                rng_state,
            )
            ctx.max_seqlen_q = max_seqlen_q
            ctx.max_seqlen_k = max_seqlen_k
            ctx.dropout_p = dropout_p
            ctx.softmax_scale = scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.softcap = softcap
            ctx.alibi_slopes = alibi_slopes
            ctx.deterministic = deterministic
            ctx.head_size = head_size
            ctx.pad_size = pad_size

        return (result, lse, softmax) if return_attn_probs else result

    @staticmethod
    def backward(ctx, dout, *unused_grads):
        q, k, v, out, lse, cu_q, cu_k, rng_state = ctx.saved_tensors
        if ctx.pad_size:
            dout = F.pad(dout, (0, ctx.pad_size))
        dout = dout.contiguous()
        dq, dk, dv = _varlen_bwd(
            dout,
            q,
            k,
            v,
            out,
            lse,
            cu_q,
            cu_k,
            ctx.alibi_slopes,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
            ctx.softcap,
            ctx.deterministic,
            rng_state,
        )
        return (
            dq[..., : ctx.head_size].contiguous(),
            dk[..., : ctx.head_size].contiguous(),
            dv[..., : ctx.head_size].contiguous(),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: torch.Tensor | None = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
):
    if deterministic:
        warnings.warn(
            "Forward is deterministic; deterministic backward is unsupported.",
            RuntimeWarning,
            stacklevel=2,
        )
        deterministic = False
    return _V100FlashAttnFunc.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        torch.is_grad_enabled(),
    )


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: torch.Tensor | None = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    block_table: torch.Tensor | None = None,
):
    if deterministic:
        warnings.warn(
            "Forward is deterministic; deterministic backward is unsupported.",
            RuntimeWarning,
            stacklevel=2,
        )
        deterministic = False
    return _V100FlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        block_table,
        torch.is_grad_enabled(),
    )


__all__ = ["flash_attn_func", "flash_attn_varlen_func"]
