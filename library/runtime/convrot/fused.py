"""Fused ConvRot forward paths: RHT + quant/GEMM in one autograd Function.

Fusion goals (training-safe, no Triton dependency):
1. Single ``autograd.Function`` for W8A16 / W8A8 so RHT → matmul (→ act quant)
   share one backward without Python-level intermediate graphs.
2. RHT backend selectable: dense (default, fastest on consumer GPUs) or FWHT
   via ``ANIMA_CONVROT_RHT=fwht``.
3. W8A16 weight path: dequant+``F.linear`` (default) or ``_weight_int8pack_mm``
   via ``ANIMA_CONVROT_W8A16_KERNEL=int8pack`` (often *slower* on 3080-class).
4. W8A8: RHT → absmax quant → ``int8_mm_scaled`` (true ``_int_mm`` or float).
"""

from __future__ import annotations

import os
from typing import Literal

import torch
from torch.nn import functional as F

from library.runtime.convrot.gemm import int8_mm_scaled, quantize_activation_absmax_int8
from library.runtime.convrot.quant import dequantize_weight, get_dequant_scratch
from library.runtime.convrot.rht import (
    assert_group_divides,
    group_fwht,
    group_rht,
    rht_backend,
)

_FUSED_ENV = "ANIMA_CONVROT_FUSED"
_W8A16_KERNEL_ENV = "ANIMA_CONVROT_W8A16_KERNEL"


def fused_enabled() -> bool:
    raw = str(os.environ.get(_FUSED_ENV, "1") or "1").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def w8a16_kernel_backend() -> Literal["auto", "int8pack", "dequant"]:
    raw = str(os.environ.get(_W8A16_KERNEL_ENV, "auto") or "auto").strip().lower()
    if raw in {"int8pack", "pack", "weight_int8pack"}:
        return "int8pack"
    if raw in {"dequant", "float", "linear", "f.linear"}:
        return "dequant"
    return "auto"


def _rotate_acts(
    x: torch.Tensor,
    group_size: int,
    *,
    hadamard: torch.Tensor | None = None,
) -> torch.Tensor:
    """RHT on activations using configured backend (dense default).

    When ``hadamard`` is provided, always use dense matmul with that matrix
    (compile-friendly: no env lookup / cache miss on the hot path).
    """
    with torch.profiler.record_function("convrot::rht"):
        if hadamard is not None:
            return group_rht(x, group_size, hadamard=hadamard)
        if rht_backend() == "fwht":
            return group_fwht(x, group_size)
        return group_rht(x, group_size)


def can_use_weight_int8pack_mm(
    *,
    device: torch.device,
    x_dtype: torch.dtype,
    m: int,
    k: int,
    n: int,
) -> bool:
    if device.type != "cuda":
        return False
    if not hasattr(torch, "_weight_int8pack_mm"):
        return False
    if x_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return False
    if m <= 0 or k < 16 or n < 16:
        return False
    return True


def _resolve_compute_dtype(x: torch.Tensor) -> torch.dtype:
    """Prefer act dtype for TC; CUDA non-half inputs fall back to bf16."""
    if x.is_floating_point() and x.dtype in (torch.float16, torch.bfloat16):
        return x.dtype
    if x.device.type == "cuda":
        return torch.bfloat16
    return torch.float32


def _w8a16_linear_core(
    x_rot: torch.Tensor,
    w_q: torch.Tensor,
    w_scale: torch.Tensor,
) -> torch.Tensor:
    """Return y in ``x_rot``'s floating dtype (no forced float32 materialize)."""
    backend = w8a16_kernel_backend()
    *leading, k = x_rot.shape
    flat = x_rot.reshape(-1, k)
    m = flat.shape[0]
    n = int(w_q.shape[0])
    # Default/auto: dequant — measured much faster than int8pack on RTX 3080.
    use_pack = backend == "int8pack" or (
        backend == "auto"
        and False  # keep int8pack opt-in only until profiled faster on target GPU
        and can_use_weight_int8pack_mm(
            device=flat.device,
            x_dtype=flat.dtype,
            m=m,
            k=k,
            n=n,
        )
    )
    # Prefer the activation dtype for dequant linear so Ampere can stay on
    # bf16/fp16 Tensor Cores. Forcing fp32 here was the main W8A16 tax on 3080.
    compute_dtype = flat.dtype if flat.is_floating_point() else torch.float32
    if compute_dtype == torch.float32 and flat.device.type == "cuda":
        # Host often promotes to fp32; fall back to bf16 TC when available.
        compute_dtype = torch.bfloat16

    if use_pack:
        try:
            x_in = flat.to(dtype=compute_dtype)
            with torch.profiler.record_function("convrot::gemm_int8pack"):
                y = torch._weight_int8pack_mm(
                    x_in.contiguous(),
                    w_q.contiguous(),
                    w_scale.to(device=flat.device, dtype=torch.float32).contiguous(),
                )
            # Pack kernel returns float; cast to compute_dtype for a single exit type.
            return y.to(dtype=compute_dtype).reshape(*leading, n)
        except RuntimeError:
            if backend == "int8pack":
                raise
    with torch.profiler.record_function("convrot::dequant"):
        # P1.8: reuse a process-level [>=N,>=K] scratch across sequential layers.
        scratch = get_dequant_scratch(
            n, k, device=flat.device, dtype=compute_dtype
        )
        weight = dequantize_weight(
            w_q, w_scale, dtype=compute_dtype, out=scratch
        )
        if weight.device != flat.device:
            weight = weight.to(device=flat.device)
    with torch.profiler.record_function("convrot::gemm_dequant_linear"):
        y = F.linear(flat.to(dtype=compute_dtype), weight, None)
    return y.reshape(*leading, n)


class _FusedW8A16Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_q, w_scale, group_size: int, hadamard: torch.Tensor | None):
        # RHT + dequant GEMM stay in act TC dtype (P1.5). Optional precomputed
        # ``hadamard`` avoids env/cache lookup (P1-H); held on ctx (not saved
        # for backward) because it is a constant buffer and not differentiated.
        # w_q / w_scale are likewise frozen buffers — hold on ctx instead of
        # save_for_backward to skip autograd packing (P1.12).
        compute_dtype = _resolve_compute_dtype(x)
        x_work = x.to(dtype=compute_dtype) if x.dtype != compute_dtype else x
        x_rot = _rotate_acts(x_work, int(group_size), hadamard=hadamard)
        y = _w8a16_linear_core(x_rot, w_q, w_scale)
        # Prefer returning act dtype so LoRA ``org + delta`` stays half under
        # torch.compile (avoids float32 QKV into FlashAttention). When input is
        # already non-half (rare host path), keep compute_dtype (bf16 on CUDA).
        out_dtype = (
            x.dtype
            if x.is_floating_point() and x.dtype in (torch.float16, torch.bfloat16)
            else compute_dtype
        )
        if y.dtype != out_dtype:
            y = y.to(dtype=out_dtype)
        ctx.group_size = int(group_size)
        ctx.hadamard = hadamard
        ctx.w_q = w_q
        ctx.w_scale = w_scale
        ctx.x_dtype = x.dtype
        ctx.compute_dtype = compute_dtype
        return y

    @staticmethod
    def backward(ctx, grad_y):
        w_q = ctx.w_q
        w_scale = ctx.w_scale
        hadamard = getattr(ctx, "hadamard", None)
        dtype = getattr(ctx, "compute_dtype", torch.float32)
        # grad_rot = (grad_y * scale) @ W_q  — avoids full [N,K] dequant temp.
        # W[n,k] = W_q[n,k] * scale[n]  ⇒  gy @ W = (gy * scale) @ W_q
        gy = grad_y.to(dtype=dtype)
        scale = w_scale.to(device=gy.device, dtype=dtype)
        if scale.dim() == 1:
            gy = gy * scale  # broadcast over last dim N
        else:
            gy = gy * scale.reshape(1, -1)
        w = w_q.to(dtype=dtype)
        if w.device != gy.device:
            w = w.to(device=gy.device)
        grad_rot = gy @ w
        # RHT is involutory; stay in compute_dtype (no fp32 hop).
        grad_x = _rotate_acts(grad_rot, ctx.group_size, hadamard=hadamard)
        if ctx.x_dtype != grad_x.dtype:
            grad_x = grad_x.to(dtype=ctx.x_dtype)
        return grad_x, None, None, None, None


class _FusedW8A8Fn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        w_q,
        w_scale,
        group_size: int,
        hadamard: torch.Tensor | None,
        weight_layout: str,
    ):
        # RHT in act TC dtype; absmax quant still promotes to fp32 internally.
        # ``weight_layout``: "nk" classic [N,K]; "kn" pre-transposed for _int_mm.
        compute_dtype = _resolve_compute_dtype(x)
        x_work = x.to(dtype=compute_dtype) if x.dtype != compute_dtype else x
        x_rot = _rotate_acts(x_work, int(group_size), hadamard=hadamard)
        layout = "kn" if weight_layout == "kn" else "nk"
        with torch.profiler.record_function("convrot::act_quant"):
            x_q, x_scale = quantize_activation_absmax_int8(x_rot)
        with torch.profiler.record_function("convrot::gemm_int8"):
            # Emit y in compute_dtype to skip a full float32→bf16 cast (P1.9).
            y = int8_mm_scaled(
                x_q,
                x_scale,
                w_q,
                w_scale,
                weight_layout=layout,
                out_dtype=compute_dtype,
            )
        out_dtype = (
            x.dtype
            if x.is_floating_point() and x.dtype in (torch.float16, torch.bfloat16)
            else compute_dtype
        )
        if y.dtype != out_dtype:
            y = y.to(dtype=out_dtype)
        ctx.group_size = int(group_size)
        ctx.hadamard = hadamard
        ctx.compute_dtype = compute_dtype
        ctx.weight_layout = layout
        # Frozen weight buffers — ctx attrs, not save_for_backward (P1.12).
        ctx.w_q = w_q
        ctx.w_scale = w_scale
        ctx.x_dtype = x.dtype
        return y

    @staticmethod
    def backward(ctx, grad_y):
        w_q = ctx.w_q
        w_scale = ctx.w_scale
        hadamard = getattr(ctx, "hadamard", None)
        layout = getattr(ctx, "weight_layout", "nk")
        # STE accumulate stays fp32 (bf16 STE failed full-ckpt grad gate).
        # kn: grad_rot = (gy * scale) @ w_kn.T
        # nk: grad_rot = (gy * scale) @ w_nk
        # Opt-in TF32: ANIMA_CONVROT_STE_TF32=1 (see gemm.ste_tf32_enabled).
        from library.runtime.convrot.gemm import ste_tf32_enabled

        gy = grad_y.to(torch.float32)
        scale = w_scale.to(device=gy.device, dtype=torch.float32)
        if scale.dim() == 1:
            gy = gy * scale
        else:
            gy = gy * scale.reshape(1, -1)
        w = w_q.to(torch.float32)
        if w.device != gy.device:
            w = w.to(device=gy.device)
        use_tf32 = ste_tf32_enabled() and gy.is_cuda
        prev = torch.get_float32_matmul_precision() if use_tf32 else None
        try:
            if use_tf32:
                torch.set_float32_matmul_precision("high")
            if layout == "kn":
                # w is [K,N]; need [N,K] as right factor → w.T
                grad_rot = gy @ w.transpose(0, 1)
            else:
                grad_rot = gy @ w
        finally:
            if use_tf32 and prev is not None:
                torch.set_float32_matmul_precision(prev)
        compute_dtype = getattr(ctx, "compute_dtype", torch.float32)
        if compute_dtype in (torch.float16, torch.bfloat16):
            grad_rot_c = grad_rot.to(dtype=compute_dtype)
            grad_x = _rotate_acts(grad_rot_c, ctx.group_size, hadamard=hadamard)
        else:
            grad_x = _rotate_acts(grad_rot, ctx.group_size, hadamard=hadamard)
        if ctx.x_dtype != grad_x.dtype:
            grad_x = grad_x.to(dtype=ctx.x_dtype)
        return grad_x, None, None, None, None, None


def fused_w8a16_forward(
    x: torch.Tensor,
    quantized_weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    group_size: int,
    hadamard: torch.Tensor | None = None,
) -> torch.Tensor:
    assert_group_divides(int(quantized_weight.shape[1]), group_size)
    # autograd.Function.apply does not accept keyword args for tensor inputs.
    # Keep scale dtype as stored (often bf16 from apply); only move device.
    scale_dev = scale.to(device=x.device) if scale.device != x.device else scale
    y = _FusedW8A16Fn.apply(
        x,
        quantized_weight.to(device=x.device),
        scale_dev,
        int(group_size),
        hadamard.to(device=x.device) if hadamard is not None else None,
    )
    if x.is_floating_point() and y.dtype != x.dtype:
        return y.to(dtype=x.dtype)
    return y


def fused_w8a8_forward(
    x: torch.Tensor,
    quantized_weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    group_size: int,
    hadamard: torch.Tensor | None = None,
    weight_layout: str = "nk",
) -> torch.Tensor:
    """W8A8 fused path.

    ``weight_layout``:
      * ``nk`` — classic Linear ``[out, in]`` (default, tests / modules)
      * ``kn`` — pre-transposed ``[in, out]`` stored by apply (P1.6 hot path)
    """
    layout = "kn" if weight_layout == "kn" else "nk"
    if layout == "kn":
        # [K, N] → K is dim 0
        assert_group_divides(int(quantized_weight.shape[0]), group_size)
    else:
        assert_group_divides(int(quantized_weight.shape[1]), group_size)
    scale_dev = scale.to(device=x.device) if scale.device != x.device else scale
    y = _FusedW8A8Fn.apply(
        x,
        quantized_weight.to(device=x.device),
        scale_dev,
        int(group_size),
        hadamard.to(device=x.device) if hadamard is not None else None,
        layout,
    )
    if x.is_floating_point() and y.dtype != x.dtype:
        return y.to(dtype=x.dtype)
    return y
