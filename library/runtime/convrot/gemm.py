"""Int8 GEMM helpers for ConvRot W8A8 (true path + float fallback)."""

from __future__ import annotations

import os
from typing import Literal

import torch
from torch.nn import functional as F

# torch._int_mm requires M (rows of A) > 16 on current CUDA builds; some shapes
# also trip cublasLt NOT_SUPPORTED. Ampere's cuBLASLt int8 path requires M to
# be 32-aligned (Anima's 4200-token family is not), so the hot path pads rows
# before GEMM and slices the result back to the native token count.
_INT_MM_MIN_M = 17
_INT_MM_M_ALIGNMENT = 32
_BACKEND_ENV = "ANIMA_CONVROT_INT8_GEMM"
# Opt-in TF32 for the large fp32 STE matmul. Default OFF: seed0 grad_rel can
# exceed the 5% full-ckpt gate on 3080 (P1.11c). Speed seekers may set =1.
_STE_TF32_ENV = "ANIMA_CONVROT_STE_TF32"

WeightLayout = Literal["nk", "kn"]


def int8_gemm_backend() -> Literal["auto", "int_mm", "float"]:
    raw = str(os.environ.get(_BACKEND_ENV, "auto") or "auto").strip().lower()
    if raw in {"auto", "int_mm", "int8", "true", "1"}:
        if raw in {"int_mm", "int8", "true", "1"}:
            return "int_mm"
        return "auto"
    if raw in {"float", "fake", "fp", "0", "false", "off"}:
        return "float"
    return "auto"


def ste_tf32_enabled() -> bool:
    raw = str(os.environ.get(_STE_TF32_ENV, "0") or "0").strip().lower()
    return raw in {"1", "true", "on", "yes", "high"}


def can_use_torch_int_mm(
    m: int,
    k: int,
    n: int,
    *,
    device: torch.device,
) -> bool:
    if device.type != "cuda":
        return False
    if not hasattr(torch, "_int_mm"):
        return False
    if m < _INT_MM_MIN_M:
        return False
    # Heuristic: cublasLt path is picky on tiny dims; require reasonable K/N.
    if k < 16 or n < 16:
        return False
    return True


def quantize_activation_absmax_int8(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token absmax quant → real int8 + scale (no STE).

    ``scale`` shape is ``x.shape[:-1] + (1,)`` such that
    ``x ≈ x_q.float() * scale``.

    Half/bf16 inputs keep absmax in native dtype then promote scale only;
    the division still runs in fp32 for stable int8 codes.

    Note: a pure half mul-round path (no fp32 ``x``) was tried in P1.11 and
    failed the full-ckpt grad gate (seed2 grad_rel >1). Keep fp32 codes.
    """
    from library.runtime.convrot.quant import INT8_MAX, SCALE_EPS

    work = x.detach()
    if work.dtype in (torch.float16, torch.bfloat16):
        amax = work.abs().amax(dim=-1, keepdim=True).to(torch.float32).clamp_min(SCALE_EPS)
        work_f = work.to(torch.float32)
    else:
        work_f = work.to(torch.float32)
        amax = work_f.abs().amax(dim=-1, keepdim=True).clamp_min(SCALE_EPS)
    scale = amax / INT8_MAX
    q = (work_f / scale).round().clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
    return q, scale


def int8_mm_scaled(
    x_q: torch.Tensor,
    x_scale: torch.Tensor,
    w_q: torch.Tensor,
    w_scale: torch.Tensor,
    *,
    prefer: Literal["auto", "int_mm", "float"] | None = None,
    weight_layout: WeightLayout = "nk",
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Compute ``(x_q @ W^T) * x_scale * w_scale``.

    Parameters
    ----------
    x_q:
        int8 activations ``[..., K]``
    x_scale:
        float scales broadcastable to ``[..., 1]``
    w_q:
        int8 weights. Layout:
        * ``nk`` (default): ``[N, K]`` out×in (classic Linear.weight)
        * ``kn``: ``[K, N]`` = ``weight.T`` contiguous for ``torch._int_mm``
    w_scale:
        float per-out-channel scales ``[N]``
    weight_layout:
        See ``w_q``. Prefer ``kn`` on the W8A8 hot path so each step skips
        ``w_q.t().contiguous()`` (same storage bytes as ``nk``).
    out_dtype:
        Result dtype. Default float32 (tests / reference). Training fused path
        passes bf16/fp16 to avoid a late cast and halve y traffic (P1.9).
    """
    if x_q.dtype is not torch.int8 or w_q.dtype is not torch.int8:
        raise TypeError("x_q and w_q must be torch.int8")
    if w_q.dim() != 2:
        raise ValueError("w_q must be 2D")
    *leading, k = x_q.shape
    if weight_layout == "kn":
        k_w, n = w_q.shape
    else:
        n, k_w = w_q.shape
    if k != k_w:
        raise ValueError(f"K mismatch: x has {k}, w has {k_w} (layout={weight_layout})")
    if w_scale.numel() != n:
        raise ValueError(f"w_scale length {w_scale.numel()} != N={n}")

    flat = x_q.reshape(-1, k)
    m = flat.shape[0]
    backend = prefer or int8_gemm_backend()
    use_int = backend == "int_mm" or (
        backend == "auto"
        and can_use_torch_int_mm(m, k, n, device=flat.device)
    )
    result_dtype = out_dtype or torch.float32

    # Post-scale in float32 (int32 acc needs headroom); final cast once.
    x_scale_flat = x_scale.reshape(-1, 1).to(device=flat.device, dtype=torch.float32)
    w_scale_row = w_scale.reshape(1, -1).to(device=flat.device, dtype=torch.float32)

    if use_int:
        try:
            # _int_mm(A[M,K], B[K,N]) -> int32[M,N]
            if weight_layout == "kn":
                b = w_q if w_q.is_contiguous() else w_q.contiguous()
            else:
                b = w_q.t().contiguous()
            a = flat if flat.is_contiguous() else flat.contiguous()
            # Keep native flatten shapes at the model boundary while satisfying
            # SM86 cuBLASLt. Zero rows contribute zero accumulators and are
            # removed before post-scaling, so model numerics are unchanged.
            pad_rows = (-m) % _INT_MM_M_ALIGNMENT
            if pad_rows:
                a = F.pad(a, (0, 0, 0, pad_rows))
            acc = torch._int_mm(a, b)
            acc = acc[:m]
            # int32→fp32 required before scale (bf16 mantissa too short for raw acc).
            y = acc.to(torch.float32)
            y.mul_(x_scale_flat)
            y.mul_(w_scale_row)
            if result_dtype != torch.float32:
                y = y.to(dtype=result_dtype)
            return y.reshape(*leading, n)
        except RuntimeError:
            if backend == "int_mm":
                raise
            # fall through to float reference

    # Float reference / fallback: exact for int8×int8 when accumulated in fp32.
    w_f = w_q.to(torch.float32)
    if weight_layout == "kn":
        y = flat.to(torch.float32) @ w_f
    else:
        y = flat.to(torch.float32) @ w_f.t()
    y.mul_(x_scale_flat)
    y.mul_(w_scale_row)
    if result_dtype != torch.float32:
        y = y.to(dtype=result_dtype)
    return y.reshape(*leading, n)


def dequant_weight_for_grad(
    w_q: torch.Tensor,
    w_scale: torch.Tensor,
    *,
    weight_layout: WeightLayout = "nk",
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Materialize ``W [N,K]`` for ``grad_x = grad_y @ W`` (STE / fused bwd)."""
    from library.runtime.convrot.quant import dequantize_weight

    if weight_layout == "kn":
        # w_q is [K,N]; scale [N] → W[n,k] = w_q[k,n] * scale[n]
        if device is not None and w_q.device != device:
            w_q = w_q.to(device=device)
            w_scale = w_scale.to(device=device)
        # Cast then mul in target dtype; transpose view is free when contiguous kn.
        w_kn = w_q.to(dtype) * w_scale.to(device=w_q.device, dtype=dtype)
        return w_kn.transpose(0, 1)
    w = dequantize_weight(w_q, w_scale, dtype=dtype)
    if device is not None and w.device != device:
        w = w.to(device=device)
    return w


class _W8A8IntLinearFn(torch.autograd.Function):
    """Forward: true/fake int8 GEMM; backward: STE via dequant weight @ grad."""

    @staticmethod
    def forward(
        ctx,
        x_rot: torch.Tensor,
        w_q: torch.Tensor,
        w_scale: torch.Tensor,
        weight_layout: str,
    ) -> torch.Tensor:
        layout: WeightLayout = "kn" if weight_layout == "kn" else "nk"
        out_dtype = (
            x_rot.dtype
            if x_rot.is_floating_point()
            and x_rot.dtype in (torch.float16, torch.bfloat16)
            else torch.float32
        )
        x_q, x_scale = quantize_activation_absmax_int8(x_rot)
        y = int8_mm_scaled(
            x_q, x_scale, w_q, w_scale, weight_layout=layout, out_dtype=out_dtype
        )
        ctx.weight_layout = layout
        # STE does not need x_rot — only dtype for casting grad_x. Saves a full
        # activation tensor per layer vs older save_for_backward(x_rot, ...).
        # Frozen w_q/w_scale: hold on ctx (P1.12), not save_for_backward.
        ctx.x_dtype = x_rot.dtype
        ctx.w_q = w_q
        ctx.w_scale = w_scale
        return y

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        w_q = ctx.w_q
        w_scale = ctx.w_scale
        layout: WeightLayout = getattr(ctx, "weight_layout", "nk")
        x_dtype = getattr(ctx, "x_dtype", grad_y.dtype)
        # Base weights frozen — only grad_x. STE ignores act quant.
        # Default: fp32 accumulate (required for full-ckpt grad gate).
        # Opt-in TF32 via ANIMA_CONVROT_STE_TF32=1 (faster, may fail 5% gate).
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
                grad_x = gy @ w.transpose(0, 1)
            else:
                grad_x = gy @ w
        finally:
            if use_tf32 and prev is not None:
                torch.set_float32_matmul_precision(prev)
        if x_dtype != grad_x.dtype:
            grad_x = grad_x.to(dtype=x_dtype)
        return grad_x, None, None, None


def w8a8_int_linear(
    x_rot: torch.Tensor,
    w_q: torch.Tensor,
    w_scale: torch.Tensor,
    *,
    weight_layout: WeightLayout = "nk",
) -> torch.Tensor:
    """Autograd-aware W8A8 linear in rotated space."""
    return _W8A8IntLinearFn.apply(x_rot, w_q, w_scale, weight_layout)
