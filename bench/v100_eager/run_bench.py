"""Benchmark V100 eager memory operators against their original expressions."""

from __future__ import annotations

import argparse
import gc
import statistics
import time
from collections.abc import Callable

import torch
import torch.nn.functional as F
from lycoris.functional.general import factorization
from lycoris.functional.lokr import bypass_forward_diff

from bench._common import make_run_dir, write_result
from library.anima.eager_autograd import (
    eager_fused_lora_mlp_tensors,
    eager_fused_lokr_mlp_tensors,
    eager_rms_norm,
    eager_rotary_qk,
)
from networks.lora_modules.custom_autograd import (
    eager_lora_down_project,
    eager_lora_up_residual,
)

MIB = 2**20
CASES = ("lora_down", "lora_up", "mlp", "lokr_mlp", "rope", "rms_norm")
CaseResult = tuple[torch.Tensor, list[torch.Tensor]]
CaseBuilder = Callable[[bool, torch.device, argparse.Namespace], CaseResult]


def _lora_linear(
    x: torch.Tensor,
    base_weight: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
) -> torch.Tensor:
    base = F.linear(x.to(base_weight.dtype), base_weight)
    rank = F.linear(x.float(), down_weight.float())
    delta = F.linear(rank, up_weight.float())
    return base + (delta * 0.75).to(base.dtype)


def _build_lora_down(
    after: bool, device: torch.device, args: argparse.Namespace
) -> CaseResult:
    x = (
        torch.randn(
            args.rows,
            args.model_dim,
            device=device,
            dtype=torch.float16,
        )
        * 0.1
    ).requires_grad_()
    weight = (
        torch.randn(
            args.rank,
            args.model_dim,
            device=device,
            dtype=torch.float32,
        )
        * 0.02
    ).requires_grad_()
    output = (
        eager_lora_down_project(x, weight, None)
        if after
        else F.linear(x.float(), weight)
    )
    return output, [x, weight]


def _build_lora_up(
    after: bool, device: torch.device, args: argparse.Namespace
) -> CaseResult:
    rank = (
        torch.randn(
            args.rows,
            args.rank,
            device=device,
            dtype=torch.float32,
        )
        * 0.1
    ).requires_grad_()
    weight = (
        torch.randn(
            args.model_dim,
            args.rank,
            device=device,
            dtype=torch.float32,
        )
        * 0.02
    ).requires_grad_()
    base_leaf = torch.zeros(
        args.rows,
        args.model_dim,
        device=device,
        dtype=torch.float16,
        requires_grad=True,
    )
    base = base_leaf + 0.0
    output = (
        eager_lora_up_residual(base, rank, weight, 0.75, args.chunk)
        if after
        else base + (F.linear(rank, weight) * 0.75).to(base.dtype)
    )
    return output, [rank, weight, base_leaf]


def _build_mlp(
    after: bool, device: torch.device, args: argparse.Namespace
) -> CaseResult:
    x = (
        torch.randn(
            args.rows,
            args.model_dim,
            device=device,
            dtype=torch.float16,
        )
        * 0.1
    ).requires_grad_()
    base1 = torch.randn(
        args.ffn_dim, args.model_dim, device=device, dtype=torch.float16
    ) * 0.02
    base2 = torch.randn(
        args.model_dim, args.ffn_dim, device=device, dtype=torch.float16
    ) * 0.02
    down1 = (
        torch.randn(
            args.rank,
            args.model_dim,
            device=device,
            dtype=torch.float16,
        )
        * 0.02
    ).requires_grad_()
    up1 = (
        torch.randn(
            args.ffn_dim,
            args.rank,
            device=device,
            dtype=torch.float16,
        )
        * 0.02
    ).requires_grad_()
    down2 = (
        torch.randn(
            args.rank,
            args.ffn_dim,
            device=device,
            dtype=torch.float16,
        )
        * 0.02
    ).requires_grad_()
    up2 = (
        torch.randn(
            args.model_dim,
            args.rank,
            device=device,
            dtype=torch.float16,
        )
        * 0.02
    ).requires_grad_()
    empty_scale = torch.empty(0, device=device, dtype=torch.float32)
    rank_mask = torch.ones(1, args.rank, device=device, dtype=torch.float32)
    if after:
        output = eager_fused_lora_mlp_tensors(
            x,
            base1,
            down1,
            up1,
            empty_scale,
            rank_mask,
            0.75,
            base2,
            down2,
            up2,
            empty_scale,
            rank_mask,
            0.75,
            args.chunk,
        )
    else:
        hidden = F.gelu(_lora_linear(x, base1, down1, up1))
        output = _lora_linear(hidden, base2, down2, up2)
    return output, [x, down1, up1, down2, up2]


def _full_lokr_factors(
    out_features: int,
    in_features: int,
    factor: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    out_a, out_b = factorization(out_features, factor)
    in_a, in_b = factorization(in_features, factor)
    w1 = (
        torch.randn(out_a, in_a, device=device, dtype=torch.float16) * 0.02
    ).requires_grad_()
    w2 = (
        torch.randn(out_b, in_b, device=device, dtype=torch.float16) * 0.02
    ).requires_grad_()
    return w1, w2


def _lokr_linear(
    x: torch.Tensor,
    base_weight: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    residual_scale: float,
) -> torch.Tensor:
    base = F.linear(x.to(base_weight.dtype), base_weight)
    delta = bypass_forward_diff(
        x.float(),
        None,
        w1.float(),
        None,
        None,
        w2.float(),
        None,
        None,
        None,
        gamma=1,
    )
    return base + (delta * residual_scale).to(base.dtype)


def _build_lokr_mlp(
    after: bool, device: torch.device, args: argparse.Namespace
) -> CaseResult:
    x = (
        torch.randn(
            args.rows,
            args.model_dim,
            device=device,
            dtype=torch.float16,
        )
        * 0.1
    ).requires_grad_()
    base1 = torch.randn(
        args.ffn_dim, args.model_dim, device=device, dtype=torch.float16
    ) * 0.02
    base2 = torch.randn(
        args.model_dim, args.ffn_dim, device=device, dtype=torch.float16
    ) * 0.02
    w1_1, w2_1 = _full_lokr_factors(
        args.ffn_dim,
        args.model_dim,
        args.lokr_factor,
        device,
    )
    w1_2, w2_2 = _full_lokr_factors(
        args.model_dim,
        args.ffn_dim,
        args.lokr_factor,
        device,
    )
    scalar = torch.ones((), device=device, dtype=torch.float32)
    if after:
        output = eager_fused_lokr_mlp_tensors(
            x,
            base1,
            w1_1,
            None,
            None,
            w2_1,
            None,
            None,
            scalar,
            0.75,
            base2,
            w1_2,
            None,
            None,
            w2_2,
            None,
            None,
            scalar,
            1.25,
            args.lokr_chunk,
        )
    else:
        hidden = F.gelu(_lokr_linear(x, base1, w1_1, w2_1, 0.75))
        output = _lokr_linear(hidden, base2, w1_2, w2_2, 1.25)
    return output, [x, w1_1, w2_1, w1_2, w2_2]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _build_rope(
    after: bool, device: torch.device, args: argparse.Namespace
) -> CaseResult:
    head_dim = args.model_dim // args.heads
    q_leaf = torch.randn(
        1,
        args.rows,
        args.heads,
        head_dim,
        device=device,
        dtype=torch.float16,
        requires_grad=True,
    )
    k_leaf = torch.randn_like(q_leaf, requires_grad=True)
    q = q_leaf + 0.0
    k = k_leaf + 0.0
    angles = torch.randn(
        1,
        args.rows,
        1,
        head_dim // 2,
        device=device,
        dtype=torch.float16,
    )
    cos_half = angles.cos()
    sin_half = angles.sin()
    cos = torch.cat((cos_half, cos_half), dim=-1)
    sin = torch.cat((sin_half, sin_half), dim=-1)
    if after:
        q_out, k_out = eager_rotary_qk(
            q,
            k,
            cos,
            sin,
            seq_axis=1,
            rot_dim=head_dim,
            chunk_size=args.chunk,
        )
    else:
        q_out = q * cos + _rotate_half(q) * sin
        k_out = k * cos + _rotate_half(k) * sin
    return q_out + k_out, [q_leaf, k_leaf]


def _build_rms_norm(
    after: bool, device: torch.device, args: argparse.Namespace
) -> CaseResult:
    head_dim = args.model_dim // args.heads
    x = torch.randn(
        1,
        args.rows,
        args.heads,
        head_dim,
        device=device,
        dtype=torch.float16,
        requires_grad=True,
    )
    weight = torch.ones(
        head_dim,
        device=device,
        dtype=torch.float16,
    )
    if after:
        output = eager_rms_norm(x, weight, eps=1e-6, chunk_size=args.rms_chunk)
    else:
        x_fp32 = x.float()
        normalized = x_fp32 * torch.rsqrt(
            x_fp32.pow(2).mean(-1, keepdim=True) + 1e-6
        )
        output = (normalized * weight.float()).to(x.dtype)
    return output, [x]


BUILDERS: dict[str, CaseBuilder] = {
    "lora_down": _build_lora_down,
    "lora_up": _build_lora_up,
    "mlp": _build_mlp,
    "lokr_mlp": _build_lokr_mlp,
    "rope": _build_rope,
    "rms_norm": _build_rms_norm,
}


def _run_once(
    builder: CaseBuilder,
    *,
    after: bool,
    device: torch.device,
    args: argparse.Namespace,
    capture_saved: bool,
) -> dict:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    torch.manual_seed(123)
    saved: list[dict] = []

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        if capture_saved:
            saved.append(
                {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "bytes": tensor.numel() * tensor.element_size(),
                }
            )
        return tensor

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        baseline_allocated = torch.cuda.memory_allocated(device)
        baseline_reserved = torch.cuda.memory_reserved(device)
        torch.cuda.synchronize(device)
    else:
        baseline_allocated = baseline_reserved = 0

    started = time.perf_counter()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        output, leaves = builder(after, device, args)
    output.mean(dtype=torch.float32).backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = {
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "saved_tensor_bytes": sum(item["bytes"] for item in saved),
        "saved_tensors": saved,
        "finite_gradients": all(
            leaf.grad is not None and bool(torch.isfinite(leaf.grad).all())
            for leaf in leaves
        ),
    }
    if device.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        result.update(
            {
                "workspace_peak_allocated_mib": (
                    peak_allocated - baseline_allocated
                )
                / MIB,
                "workspace_peak_reserved_mib": (
                    peak_reserved - baseline_reserved
                )
                / MIB,
            }
        )
    return result


def _measure(
    builder: CaseBuilder,
    *,
    after: bool,
    device: torch.device,
    args: argparse.Namespace,
) -> dict:
    runs = [
        _run_once(
            builder,
            after=after,
            device=device,
            args=args,
            capture_saved=index == 0,
        )
        for index in range(args.repeats)
    ]
    result = dict(runs[0])
    result["elapsed_ms"] = statistics.median(
        run["elapsed_ms"] for run in runs
    )
    for key in ("workspace_peak_allocated_mib", "workspace_peak_reserved_mib"):
        if key in result:
            result[key] = max(run[key] for run in runs)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--ffn-dim", type=int, default=768)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--lokr-chunk", type=int, default=128)
    parser.add_argument("--lokr-factor", type=int, default=8)
    parser.add_argument("--rms-chunk", type=int, default=8192)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
    parser.add_argument("--label", default=None)
    args = parser.parse_args()
    positive = (
        args.rows,
        args.model_dim,
        args.ffn_dim,
        args.rank,
        args.heads,
        args.chunk,
        args.lokr_chunk,
        args.lokr_factor,
        args.rms_chunk,
        args.repeats,
    )
    if any(value < 1 for value in positive):
        parser.error("shape, chunk, and repeat values must be positive")
    head_dim = args.model_dim // args.heads
    if args.model_dim % args.heads != 0 or head_dim % 2:
        parser.error("--model-dim / --heads must be an even head dimension")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    return args


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    metrics = {
        "shape": {
            "rows": args.rows,
            "model_dim": args.model_dim,
            "ffn_dim": args.ffn_dim,
            "rank": args.rank,
            "heads": args.heads,
            "chunk": args.chunk,
            "lokr_chunk": args.lokr_chunk,
            "lokr_factor": args.lokr_factor,
            "rms_chunk": args.rms_chunk,
        },
        "cases": {},
    }
    for name in args.cases:
        builder = BUILDERS[name]
        before = _measure(builder, after=False, device=device, args=args)
        after = _measure(builder, after=True, device=device, args=args)
        before_saved = before["saved_tensor_bytes"]
        reduction = (
            100.0 * (before_saved - after["saved_tensor_bytes"]) / before_saved
            if before_saved
            else 0.0
        )
        metrics["cases"][name] = {
            "before": before,
            "after": after,
            "saved_tensor_reduction_pct": reduction,
        }

    run_dir = make_run_dir("v100_eager", label=args.label)
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label,
        device=device,
    )

    print(f"=== V100 bounded eager bench -> {run_dir} ===")
    print("case       before saved  after saved   reduction  before ms  after ms")
    for name, result in metrics["cases"].items():
        before = result["before"]
        after = result["after"]
        print(
            f"{name:10} "
            f"{before['saved_tensor_bytes'] / MIB:10.1f} MiB "
            f"{after['saved_tensor_bytes'] / MIB:10.1f} MiB "
            f"{result['saved_tensor_reduction_pct']:8.1f}% "
            f"{before['elapsed_ms']:9.2f} "
            f"{after['elapsed_ms']:8.2f}"
        )
    if device.type == "cuda":
        print("case       before alloc  after alloc  before reserv  after reserv")
        for name, result in metrics["cases"].items():
            before = result["before"]
            after = result["after"]
            print(
                f"{name:10} "
                f"{before['workspace_peak_allocated_mib']:10.1f} MiB "
                f"{after['workspace_peak_allocated_mib']:9.1f} MiB "
                f"{before['workspace_peak_reserved_mib']:11.1f} MiB "
                f"{after['workspace_peak_reserved_mib']:10.1f} MiB"
            )


if __name__ == "__main__":
    main()
