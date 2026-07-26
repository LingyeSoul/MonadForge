"""Benchmark the eager FP32 LoKr bypass before and after rematerialization."""

from __future__ import annotations

import argparse
import gc
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import torch
from lycoris.functional.general import factorization
from lycoris.functional.lokr import bypass_forward_diff

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench._common import make_run_dir, write_result  # noqa: E402
from networks.lora_modules.custom_autograd import (  # noqa: E402
    eager_lokr_residual,
)

MIB = 2**20


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _run_once(
    *,
    device: torch.device,
    rows: int,
    out_features: int,
    in_features: int,
    factor: int,
    custom_chunk: int | None,
    capture_saved: bool,
) -> dict:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    torch.manual_seed(123)

    out_a, out_b = factorization(out_features, factor)
    in_a, in_b = factorization(in_features, factor)
    x = torch.randn(
        rows,
        in_features,
        dtype=torch.float16,
        device=device,
        requires_grad=True,
    )
    w1 = torch.randn(
        out_a,
        in_a,
        dtype=torch.float16,
        device=device,
        requires_grad=True,
    )
    w2 = torch.randn(
        out_b,
        in_b,
        dtype=torch.float16,
        device=device,
        requires_grad=True,
    )
    base_leaf = torch.zeros(
        rows,
        out_features,
        dtype=torch.float16,
        device=device,
        requires_grad=True,
    )
    base = base_leaf + 0.0
    scalar = torch.tensor(1.0, device=device)

    saved: list[dict] = []

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        if capture_saved:
            saved.append(
                {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "bytes": _tensor_bytes(tensor),
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
        if custom_chunk is None:
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
            output = base + delta.to(base.dtype)
        else:
            output = eager_lokr_residual(
                base,
                x,
                w1,
                None,
                None,
                w2,
                None,
                None,
                scalar,
                1.0,
                custom_chunk,
            )
    output.mean(dtype=torch.float32).backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    factor_shapes = {
        (out_a, in_a),
        (in_a, out_a),
        (out_b, in_b),
        (in_b, out_b),
    }
    activation_saved = sum(
        item["bytes"]
        for item in saved
        if item["shape"] and tuple(item["shape"]) not in factor_shapes
    )
    result = {
        "elapsed_ms": elapsed * 1000.0,
        "saved_tensor_bytes": sum(item["bytes"] for item in saved),
        "saved_activation_bytes": activation_saved,
        "saved_tensors": saved,
        "finite_gradients": all(
            grad is not None and bool(torch.isfinite(grad).all())
            for grad in (x.grad, w1.grad, w2.grad)
        ),
    }
    if device.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        result.update(
            {
                "baseline_allocated_mib": baseline_allocated / MIB,
                "baseline_reserved_mib": baseline_reserved / MIB,
                "peak_allocated_mib": peak_allocated / MIB,
                "peak_reserved_mib": peak_reserved / MIB,
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
    runner: Callable[..., dict],
    *,
    repeats: int,
    **kwargs,
) -> dict:
    runs = [
        runner(capture_saved=index == 0, **kwargs)
        for index in range(repeats)
    ]
    result = dict(runs[0])
    result["elapsed_ms"] = statistics.median(
        run["elapsed_ms"] for run in runs
    )
    for key in (
        "peak_allocated_mib",
        "peak_reserved_mib",
        "workspace_peak_allocated_mib",
        "workspace_peak_reserved_mib",
    ):
        if key in result:
            result[key] = max(run[key] for run in runs)
    return result


def _mib(value: int) -> str:
    return f"{value / MIB:.1f} MiB"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Execution device (default: %(default)s).",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=128,
        help="Flattened token rows (use about 4200 for a 1024px V100 run).",
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=2,
        default=(9216, 3072),
        metavar=("OUT", "IN"),
        help="Linear shape as OUT IN (default: %(default)s).",
    )
    parser.add_argument(
        "--factor",
        type=int,
        default=8,
        help="LyCORIS factorization hint (default: %(default)s).",
    )
    parser.add_argument(
        "--chunks",
        type=int,
        nargs="+",
        default=(1024,),
        help="Custom-path row chunks to compare (default: %(default)s).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Runs per path; timing reports the median (default: %(default)s).",
    )
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    if args.rows < 1 or args.repeats < 1:
        parser.error("--rows and --repeats must be positive")
    if any(chunk < 1 for chunk in args.chunks):
        parser.error("--chunks values must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")

    device = torch.device(args.device)
    out_features, in_features = args.shape
    common = {
        "device": device,
        "rows": args.rows,
        "out_features": out_features,
        "in_features": in_features,
        "factor": args.factor,
        "repeats": args.repeats,
    }
    official = _measure(_run_once, custom_chunk=None, **common)
    custom = {
        str(chunk): _measure(_run_once, custom_chunk=chunk, **common)
        for chunk in args.chunks
    }
    metrics = {
        "shape": {
            "rows": args.rows,
            "out_features": out_features,
            "in_features": in_features,
            "factor": args.factor,
        },
        "official_lycoris": official,
        "custom_chunks": custom,
    }

    run_dir = make_run_dir("lokr", label=args.label)
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label,
        device=device,
    )

    print(f"=== eager LoKr memory bench -> {run_dir} ===")
    print(
        f"shape={args.rows}x{in_features}->{out_features}, "
        f"factor={args.factor}, device={device}"
    )
    print(
        "official: "
        f"saved={_mib(official['saved_tensor_bytes'])}, "
        f"saved activations={_mib(official['saved_activation_bytes'])}, "
        f"time={official['elapsed_ms']:.2f} ms"
    )
    for chunk, result in custom.items():
        line = (
            f"custom[{chunk}]: saved={_mib(result['saved_tensor_bytes'])}, "
            f"saved activations={_mib(result['saved_activation_bytes'])}, "
            f"time={result['elapsed_ms']:.2f} ms"
        )
        if device.type == "cuda":
            line += (
                f", peak allocated={result['peak_allocated_mib']:.1f} MiB"
                f", peak reserved={result['peak_reserved_mib']:.1f} MiB"
            )
        print(line)


if __name__ == "__main__":
    main()
