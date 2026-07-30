"""Run the issue-43 dense M/N modulo-16 forward/backward validation matrix."""

from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path
from typing import Any

import torch

from bench._common import make_run_dir
from bench.v100_flash._validation import (
    append_jsonl,
    environment_manifest,
    require_v100,
    resolve_device,
    tensor_stats,
    write_json,
)


def _reference_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    causal: bool,
    upcast: bool,
) -> torch.Tensor:
    original_dtype = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    scores = torch.einsum("bhmd,bhnd->bhmn", q, k) * scale
    if causal:
        diagonal = 1 + scores.shape[-1] - scores.shape[-2]
        mask = torch.triu(
            torch.ones(
                scores.shape[-2],
                scores.shape[-1],
                device=scores.device,
                dtype=torch.bool,
            ),
            diagonal=diagonal,
        )
        scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("bhmn,bhnd->bhmd", probs, v).to(original_dtype)


def _reference_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    *,
    scale: float,
    causal: bool,
    upcast: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    original_dtype = q.dtype
    q_leaf = q.detach().clone().requires_grad_(True)
    k_leaf = k.detach().clone().requires_grad_(True)
    v_leaf = v.detach().clone().requires_grad_(True)
    q_work, k_work, v_work = q_leaf, k_leaf, v_leaf
    if upcast:
        q_work, k_work, v_work = q_work.float(), k_work.float(), v_work.float()
    scores = torch.einsum("bhmd,bhnd->bhmn", q_work, k_work) * scale
    if causal:
        diagonal = 1 + scores.shape[-1] - scores.shape[-2]
        mask = torch.triu(
            torch.ones(
                scores.shape[-2],
                scores.shape[-1],
                device=scores.device,
                dtype=torch.bool,
            ),
            diagonal=diagonal,
        )
        scores = scores.masked_fill(mask, float("-inf"))
    output = torch.einsum("bhmn,bhnd->bhmd", torch.softmax(scores, dim=-1), v_work)
    grads = torch.autograd.grad(
        output.to(original_dtype), (q_leaf, k_leaf, v_leaf), dout
    )
    return tuple(grad.to(original_dtype) for grad in grads)


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max().item())


def _case_shapes(base: int) -> list[tuple[str, int, int]]:
    rows: list[tuple[str, int, int]] = []
    for residue in range(16):
        rows.append(("self", base + residue, base + residue))
        rows.append(("cross_k_tail", base, base * 2 + residue))
        rows.append(("cross_q_tail", base + residue, base * 2))
    return rows


def _run_case(
    flash_attn_func,
    *,
    head_dim: int,
    heads: int,
    m: int,
    n: int,
    causal: bool,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    q_bhmd = torch.randn(1, heads, m, head_dim, device=device, dtype=torch.float16)
    k_bhnd = torch.randn(1, heads, n, head_dim, device=device, dtype=torch.float16)
    v_bhnd = torch.randn(1, heads, n, head_dim, device=device, dtype=torch.float16)
    dout_bhmd = torch.randn(1, heads, m, head_dim, device=device, dtype=torch.float16)
    scale = 1.0 / math.sqrt(head_dim)

    ref32 = _reference_forward(
        q_bhmd, k_bhnd, v_bhnd, scale=scale, causal=causal, upcast=True
    )
    ref16 = _reference_forward(
        q_bhmd, k_bhnd, v_bhnd, scale=scale, causal=causal, upcast=False
    )
    ref32_grads = _reference_backward(
        q_bhmd, k_bhnd, v_bhnd, dout_bhmd, scale=scale, causal=causal, upcast=True
    )
    ref16_grads = _reference_backward(
        q_bhmd, k_bhnd, v_bhnd, dout_bhmd, scale=scale, causal=causal, upcast=False
    )

    q = q_bhmd.transpose(1, 2).contiguous().detach().requires_grad_(True)
    k = k_bhnd.transpose(1, 2).contiguous().detach().requires_grad_(True)
    v = v_bhnd.transpose(1, 2).contiguous().detach().requires_grad_(True)
    dout = dout_bhmd.transpose(1, 2).contiguous()
    output = flash_attn_func(
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=scale,
        causal=causal,
    )
    output.backward(dout)
    flash_bhmd = output.transpose(1, 2)
    flash_grads = (
        q.grad.transpose(1, 2),
        k.grad.transpose(1, 2),
        v.grad.transpose(1, 2),
    )

    forward_error = _max_abs(flash_bhmd, ref32)
    native_forward_error = _max_abs(ref16, ref32)
    forward_limit = 2.0 * native_forward_error + 1e-5
    grad_rows: dict[str, Any] = {}
    grads_ok = True
    for name, actual, reference, native in zip(
        ("dq", "dk", "dv"), flash_grads, ref32_grads, ref16_grads, strict=True
    ):
        error = _max_abs(actual, reference)
        native_error = _max_abs(native, reference)
        limit = 3.0 * native_error + 1e-4
        finite = bool(torch.isfinite(actual).all().item())
        passed = finite and error <= limit
        grads_ok = grads_ok and passed
        grad_rows[name] = {
            "finite": finite,
            "max_abs": error,
            "native_fp16_max_abs": native_error,
            "limit": limit,
            "passed": passed,
        }

    output_finite = bool(torch.isfinite(output).all().item())
    forward_ok = output_finite and forward_error <= forward_limit
    return {
        "head_dim": head_dim,
        "heads": heads,
        "m": m,
        "n": n,
        "m_mod_16": m % 16,
        "n_mod_16": n % 16,
        "causal": causal,
        "forward": {
            "finite": output_finite,
            "max_abs": forward_error,
            "native_fp16_max_abs": native_forward_error,
            "limit": forward_limit,
            "passed": forward_ok,
        },
        "backward": grad_rows,
        "passed": forward_ok and grads_ok,
    }


def _event_ms(fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def _benchmark_aligned(
    flash_attn_func,
    *,
    length: int,
    warmup: int,
    runs: int,
    rounds: int,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(421)
    q = torch.randn(
        1, length, 16, 128, device=device, dtype=torch.float16, requires_grad=True
    )
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    dout = torch.randn_like(q)

    def forward_only():
        with torch.no_grad():
            flash_attn_func(q, k, v, dropout_p=0.0)

    def full_step():
        q.grad = None
        k.grad = None
        v.grad = None
        output = flash_attn_func(q, k, v, dropout_p=0.0)
        output.backward(dout)

    for _ in range(warmup):
        full_step()
    torch.cuda.synchronize(device)

    round_rows: list[dict[str, float]] = []
    for _ in range(rounds):
        forward_times = [_event_ms(forward_only) for _ in range(runs)]
        backward_times: list[float] = []
        for _ in range(runs):
            q.grad = None
            k.grad = None
            v.grad = None
            output = flash_attn_func(q, k, v, dropout_p=0.0)
            backward_times.append(
                _event_ms(lambda output=output: output.backward(dout))
            )
        round_rows.append(
            {
                "forward_median_ms": statistics.median(forward_times),
                "backward_median_ms": statistics.median(backward_times),
            }
        )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    full_step()
    torch.cuda.synchronize(device)
    return {
        "length": length,
        "warmup": warmup,
        "runs": runs,
        "rounds": round_rows,
        "forward_median_ms": statistics.median(
            row["forward_median_ms"] for row in round_rows
        ),
        "backward_median_ms": statistics.median(
            row["backward_median_ms"] for row in round_rows
        ),
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / (1024**2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--progress-jsonl", type=Path)
    parser.add_argument("--candidate", default="candidate")
    parser.add_argument("--source-sha")
    parser.add_argument("--wheel-sha")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--head-dims", default="16,32,64,128,256")
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--base-seq", type=int, default=32)
    parser.add_argument("--benchmark-warmup", type=int, default=3)
    parser.add_argument("--benchmark-runs", type=int, default=10)
    parser.add_argument("--benchmark-rounds", type=int, default=3)
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--expect-fixed", action="store_true")
    args = parser.parse_args()

    if args.benchmark_only and args.skip_benchmark:
        parser.error("--benchmark-only and --skip-benchmark are mutually exclusive")
    if args.benchmark_only and args.expect_fixed:
        parser.error("--expect-fixed requires the correctness matrix")

    device = resolve_device(args.device)
    require_v100(device)
    torch.cuda.set_device(device)
    from flash_attn import flash_attn_func

    head_dims = [int(value) for value in args.head_dims.split(",") if value]
    cases: list[dict[str, Any]] = []
    failures: list[str] = []
    case_index = 0
    if not args.benchmark_only:
        for head_dim in head_dims:
            for kind, m, n in _case_shapes(args.base_seq):
                for causal in (False, True):
                    case_index += 1
                    try:
                        row = _run_case(
                            flash_attn_func,
                            head_dim=head_dim,
                            heads=args.heads,
                            m=m,
                            n=n,
                            causal=causal,
                            seed=421 + case_index,
                            device=device,
                        )
                        row["kind"] = kind
                    except Exception as exc:  # noqa: BLE001 - preserve every matrix failure in JSON.
                        row = {
                            "head_dim": head_dim,
                            "m": m,
                            "n": n,
                            "m_mod_16": m % 16,
                            "n_mod_16": n % 16,
                            "causal": causal,
                            "kind": kind,
                            "passed": False,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    cases.append(row)
                    append_jsonl(
                        args.progress_jsonl,
                        {"event": "case", "index": case_index, **row},
                    )
                    if not row["passed"]:
                        failures.append(
                            f"D={head_dim} M={m} N={n} causal={causal} kind={kind}"
                        )

    torch.manual_seed(421)
    parity_q = torch.randn(1, 64, 2, 128, device=device, dtype=torch.float16)
    parity_k = torch.randn_like(parity_q)
    parity_v = torch.randn_like(parity_q)
    implicit = flash_attn_func(parity_q, parity_k, parity_v, dropout_p=0.0)
    explicit = flash_attn_func(
        parity_q,
        parity_k,
        parity_v,
        dropout_p=0.0,
        softmax_scale=1.0 / math.sqrt(128),
    )
    parity_error = _max_abs(implicit, explicit)
    scale_parity = {
        "implicit": tensor_stats(implicit),
        "explicit": tensor_stats(explicit),
        "max_abs": parity_error,
        "passed": bool(torch.isfinite(implicit).all().item()) and parity_error == 0.0,
    }
    if not scale_parity["passed"]:
        failures.append("implicit and explicit default softmax scale differed")

    benchmarks: list[dict[str, Any]] = []
    if not args.skip_benchmark:
        for length in (4112, 4128):
            benchmark = _benchmark_aligned(
                flash_attn_func,
                length=length,
                warmup=args.benchmark_warmup,
                runs=args.benchmark_runs,
                rounds=args.benchmark_rounds,
                device=device,
            )
            benchmarks.append(benchmark)
            append_jsonl(args.progress_jsonl, {"event": "benchmark", **benchmark})

    accepted = not failures if args.expect_fixed else None
    report = {
        "schema_version": 1,
        "candidate": args.candidate,
        "source_sha": args.source_sha,
        "wheel_sha": args.wheel_sha,
        "environment": environment_manifest(device),
        "matrix": {
            "head_dims": head_dims,
            "heads": args.heads,
            "base_seq": args.base_seq,
            "benchmark_only": args.benchmark_only,
            "case_count": len(cases),
            "passed_count": sum(bool(row["passed"]) for row in cases),
            "failed_count": sum(not bool(row["passed"]) for row in cases),
            "cases": cases,
        },
        "scale_parity": scale_parity,
        "aligned_benchmarks": benchmarks,
        "expect_fixed": args.expect_fixed,
        "accepted": accepted,
        "acceptance_failures": failures,
    }
    if args.output is None:
        run_dir = make_run_dir("v100_flash", label=f"{args.candidate}-tail-matrix")
        output = run_dir / "tail_matrix.json"
    else:
        output = args.output
    write_json(output, report)
    print(f"wrote {output}")
    print(f"cases={len(cases)} failures={len(failures)} accepted={accepted}")
    for failure in failures[:20]:
        print(f"FAIL: {failure}")
    if len(failures) > 20:
        print(f"... {len(failures) - 20} additional failures")
    return 0 if accepted is not False else 2


if __name__ == "__main__":
    raise SystemExit(main())
