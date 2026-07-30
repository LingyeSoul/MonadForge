"""Replay an issue-43 Q/K/V capture through raw and MonadForge attention paths."""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
from typing import Any

import torch

from bench._common import make_run_dir
from bench.v100_flash._validation import (
    compare_tensors,
    environment_manifest,
    load_capture,
    require_v100,
    resolve_device,
    run_cuda_path,
    sha256_file,
    tensor_stats,
    torch_sdpa_blhd,
    write_json,
)

EXPECTED_CAPTURE_SHA256 = (
    "91f67505dd66914718a3de61d361c71a1d621fcda46f0fa0d43731d11f05fa0d"
)
EXPECTED_BAD_COUNTS = {"nan_count": 1412224, "pos_inf_count": 562, "neg_inf_count": 590}
DEFAULT_FULL_REPEATS = 10


def _default_output(candidate: str) -> Path:
    run_dir = make_run_dir("v100_flash", label=f"{candidate}-capture")
    return run_dir / "replay.json"


def _run_prefix_path(name, fn, device: torch.device):
    try:
        output = fn()
        torch.cuda.synchronize(device)
        return {"name": name, "ok": True, "stats": tensor_stats(output)}, output
    except Exception as exc:  # noqa: BLE001 - retain per-path integration errors.
        torch.cuda.synchronize(device)
        return {
            "name": name,
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }, None


def _acceptance(
    expectation: str,
    replays: dict[str, dict[str, Any]],
    prefix_sweep: list[dict[str, Any]],
    repeat_replays: dict[str, list[dict[str, Any]]] | None = None,
) -> tuple[bool | None, bool | None, bool | None, list[str], list[str]]:
    if expectation == "none":
        return None, None, None, [], []
    numeric_failures: list[str] = []
    integration_failures: list[str] = []
    torch_stats = replays.get("torch_sdpa_fp16", {}).get("stats", {})
    if not torch_stats.get("finite"):
        numeric_failures.append("Torch SDPA FP16 reference was not finite")
    bad_sdpa_prefixes = [
        row["length"] for row in prefix_sweep if not row["sdpa"]["finite"]
    ]
    if bad_sdpa_prefixes:
        numeric_failures.append(
            f"Torch SDPA returned non-finite prefixes: {bad_sdpa_prefixes}"
        )

    raw = replays.get("raw_flash_eager", {})
    if expectation == "known-bad":
        stats = raw.get("stats", {})
        if stats.get("finite") is not False:
            numeric_failures.append(
                "published wheel did not reproduce a non-finite full capture"
            )
        for key, expected in EXPECTED_BAD_COUNTS.items():
            if stats.get(key) != expected:
                numeric_failures.append(
                    f"published wheel {key}={stats.get(key)!r}, expected {expected}"
                )
        finite_lengths = [
            row["length"] for row in prefix_sweep if row["flash"]["finite"]
        ]
        if finite_lengths != [4112, 4128]:
            numeric_failures.append(
                f"known-bad finite prefix lengths were {finite_lengths}, "
                "expected [4112, 4128]"
            )
        for name in ("compat_flash_eager", "compat_flash_compiled"):
            replay = replays.get(name, {})
            if not replay.get("ok"):
                integration_failures.append(f"{name} failed: {replay.get('error')}")
                continue
            parity = replay.get("vs_raw_flash", {})
            if not parity.get("same_nonfinite_mask"):
                integration_failures.append(
                    f"{name} did not reproduce the raw Flash non-finite mask"
                )
    else:
        for name in ("raw_flash_eager", "compat_flash_eager"):
            replay = replays.get(name, {})
            if not replay.get("ok"):
                numeric_failures.append(f"{name} failed: {replay.get('error')}")
                continue
            if not replay.get("stats", {}).get("finite"):
                numeric_failures.append(f"{name} returned non-finite output")
            tolerance = replay.get("vs_fp32_tolerance", {})
            if tolerance and not tolerance.get("passed"):
                numeric_failures.append(
                    f"{name} exceeded the upstream FP16-relative tolerance"
                )

        compiled = replays.get("compat_flash_compiled", {})
        if not compiled.get("ok"):
            integration_failures.append(
                f"compat_flash_compiled failed: {compiled.get('error')}"
            )
        else:
            if not compiled.get("stats", {}).get("finite"):
                integration_failures.append(
                    "compat_flash_compiled returned non-finite output"
                )
            tolerance = compiled.get("vs_fp32_tolerance", {})
            if tolerance and not tolerance.get("passed"):
                integration_failures.append(
                    "compat_flash_compiled exceeded the upstream FP16-relative tolerance"
                )
        for name in ("raw_flash_eager", "compat_flash_eager"):
            bad_prefixes = [
                row["length"]
                for row in prefix_sweep
                if not row.get("paths", {}).get(name, {}).get("stats", {}).get("finite")
            ]
            if bad_prefixes:
                numeric_failures.append(
                    f"{name} had failed or non-finite prefixes: {bad_prefixes}"
                )
        compiled_bad_prefixes = [
            row["length"]
            for row in prefix_sweep
            if not row.get("paths", {})
            .get("compat_flash_compiled", {})
            .get("stats", {})
            .get("finite")
        ]
        if compiled_bad_prefixes:
            integration_failures.append(
                "compat_flash_compiled had failed or non-finite prefixes: "
                f"{compiled_bad_prefixes}"
            )
        for name in ("raw_flash_eager", "compat_flash_eager"):
            failed_repeats = [
                int(row.get("repeat", -1))
                for row in (repeat_replays or {}).get(name, [])
                if not row.get("ok")
                or not row.get("stats", {}).get("finite")
                or not row.get("vs_fp32_tolerance", {}).get("passed", True)
            ]
            if failed_repeats:
                numeric_failures.append(
                    f"{name} repeated full-capture runs failed: {failed_repeats}"
                )
        failed_compiled_repeats = [
            int(row.get("repeat", -1))
            for row in (repeat_replays or {}).get("compat_flash_compiled", [])
            if not row.get("ok")
            or not row.get("stats", {}).get("finite")
            or not row.get("vs_fp32_tolerance", {}).get("passed", True)
        ]
        if failed_compiled_repeats:
            integration_failures.append(
                "compat_flash_compiled repeated full-capture runs failed: "
                f"{failed_compiled_repeats}"
            )

    numeric_accepted = not numeric_failures
    integration_accepted = not integration_failures
    return (
        numeric_accepted and integration_accepted,
        numeric_accepted,
        integration_accepted,
        numeric_failures,
        integration_failures,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--candidate", default="candidate")
    parser.add_argument("--source-sha")
    parser.add_argument("--wheel-sha")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prefix-start", type=int, default=4112)
    parser.add_argument("--prefix-end", type=int, default=4128)
    parser.add_argument("--skip-compiled", action="store_true")
    parser.add_argument(
        "--compiled-fullgraph",
        action="store_true",
        help="Require the MonadForge path to compile without any Dynamo graph break.",
    )
    parser.add_argument("--skip-fp32-reference", action="store_true")
    parser.add_argument(
        "--full-repeats",
        type=int,
        default=DEFAULT_FULL_REPEATS,
        help="Repeat each Flash full-capture path to detect nondeterministic output.",
    )
    parser.add_argument(
        "--expect", choices=("none", "known-bad", "fixed"), default="none"
    )
    args = parser.parse_args()
    if args.full_repeats < 1:
        parser.error("--full-repeats must be positive")

    device = resolve_device(args.device)
    require_v100(device)
    torch.cuda.set_device(device)
    capture_path = Path(args.capture).resolve()
    capture_sha = sha256_file(capture_path)
    if capture_sha != EXPECTED_CAPTURE_SHA256:
        raise SystemExit(
            f"capture SHA-256 mismatch: {capture_sha}, expected {EXPECTED_CAPTURE_SHA256}"
        )

    metadata, q_cpu, k_cpu, v_cpu = load_capture(capture_path)
    q = q_cpu.to(device=device, dtype=torch.float16).contiguous()
    k = k_cpu.to(device=device, dtype=torch.float16).contiguous()
    v = v_cpu.to(device=device, dtype=torch.float16).contiguous()
    del q_cpu, k_cpu, v_cpu
    if q.shape != k.shape or q.shape != v.shape:
        raise SystemExit(
            f"capture Q/K/V shapes differ: {q.shape}, {k.shape}, {v.shape}"
        )
    if q.ndim != 4 or q.shape[-1] != 128:
        raise SystemExit(f"unexpected capture shape: {tuple(q.shape)}")
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if not bool(torch.isfinite(tensor).all().item()):
            raise SystemExit(f"capture {name} is not finite")

    from flash_attn import flash_attn_func

    from networks.attention_dispatch import AttentionParams, dispatch_attention

    scale = None
    raw_metrics, raw_output = run_cuda_path(
        "raw_flash_eager",
        lambda: flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=scale),
        device,
    )

    reference_metrics, reference = run_cuda_path(
        "torch_sdpa_fp16",
        lambda: torch_sdpa_blhd(q, k, v, softmax_scale=scale),
        device,
    )
    if reference is None:
        raise SystemExit(f"Torch SDPA reference failed: {reference_metrics}")

    fp32_metrics: dict[str, Any] | None = None
    fp32_reference: torch.Tensor | None = None
    if not args.skip_fp32_reference:
        fp32_metrics, fp32_reference = run_cuda_path(
            "torch_sdpa_fp32",
            lambda: torch_sdpa_blhd(
                q.float(), k.float(), v.float(), softmax_scale=scale
            ).half(),
            device,
        )

    params = AttentionParams.create_attention_params("flash", softmax_scale=scale)

    def compat_call(
        q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor
    ) -> torch.Tensor:
        flattened = dispatch_attention(q_in, k_in, v_in, params, 0.0)
        return flattened.view(
            q_in.shape[0], q_in.shape[1], q_in.shape[2], q_in.shape[3]
        )

    compat_metrics, compat_output = run_cuda_path(
        "compat_flash_eager", lambda: compat_call(q, k, v), device
    )
    compiled_metrics: dict[str, Any]
    compiled_output: torch.Tensor | None
    compiled_call = None
    if args.skip_compiled:
        compiled_metrics = {
            "name": "compat_flash_compiled",
            "ok": False,
            "skipped": True,
        }
        compiled_output = None
    else:
        compiled_call = torch.compile(
            compat_call,
            backend="inductor",
            fullgraph=args.compiled_fullgraph,
        )
        compiled_metrics, compiled_output = run_cuda_path(
            "compat_flash_compiled", lambda: compiled_call(q, k, v), device
        )

    replay_rows = [reference_metrics]
    if fp32_metrics is not None:
        replay_rows.append(fp32_metrics)
    replay_rows.extend((raw_metrics, compat_metrics, compiled_metrics))
    outputs = {
        "raw_flash_eager": raw_output,
        "compat_flash_eager": compat_output,
        "compat_flash_compiled": compiled_output,
    }
    baseline_error = None
    if fp32_reference is not None:
        baseline_error = compare_tensors(reference, fp32_reference).get("max_abs")

    def annotate(row: dict[str, Any], output: torch.Tensor | None) -> None:
        if output is None:
            return
        row["vs_torch_sdpa"] = compare_tensors(output, reference)
        if raw_output is not None:
            row["vs_raw_flash"] = compare_tensors(output, raw_output)
        if fp32_reference is not None and baseline_error is not None:
            comparison = compare_tensors(output, fp32_reference)
            candidate_error = comparison.get("max_abs")
            limit = 2.0 * float(baseline_error) + 1e-5
            row["vs_fp32"] = comparison
            row["vs_fp32_tolerance"] = {
                "candidate_max_abs": candidate_error,
                "native_fp16_max_abs": baseline_error,
                "limit": limit,
                "passed": candidate_error is not None
                and float(candidate_error) <= limit,
            }

    for row in replay_rows:
        annotate(row, outputs.get(row["name"]))

    repeat_calls = {
        "raw_flash_eager": partial(
            flash_attn_func, q, k, v, dropout_p=0.0, softmax_scale=scale
        ),
        "compat_flash_eager": partial(compat_call, q, k, v),
    }
    if compiled_call is not None:
        repeat_calls["compat_flash_compiled"] = partial(compiled_call, q, k, v)
    repeat_replays: dict[str, list[dict[str, Any]]] = {}
    for name, call in repeat_calls.items():
        rows: list[dict[str, Any]] = []
        for repeat in range(args.full_repeats):
            row, output = run_cuda_path(name, call, device)
            row["repeat"] = repeat
            annotate(row, output)
            rows.append(row)
        repeat_replays[name] = rows

    prefix_sweep: list[dict[str, Any]] = []
    for length in range(args.prefix_start, args.prefix_end + 1):
        q_prefix = q[:, :length]
        k_prefix = k[:, :length]
        v_prefix = v[:, :length]
        sdpa_metrics, sdpa_out = _run_prefix_path(
            "torch_sdpa_fp16",
            partial(torch_sdpa_blhd, q_prefix, k_prefix, v_prefix),
            device,
        )
        path_calls = {
            "raw_flash_eager": partial(
                flash_attn_func, q_prefix, k_prefix, v_prefix, dropout_p=0.0
            ),
            "compat_flash_eager": partial(compat_call, q_prefix, k_prefix, v_prefix),
        }
        if compiled_call is not None:
            path_calls["compat_flash_compiled"] = partial(
                compiled_call, q_prefix, k_prefix, v_prefix
            )

        path_rows: dict[str, dict[str, Any]] = {}
        for name, call in path_calls.items():
            metrics, output = _run_prefix_path(name, call, device)
            if output is not None and sdpa_out is not None:
                metrics["vs_torch_sdpa"] = compare_tensors(output, sdpa_out)
            path_rows[name] = metrics
            del output
        if compiled_call is None:
            path_rows["compat_flash_compiled"] = {
                "name": "compat_flash_compiled",
                "ok": False,
                "skipped": True,
            }

        raw_stats = path_rows["raw_flash_eager"].get("stats", {"finite": False})
        sdpa_stats = sdpa_metrics.get("stats", {"finite": False})
        prefix_sweep.append(
            {
                "length": length,
                "residue": length % 16,
                "paths": path_rows,
                "flash": raw_stats,
                "sdpa": sdpa_stats,
                "flash_vs_sdpa": path_rows["raw_flash_eager"].get("vs_torch_sdpa"),
            }
        )
        del sdpa_out

    replay_map = {row["name"]: row for row in replay_rows}
    (
        accepted,
        numeric_accepted,
        integration_accepted,
        numeric_failures,
        integration_failures,
    ) = _acceptance(args.expect, replay_map, prefix_sweep, repeat_replays)
    failures = numeric_failures + integration_failures
    report = {
        "schema_version": 1,
        "candidate": args.candidate,
        "source_sha": args.source_sha,
        "wheel_sha": args.wheel_sha,
        "capture": str(capture_path),
        "capture_sha256": capture_sha,
        "capture_metadata": metadata,
        "environment": environment_manifest(device),
        "input_stats": {
            "q": tensor_stats(q),
            "k": tensor_stats(k),
            "v": tensor_stats(v),
        },
        "softmax_scale": scale,
        "full_repeats": args.full_repeats,
        "compiled_fullgraph": args.compiled_fullgraph,
        "replays": replay_rows,
        "repeat_replays": repeat_replays,
        "prefix_sweep": prefix_sweep,
        "expectation": args.expect,
        "accepted": accepted,
        "kernel_numeric_accepted": numeric_accepted,
        "compiled_integration_accepted": integration_accepted,
        "numeric_acceptance_failures": numeric_failures,
        "integration_acceptance_failures": integration_failures,
        "acceptance_failures": failures,
    }
    output_path = args.output or _default_output(args.candidate)
    write_json(output_path, report)
    print(f"wrote {output_path}")
    print(f"accepted={accepted} failures={len(failures)}")
    for failure in failures:
        print(f"FAIL: {failure}")
    return 0 if accepted is not False else 2


if __name__ == "__main__":
    raise SystemExit(main())
