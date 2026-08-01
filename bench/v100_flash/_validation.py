"""Shared helpers for the V100 FlashAttention issue-43 validation tools."""

from __future__ import annotations

import hashlib
import importlib
import json
import platform
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return output


def append_jsonl(path: str | Path | None, payload: Mapping[str, Any]) -> None:
    if path is None:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, allow_nan=False) + "\n")


def environment_manifest(device: torch.device) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(device)
    manifest: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": str(device),
        "gpu": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
    }
    try:
        flash_attn = importlib.import_module("flash_attn")
        manifest["flash_attn_version"] = getattr(flash_attn, "__version__", None)
        manifest["flash_attn_doc"] = getattr(flash_attn, "__doc__", None)
        manifest["flash_attn_path"] = getattr(flash_attn, "__file__", None)
    except ImportError:
        manifest["flash_attn_version"] = None
    try:
        backend = importlib.import_module("flash_attn_v100")
        manifest["flash_attn_v100_version"] = getattr(backend, "__version__", None)
        manifest["flash_attn_v100_path"] = getattr(backend, "__file__", None)
    except ImportError:
        manifest["flash_attn_v100_version"] = None
    try:
        extension = importlib.import_module("flash_attn_v100_cuda")
        extension_path = getattr(extension, "__file__", None)
        manifest["flash_attn_extension_path"] = extension_path
        manifest["flash_attn_extension_sha256"] = (
            sha256_file(extension_path) if extension_path else None
        )
    except ImportError:
        manifest["flash_attn_extension_path"] = None
        manifest["flash_attn_extension_sha256"] = None
    try:
        dispatcher = importlib.import_module("networks.attention_dispatch")
        manifest["v100_compile_compat_active"] = bool(
            getattr(dispatcher, "flash_attn_v100_compat_active", False)
        )
    except ImportError:
        manifest["v100_compile_compat_active"] = False
    return manifest


def resolve_device(value: str | torch.device) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def require_v100(device: torch.device) -> None:
    if device.type != "cuda":
        raise SystemExit("issue-43 validation requires --device cuda")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    props = torch.cuda.get_device_properties(device)
    if (props.major, props.minor) != (7, 0):
        raise SystemExit(
            f"issue-43 validation requires SM 7.0, got {props.major}.{props.minor}"
        )


def _find_tensor(payload: Mapping[str, Any], name: str) -> torch.Tensor | None:
    value = payload.get(name)
    if isinstance(value, torch.Tensor):
        return value
    for group_name in ("tensors", "inputs", "attention", "capture", "payload"):
        group = payload.get(group_name)
        if isinstance(group, Mapping):
            value = group.get(name)
            if isinstance(value, torch.Tensor):
                return value
    return None


def load_capture(
    path: str | Path,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
    capture_path = Path(path)
    payload = torch.load(capture_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError(f"capture root must be a mapping, got {type(payload).__name__}")
    q = _find_tensor(payload, "q")
    k = _find_tensor(payload, "k")
    v = _find_tensor(payload, "v")
    missing = [
        name for name, tensor in (("q", q), ("k", k), ("v", v)) if tensor is None
    ]
    if missing:
        raise KeyError(f"capture is missing tensors: {', '.join(missing)}")
    assert q is not None and k is not None and v is not None
    metadata = {
        "schema_version": payload.get("schema_version"),
        "failure": payload.get("failure"),
        "metadata": payload.get("metadata"),
        "stats": payload.get("stats"),
        "environment": payload.get("environment"),
    }
    return metadata, q, k, v


def tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    detached = tensor.detach()
    finite = torch.isfinite(detached)
    nan_count = int(torch.isnan(detached).sum().item())
    pos_inf_count = int(torch.isposinf(detached).sum().item())
    neg_inf_count = int(torch.isneginf(detached).sum().item())
    finite_count = int(finite.sum().item())
    result: dict[str, Any] = {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "finite": finite_count == detached.numel(),
        "finite_count": finite_count,
        "nan_count": nan_count,
        "pos_inf_count": pos_inf_count,
        "neg_inf_count": neg_inf_count,
    }
    if finite_count:
        values = detached[finite]
        result["finite_min"] = float(values.min().item())
        result["finite_max"] = float(values.max().item())
    else:
        result["finite_min"] = None
        result["finite_max"] = None
    return result


def compare_tensors(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    left = candidate.detach().float()
    right = reference.detach().float()
    if left.shape != right.shape:
        return {
            "shape_match": False,
            "candidate_shape": list(left.shape),
            "reference_shape": list(right.shape),
        }
    left_finite = torch.isfinite(left)
    right_finite = torch.isfinite(right)
    both = left_finite & right_finite
    result: dict[str, Any] = {
        "shape_match": True,
        "same_nonfinite_mask": bool(torch.equal(left_finite, right_finite)),
        "finite_pair_count": int(both.sum().item()),
    }
    if bool(both.any().item()):
        diff = (left[both] - right[both]).abs()
        result["max_abs"] = float(diff.max().item())
        result["mean_abs"] = float(diff.mean().item())
    else:
        result["max_abs"] = None
        result["mean_abs"] = None
    return result


def torch_sdpa_blhd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    causal: bool = False,
) -> torch.Tensor:
    output = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        dropout_p=0.0,
        is_causal=causal,
        scale=softmax_scale,
    )
    return output.transpose(1, 2).contiguous()


def run_cuda_path(
    name: str,
    fn: Callable[[], torch.Tensor],
    device: torch.device,
) -> tuple[dict[str, Any], torch.Tensor | None]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    try:
        output = fn()
        torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        metrics = {
            "name": name,
            "ok": True,
            "elapsed_ms": round(elapsed_ms, 3),
            "peak_allocated_mib": round(
                torch.cuda.max_memory_allocated(device) / (1024**2), 3
            ),
            "peak_reserved_mib": round(
                torch.cuda.max_memory_reserved(device) / (1024**2), 3
            ),
            "stats": tensor_stats(output),
        }
        return metrics, output
    except Exception as exc:  # noqa: BLE001 - each backend path must be reported independently.
        torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return (
            {
                "name": name,
                "ok": False,
                "elapsed_ms": round(elapsed_ms, 3),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
            None,
        )
