"""Merge service — adapter directory listing and bakeability scanning."""

from __future__ import annotations

from datetime import datetime, timezone

from webui.services.config_service import ROOT
from webui.services.paths import resolve_path


def list_adapter_dirs() -> list[dict]:
    """List directories likely to contain supported adapter weights."""
    candidates = [
        ("output/ckpt", ROOT / "output" / "ckpt"),
        ("output_temp", ROOT / "output_temp"),
        ("models/diffusion_models", ROOT / "models" / "diffusion_models"),
    ]
    dirs: list[dict] = []
    seen: set[str] = set()

    from library.io.output_layout import discover_weights

    for name, path in candidates:
        if path.exists() and discover_weights(path):
            dirs.append({"name": name, "path": str(path)})
            seen.add(str(path))

    # Subdirectories of output/ckpt/ and output_temp/
    for parent, label in [
        (ROOT / "output" / "ckpt", "output/ckpt"),
        (ROOT / "output_temp", "output_temp"),
    ]:
        if not parent.exists():
            continue
        for p in sorted(parent.iterdir()):
            if (
                p.is_dir()
                and not p.name.endswith("-checkpoint-state")
                and discover_weights(p)
            ):
                key = str(p)
                if key not in seen:
                    dirs.append({"name": f"{label}/{p.name}", "path": key})
                    seen.add(key)

    return dirs


def list_files(dir_path: str) -> list[dict]:
    """List supported adapter weight files in a directory, newest first."""
    d = resolve_path(dir_path, expect_file=False)
    if d is None:
        return []
    files = []
    from library.io.output_layout import (
        OUTPUT_WEIGHT_EXTENSIONS,
        is_checkpoint_weight,
        read_run_manifest,
        resolve_manifest_path,
    )
    manifest = read_run_manifest(d)
    manifest_final = resolve_manifest_path(
        d / "run_manifest.json", (manifest or {}).get("final_weight")
    )
    # An incomplete/torn manifest must not hide otherwise readable legacy
    # weights. Once a manifest names a real final file it is authoritative.
    manifest_authoritative = manifest_final is not None and manifest_final.is_file()
    for p in sorted(
        (
            f
            for f in d.iterdir()
            if f.is_file()
            and f.suffix.lower() in OUTPUT_WEIGHT_EXTENSIONS
            and (
                f == manifest_final
                if manifest_authoritative
                else is_checkpoint_weight(f)
            )
        ),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    ):
        stat = p.stat()
        size = stat.st_size
        mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
        files.append(
            {
                "name": p.name,
                "path": str(p),
                "size": size,
                "size_human": _human_size(size),
                "mtime": mtime,
            }
        )
    return files


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} TB"


def scan_adapter(file_path: str) -> dict:
    """Scan an adapter weight file for bakeability.

    Classifies keys into families and returns a severity verdict.
    """
    p = resolve_path(file_path, expect_file=True)
    if p is None:
        return {"error": f"File not found: {file_path}", "verdict": "unknown"}

    try:
        if p.suffix.lower() == ".safetensors":
            from safetensors import safe_open

            with safe_open(str(p), framework="numpy") as f:
                keys = list(f.keys())
                metadata = f.metadata() or {}
        else:
            import torch

            state = torch.load(
                str(p),
                map_location="cpu",
                weights_only=True,
            )
            if not isinstance(state, dict):
                raise TypeError("checkpoint does not contain a state dict")
            nested = state.get("state_dict")
            if isinstance(nested, dict):
                state = nested
            keys = list(state.keys())
            metadata = {}
    except Exception as e:
        return {"error": str(e), "verdict": "unknown"}

    # Classify keys
    counts = {
        "lora_down": 0,
        "ortho_sp": 0,
        "dora": 0,
        "glokr": 0,
        "lora_ups": 0,
        "lora_up_weight": 0,
        "reft": 0,
        "postfix": 0,
        "other": 0,
    }

    for key in keys:
        if key.startswith("reft_"):
            counts["reft"] += 1
        elif ".lora_up_weight" in key:
            counts["lora_up_weight"] += 1
        elif ".lora_ups." in key:
            counts["lora_ups"] += 1
        elif key.endswith(".lora_down.weight"):
            counts["lora_down"] += 1
        elif key.endswith(".S_p"):
            counts["ortho_sp"] += 1
        elif ".glokr_w" in key or key.endswith((".bora_m_row", ".bora_m_col")):
            # GLoKr (Kronecker + BoRA weight decomposition) — bakeable via
            # GLoKRModule.merge_to (weight replacement + multiplier lerp).
            counts["glokr"] += 1
        elif key.endswith(".dora_scale") or key.endswith(".magnitude"):
            counts["dora"] += 1
        else:
            counts["other"] += 1

    # Check metadata for postfix
    ss_mode = metadata.get("ss_mode", "")
    if ss_mode in ("postfix", "postfix_exp", "postfix_func", "cond"):
        counts["postfix"] = len(keys)

    # Severity verdict
    has_hydra = counts["lora_up_weight"] > 0 or counts["lora_ups"] > 0
    has_lora_like = (
        counts["lora_down"] > 0 or counts["ortho_sp"] > 0 or counts["glokr"] > 0
    )
    has_reft = counts["reft"] > 0
    has_postfix = counts["postfix"] > 0

    if has_hydra:
        verdict = "block"
    elif has_postfix and not has_lora_like:
        verdict = "block"
    elif has_reft and not has_lora_like:
        verdict = "block"
    elif has_reft and has_lora_like:
        verdict = "partial"
    elif has_lora_like:
        verdict = "ok"
    else:
        verdict = "unknown"

    return {
        "verdict": verdict,
        "counts": {k: v for k, v in counts.items() if v > 0},
        "total_keys": len(keys),
        "metadata": {k: v for k, v in (metadata or {}).items() if k.startswith("ss_")},
    }
