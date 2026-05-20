"""Merge service — adapter directory listing and bakeability scanning."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from webui.services.config_service import ROOT


def list_adapter_dirs() -> list[dict]:
    """List directories likely to contain .safetensors adapter files."""
    candidates = [
        ("output/ckpt", ROOT / "output" / "ckpt"),
        ("output_temp", ROOT / "output_temp"),
        ("models/diffusion_models", ROOT / "models" / "diffusion_models"),
    ]
    dirs: list[dict] = []
    seen: set[str] = set()

    for name, path in candidates:
        if path.exists() and any(path.glob("*.safetensors")):
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
                and any(p.glob("*.safetensors"))
            ):
                key = str(p)
                if key not in seen:
                    dirs.append({"name": f"{label}/{p.name}", "path": key})
                    seen.add(key)

    return dirs


def list_files(dir_path: str) -> list[dict]:
    """List .safetensors files in a directory, newest first."""
    d = Path(dir_path)
    if not d.is_absolute():
        d = ROOT / d
    if not d.is_dir():
        return []
    files = []
    for p in sorted(
        (f for f in d.iterdir() if f.is_file() and f.suffix == ".safetensors"),
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
    """Scan a .safetensors file for bakeability.

    Classifies keys into families and returns a severity verdict.
    """
    try:
        from safetensors import safe_open
    except ImportError:
        return {"error": "safetensors package not installed", "verdict": "unknown"}

    p = Path(file_path)
    if not p.is_absolute():
        p = ROOT / file_path
    if not p.is_file():
        return {"error": f"File not found: {file_path}", "verdict": "unknown"}

    try:
        with safe_open(str(p), framework="numpy") as f:
            keys = list(f.keys())
            metadata = f.metadata() or {}
    except Exception as e:
        return {"error": str(e), "verdict": "unknown"}

    # Classify keys
    counts = {
        "lora_down": 0,
        "ortho_sp": 0,
        "dora": 0,
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
    has_lora_like = counts["lora_down"] > 0 or counts["ortho_sp"] > 0
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
