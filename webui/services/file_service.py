"""File browser service — directory listing and path validation for model file picker."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from webui.services.config_service import ROOT

_MODEL_EXTS = {".safetensors", ".pt", ".pth", ".bin", ".ckpt"}


def _resolve(dir_path: str) -> Path | None:
    """Resolve a directory path relative to ROOT or as absolute.

    Returns None if the path doesn't exist or isn't a directory.
    Rejects ``..`` components in relative paths for safety.
    """
    p = Path(dir_path)
    if p.is_absolute():
        return p if p.is_dir() else None
    if ".." in p.parts:
        return None
    resolved = (ROOT / p).resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError:
        return None
    return resolved if resolved.is_dir() else None


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} TB"


def browse_directory(dir_path: str, ext: str | None = None) -> dict:
    """List subdirectories and files in *dir_path*.

    Parameters
    ----------
    dir_path:
        Relative to ROOT or absolute. Empty string falls back to ROOT.
    ext:
        If set, only include files with this extension (e.g. ``.safetensors``).

    Returns
    -------
    dict with keys ``current_dir``, ``parent``, ``subdirs``, ``files``.
    """
    if not dir_path or not dir_path.strip():
        dir_path = str(ROOT)
    resolved = _resolve(dir_path)
    if resolved is None:
        return {"current_dir": dir_path, "parent": None, "subdirs": [], "files": [], "error": "Directory not found"}

    subdirs = []
    for p in sorted(resolved.iterdir()):
        if p.is_dir() and not p.name.startswith("."):
            subdirs.append({"name": p.name, "path": str(p)})

    files = []
    for p in sorted(
        (f for f in resolved.iterdir() if f.is_file()),
        key=lambda f: f.name,
    ):
        if ext and p.suffix != ext:
            continue
        if not ext and p.suffix not in _MODEL_EXTS:
            continue
        stat = p.stat()
        size = stat.st_size
        mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
        files.append({
            "name": p.name,
            "path": str(p),
            "size": size,
            "size_human": _human_size(size),
            "mtime": mtime,
        })

    parent = str(resolved.parent) if resolved.parent != resolved else None
    try:
        rel = str(resolved.relative_to(ROOT))
    except ValueError:
        rel = str(resolved)

    return {
        "current_dir": rel,
        "current_dir_abs": str(resolved),
        "parent": parent,
        "subdirs": subdirs,
        "files": files,
    }


def validate_model_path(path: str) -> dict:
    """Check whether a model file exists at *path*.

    Parameters
    ----------
    path:
        Relative to ROOT or absolute.

    Returns
    -------
    dict with ``exists`` (bool) and ``resolved`` (str).
    """
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return {"exists": p.is_file(), "resolved": str(p)}
