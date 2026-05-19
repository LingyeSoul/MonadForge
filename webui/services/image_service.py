"""Image / caption / version-history service — no Qt dependencies.

Extracts the logic from ``gui/tabs/image_tab.py`` so the WebUI can serve
dataset browsing, caption editing, and mask overlay over REST.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from webui.services.config_service import ROOT

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# Allow safe relative paths only — no traversal.
_SAFE_REL = re.compile(r"^[a-zA-Z0-9_./-]+$")

_MASK_SEARCH_ROOTS = [
    ROOT / "post_image_dataset" / "masks",
    ROOT / "masks" / "merged",
]


# ── directory discovery ─────────────────────────────────────────


def list_directories() -> list[dict[str, str]]:
    """Return available dataset directories (mirrors ``gui._image_dirs``)."""
    candidates = [
        ("image_dataset", ROOT / "image_dataset"),
        ("post_image_dataset/resized", ROOT / "post_image_dataset" / "resized"),
        ("ip-adapter-dataset", ROOT / "ip-adapter-dataset"),
        ("easycontrol-dataset", ROOT / "easycontrol-dataset"),
        ("output/tests", ROOT / "output" / "tests"),
    ]
    return [
        {"name": name, "path": str(path)} for name, path in candidates if path.exists()
    ]


def resolve_directory(name: str) -> Path | None:
    """Resolve a directory name to an absolute path.

    Accepts the short names returned by :func:`list_directories` as well as
    absolute paths for user-added directories.
    """
    candidates = {
        "image_dataset": ROOT / "image_dataset",
        "post_image_dataset/resized": ROOT / "post_image_dataset" / "resized",
        "ip-adapter-dataset": ROOT / "ip-adapter-dataset",
        "easycontrol-dataset": ROOT / "easycontrol-dataset",
        "output/tests": ROOT / "output" / "tests",
    }
    if name in candidates:
        p = candidates[name]
        return p if p.exists() else None
    # Absolute path fallback (user-added directory)
    p = Path(name)
    return p if p.exists() and p.is_dir() else None


# ── image scanning ──────────────────────────────────────────────


def _scan_images(directory: Path) -> list[Path]:
    """Return every image file under *directory* (recursively), sorted."""
    if not directory.exists():
        return []
    return sorted(
        p
        for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def list_images(
    directory: str,
    search: str = "",
    sort_desc: bool = False,
    page: int = 1,
    page_size: int = 50,
) -> dict[str, Any]:
    """Paginated image listing with optional search filter.

    Returns ``{"items": [...], "total": N, "page": P, "pages": N}``.
    Each item is an ``ImageInfo``-like dict with *relative* path (used as
    the key for caption / version / mask endpoints).
    """
    base = resolve_directory(directory)
    if base is None:
        return {"items": [], "total": 0, "page": 1, "pages": 0}

    all_images = _scan_images(base)

    # Filter
    q = search.strip().lower()
    if q:
        all_images = [p for p in all_images if q in _display_label(p, base).lower()]

    # Sort
    if sort_desc:
        all_images = list(reversed(all_images))

    total = len(all_images)
    pages = max(1, (total + page_size - 1) // page_size)
    page = max(1, min(page, pages))
    start = (page - 1) * page_size
    page_images = all_images[start : start + page_size]

    items = [_image_info(p, base) for p in page_images]
    return {"items": items, "total": total, "page": page, "pages": pages}


def _display_label(p: Path, base: Path) -> str:
    try:
        rel = p.relative_to(base)
    except ValueError:
        return p.stem
    if rel.parent == Path("."):
        return p.stem
    return f"{rel.parent.as_posix()}/{p.stem}"


def _image_info(p: Path, base: Path) -> dict[str, Any]:
    rel = str(p.relative_to(base)).replace("\\", "/")
    caption_path = p.with_suffix(".txt")
    mask_path = resolve_mask_path(p, base)
    return {
        "path": rel,
        "filename": p.name,
        "stem": p.stem,
        "caption": caption_path.read_text(encoding="utf-8")
        if caption_path.exists()
        else None,
        "has_mask": mask_path is not None,
    }


# ── image file resolution ──────────────────────────────────────


def resolve_image_path(directory: str, rel_path: str) -> Path | None:
    """Safely resolve a relative image path under *directory*.

    Returns ``None`` if the path doesn't point to an existing image file.
    """
    base = resolve_directory(directory)
    if base is None:
        return None
    if ".." in rel_path or not _SAFE_REL.match(rel_path):
        return None
    candidate = (base / rel_path).resolve()
    # Ensure the resolved path is still under base
    try:
        candidate.relative_to(base)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


# ── caption CRUD ───────────────────────────────────────────────


def get_caption(directory: str, rel_path: str) -> dict[str, Any]:
    """Read the caption (.txt sidecar) for an image."""
    img = resolve_image_path(directory, rel_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {rel_path}")
    cp = img.with_suffix(".txt")
    content = cp.read_text(encoding="utf-8") if cp.exists() else ""
    return {"path": rel_path, "content": content}


def save_caption(directory: str, rel_path: str, content: str) -> dict[str, Any]:
    """Write caption + append previous on-disk text to history."""
    img = resolve_image_path(directory, rel_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {rel_path}")
    cp = img.with_suffix(".txt")

    # Snapshot current disk text before overwriting
    if cp.exists():
        prev = cp.read_text(encoding="utf-8")
        if prev != content:
            _append_history(cp, prev)
    cp.write_text(content, encoding="utf-8")
    return {"path": rel_path, "content": content}


# ── version history (JSONL) ────────────────────────────────────


def _history_path(caption_path: Path) -> Path:
    return caption_path.with_suffix(caption_path.suffix + ".history.jsonl")


def _read_history(caption_path: Path) -> list[dict]:
    hp = _history_path(caption_path)
    if not hp.exists():
        return []
    out: list[dict] = []
    for line in hp.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict) and "ts" in entry and "text" in entry:
            out.append(entry)
    return out


def _append_history(caption_path: Path, prev_text: str) -> None:
    hp = _history_path(caption_path)
    entry = {"ts": datetime.now().isoformat(timespec="seconds"), "text": prev_text}
    with hp.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def get_versions(directory: str, rel_path: str) -> list[dict]:
    """Return caption version history (newest first)."""
    img = resolve_image_path(directory, rel_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {rel_path}")
    cp = img.with_suffix(".txt")
    history = _read_history(cp)
    history.reverse()  # newest first
    return history


# ── mask resolution ────────────────────────────────────────────


def resolve_mask_path(image_path: Path, base: Path | None = None) -> Path | None:
    """Locate the merged mask PNG for *image_path*.

    Mirrors ``gui.tabs.image_tab._resolve_mask_path``.
    """
    if base is None:
        try:
            base = image_path.parent
            _ = image_path.relative_to(base)
        except ValueError:
            return None
    try:
        rel = image_path.relative_to(base)
    except ValueError:
        return None
    rel_parent = rel.parent
    name = f"{image_path.stem}_mask.png"
    for mask_root in _MASK_SEARCH_ROOTS:
        candidate = mask_root / rel_parent / name
        if candidate.is_file():
            return candidate
    return None


def get_mask_info(directory: str, rel_path: str) -> dict[str, Any]:
    """Return mask path info for an image (or ``{"has_mask": false}``)."""
    img = resolve_image_path(directory, rel_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {rel_path}")
    base = resolve_directory(directory)
    mask = resolve_mask_path(img, base)
    return {
        "image_path": rel_path,
        "has_mask": mask is not None,
        "mask_path": str(mask) if mask else None,
    }
