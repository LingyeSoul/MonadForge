"""Image / caption / version-history service — no Qt dependencies.

Handles dataset browsing, caption editing, and mask overlay for the WebUI.
"""

from __future__ import annotations

import json
import re
import tarfile
import tempfile
import threading
import time
import uuid
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Callable

from library.captioning.position_clauses import (
    PositionClause,
    compose_caption,
    parse_caption,
)
from webui.services.config_service import ROOT, get_path_overrides

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

ARCHIVE_SUFFIXES = (
    ".zip",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".tar.bz2",
    ".tbz2",
    ".tar.xz",
    ".txz",
)

# Reject only the path-traversal vectors: NUL/control bytes and a ``..``
# segment. We deliberately do NOT use a positive character whitelist here:
# dataset filenames legitimately contain characters outside
# ``[a-zA-Z0-9_./-]`` (spaces, CJK, ``@``, parens, …), and the old whitelist
# silently broke browsing for such images — the file appeared in the listing
# (``list_images`` never applied it) but its ``<img>``/caption fetch 404'd.
# Relative paths MAY contain ``/`` (subdirectories, via ``rglob`` in
# ``list_images``), so separators are allowed; containment is enforced by the
# ``resolve() / relative_to(base)`` check in :func:`resolve_image_path`.
_UNSAFE_REL = re.compile(r"[\x00-\x1f]")
_caption_lock = threading.RLock()

_MASK_SEARCH_ROOTS_FALLBACK = [
    ROOT / "post_image_dataset" / "masks",
    ROOT / "masks" / "merged",
]


def _get_mask_search_roots() -> list[Path]:
    """Mask search roots derived from the config chain's ``resized_image_dir``."""
    paths = get_path_overrides()
    resized = Path(paths["resized_image_dir"])
    if not resized.is_absolute():
        resized = ROOT / resized
    return [
        resized.parent / "masks",
        ROOT / "masks" / "merged",
    ]


# ── directory discovery ─────────────────────────────────────────


def list_directories() -> list[dict[str, str]]:
    """Return available dataset directories (mirrors ``gui._image_dirs``).

    Includes paths from the config chain (``source_image_dir``,
    ``resized_image_dir``, ``lora_cache_dir``) alongside the fixed entries.
    """
    paths = get_path_overrides()
    seen: set[str] = set()
    candidates: list[tuple[str, Path]] = []

    # Config-driven paths (may be absolute or relative)
    for key in ("source_image_dir", "resized_image_dir", "lora_cache_dir"):
        raw = paths[key]
        p = Path(raw) if Path(raw).is_absolute() else ROOT / raw
        if raw not in seen:
            candidates.append((raw, p))
            seen.add(raw)

    # Fixed entries
    for name, path in [
        ("ip-adapter-dataset", ROOT / "ip-adapter-dataset"),
        ("easycontrol-dataset", ROOT / "easycontrol-dataset"),
        ("output/tests", ROOT / "output" / "tests"),
    ]:
        if name not in seen:
            candidates.append((name, path))
            seen.add(name)

    return [
        {"name": name, "path": str(path)} for name, path in candidates if path.exists()
    ]


def resolve_directory(name: str) -> Path | None:
    """Resolve a directory name to an absolute path.

    Accepts the short names returned by :func:`list_directories` as well as
    absolute paths for user-added directories.
    """
    paths = get_path_overrides()
    candidates: dict[str, Path] = {}

    # Config-driven paths
    for key in ("source_image_dir", "resized_image_dir", "lora_cache_dir"):
        raw = paths[key]
        candidates[raw] = Path(raw) if Path(raw).is_absolute() else ROOT / raw

    # Fixed entries
    candidates["ip-adapter-dataset"] = ROOT / "ip-adapter-dataset"
    candidates["easycontrol-dataset"] = ROOT / "easycontrol-dataset"
    candidates["output/tests"] = ROOT / "output" / "tests"

    if name in candidates:
        p = candidates[name]
        return p.resolve() if p.is_dir() else None
    # Absolute path fallback (user-added directory); relative names are
    # anchored to ROOT so uploaded/extracted dataset paths stay browsable.
    p = Path(name)
    if not p.is_absolute():
        if ".." in p.parts:
            return None
        p = ROOT / p
    return p.resolve() if p.is_dir() else None


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
    relative = Path(rel_path)
    if _UNSAFE_REL.search(rel_path) or relative.is_absolute() or ".." in relative.parts:
        return None
    candidate = (base / rel_path).resolve()
    # Ensure the resolved path is still under base. This is the real
    # containment guarantee — the regex above is defense-in-depth only.
    try:
        candidate.relative_to(base)
    except ValueError:
        return None
    return (
        candidate
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTS
        else None
    )


# ── caption CRUD ───────────────────────────────────────────────


def _caption_path(image: Path, directory: str) -> Path:
    base = resolve_directory(directory)
    caption = image.with_suffix(".txt").resolve()
    history = _history_path(caption).resolve()
    if (
        base is None
        or not caption.is_relative_to(base)
        or not history.is_relative_to(base)
    ):
        raise ValueError("Caption or history points outside the dataset directory")
    return caption


def get_caption(directory: str, rel_path: str) -> dict[str, Any]:
    """Read the caption (.txt sidecar) for an image."""
    img = resolve_image_path(directory, rel_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {rel_path}")
    cp = _caption_path(img, directory)
    content = cp.read_text(encoding="utf-8") if cp.exists() else ""
    return {"path": rel_path, "content": content}


def save_caption(directory: str, rel_path: str, content: str) -> dict[str, Any]:
    """Write caption + append previous on-disk text to history."""
    img = resolve_image_path(directory, rel_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {rel_path}")
    cp = _caption_path(img, directory)
    with _caption_lock:
        _save_caption_file(cp, content)
    return {"path": rel_path, "content": content}


def _save_caption_file(caption: Path, content: str) -> None:
    """Stage the complete caption before replacing it; preserve its old version."""
    previous = caption.read_text(encoding="utf-8") if caption.exists() else None
    if previous == content:
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=caption.parent, suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        if previous is not None:
            temporary.chmod(caption.stat().st_mode)
            _append_history(caption, previous)
        temporary.replace(caption)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


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
    cp = _caption_path(img, directory)
    history = _read_history(cp)
    history.reverse()  # newest first
    return history


def build_tag_index(directory: str) -> dict[str, Any]:
    """Build a tag frequency table from all .txt sidecars in *directory*.

    Returns ``{"tags": {"tag": count, ...}, "total_images": N}`` sorted by
    count descending.
    """
    base = resolve_directory(directory)
    if base is None:
        return {"tags": {}, "total_images": 0}

    all_images = _scan_images(base)
    tag_counts: dict[str, int] = {}
    total = 0

    for img_path in all_images:
        caption_path = img_path.with_suffix(".txt")
        if not caption_path.exists():
            continue
        text = caption_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        total += 1
        parsed = parse_caption(text)
        for tag in (
            *parsed.flat_tags,
            *(tag for clause in parsed.clauses for tag in clause.tags),
        ):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    sorted_tags = dict(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True))
    return {"tags": sorted_tags, "total_images": total}


def batch_update_captions(
    directory: str,
    paths: list[str],
    action: str,
    tag: str | None = None,
    find: str | None = None,
    replace: str | None = None,
    use_regex: bool = False,
) -> dict[str, Any]:
    """Batch update captions for multiple images.

    *action* is one of ``"append"``, ``"remove"``, or ``"replace"``.
    Returns ``{"updated": N, "failed": M, "errors": [...]}``.
    """
    if action not in {"append", "remove", "replace"}:
        raise ValueError(f"Unknown action: {action}")
    pattern = None
    if action == "replace":
        if not find:
            raise ValueError("Replacement search text must not be empty")
        if use_regex:
            try:
                pattern = re.compile(find)
                pattern.sub(replace or "", "")
            except re.error as exc:
                raise ValueError(f"Invalid replacement expression: {exc}") from exc

    updated = 0
    failed = 0
    errors: list[str] = []

    for rel_path in dict.fromkeys(paths):
        try:
            img = resolve_image_path(directory, rel_path)
            if img is None:
                raise FileNotFoundError(f"Image not found: {rel_path}")
            cp = _caption_path(img, directory)
            with _caption_lock:
                current = cp.read_text(encoding="utf-8") if cp.exists() else ""
                parsed = parse_caption(current)
                if action == "append":
                    added = parse_caption(tag or "")
                    new_content = compose_caption(
                        (*parsed.flat_tags, *added.flat_tags),
                        (*parsed.clauses, *added.clauses),
                    )
                elif action == "remove":
                    remove_set = set(parse_caption(tag or "").flat_tags)
                    clauses = [
                        PositionClause(
                            clause.position,
                            tuple(t for t in clause.tags if t not in remove_set),
                            clause.prefix,
                        )
                        for clause in parsed.clauses
                    ]
                    new_content = compose_caption(
                        (t for t in parsed.flat_tags if t not in remove_set), clauses
                    )
                else:
                    new_content = (
                        pattern.sub(replace or "", current)
                        if pattern
                        else current.replace(find, replace or "")
                    )
                _save_caption_file(cp, new_content)
            updated += 1
        except Exception as e:
            failed += 1
            errors.append(f"{rel_path}: {e}")

    return {"updated": updated, "failed": failed, "errors": errors}


# ── archive upload / extraction ────────────────────────────────


def _resolve_extract_target(target: str) -> Path:
    """Resolve the extraction target directory without requiring it to exist.

    Relative paths are anchored to ROOT and rejected on ``..`` components or
    containment escapes; absolute paths are accepted as-is (localhost tool,
    mirrors ``resolve_path``). Raises ``ValueError`` for invalid targets.
    """
    target = (target or "").strip()
    if not target:
        raise ValueError("Extraction target path is required")
    p = Path(target)
    if p.is_absolute():
        resolved = p
    else:
        if ".." in p.parts:
            raise ValueError("Target path may not contain '..'")
        resolved = (ROOT / p).resolve()
        try:
            resolved.relative_to(ROOT)
        except ValueError as e:
            raise ValueError("Target path escapes the project root") from e
    if resolved.exists() and not resolved.is_dir():
        raise ValueError(f"Target exists and is not a directory: {target}")
    return resolved


def _zip_member_dest(member: zipfile.ZipInfo) -> PurePosixPath | None:
    """Return the sanitized relative destination for a zip member, or None.

    Zip archives are extracted member-by-member (no built-in safe filter),
    so this enforces the traversal checks ourselves: reject absolute paths,
    drive letters, ``..`` segments, and NUL/control characters.
    """
    name = member.filename.replace("\\", "/")
    if "\x00" in name or any(ord(c) < 32 for c in name):
        return None
    parts = PurePosixPath(name).parts
    if not parts:
        return None
    if parts[0].startswith("/") or ":" in parts[0]:
        return None
    if any(part == ".." for part in parts):
        return None
    return PurePosixPath(*parts)


def _archive_display(target_dir: Path) -> str:
    """Target path for display/API use: ROOT-relative when possible."""
    try:
        display = str(target_dir.relative_to(ROOT))
    except ValueError:
        display = str(target_dir)
    return display.replace("\\", "/")


def extract_archive(
    archive_path: Path,
    target: str,
    *,
    progress: Callable[[int, int], None] | None = None,
    archive_name: str | None = None,
) -> dict[str, Any]:
    """Extract an uploaded archive into *target* as a dataset directory.

    Creates the target directory (including parents) if needed. Supports
    zip and tar (``.tar``, ``.tar.gz``/``.tgz``, ``.tar.bz2``/``.tbz2``,
    ``.tar.xz``/``.txz``). Unsafe members are skipped and counted, never
    extracted. Returns ``{"target", "extracted", "skipped"}``.

    *progress*, when given, is called as ``progress(extracted, skipped)``
    after each member so long extractions can surface live progress.
    *archive_name* overrides the name used for format detection — needed
    when *archive_path* is a temp file whose suffix was truncated
    (``pack.tar.gz`` → ``.gz``).
    """
    target_dir = _resolve_extract_target(target)
    name = (archive_name or archive_path.name).lower()

    if name.endswith(".zip"):
        extracted, skipped = _extract_zip(archive_path, target_dir, progress=progress)
    elif name.endswith(ARCHIVE_SUFFIXES):
        extracted, skipped = _extract_tar(archive_path, target_dir, progress=progress)
    else:
        raise ValueError(
            "Unsupported archive format: expected .zip, .tar, .tar.gz, .tgz, .tar.bz2 or .tar.xz"
        )

    return {
        "target": _archive_display(target_dir),
        "extracted": extracted,
        "skipped": skipped,
    }


def _extract_zip(
    archive_path: Path,
    target_dir: Path,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[int, int]:
    extracted = 0
    skipped = 0
    with zipfile.ZipFile(archive_path) as zf:
        for member in zf.infolist():
            dest = _zip_member_dest(member)
            if dest is None:
                skipped += 1
            else:
                final = (target_dir / dest).resolve()
                try:
                    final.relative_to(target_dir.resolve())
                except ValueError:
                    skipped += 1
                else:
                    if member.is_dir():
                        final.mkdir(parents=True, exist_ok=True)
                    else:
                        final.parent.mkdir(parents=True, exist_ok=True)
                        with zf.open(member) as src:
                            _extract_file(src, final)
                        extracted += 1
            if progress is not None:
                progress(extracted, skipped)
    return extracted, skipped


def _extract_file(source: BinaryIO, destination: Path) -> None:
    """Do not publish partial archive members or truncate an existing file."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, suffix=".tmp", delete=False
        ) as output:
            temporary = Path(output.name)
            while chunk := source.read(1 << 20):
                output.write(chunk)
        if destination.is_file():
            temporary.chmod(destination.stat().st_mode)
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _extract_tar(
    archive_path: Path,
    target_dir: Path,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[int, int]:
    extracted = 0
    skipped = 0
    with tarfile.open(archive_path) as tf:
        for member in tf:
            # filter="data" (Python 3.12+) blocks path traversal and link
            # targets outside the extraction directory; absolute paths are
            # sanitized to land inside the target.
            try:
                if member.isfile():
                    safe_member = tarfile.data_filter(member, target_dir)
                    if safe_member is None:
                        skipped += 1
                        continue
                    destination = (target_dir / safe_member.name).resolve()
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with tf.extractfile(member) as source:
                        _extract_file(source, destination)
                    extracted += 1
                elif member.isdir():
                    tf.extract(member, target_dir, filter="data")
                else:
                    skipped += 1
            except (tarfile.FilterError, tarfile.TarError, OSError):
                skipped += 1
            if progress is not None:
                progress(extracted, skipped)
    return extracted, skipped


# ── background extraction tasks ────────────────────────────────
#
# In-memory registry: the WebUI is a single localhost process, so task
# state need not survive a restart — clients treat an unknown id as a
# failure (the server is typically gone too).

_EXTRACT_TASK_TTL = 3600.0  # seconds to keep finished tasks around
_extract_tasks: dict[str, dict[str, Any]] = {}
_extract_tasks_lock = threading.Lock()

_TASK_PUBLIC_KEYS = ("task_id", "status", "target", "extracted", "skipped", "error")


def _public_task(task: dict[str, Any]) -> dict[str, Any]:
    return {k: task[k] for k in _TASK_PUBLIC_KEYS}


def _purge_stale_tasks() -> None:
    cutoff = time.monotonic() - _EXTRACT_TASK_TTL
    for tid in [t for t, v in _extract_tasks.items() if v["created"] < cutoff]:
        _extract_tasks.pop(tid, None)


def start_extract_task(
    archive_path: Path,
    target: str,
    *,
    archive_name: str | None = None,
) -> dict[str, Any]:
    """Validate *target*, register a background extraction task, start it.

    Raises ``ValueError`` synchronously for invalid targets so the caller
    can reject the upload with 400 before any work starts. Extraction runs
    in a daemon thread; poll :func:`get_extract_task` until *status* is
    ``"done"`` or ``"error"``. The archive temp file is removed by the
    worker when it finishes.
    """
    display = _archive_display(_resolve_extract_target(target))
    with _extract_tasks_lock:
        _purge_stale_tasks()
        task: dict[str, Any] = {
            "task_id": uuid.uuid4().hex,
            "status": "extracting",
            "target": display,
            "extracted": 0,
            "skipped": 0,
            "error": None,
            "created": time.monotonic(),
        }
        _extract_tasks[task["task_id"]] = task
    threading.Thread(
        target=_extract_task_worker,
        args=(task["task_id"], archive_path, target, archive_name),
        daemon=True,
    ).start()
    return _public_task(task)


def get_extract_task(task_id: str) -> dict[str, Any] | None:
    """Return the current state of an extraction task, or None."""
    with _extract_tasks_lock:
        task = _extract_tasks.get(task_id)
        return _public_task(task) if task is not None else None


def _extract_task_worker(
    task_id: str,
    archive_path: Path,
    target: str,
    archive_name: str | None,
) -> None:
    def progress(extracted: int, skipped: int) -> None:
        with _extract_tasks_lock:
            task = _extract_tasks.get(task_id)
            if task is not None:
                task["extracted"] = extracted
                task["skipped"] = skipped

    try:
        result = extract_archive(
            archive_path, target, progress=progress, archive_name=archive_name
        )
        with _extract_tasks_lock:
            task = _extract_tasks.get(task_id)
            if task is not None:
                task.update(
                    status="done",
                    target=result["target"],
                    extracted=result["extracted"],
                    skipped=result["skipped"],
                    error=None,
                )
    except Exception as e:  # noqa: BLE001 — surfaced verbatim to the poller
        with _extract_tasks_lock:
            task = _extract_tasks.get(task_id)
            if task is not None:
                task.update(status="error", error=str(e))
    finally:
        archive_path.unlink(missing_ok=True)


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
    for mask_root in _get_mask_search_roots():
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
