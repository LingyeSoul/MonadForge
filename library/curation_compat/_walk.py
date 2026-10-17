"""Curation-side dataset walking (pure ``pathlib`` / ``os``; torch-free).

Must stay behaviour-identical to the trainer's ``image_utils`` walkers — both
sides walk the same caption master.
"""

from __future__ import annotations

import fnmatch
import os
from collections.abc import Iterator
from pathlib import Path

CAPTION_EXTENSIONS = (".txt", ".caption")
"""The sidecar suffixes a caption can wear beside its image."""

IMAGE_EXTENSIONS: list[str] = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]
IMAGE_EXTENSIONS.extend([ext.upper() for ext in IMAGE_EXTENSIONS])

# Optional-plugin formats, mirrored from the trainer's image_utils so both
# walkers see the same files. The excepts are blind on purpose: these Pillow
# plugins have native dependencies, and a broken one must degrade to "that
# format is not walkable here" rather than take every stage down.
try:
    import pillow_avif  # noqa: F401

    IMAGE_EXTENSIONS.extend([".avif", ".AVIF"])
except Exception:  # noqa: BLE001, S110
    pass
for _jxl_plugin in ("jxlpy", "pillow_jxl"):
    try:
        __import__(_jxl_plugin)
    except Exception:  # noqa: BLE001, S112
        continue
    IMAGE_EXTENSIONS.extend([".jxl", ".JXL"])


def filter_paths_by_glob(
    img_paths: list[str],
    image_dir: str | None,
    pattern: str | None,
) -> list[bool]:
    """Return a per-path boolean mask: True keeps the file, False drops it.

    The pattern is matched against each file's path relative to ``image_dir``
    (with forward slashes, no leading "./") via ``fnmatch``. ``|`` separates
    alternatives — ``char_a/*|char_b/*`` keeps anything under either folder.
    Default ``*``, empty, or None all keep everything. Returns a mask rather
    than a filtered list so callers can keep parallel arrays (sizes,
    captions) aligned.
    """
    if not pattern:
        return [True] * len(img_paths)
    alternatives = [alt.strip() for alt in pattern.split("|")]
    alternatives = [alt for alt in alternatives if alt]
    if not alternatives or any(alt == "*" for alt in alternatives):
        return [True] * len(img_paths)
    base = os.path.abspath(image_dir) if image_dir else None
    keep: list[bool] = []
    for p in img_paths:
        if base is not None:
            try:
                rel = os.path.relpath(p, base)
            except ValueError:
                rel = os.path.basename(p)
        else:
            rel = os.path.basename(p)
        rel = rel.replace(os.sep, "/")
        keep.append(any(fnmatch.fnmatchcase(rel, alt) for alt in alternatives))
    return keep


def glob_images_pathlib(dir_path: Path, recursive: bool) -> list[Path]:
    out: list[Path] = []
    for ext in IMAGE_EXTENSIONS:
        out += list(
            dir_path.rglob("*" + ext) if recursive else dir_path.glob("*" + ext)
        )
    return sorted(set(out))


def assert_unique_stems(img_paths, source_label: str = "directory") -> None:
    """Raise if two image paths share a stem *within the same subfolder*
    (sidecars are stem-keyed per folder)."""
    seen: dict[tuple[str, str], str] = {}
    collisions: dict[tuple[str, str], list[str]] = {}
    for p in img_paths:
        key = (os.path.dirname(p), os.path.splitext(os.path.basename(p))[0])
        if key in seen:
            collisions.setdefault(key, [seen[key]]).append(p)
        else:
            seen[key] = p
    if collisions:
        lines = [
            f"  stem '{stem}' in {parent}: " + ", ".join(paths)
            for (parent, stem), paths in sorted(collisions.items())
        ]
        raise ValueError(
            f"Duplicate image stems within a single folder of {source_label}. "
            "Cache filenames are stem-keyed; rename one of the colliding "
            "files (or move it to a different subfolder).\n" + "\n".join(lines)
        )


def walk_images(
    data_dir: Path, recursive: bool = False, pattern: str | None = None
) -> list[Path]:
    """Enumerate dataset images, sorted; ``pattern`` is the ``path_pattern``
    glob (``|``-OR) against paths relative to ``data_dir``; ``None``/``"*"``
    keeps all. Same-folder stem collisions raise."""
    paths = glob_images_pathlib(Path(data_dir), recursive)
    if pattern and pattern != "*":
        keep = filter_paths_by_glob([str(p) for p in paths], str(data_dir), pattern)
        paths = [p for p, k in zip(paths, keep) if k]
    assert_unique_stems([str(p) for p in paths], source_label=str(data_dir))
    return paths


def suffix_key(path: Path) -> str:
    """A path's extension, case-folded: ``.JPG`` and ``.jpg`` are one format."""
    return path.suffix.lower()


def sibling_image(directory: Path, stem: str) -> Path | None:
    """The image named ``stem`` in ``directory``, whatever its extension: the
    resize step may re-encode (``.jpg`` master → ``.png`` resized), so the
    source and resized trees are matched on stem, not on the full relative
    path."""
    for ext in IMAGE_EXTENSIONS:
        p = directory / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def safe_walk(
    top: str | os.PathLike, *, followlinks: bool = True
) -> Iterator[tuple[str, list[str], list[str]]]:
    """``os.walk`` that follows symlinks but never revisits a directory."""
    seen: set[str] = {os.path.realpath(top)}
    for dirpath, dirnames, filenames in os.walk(top, followlinks=followlinks):
        kept: list[str] = []
        for d in dirnames:
            real = os.path.realpath(os.path.join(dirpath, d))
            if real in seen:
                continue
            seen.add(real)
            kept.append(d)
        dirnames[:] = kept
        yield dirpath, dirnames, filenames


def caption_key(
    image_path: str | os.PathLike, image_dir: str | os.PathLike | None = None
) -> str:
    """Stable subdir-aware caption-index key: path relative to ``image_dir``,
    extension stripped, ``/``-separated; bare stem when unrelated."""
    stem_no_ext = os.path.splitext(os.fspath(image_path))[0]
    if image_dir is not None:
        try:
            rel = os.path.relpath(stem_no_ext, os.fspath(image_dir))
        except ValueError:
            rel = ""
        if rel and rel != "." and not rel.startswith(".."):
            return rel.replace(os.sep, "/")
    return os.path.basename(stem_no_ext)
