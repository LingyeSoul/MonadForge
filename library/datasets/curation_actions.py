"""Dataset curation helpers shared by GUI and preprocess."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from library.datasets.buckets import (
    DEFAULT_TARGET_RES,
    BucketManager,
    buckets_for_edges,
    choose_edge,
)

SIDECAR_EXTENSIONS = (".txt", ".caption", ".json", ".txt.history.jsonl")


def linked_paths(image_path: Path) -> list[Path]:
    """Return an image and existing caption/metadata sidecars."""

    paths = [image_path]
    for ext in SIDECAR_EXTENSIONS:
        sidecar = image_path.with_suffix(ext)
        if sidecar.exists():
            paths.append(sidecar)
    return paths


def move_linked_files(
    image_path: Path,
    *,
    source_root: Path,
    target_root: Path,
) -> list[Path]:
    """Move an image and sidecars to ``target_root`` preserving layout."""

    moved: list[Path] = []
    for source in linked_paths(image_path):
        if not source.exists():
            continue
        try:
            rel = source.relative_to(source_root)
        except ValueError:
            rel = Path(source.name)
        target = _unique_path(target_root / rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        moved.append(target)
    return moved


def _unique_path(path: Path) -> Path:
    """Return a non-existing sibling path without overwriting prior moves."""

    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    for index in range(1, 10_000):
        candidate = parent / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"could not find a free target path for {path}")


def rel_key(path: Path, root: Path) -> str:
    """Stable JSON key for an image path relative to a dataset root."""

    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def load_curation_decisions(
    path: Path | str | None,
    *,
    source_dir: Path | str | None = None,
) -> dict[str, dict[str, Any]]:
    """Load per-image preprocess decisions from JSON.

    The file is intentionally optional. Missing or malformed files behave as an
    empty decision set so normal CLI preprocess remains unchanged unless a GUI
    decision file is present.
    """

    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    images = data.get("images") if isinstance(data, dict) else None
    if not isinstance(images, dict):
        return {}
    strip_prefix = ""
    add_prefix = ""
    if source_dir is not None:
        saved_source = str(data.get("source_dir") or "").replace("\\", "/")
        source = Path(source_dir)
        if saved_source:
            try:
                cwd = Path.cwd().resolve()
                saved_abs = (cwd / saved_source).resolve()
                source_abs = (
                    source.resolve()
                    if source.is_absolute()
                    else (cwd / source).resolve()
                )
                if source_abs == saved_abs:
                    pass
                elif source_abs.is_relative_to(saved_abs):
                    strip_prefix = source_abs.relative_to(saved_abs).as_posix()
                elif saved_abs.is_relative_to(source_abs):
                    add_prefix = saved_abs.relative_to(source_abs).as_posix()
                else:
                    return {}
            except OSError:
                candidates = {str(source).replace("\\", "/"), source.as_posix()}
                if saved_source not in candidates:
                    return {}
    out: dict[str, dict[str, Any]] = {}
    for key, value in images.items():
        if isinstance(key, str) and isinstance(value, dict):
            norm_key = key.replace("\\", "/")
            if strip_prefix:
                prefix = strip_prefix.rstrip("/") + "/"
                if not norm_key.startswith(prefix):
                    continue
                norm_key = norm_key[len(prefix) :]
            elif add_prefix:
                norm_key = f"{add_prefix.rstrip('/')}/{norm_key}"
            out[norm_key] = dict(value)
    return out


def save_curation_decisions(
    path: Path,
    *,
    source_dir: str,
    images: dict[str, dict[str, Any]],
) -> None:
    """Write GUI decisions consumed by preprocess resize."""

    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version": 1,
        "source_dir": source_dir.replace("\\", "/"),
        "images": images,
    }
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def center_crop_rect_for_aspect(
    width: int,
    height: int,
    aspect: float,
) -> tuple[int, int, int, int]:
    """Return a center crop rect in source-image coordinates."""

    if width <= 0 or height <= 0 or aspect <= 0:
        raise ValueError(f"invalid crop inputs: {width}x{height}, aspect={aspect}")
    source_aspect = width / height
    if source_aspect > aspect:
        crop_h = height
        crop_w = max(1, round(height * aspect))
        x = max(0, (width - crop_w) // 2)
        return (x, 0, min(crop_w, width), crop_h)
    crop_w = width
    crop_h = max(1, round(width / aspect))
    y = max(0, (height - crop_h) // 2)
    return (0, y, crop_w, min(crop_h, height))


def center_crop_rect_for_resize_bucket(
    width: int,
    height: int,
    *,
    target_res: list[int] | None = None,
) -> tuple[int, int, int, int]:
    """Center crop to the aspect ratio preprocess would bucket into."""

    tier = target_res or list(DEFAULT_TARGET_RES)
    edge = choose_edge(width, height, tier)
    mgr = BucketManager(
        max_reso=(max(tier), max(tier)),
        min_size=512,
        max_size=2048,
        reso_steps=64,
    )
    mgr.set_predefined_resos(buckets_for_edges([edge]))
    bucket_reso, _, _ = mgr.select_bucket(width, height)
    return center_crop_rect_for_aspect(width, height, bucket_reso[0] / bucket_reso[1])


def center_crop_rect_for_resize_bucket_within(
    bounds: tuple[int, int, int, int],
    *,
    target_res: list[int] | None = None,
) -> tuple[int, int, int, int]:
    """Return the bucket-aspect crop inside an already selected larger bounds."""

    x, y, width, height = bounds
    inner_x, inner_y, inner_w, inner_h = center_crop_rect_for_resize_bucket(
        width, height, target_res=target_res
    )
    return (x + inner_x, y + inner_y, inner_w, inner_h)


def clamp_crop_rect(
    rect: tuple[int, int, int, int],
    *,
    image_width: int,
    image_height: int,
    min_size: int = 8,
) -> tuple[int, int, int, int]:
    """Clamp a crop rect to image bounds while preserving a minimum size."""

    x, y, width, height = rect
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"invalid image size: {image_width}x{image_height}")
    min_w = min(max(1, min_size), image_width)
    min_h = min(max(1, min_size), image_height)
    width = max(min_w, min(width, image_width))
    height = max(min_h, min(height, image_height))
    x = max(0, min(x, image_width - width))
    y = max(0, min(y, image_height - height))
    return (x, y, width, height)


def inset_crop_rect_by_percent(
    *,
    image_width: int,
    image_height: int,
    left: float = 0.0,
    top: float = 0.0,
    right: float = 0.0,
    bottom: float = 0.0,
) -> tuple[int, int, int, int]:
    """Build a crop rect by trimming percentages from each image side."""

    x = round(image_width * max(0.0, left) / 100)
    y = round(image_height * max(0.0, top) / 100)
    w = round(image_width * (100 - max(0.0, left) - max(0.0, right)) / 100)
    h = round(image_height * (100 - max(0.0, top) - max(0.0, bottom)) / 100)
    return clamp_crop_rect(
        (x, y, w, h),
        image_width=image_width,
        image_height=image_height,
    )


def crop_rect_from_decision(value: Any) -> tuple[int, int, int, int] | None:
    """Normalize a JSON crop_bounds value."""

    if not isinstance(value, list | tuple) or len(value) != 4:
        return None
    try:
        x, y, width, height = (int(v) for v in value)
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return (x, y, width, height)
