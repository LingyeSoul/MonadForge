"""Resize-preview helpers shared by preprocess GUI surfaces.

The resize step scales images to cover the selected constant-token bucket and
then center-crops to that bucket. This module exposes the same geometry without
touching files, so GUI previews can show what preprocessing will keep.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from library.datasets.buckets import DEFAULT_TARGET_RES, buckets_for_edges, choose_edge

DEFAULT_RESIZE_CROP_ANCHOR = "center"
RESIZE_CROP_ANCHORS = {
    "top_left": (0.0, 0.0),
    "top": (0.5, 0.0),
    "top_right": (1.0, 0.0),
    "left": (0.0, 0.5),
    "center": (0.5, 0.5),
    "right": (1.0, 0.5),
    "bottom_left": (0.0, 1.0),
    "bottom": (0.5, 1.0),
    "bottom_right": (1.0, 1.0),
}


@dataclass(frozen=True)
class CropRect:
    left: float
    top: float
    width: float
    height: float


@dataclass(frozen=True)
class ResizePreview:
    source_size: tuple[int, int]
    target_edge: int
    bucket_size: tuple[int, int]
    kept_rect: CropRect
    margin_rect: CropRect
    crop_anchor: str
    crop_margins: dict[str, float]


def normalize_target_res(target_res: Iterable[int] | int | str | None) -> list[int]:
    """Normalize config-style ``target_res`` values into a non-empty int list."""
    if target_res is None:
        return list(DEFAULT_TARGET_RES)
    if isinstance(target_res, int):
        return [target_res]
    if isinstance(target_res, str):
        raw = target_res.strip()
        if not raw:
            return list(DEFAULT_TARGET_RES)
        return [int(part.strip()) for part in raw.split(",") if part.strip()]
    values = [int(value) for value in target_res]
    return values or list(DEFAULT_TARGET_RES)


def normalize_crop_anchor(crop_anchor: str | None) -> str:
    value = str(crop_anchor or DEFAULT_RESIZE_CROP_ANCHOR).strip().lower()
    return value if value in RESIZE_CROP_ANCHORS else DEFAULT_RESIZE_CROP_ANCHOR


def normalize_crop_margins(raw) -> dict[str, float]:
    if raw is None:
        margins = {}
    elif isinstance(raw, dict):
        margins = raw
    elif isinstance(raw, str):
        parts = [part.strip() for part in raw.split(",") if part.strip()]
        margins = dict(zip(("top", "right", "bottom", "left"), parts, strict=False))
    elif isinstance(raw, (list, tuple)):
        margins = dict(zip(("top", "right", "bottom", "left"), raw, strict=False))
    else:
        margins = {}

    out = {}
    for key in ("top", "right", "bottom", "left"):
        try:
            out[key] = max(0.0, float(margins.get(key, 0.0)))
        except (TypeError, ValueError):
            out[key] = 0.0

    h_sum = out["left"] + out["right"]
    if h_sum >= 95.0:
        scale = 95.0 / h_sum
        out["left"] *= scale
        out["right"] *= scale
    v_sum = out["top"] + out["bottom"]
    if v_sum >= 95.0:
        scale = 95.0 / v_sum
        out["top"] *= scale
        out["bottom"] *= scale
    return out


def margin_crop_rect(width: int, height: int, crop_margins=None) -> CropRect:
    margins = normalize_crop_margins(crop_margins)
    left = width * margins["left"] / 100.0
    top = height * margins["top"] / 100.0
    right = width - width * margins["right"] / 100.0
    bottom = height - height * margins["bottom"] / 100.0
    return CropRect(left=left, top=top, width=max(1.0, right - left), height=max(1.0, bottom - top))


def parse_bucket_resos(raw) -> list[tuple[int, int]]:
    """Normalize bucket filters from TOML/CLI values.

    Accepts ``["1008x1024"]``, ``["1008,1024"]``, ``[(1008, 1024)]``, or a
    comma-separated string. Empty input means "all supported buckets".
    """
    if raw is None:
        return []
    values = raw
    if isinstance(raw, str):
        values = [part.strip() for part in raw.split(",") if part.strip()]
    out: list[tuple[int, int]] = []
    for item in values:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            width, height = int(item[0]), int(item[1])
        else:
            text = str(item).strip().lower().replace("×", "x")
            if "x" in text:
                left, right = text.split("x", 1)
            elif ":" in text:
                left, right = text.split(":", 1)
            else:
                continue
            width, height = int(left.strip()), int(right.strip())
        if width > 0 and height > 0:
            out.append((width, height))
    return sorted(set(out))


def format_bucket_resos(bucket_resos: Iterable[tuple[int, int]]) -> list[str]:
    return [f"{width}x{height}" for width, height in bucket_resos]


def select_resize_bucket(
    width: int,
    height: int,
    target_res: Iterable[int] | int | str | None = None,
    bucket_resos=None,
) -> tuple[int, tuple[int, int]]:
    tiers = normalize_target_res(target_res)
    allowed = set(parse_bucket_resos(bucket_resos))
    if not allowed:
        edge = choose_edge(width, height, tiers)
        return edge, _nearest_aspect_bucket(width, height, buckets_for_edges([edge]))

    edge = choose_edge(width, height, tiers)
    buckets = [bucket for bucket in buckets_for_edges([edge]) if bucket in allowed]
    if buckets:
        return edge, _nearest_aspect_bucket(width, height, buckets)

    fallback: list[tuple[int, tuple[int, int]]] = []
    for candidate_edge in tiers:
        fallback.extend(
            (candidate_edge, bucket)
            for bucket in buckets_for_edges([candidate_edge])
            if bucket in allowed
        )
    if not fallback:
        raise ValueError("resize_bucket_resos has no buckets for selected target_res")
    ar = width / height
    return min(
        fallback,
        key=lambda item: (
            abs(item[1][0] / item[1][1] - ar),
            abs(math.log(_cover_scale(width, height, item[1][0], item[1][1]))),
            item[0],
            item[1],
        ),
    )


def compute_resize_preview(
    width: int,
    height: int,
    target_res: Iterable[int] | int | str | None = None,
    *,
    crop_anchor: str | None = None,
    bucket_resos=None,
    crop_margins=None,
) -> ResizePreview:
    """Return the bucket and source-space crop rect used by preprocessing."""
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")

    anchor = normalize_crop_anchor(crop_anchor)
    anchor_x, anchor_y = RESIZE_CROP_ANCHORS[anchor]
    margins = normalize_crop_margins(crop_margins)
    margin_rect = margin_crop_rect(width, height, margins)
    work_w = max(1, round(margin_rect.width))
    work_h = max(1, round(margin_rect.height))
    edge, (bucket_w, bucket_h) = select_resize_bucket(
        work_w, work_h, target_res, bucket_resos
    )

    source_ar = work_w / work_h
    bucket_ar = bucket_w / bucket_h
    if source_ar > bucket_ar:
        kept_h = float(work_h)
        kept_w = kept_h * bucket_ar
        left = margin_rect.left + (work_w - kept_w) * anchor_x
        top = margin_rect.top
    else:
        kept_w = float(work_w)
        kept_h = kept_w / bucket_ar
        left = margin_rect.left
        top = margin_rect.top + (work_h - kept_h) * anchor_y

    return ResizePreview(
        source_size=(width, height),
        target_edge=edge,
        bucket_size=(bucket_w, bucket_h),
        kept_rect=CropRect(left=left, top=top, width=kept_w, height=kept_h),
        margin_rect=margin_rect,
        crop_anchor=anchor,
        crop_margins=margins,
    )


def _nearest_aspect_bucket(
    width: int, height: int, buckets: Iterable[tuple[int, int]]
) -> tuple[int, int]:
    ar = width / height
    return min(buckets, key=lambda bucket: abs(bucket[0] / bucket[1] - ar))


def _cover_scale(width: int, height: int, bw: int, bh: int) -> float:
    return max(bw / width, bh / height)
