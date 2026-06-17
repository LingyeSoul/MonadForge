"""Resize a dataset directory to constant-token bucket resolutions.

Orchestration extracted from ``preprocess/resize_images.py`` (see
``docs/proposal/tooling_architecture.md`` §A). The script keeps only argparse;
the walk → min-pixel filter → parallel resize+crop → caption mirror loop lives
here. ``process_image`` stays a module-level function so it remains picklable
for ``ProcessPoolExecutor`` workers.
"""

from __future__ import annotations

import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image
from PIL import ImageOps
from PIL.PngImagePlugin import PngInfo

from library.datasets.buckets import (
    DEFAULT_TARGET_RES,
    BucketManager,
)
from library.preprocess._dataset import PreprocessStats, walk_images
from library.preprocess._progress import ProgressFn
from library.preprocess.resize_preview import (
    DEFAULT_RESIZE_CROP_ANCHOR,
    RESIZE_CROP_ANCHORS,
    format_bucket_resos,
    margin_crop_rect,
    normalize_crop_anchor,
    normalize_crop_margins,
    parse_bucket_resos,
    select_resize_bucket,
)

CAPTION_EXTENSIONS = {".txt", ".caption"}
_RESIZE_ANCHOR_KEY = "anima_resize_crop_anchor"
_RESIZE_BUCKET_RESOS_KEY = "anima_resize_bucket_resos"
_RESIZE_MARGINS_KEY = "anima_resize_crop_margins"


def _collect_metadata(src: Image.Image) -> dict:
    """Pull through metadata that ``convert("RGB")`` + a bare ``save()`` drops.

    Captured from the *original* opened image (before resize/crop produces a
    fresh object that no longer carries ``.text``): the ICC color profile, raw
    EXIF, and PNG text chunks — the last is where ComfyUI / A1111 stash the
    generation prompt + params. Returned as ``save()`` kwargs. Each field is
    best-effort so a malformed chunk never kills the worker.
    """
    save_kwargs: dict = {}

    icc = src.info.get("icc_profile")
    if icc:
        save_kwargs["icc_profile"] = icc

    exif = src.info.get("exif")
    if exif:
        save_kwargs["exif"] = exif

    text_chunks = getattr(src, "text", None)
    if text_chunks:
        pnginfo = PngInfo()
        for key, value in text_chunks.items():
            try:
                pnginfo.add_text(key, str(value))
            except Exception:
                continue
        save_kwargs["pnginfo"] = pnginfo

    return save_kwargs


def _format_margins(crop_margins) -> str:
    margins = normalize_crop_margins(crop_margins)
    return ",".join(f"{margins[key]:g}" for key in ("top", "right", "bottom", "left"))


def _resize_metadata_signature(
    crop_anchor: str, bucket_resos, crop_margins=None
) -> dict[str, str]:
    anchor = normalize_crop_anchor(crop_anchor)
    buckets = format_bucket_resos(parse_bucket_resos(bucket_resos))
    margins = _format_margins(crop_margins)
    if anchor == DEFAULT_RESIZE_CROP_ANCHOR and not buckets and margins == "0,0,0,0":
        return {}
    return {
        _RESIZE_ANCHOR_KEY: anchor,
        _RESIZE_BUCKET_RESOS_KEY: "|".join(buckets),
        _RESIZE_MARGINS_KEY: margins,
    }


def _add_resize_metadata(save_kwargs: dict, signature: dict[str, str]) -> None:
    if not signature:
        return
    pnginfo = save_kwargs.get("pnginfo")
    if pnginfo is None:
        pnginfo = PngInfo()
        save_kwargs["pnginfo"] = pnginfo
    for key, value in signature.items():
        pnginfo.add_text(key, value)


def _resize_metadata_matches(image: Image.Image, signature: dict[str, str]) -> bool:
    if not signature:
        return True
    text = getattr(image, "text", {}) or {}
    return all(text.get(key) == value for key, value in signature.items())


def process_image(
    image_path: Path,
    out_dir: Path,
    bucket_args: tuple,
    copy_captions: bool = True,
    rel_dir: str = "",
    overwrite: bool = False,
) -> tuple[str, tuple[int, int], bool]:
    """Worker — receives bucket params (not a BucketManager) to stay picklable.

    ``rel_dir`` is the (possibly empty) relative subdir under the source root;
    the output mirrors it as ``out_dir / rel_dir / stem.png``. Empty ``rel_dir``
    collapses to the flat layout.

    Returns ``(name, bucket_reso, skipped)``. Unless ``overwrite`` is set, an
    image whose resized PNG already exists *at the correct bucket size* is
    skipped (no re-decode/resize) — so a re-run is near-free, while a bucket
    change (e.g. adding a ``--target_res`` tier) still re-resizes only the
    images whose target bucket actually moved.
    """
    # 6th+ elements are optional so pre-multiscale callers still work.
    max_reso, min_size, max_size, reso_steps, use_constant, *rest = bucket_args
    target_res = rest[0] if rest else None
    crop_anchor = rest[1] if len(rest) > 1 else DEFAULT_RESIZE_CROP_ANCHOR
    bucket_resos = rest[2] if len(rest) > 2 else None
    crop_margins = rest[3] if len(rest) > 3 else None
    bucket_mgr = BucketManager(
        max_reso=max_reso,
        min_size=min_size,
        max_size=max_size,
        reso_steps=reso_steps,
    )

    src_img = Image.open(image_path)
    save_kwargs = _collect_metadata(src_img)
    img = ImageOps.exif_transpose(src_img)
    w, h = img.size
    margin_rect = margin_crop_rect(w, h, crop_margins)
    margin_box = (
        round(margin_rect.left),
        round(margin_rect.top),
        round(margin_rect.left + margin_rect.width),
        round(margin_rect.top + margin_rect.height),
    )
    work_w = max(1, margin_box[2] - margin_box[0])
    work_h = max(1, margin_box[3] - margin_box[1])

    if use_constant:
        # Default to the canonical 1024 tier (not the full multi-tier catalog):
        # all_constant_token_buckets()'s aspect-only select_bucket would UPSCALE
        # a 0.7MP portrait into a 1536-tier bucket — the multi-tier resize regression.
        tier = target_res or list(DEFAULT_TARGET_RES)
        _, bucket_reso = select_resize_bucket(work_w, work_h, tier, bucket_resos)
    else:
        bucket_mgr.make_buckets(constant_token_buckets=False)
        bucket_reso, _, _ = bucket_mgr.select_bucket(w, h)

    bw, bh = bucket_reso
    crop_anchor = normalize_crop_anchor(crop_anchor)
    anchor_x, anchor_y = RESIZE_CROP_ANCHORS[crop_anchor]
    signature = _resize_metadata_signature(crop_anchor, bucket_resos, crop_margins)

    target_dir = out_dir / rel_dir if rel_dir else out_dir
    out_path = target_dir / f"{image_path.stem}.png"

    if not overwrite and out_path.exists():
        try:
            with Image.open(out_path) as ex:
                if ex.size == (bw, bh) and _resize_metadata_matches(ex, signature):
                    return image_path.name, bucket_reso, True
        except Exception:
            pass

    img = img.convert("RGB")
    img = img.crop(margin_box)

    ar_img = work_w / work_h
    ar_bucket = bw / bh
    if ar_img > ar_bucket:
        new_h = bh
        new_w = round(bh * ar_img)
    else:
        new_w = bw
        new_h = round(bw / ar_img)

    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    left = round((new_w - bw) * anchor_x)
    top = round((new_h - bh) * anchor_y)
    img = img.crop((left, top, left + bw, top + bh))

    target_dir.mkdir(parents=True, exist_ok=True)
    _add_resize_metadata(save_kwargs, signature)
    img.save(out_path, format="PNG", **save_kwargs)

    if copy_captions:
        for ext in CAPTION_EXTENSIONS:
            cap = image_path.with_suffix(ext)
            if cap.exists():
                shutil.copy2(cap, target_dir / f"{image_path.stem}{ext}")

    return image_path.name, bucket_reso, False


def resize_to_buckets(
    src: Path,
    dst: Path,
    *,
    resolution: int = 1024,
    min_bucket_reso: int = 512,
    max_bucket_reso: int = 2048,
    bucket_reso_steps: int = 64,
    constant_token_buckets: bool = True,
    target_res: list[int] | None = None,
    workers: int = 4,
    min_pixels: int = 500_000,
    copy_captions: bool = True,
    recursive: bool = False,
    path_pattern: str | None = None,
    verbose: bool = True,
    overwrite: bool = False,
    curation_decisions: dict[str, dict] | None = None,
    crop_anchor: str = DEFAULT_RESIZE_CROP_ANCHOR,
    bucket_resos=None,
    crop_margins=None,
    progress: ProgressFn | None = None,
) -> tuple[PreprocessStats, dict[tuple[int, int], int]]:
    """Resize+crop every image under ``src`` into bucket resolutions under ``dst``.

    Mirrors the source subdir layout, copies caption sidecars, and skips images
    below ``min_pixels``. Returns ``(stats, bucket_counts)`` where
    ``bucket_counts`` maps each ``(W, H)`` bucket to its image count (over the
    full dataset, skipped + written). Pass ``progress`` for a per-image bar.

    Unless ``overwrite`` is set, images whose resized PNG already exists at the
    correct bucket are skipped — a re-run only touches images whose target
    bucket changed (e.g. after adding a ``--target_res`` tier).
    """
    dst.mkdir(parents=True, exist_ok=True)

    bucket_args = (
        (resolution, resolution),
        min_bucket_reso,
        max_bucket_reso,
        bucket_reso_steps,
        constant_token_buckets,
        target_res,
        crop_anchor,
        parse_bucket_resos(bucket_resos),
        normalize_crop_margins(crop_margins),
    )

    # walk_images enforces per-subfolder stem uniqueness (collisions would collide the resized output).
    image_files = walk_images(src, recursive=recursive, pattern=path_pattern)
    stats = PreprocessStats(seen=len(image_files))

    decisions = curation_decisions or {}

    def _rel_key(p: Path) -> str:
        try:
            return p.relative_to(src).as_posix()
        except ValueError:
            return p.name

    if decisions:
        kept: list[Path] = []
        skipped_by_decision: list[Path] = []
        for p in image_files:
            decision = decisions.get(_rel_key(p), {})
            if decision.get("action") in {"skip", "move"}:
                skipped_by_decision.append(p)
                continue
            kept.append(p)
        if skipped_by_decision and verbose:
            print(
                f"Skipping {len(skipped_by_decision)} image(s) marked by "
                "curation decisions:"
            )
            for p in skipped_by_decision:
                print(f"  {_rel_key(p)}")
        stats.skipped += len(skipped_by_decision)
        image_files = kept

    if min_pixels > 0:
        kept: list[Path] = []
        skipped: list[tuple[Path, int, int]] = []
        for p in image_files:
            try:
                with Image.open(p) as im:
                    w, h = im.size
            except Exception as e:
                if verbose:
                    print(f"  warn: could not read {p.name}: {e}")
                continue
            if w * h < min_pixels:
                skipped.append((p, w, h))
            else:
                kept.append(p)
        if skipped and verbose:
            print(
                f"Skipping {len(skipped)} images below {min_pixels:,} pixels "
                f"({min_pixels / 1e6:.2f}MP):"
            )
            for p, w, h in skipped:
                print(f"  {p.name}  {w}x{h}  ({w * h / 1e6:.3f}MP)")
        stats.skipped += len(skipped)
        image_files = kept

    if verbose:
        mode = "standard"
        if constant_token_buckets:
            mode = (
                f"multi-scale constant-token (tiers {sorted(target_res)})"
                if target_res
                else "constant-token"
            )
        print(f"Resizing {len(image_files)} images to {mode} buckets")

    def _rel_for(p: Path) -> str:
        try:
            rel = p.parent.relative_to(src)
        except ValueError:
            return ""
        rel_str = str(rel)
        return "" if rel_str == "." else rel_str

    if progress is not None:
        progress(0, total=len(image_files))

    bucket_counts: dict[tuple[int, int], int] = {}
    resize_skipped = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                process_image,
                img_path,
                dst,
                bucket_args,
                copy_captions,
                _rel_for(img_path),
                overwrite,
            ): img_path
            for img_path in image_files
        }
        for future in as_completed(futures):
            name, reso, skipped = future.result()
            bucket_counts[reso] = bucket_counts.get(reso, 0) + 1
            if skipped:
                resize_skipped += 1
            else:
                stats.written += 1
            if progress is not None:
                tag = "skip" if skipped else f"→ {reso[0]}x{reso[1]}"
                progress(1, detail=f"{name} {tag}")
    stats.skipped += resize_skipped
    if verbose and resize_skipped:
        print(
            f"Skipped {resize_skipped} image(s) already at their target bucket "
            f"(pass --overwrite to force re-resize); {stats.written} (re)written."
        )

    if verbose:
        print("\nBucket distribution:")
        for reso in sorted(bucket_counts):
            tokens = (reso[0] // 16) * (reso[1] // 16)
            print(
                f"  {reso[0]:>4d}x{reso[1]:<4d}: {bucket_counts[reso]:>3d} "
                f"images  ({tokens} tokens)"
            )

    return stats, bucket_counts
