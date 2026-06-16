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
    buckets_for_edges,
    choose_edge,
)
from library.datasets.curation_actions import clamp_crop_rect, crop_rect_from_decision
from library.preprocess._dataset import PreprocessStats, walk_images
from library.preprocess._progress import ProgressFn

CAPTION_EXTENSIONS = {".txt", ".caption"}


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


def process_image(
    image_path: Path,
    out_dir: Path,
    bucket_args: tuple,
    copy_captions: bool = True,
    rel_dir: str = "",
    overwrite: bool = False,
    crop_rect: tuple[int, int, int, int] | None = None,
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
    # 6th element (target_res) is optional so pre-multiscale 5-tuple callers still work.
    max_reso, min_size, max_size, reso_steps, use_constant, *rest = bucket_args
    target_res = rest[0] if rest else None
    bucket_mgr = BucketManager(
        max_reso=max_reso,
        min_size=min_size,
        max_size=max_size,
        reso_steps=reso_steps,
    )

    src_img = Image.open(image_path)
    save_kwargs = _collect_metadata(src_img)
    img = ImageOps.exif_transpose(src_img)
    if crop_rect is not None:
        crop_rect = clamp_crop_rect(
            crop_rect,
            image_width=img.width,
            image_height=img.height,
        )
        x, y, width, height = crop_rect
        img = img.crop((x, y, x + width, y + height))
    w, h = img.size

    if use_constant:
        # Default to the canonical 1024 tier (not the full multi-tier catalog):
        # all_constant_token_buckets()'s aspect-only select_bucket would UPSCALE
        # a 0.7MP portrait into a 1536-tier bucket — the multi-tier resize regression.
        tier = target_res or list(DEFAULT_TARGET_RES)
        edge = choose_edge(w, h, tier)
        bucket_mgr.set_predefined_resos(buckets_for_edges([edge]))
    else:
        bucket_mgr.make_buckets(constant_token_buckets=False)

    bucket_reso, _, _ = bucket_mgr.select_bucket(w, h)
    bw, bh = bucket_reso

    target_dir = out_dir / rel_dir if rel_dir else out_dir
    out_path = target_dir / f"{image_path.stem}.png"

    if crop_rect is None and not overwrite and out_path.exists():
        try:
            with Image.open(out_path) as ex:
                if ex.size == (bw, bh):
                    return image_path.name, bucket_reso, True
        except Exception:
            pass

    img = img.convert("RGB")

    ar_img = w / h
    ar_bucket = bw / bh
    if ar_img > ar_bucket:
        new_h = bh
        new_w = round(bh * ar_img)
    else:
        new_w = bw
        new_h = round(bw / ar_img)

    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    left = (new_w - bw) // 2
    top = (new_h - bh) // 2
    img = img.crop((left, top, left + bw, top + bh))

    target_dir.mkdir(parents=True, exist_ok=True)
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

    crop_by_path: dict[Path, tuple[int, int, int, int]] = {}
    if decisions:
        kept: list[Path] = []
        skipped_by_decision: list[Path] = []
        for p in image_files:
            decision = decisions.get(_rel_key(p), {})
            if decision.get("action") in {"skip", "move"}:
                skipped_by_decision.append(p)
                continue
            crop_rect = crop_rect_from_decision(decision.get("crop_bounds"))
            if crop_rect is not None:
                crop_by_path[p] = crop_rect
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
        stats.skipped = len(skipped)
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
                crop_by_path.get(img_path),
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
