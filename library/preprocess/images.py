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
from PIL.PngImagePlugin import PngInfo

from library.datasets.buckets import BucketManager
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
) -> tuple[str, tuple[int, int]]:
    """Worker — receives bucket params (not a BucketManager) to stay picklable.

    ``rel_dir`` is the (possibly empty) relative subdir under the source root;
    the output mirrors it as ``out_dir / rel_dir / stem.png``. Empty ``rel_dir``
    collapses to the flat layout.
    """
    max_reso, min_size, max_size, reso_steps, use_constant = bucket_args
    bucket_mgr = BucketManager(
        max_reso=max_reso,
        min_size=min_size,
        max_size=max_size,
        reso_steps=reso_steps,
    )
    bucket_mgr.make_buckets(constant_token_buckets=use_constant)

    src_img = Image.open(image_path)
    save_kwargs = _collect_metadata(src_img)
    img = src_img.convert("RGB")
    w, h = img.size

    bucket_reso, _, _ = bucket_mgr.select_bucket(w, h)
    bw, bh = bucket_reso

    # Resize preserving aspect ratio so the image covers the bucket.
    ar_img = w / h
    ar_bucket = bw / bh
    if ar_img > ar_bucket:
        new_h = bh
        new_w = round(bh * ar_img)
    else:
        new_w = bw
        new_h = round(bw / ar_img)

    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Center crop to bucket resolution.
    left = (new_w - bw) // 2
    top = (new_h - bh) // 2
    img = img.crop((left, top, left + bw, top + bh))

    target_dir = out_dir / rel_dir if rel_dir else out_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    out_path = target_dir / f"{image_path.stem}.png"
    img.save(out_path, format="PNG", **save_kwargs)

    if copy_captions:
        for ext in CAPTION_EXTENSIONS:
            cap = image_path.with_suffix(ext)
            if cap.exists():
                shutil.copy2(cap, target_dir / f"{image_path.stem}{ext}")

    return image_path.name, bucket_reso


def resize_to_buckets(
    src: Path,
    dst: Path,
    *,
    resolution: int = 1024,
    min_bucket_reso: int = 512,
    max_bucket_reso: int = 2048,
    bucket_reso_steps: int = 64,
    constant_token_buckets: bool = True,
    workers: int = 4,
    min_pixels: int = 500_000,
    copy_captions: bool = True,
    recursive: bool = False,
    verbose: bool = True,
    progress: ProgressFn | None = None,
) -> tuple[PreprocessStats, dict[tuple[int, int], int]]:
    """Resize+crop every image under ``src`` into bucket resolutions under ``dst``.

    Mirrors the source subdir layout, copies caption sidecars, and skips images
    below ``min_pixels``. Returns ``(stats, bucket_counts)`` where
    ``bucket_counts`` maps each ``(W, H)`` bucket to its image count. Pass
    ``progress`` for a per-image bar.
    """
    dst.mkdir(parents=True, exist_ok=True)

    bucket_args = (
        (resolution, resolution),
        min_bucket_reso,
        max_bucket_reso,
        bucket_reso_steps,
        constant_token_buckets,
    )

    # walk_images enforces per-subfolder stem uniqueness (same-folder stem
    # collisions would collide the resized output).
    image_files = walk_images(src, recursive=recursive)
    stats = PreprocessStats(seen=len(image_files))

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
        print(
            f"Resizing {len(image_files)} images to "
            f"{'constant-token' if constant_token_buckets else 'standard'} buckets"
        )

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
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                process_image,
                img_path,
                dst,
                bucket_args,
                copy_captions,
                _rel_for(img_path),
            ): img_path
            for img_path in image_files
        }
        for future in as_completed(futures):
            name, reso = future.result()
            bucket_counts[reso] = bucket_counts.get(reso, 0) + 1
            stats.written += 1
            if progress is not None:
                progress(1, detail=f"{name} → {reso[0]}x{reso[1]}")

    if verbose:
        print("\nBucket distribution:")
        for reso in sorted(bucket_counts):
            tokens = (reso[0] // 16) * (reso[1] // 16)
            print(
                f"  {reso[0]:>4d}x{reso[1]:<4d}: {bucket_counts[reso]:>3d} "
                f"images  ({tokens} tokens)"
            )

    return stats, bucket_counts
