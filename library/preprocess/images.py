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

from library.datasets.buckets import DEFAULT_TARGET_RES
from library.preprocess._dataset import PreprocessStats, walk_images
from library.preprocess._progress import ProgressFn
from library.datasets.buckets import DEFAULT_FREEFIT_MAX_RATIO, FREEFIT_BAND_VERSION
from library.preprocess.resize_preview import (
    DEFAULT_FIT_MODE,
    DEFAULT_RESIZE_CROP_ANCHOR,
    RESIZE_CROP_ANCHORS,
    format_bucket_resos,
    margin_crop_rect,
    normalize_crop_anchor,
    normalize_crop_margins,
    normalize_fit_mode,
    parse_bucket_resos,
    select_resize_bucket,
)

CAPTION_EXTENSIONS = {".txt", ".caption"}
_RESIZE_ANCHOR_KEY = "anima_resize_crop_anchor"
_RESIZE_BUCKET_RESOS_KEY = "anima_resize_bucket_resos"
_RESIZE_MARGINS_KEY = "anima_resize_crop_margins"
_RESIZE_FIT_MODE_KEY = "anima_resize_fit_mode"
_RESIZE_MAX_RATIO_KEY = "anima_resize_max_ratio"
_RESIZE_BAND_VERSION_KEY = "anima_resize_band_version"


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
    crop_anchor: str,
    bucket_resos,
    crop_margins=None,
    fit_mode: str = DEFAULT_FIT_MODE,
    max_ratio: float = DEFAULT_FREEFIT_MAX_RATIO,
) -> dict[str, str]:
    anchor = normalize_crop_anchor(crop_anchor)
    buckets = format_bucket_resos(parse_bucket_resos(bucket_resos))
    margins = _format_margins(crop_margins)
    fit = normalize_fit_mode(fit_mode)
    # Default snap path keeps the empty (legacy-compatible) signature so existing
    # resized PNGs are not invalidated. Free-fit folds fit_mode + max_ratio in so
    # changing either re-resizes (the size-aware skip in process_image relies on
    # the signature matching).
    if (
        fit == DEFAULT_FIT_MODE
        and anchor == DEFAULT_RESIZE_CROP_ANCHOR
        and not buckets
        and margins == "0,0,0,0"
    ):
        return {}
    sig = {
        _RESIZE_ANCHOR_KEY: anchor,
        _RESIZE_BUCKET_RESOS_KEY: "|".join(buckets),
        _RESIZE_MARGINS_KEY: margins,
    }
    if fit != DEFAULT_FIT_MODE:
        sig[_RESIZE_FIT_MODE_KEY] = fit
        sig[_RESIZE_MAX_RATIO_KEY] = f"{float(max_ratio):g}"
        # The free-fit band derivation (freefit_band_for_edge) decides the resized
        # (W, H); fold its version in so widening the band re-resizes stale PNGs.
        sig[_RESIZE_BAND_VERSION_KEY] = str(FREEFIT_BAND_VERSION)
    return sig


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


def resize_to_bucket(
    img: Image.Image,
    bucket: tuple[int, int],
    *,
    crop_anchor: str = DEFAULT_RESIZE_CROP_ANCHOR,
) -> Image.Image:
    """Cover-scale ``img`` to ``bucket`` (LANCZOS) then anchor-crop to it.

    The exact pixel geometry preprocessing caches, factored out of
    ``process_image`` so embedders/previews resize identically (pair with
    ``resize_preview.select_resize_bucket`` to pick ``bucket``). ``img`` is taken
    as the already-EXIF-transposed, margin-cropped working region — pass what you
    want kept; aspect is read from ``img.size``.
    """
    bw, bh = bucket
    anchor_x, anchor_y = RESIZE_CROP_ANCHORS[normalize_crop_anchor(crop_anchor)]
    w, h = img.size
    ar_img = w / h
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
    return img.crop((left, top, left + bw, top + bh))


def process_image(
    image_path: Path,
    out_dir: Path,
    bucket_args: tuple,
    copy_captions: bool = True,
    rel_dir: str = "",
    overwrite: bool = False,
    force_edge: int | None = None,
    out_stem_suffix: str = "",
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
    # Leading (resolution, min/max bucket, step) elements are vestigial under
    # free-fit (the legacy aspect-ratio BucketManager path is gone); kept in the
    # tuple for layout stability. 5th+ elements are the active free-fit params.
    _max_reso, _min_size, _max_size, _reso_steps, *rest = bucket_args
    target_res = rest[0] if rest else None
    crop_anchor = rest[1] if len(rest) > 1 else DEFAULT_RESIZE_CROP_ANCHOR
    bucket_resos = rest[2] if len(rest) > 2 else None
    crop_margins = rest[3] if len(rest) > 3 else None
    fit_mode = rest[4] if len(rest) > 4 else DEFAULT_FIT_MODE
    max_ratio = rest[5] if len(rest) > 5 else DEFAULT_FREEFIT_MAX_RATIO

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

    # Free-fit (the only mode): choose_edge assigns the tier (default = canonical
    # 1024), then free-fit lands the native-aspect (W, H) inside that tier's band.
    # ``force_edge`` (autoscale curriculum emit) pins the tier instead of letting
    # choose_edge pick, so one source image is emitted at every ladder tier.
    if force_edge is not None:
        tier = [force_edge]
    else:
        tier = target_res or list(DEFAULT_TARGET_RES)
    _, bucket_reso = select_resize_bucket(
        work_w, work_h, tier, bucket_resos, fit_mode=fit_mode, max_ratio=max_ratio
    )

    bw, bh = bucket_reso
    crop_anchor = normalize_crop_anchor(crop_anchor)
    signature = _resize_metadata_signature(
        crop_anchor, bucket_resos, crop_margins, fit_mode, max_ratio
    )

    out_stem = f"{image_path.stem}{out_stem_suffix}"
    target_dir = out_dir / rel_dir if rel_dir else out_dir
    out_path = target_dir / f"{out_stem}.png"

    if not overwrite and out_path.exists():
        try:
            with Image.open(out_path) as ex:
                if ex.size == (bw, bh) and _resize_metadata_matches(ex, signature):
                    return f"{out_stem}.png", bucket_reso, True
        except Exception:
            pass

    img = img.convert("RGB")
    img = img.crop(margin_box)
    img = resize_to_bucket(img, bucket_reso, crop_anchor=crop_anchor)

    target_dir.mkdir(parents=True, exist_ok=True)
    _add_resize_metadata(save_kwargs, signature)
    img.save(out_path, format="PNG", **save_kwargs)

    if copy_captions:
        for ext in CAPTION_EXTENSIONS:
            cap = image_path.with_suffix(ext)
            if cap.exists():
                shutil.copy2(cap, target_dir / f"{out_stem}{ext}")

    return f"{out_stem}.png", bucket_reso, False


def resize_to_buckets(
    src: Path,
    dst: Path,
    *,
    resolution: int = 1024,
    min_bucket_reso: int = 512,
    max_bucket_reso: int = 2048,
    bucket_reso_steps: int = 64,
    target_res: list[int] | None = None,
    autoscale_tiers: list[int] | None = None,
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
    fit_mode: str = DEFAULT_FIT_MODE,
    max_ratio: float = DEFAULT_FREEFIT_MAX_RATIO,
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
        target_res,
        crop_anchor,
        parse_bucket_resos(bucket_resos),
        normalize_crop_margins(crop_margins),
        normalize_fit_mode(fit_mode),
        float(max_ratio),
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
        tiers = sorted(target_res) if target_res else list(DEFAULT_TARGET_RES)
        mode = f"free-fit (tiers {tiers}, band, max_ratio {float(max_ratio):g})"
        print(f"Resizing {len(image_files)} images to {mode} buckets")

    def _rel_for(p: Path) -> str:
        try:
            rel = p.parent.relative_to(src)
        except ValueError:
            return ""
        rel_str = str(rel)
        return "" if rel_str == "." else rel_str

    # Autoscale curriculum emit: one resized PNG per ladder tier per image
    # (stem-suffixed ``.as{edge}``) so each source image trains as an independent
    # sample at every tier. ``None``/empty → normal single-tier emit (choose_edge).
    if autoscale_tiers:
        emit_specs = [(edge, f".as{edge}") for edge in sorted(set(autoscale_tiers))]
    else:
        emit_specs = [(None, "")]

    if progress is not None:
        progress(0, total=len(image_files) * len(emit_specs))

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
                force_edge,
                stem_suffix,
            ): img_path
            for img_path in image_files
            for force_edge, stem_suffix in emit_specs
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
