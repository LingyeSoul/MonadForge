#!/usr/bin/env python3
"""Resize training images to constant-token bucket resolutions.

Reads images from a source directory, resizes and center-crops them to the
nearest bucket resolution, writes the results plus caption sidecars to an
output directory (mirroring the source subdir layout).

The walk → filter → parallel resize → caption-mirror loop lives in
``library/preprocess/images.py``; this file is argparse only.
"""

import argparse
from pathlib import Path


from library.datasets.curation_actions import load_curation_decisions
from library.preprocess import resize_to_buckets, tqdm_progress

# Re-exported for callers/tests that import the picklable worker directly
# (the loop moved to library/preprocess/images.py).
from library.preprocess.images import process_image  # noqa: F401,E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=str, required=True, help="Source image directory")
    parser.add_argument("--dst", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--resolution", type=int, default=1024, help="Max resolution (default: 1024)"
    )
    parser.add_argument(
        "--min_bucket_reso",
        type=int,
        default=512,
        help="Min bucket size (default: 512)",
    )
    parser.add_argument(
        "--max_bucket_reso",
        type=int,
        default=2048,
        help="Max bucket size (default: 2048)",
    )
    parser.add_argument(
        "--bucket_reso_steps",
        type=int,
        default=64,
        help="Bucket step size (default: 64)",
    )
    parser.add_argument(
        "--constant_token_buckets",
        action="store_true",
        default=True,
        help="Use constant-token buckets (default: True)",
    )
    parser.add_argument(
        "--no_constant_token_buckets",
        action="store_true",
        help="Disable constant-token buckets",
    )
    parser.add_argument(
        "--target_res",
        type=int,
        nargs="+",
        default=None,
        metavar="EDGE",
        help=(
            "Multi-scale constant-token tiers (allowed: 512 768 896 1024 1280 1536). "
            "Each image lands in the tier that resizes it the least (nearest). "
            "Default (unset) = single 1024 tier (current behavior). "
            "Each extra tier adds 1 compiled block graph (1024 contributes 2). "
            "e.g. --target_res 1024 1536 for ~1MP + ~2.25MP training."
        ),
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--min_pixels",
        type=int,
        default=500_000,
        help="Skip images with fewer than this many pixels (default: 500_000 = 0.5MP). "
        "Set to 0 to disable.",
    )
    parser.add_argument(
        "--no_copy_captions",
        action="store_true",
        help=(
            "Skip copying .txt / .caption sidecars to the output directory. "
            "Use when captions live elsewhere (e.g. text-encoder caching reads "
            "them from the original raw dataset directly)."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help=(
            "Walk subfolders under --src. Output mirrors the source subdir "
            "structure under --dst (image_dataset/charA/img.png → "
            "post_image_dataset/resized/charA/img.png). Stems must be unique "
            "within each subfolder; the same stem can repeat across folders."
        ),
    )
    parser.add_argument(
        "--path_pattern",
        "--path-pattern",
        dest="path_pattern",
        default="*",
        help=(
            "Only preprocess images whose path relative to --src matches this "
            "fnmatch glob. Use | to separate alternatives. Default: *"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Re-resize every image even if a resized PNG already exists at the "
            "correct bucket. Default skips up-to-date outputs; a bucket change "
            "(e.g. adding a --target_res tier) still re-resizes affected images."
        ),
    )
    parser.add_argument(
        "--curation_decisions",
        "--curation-decisions",
        dest="curation_decisions",
        default=None,
        help=(
            "Optional GUI curation decision JSON. Images marked action=skip/move are "
            "left out, and crop_bounds are applied only to generated resized "
            "outputs. Source images are not modified."
        ),
    )
    args = parser.parse_args()

    constant_token_buckets = (
        args.constant_token_buckets and not args.no_constant_token_buckets
    )

    if args.target_res is not None:
        from library.datasets.buckets import ALLOWED_TARGET_RES

        bad = [e for e in args.target_res if e not in ALLOWED_TARGET_RES]
        if bad:
            parser.error(
                f"--target_res {bad} not in allowed tiers {list(ALLOWED_TARGET_RES)}"
            )

    resize_to_buckets(
        Path(args.src),
        Path(args.dst),
        resolution=args.resolution,
        min_bucket_reso=args.min_bucket_reso,
        max_bucket_reso=args.max_bucket_reso,
        bucket_reso_steps=args.bucket_reso_steps,
        constant_token_buckets=constant_token_buckets,
        target_res=args.target_res,
        workers=args.workers,
        min_pixels=args.min_pixels,
        copy_captions=not args.no_copy_captions,
        recursive=args.recursive,
        path_pattern=args.path_pattern,
        overwrite=args.overwrite,
        curation_decisions=load_curation_decisions(
            args.curation_decisions,
            source_dir=Path(args.src),
        ),
        progress=tqdm_progress("Resizing"),
    )


if __name__ == "__main__":
    main()
