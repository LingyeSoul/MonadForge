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
    args = parser.parse_args()

    constant_token_buckets = (
        args.constant_token_buckets and not args.no_constant_token_buckets
    )

    resize_to_buckets(
        Path(args.src),
        Path(args.dst),
        resolution=args.resolution,
        min_bucket_reso=args.min_bucket_reso,
        max_bucket_reso=args.max_bucket_reso,
        bucket_reso_steps=args.bucket_reso_steps,
        constant_token_buckets=constant_token_buckets,
        workers=args.workers,
        min_pixels=args.min_pixels,
        copy_captions=not args.no_copy_captions,
        recursive=args.recursive,
        progress=tqdm_progress("Resizing"),
    )


if __name__ == "__main__":
    main()
