#!/usr/bin/env python3
"""Remove resized/latent/PE/mask caches that are stale for the active target_res.

Recomputes each image's correct bucket from its native size + the requested
``--target_res`` tiers and deletes any cache at the wrong bucket, so the next
``make preprocess`` / ``make mask`` regenerates it cleanly. Useful after adding
or dropping a ``target_res`` tier. Dry-run by default; pass ``--delete`` to act.

The scan/delete logic lives in ``library/preprocess/reconcile.py``; this file is
argparse only. Driven from ``make preprocess-reconcile`` (paths + target_res
resolved from the config chain).
"""

import argparse
from pathlib import Path

from library.datasets.buckets import ALLOWED_TARGET_RES
from library.preprocess.reconcile import (
    delete_orphans,
    find_orphan_caches,
    reconcile_caches,
)


def _run_orphans(args) -> None:
    """Remove caches whose source image no longer exists in --image-dir."""
    orphans = find_orphan_caches(
        Path(args.image_dir),
        Path(args.resized_dir),
        Path(args.cache_dir),
        Path(args.mask_dir),
    )
    print(
        f"orphan caches (source image gone from {args.image_dir}): "
        f"{len(orphans.npz)} latent npz, {len(orphans.te)} te, {len(orphans.pe)} pe, "
        f"{len(orphans.png)} resized png, {len(orphans.mask)} mask"
        f"  ({orphans.n_files} files)"
    )
    if not args.delete:
        for p in orphans.all_paths()[:20]:
            print(f"  {p}")
        if orphans.n_files > 20:
            print(f"  … and {orphans.n_files - 20} more")
        print("\n(dry run — pass --delete to remove)")
        return
    removed = delete_orphans(orphans)
    print(f"\nremoved: {dict(removed)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-dir", default="image_dataset", help="Native source images"
    )
    parser.add_argument(
        "--resized-dir", default="post_image_dataset/resized", help="Resized PNGs"
    )
    parser.add_argument(
        "--cache-dir",
        default="post_image_dataset/lora",
        help="Latent + PE cache dir",
    )
    parser.add_argument(
        "--mask-dir", default="post_image_dataset/masks", help="Mask dir"
    )
    parser.add_argument(
        "--target_res",
        type=int,
        nargs="+",
        default=[1024],
        metavar="EDGE",
        help=f"Active tiers (allowed: {' '.join(map(str, ALLOWED_TARGET_RES))})",
    )
    parser.add_argument(
        "--delete", action="store_true", help="Actually remove stale caches"
    )
    parser.add_argument(
        "--orphans",
        action="store_true",
        help="Instead of bucket reconcile, remove caches whose source image is "
        "gone from --image-dir (latent npz + te + pe + resized png + mask). "
        "Ignores --target_res.",
    )
    args = parser.parse_args()

    if args.orphans:
        _run_orphans(args)
        return

    bad = [e for e in args.target_res if e not in ALLOWED_TARGET_RES]
    if bad:
        parser.error(
            f"--target_res {bad} not in allowed tiers {list(ALLOWED_TARGET_RES)}"
        )

    print(f"target_res = {args.target_res}")
    stale, removed = reconcile_caches(
        Path(args.image_dir),
        Path(args.resized_dir),
        Path(args.cache_dir),
        Path(args.mask_dir),
        args.target_res,
        delete=args.delete,
    )

    print(f"\n{stale.n_images} images are at the wrong bucket for this target_res")
    if stale.changed:
        print("bucket changes (current → correct : count):")
        for (cur, cor), c in sorted(stale.changed.items(), key=lambda kv: -kv[1]):
            cur_s = (
                f"{cur[0]:>4}x{cur[1]:<4}" if isinstance(cur, tuple) else f"{cur:>9}"
            )
            print(f"  {cur_s} → {cor[0]:>4}x{cor[1]:<4} : {c}")
    print(
        f"\nstale files: {len(stale.npz)} latent npz, {len(stale.png)} resized png, "
        f"{len(stale.pe)} pe, {len(stale.mask)} mask"
    )

    if not args.delete:
        print("\n(dry run — pass --delete to remove)")
        return
    print(f"\nremoved: {dict(removed)}")
    print(
        "now re-run `make preprocess` (resize skips up-to-date images) and `make mask`."
    )


if __name__ == "__main__":
    main()
