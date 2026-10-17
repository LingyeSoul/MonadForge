#!/usr/bin/env python3
"""Build a tag-selected image pool (symlinks) out of a raw crawl tree.

The colorize task normally trains on the shared curated corpus
(``post_image_dataset/resized`` ← ``image_dataset`` ← the crawler's
``selected/``). Some tag slices are too thin there to teach anything — e.g.
``korean text`` has 22 pages in ``selected/`` but 349 in the crawler's
``retrieved/`` pool. This script carves such a slice out of a raw pool into a
standalone directory of **symlinks** (image + ``.txt`` sidecar), which then
goes through the ordinary resize → mangafy → cache path as its own dataset
subset, leaving the shared corpus untouched.

Selection = (any of ``--include-tags``) − (any of ``--exclude-tags``) − (stems
already present under any ``--skip-existing-in`` tree), matched against the
standalone comma-separated tags of each caption's first line (same matcher the
staging tag filters use, ``library.datasets.dreambooth.stems_with_any_tag``).

Example — the Korean-text colorize slice::

    python easycontrol_adapters/colorization/pool_from_tags.py \\
        --src ~/gelcrawl/retrieved \\
        --dst post_image_dataset/easycontrol/colorize/korean/src \\
        --include-tags "korean text" \\
        --skip-existing-in image_dataset

``monochrome``/``greyscale`` pages are deliberately *not* excluded here — the
colorize descriptor's ``[staging].exclude_data_includes`` and
``--target_drop_sat`` already pair them out downstream, and keeping them in the
pool makes the drop visible in one place.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from library.datasets.dreambooth import stems_with_any_tag
from library.datasets.image_utils import IMAGE_EXTENSIONS


def _image_for(caption: Path) -> Path | None:
    """The image sidecar-partner of ``caption`` (first extension that exists)."""
    for ext in IMAGE_EXTENSIONS:
        img = caption.with_suffix(ext)
        if img.is_file():
            return img
    return None


def _link(src: Path, dst: Path, *, overwrite: bool) -> bool:
    """Symlink ``src`` → ``dst`` (absolute target). Returns True if created."""
    if dst.is_symlink() or dst.exists():
        if not overwrite:
            return False
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src.resolve())
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src", required=True, help="raw pool root (nested artist dirs)"
    )
    parser.add_argument(
        "--dst", required=True, help="pool dir to populate with symlinks"
    )
    parser.add_argument(
        "--include-tags",
        nargs="+",
        required=True,
        metavar="TAG",
        help="keep pages carrying any of these standalone tags",
    )
    parser.add_argument(
        "--exclude-tags",
        nargs="*",
        default=[],
        metavar="TAG",
        help="drop pages carrying any of these standalone tags",
    )
    parser.add_argument(
        "--skip-existing-in",
        nargs="*",
        default=[],
        metavar="DIR",
        help="drop stems that already exist under these trees (e.g. the curated "
        "image_dataset master) so the slice adds no duplicates",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace existing links"
    )
    parser.add_argument("--limit", type=int, default=None, help="cap #pages (QA)")
    parser.add_argument(
        "--dry-run", action="store_true", help="report the selection, link nothing"
    )
    args = parser.parse_args()

    src, dst = Path(args.src).expanduser(), Path(args.dst).expanduser()
    if not src.is_dir():
        raise SystemExit(f"--src {src} is not a directory")

    keep = stems_with_any_tag(str(src), args.include_tags)
    drop = stems_with_any_tag(str(src), args.exclude_tags)
    seen: set[str] = set()
    for tree in args.skip_existing_in:
        for dirpath, _dirs, files in os.walk(Path(tree).expanduser(), followlinks=True):
            del dirpath
            seen.update(os.path.splitext(f)[0] for f in files if f.endswith(".txt"))

    stems = keep - drop - seen
    print(
        f"[pool] {len(keep)} tagged, -{len(keep & drop)} excluded-tag, "
        f"-{len(keep - drop) - len(stems)} already-present → {len(stems)} selected"
    )

    linked = no_image = 0
    for caption in sorted(src.rglob("*.txt")):
        if caption.stem not in stems:
            continue
        img = _image_for(caption)
        if img is None:
            no_image += 1
            continue
        rel = caption.relative_to(src)
        if args.dry_run:
            linked += 1
        else:
            _link(img, dst / rel.parent / img.name, overwrite=args.overwrite)
            _link(caption, dst / rel, overwrite=args.overwrite)
            linked += 1
        if args.limit and linked >= args.limit:
            break

    verb = "would link" if args.dry_run else "linked"
    print(
        f"[pool] {verb} {linked} pages into {dst}"
        + (f" ({no_image} had no image)" if no_image else "")
    )


if __name__ == "__main__":
    main()
