"""``python -m anime_tools.exclude`` — take images out of the pipeline, or put
them back, or print the ledger. Dry run by default.

The one check the library does not do lives here: an *exclusion* must name a
file under ``--src``, so a typo is caught before anything moves. ``--restore``
skips that check on purpose, since the source may be gone by then.
"""

from __future__ import annotations

import argparse
import sys
import time

from library.curation_compat import workspace as WS
from library.curation_compat._env import resolve_path
from library.curation_compat.exclude._ledger import (
    ExclusionError,
    Trees,
    manifest_path,
    read_entries,
    rel_key,
)
from library.curation_compat.exclude._move import Result, exclude_many, restore_many

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m anime_tools.exclude",
        description="Take images out of the pipeline, or put them back. An "
        "excluded image's files move into workspace/_excluded/ and its path goes "
        "into the ledger there, which `resize` reads as extra --skip entries — so "
        "no later stage ever sees it again, and Export republishes it under "
        "<out>/_excluded/ rather than into the tree the trainer reads. Dry run by "
        "default.",
    )
    p.add_argument(
        "rels",
        nargs="*",
        metavar="REL",
        help="Image paths relative to --src, forward slashes (char_aki/a.jpg)",
    )
    p.add_argument(
        "--restore",
        action="store_true",
        help="Put these images back instead: every file returns to the tree it "
        "came from and the rel leaves the ledger",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="Print the ledger — one rel per line, with what moved — and exit",
    )
    p.add_argument("--note", default="", help="Why, recorded on the ledger row")
    p.add_argument(
        "--src",
        default=WS.SOURCE_ROOT,
        help=f"Caption master dir the rels are relative to (default: "
        f"{WS.SOURCE_ROOT}). An excluded rel must name a file under it; "
        "--restore does not check, since the source may be gone by then.",
    )
    p.add_argument("--dst", default=WS.RESIZED, help=f"Resized tree ({WS.RESIZED})")
    p.add_argument("--masks", default=WS.MASKS, help=f"Mask tree ({WS.MASKS})")
    p.add_argument("--ocr_dir", default=WS.OCR, help=f"OCR sidecar tree ({WS.OCR})")
    p.add_argument(
        "--excluded_dir",
        default=WS.EXCLUDED,
        help=f"The excluded tree and its ledger (default: {WS.EXCLUDED})",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Move for real (default: say what would move and stop)",
    )
    return p


def _print_ledger(trees: Trees) -> None:
    entries = read_entries(trees.excluded)
    if not entries:
        print(f"nothing excluded ({manifest_path(trees.excluded)})")
        return
    for rel in sorted(entries):
        e = entries[rel]
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(e.at)) if e.at else "?"
        print(
            f"{rel}  [{when}] {len(e.moved)} file(s){'  — ' + e.note if e.note else ''}"
        )
        for slot in e.moved:
            print(f"    {slot}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    trees = Trees.build(
        dst=args.dst, masks=args.masks, ocr=args.ocr_dir, excluded=args.excluded_dir
    )

    if args.list:
        _print_ledger(trees)
        return 0
    if not args.rels:
        print("no images named; pass a REL or --list", file=sys.stderr)
        return 2

    src = resolve_path(args.src)
    keys: list[str] = []
    for raw in args.rels:
        try:
            key = rel_key(raw)
            if not args.restore and not (src / key).is_file():
                raise ExclusionError(f"not in the dataset: {key} (under {src})")
        except ExclusionError as e:
            print(f"{raw}: {e}", file=sys.stderr)
            return 2
        keys.append(key)

    # Every rel checked first, then one pass over one ledger: N images named
    # here cost one read and one write, not N of each.
    results: list[Result] = (
        restore_many(trees, keys, apply=args.apply)
        if args.restore
        else exclude_many(trees, keys, note=args.note, apply=args.apply)
    )

    verb = "would move" if not args.apply else "moved"
    for r in results:
        print(f"{r.rel}: {r.action}, {verb} {len(r.moved)} file(s)")
        for slot in r.moved:
            print(f"    {slot}")
        for slot in r.skipped:
            print(f"    kept in _excluded (live path occupied): {slot}")
    if not args.apply:
        print("\nDry run — nothing moved. Re-run with --apply.")
    return 0
