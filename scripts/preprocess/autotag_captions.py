#!/usr/bin/env python3
"""Auto-tag the dataset with the Anima Tagger and write the caption master.

Thin CLI over ``library.preprocess.autotag``: loads the tagger (auto-downloading
the checkpoint if missing), walks the resized tree, and proposes a caption per
image. The dataset-wide counterpart to the Dataset tab's per-image autotag
button.

**Dry-run is the default.** Nothing is written until ``--apply`` is passed; a
dry run emits ``report.json`` so the proposals can be eyeballed first.

Three modes (``--mode``):

    missing    only images with no caption sidecar (default — never destructive)
    merge      tag everything, append only tags the caption lacks (clauses kept)
    overwrite  replace the caption with the tagger's output (destructive)

After an ``--apply`` run the TE caches are stale but *look* current — caption
edits do not invalidate them. Follow with ``make preprocess-te``, which
regenerates the ``.variants.txt`` sidecars first (those override the CLI dropout
rate, so a stale sidecar would keep training the pre-tag caption).

    make caption-autotag                              # dry run, missing-only
    make caption-autotag ARGS="--apply"               # write them
    make caption-autotag ARGS="--mode merge --apply"  # top up existing captions
    make preprocess-te                                # re-encode (required)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from library.env import resolve_under_home  # noqa: E402
from library.preprocess.autotag import (  # noqa: E402
    MODES,
    AutotagOptions,
    build_tag_fn,
    run_autotag_captions,
)

DEFAULT_REPORT_DIR = "post_image_dataset/captions/autotag"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", default="image_dataset", help="Caption master dir")
    p.add_argument("--dst", default="post_image_dataset/resized", help="Resized images")
    p.add_argument(
        "--path_pattern",
        "--path-pattern",
        dest="path_pattern",
        default="*",
        help="fnmatch glob (| to OR-combine) on the path relative to --dst",
    )
    p.add_argument(
        "--mode",
        choices=MODES,
        default="missing",
        help="missing: only uncaptioned images (default). merge: append novel "
        "tags to every caption, keeping its position clauses. overwrite: "
        "replace the caption outright (discards hand-written text)",
    )
    p.add_argument(
        "--min_confidence",
        "--min-confidence",
        dest="min_confidence",
        type=float,
        default=0.0,
        help="Extra probability floor on top of the tagger's per-tag F1 "
        "thresholds (0-1). 0 leaves its own decisions untouched",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Write the proposed captions into the master (default: dry run)",
    )
    p.add_argument(
        "--report_dir",
        "--report-dir",
        dest="report_dir",
        default=DEFAULT_REPORT_DIR,
        help=f"Where report.json lands (default: {DEFAULT_REPORT_DIR})",
    )
    p.add_argument("--tagger_dir", "--tagger-dir", dest="tagger_dir", default=None)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    resized_dir = resolve_under_home(args.dst)
    source_dir = resolve_under_home(args.src)
    if not resized_dir.exists():
        raise SystemExit(
            f"resized dir not found: {resized_dir} — run `make preprocess-resize` first"
        )

    options = AutotagOptions(mode=args.mode, min_confidence=args.min_confidence)
    print(f"Loading Anima Tagger ({args.device})...", flush=True)
    tag_fn, info = build_tag_fn(
        args.tagger_dir, device=args.device, min_confidence=args.min_confidence
    )

    def progress(index: int, total: int, rel: str) -> None:
        if index == 1 or index == total or index % 50 == 0:
            print(f"  [{index}/{total}] {rel}", flush=True)

    rows, stats = run_autotag_captions(
        resized_dir=resized_dir,
        source_dir=source_dir,
        tag_fn=tag_fn,
        options=options,
        path_pattern=args.path_pattern,
        apply=args.apply,
        progress=progress,
    )

    report_dir = resolve_under_home(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "mode": args.mode,
        "min_confidence": args.min_confidence,
        "apply": args.apply,
        "src": str(source_dir),
        "dst": str(resized_dir),
        "path_pattern": args.path_pattern,
        **dict(info),
        "stats": {
            "seen": stats.seen,
            "candidates": stats.candidates,
            "proposed": stats.proposed,
            "written": stats.written,
            "skipped": dict(stats.skipped),
        },
        "rows": [asdict(r) for r in rows],
    }
    report_path = report_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"\nseen={stats.seen} candidates={stats.candidates} "
        f"proposed={stats.proposed} written={stats.written}"
    )
    for reason, count in stats.skipped.most_common():
        print(f"  skip:{reason} {count}")
    print(f"report → {report_path}")
    if not args.apply:
        print("dry run — nothing written. Re-run with --apply to commit.")
    elif stats.written:
        print("captions changed — run `make preprocess-te` to re-encode.")


if __name__ == "__main__":
    main()
