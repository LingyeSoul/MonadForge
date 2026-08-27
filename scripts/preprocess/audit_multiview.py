"""Audit `1girl` captions for images that are really several views of one girl.

Thin CLI over ``library.preprocess.multiview_audit``: loads SAM3 and the Anima
Tagger (the same two models ``caption-position`` uses, via the same builders),
sweeps the images that pipeline skips as ``single-subject``, and reports every
one where the ``girl`` prompt finds two or more subjects.

Dry-run by default. ``--apply`` writes the missing tag into the **caption
master** — unlike the clause rewrite, which only ever touches the derived
caption — because a missing ``multiple views`` is a fact about the picture that
every later stage should read down from. Follow it with ``make preprocess-te``.

GOTCHA: ``image_dataset/`` is gitignored, so an ``--apply`` is not
git-recoverable. ``report.json`` carries the verbatim before-text of every
caption it touched; keep it.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

# Monkey-patch numpy for sam3 compatibility (upstream pins numpy<2 and uses np.bool)
if not hasattr(np, "bool"):
    np.bool = np.bool_

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from library.env import resolve_under_home  # noqa: E402
from library.preprocess.instance_detection import (  # noqa: E402
    DEFAULT_SUBJECT_PROMPT_EMBED,
)
from library.preprocess.multiview_audit import (  # noqa: E402
    DEFAULT_IDENTITY_CONFIDENCE,
    DEFAULT_MULTIVIEW_PROB,
    EXTRA_CHARACTER,
    MULTIPLE_VIEWS,
    apply_findings,
    run_multiview_audit,
)
from library.preprocess.position_captions import (  # noqa: E402
    PositionCaptionOptions,
    load_clause_vocabulary,
)
from scripts.preprocess.position_captions import build_detect_fn  # noqa: E402

DEFAULT_REPORT_DIR = "post_image_dataset/captions/multiview_audit"


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
        "--apply",
        action="store_true",
        help="Write the suggested tag into the caption master (default: dry run)",
    )
    p.add_argument(
        "--apply_verdicts",
        "--apply-verdicts",
        dest="apply_verdicts",
        default=MULTIPLE_VIEWS,
        help=f"Comma-separated verdicts --apply may write "
        f"('{MULTIPLE_VIEWS}', '{EXTRA_CHARACTER}')",
    )
    p.add_argument(
        "--apply_confidence",
        "--apply-confidence",
        dest="apply_confidence",
        default="strong",
        help="Comma-separated confidence tiers --apply may write (strong, weak). "
        "A weak finding has only the geometry behind it — review the crops first",
    )
    p.add_argument(
        "--report_dir",
        "--report-dir",
        dest="report_dir",
        default=DEFAULT_REPORT_DIR,
        help=f"Where report.json lands (default: {DEFAULT_REPORT_DIR})",
    )
    p.add_argument(
        "--crops",
        action="store_true",
        help="Export the per-instance crops next to the report (review aid)",
    )
    p.add_argument(
        "--no_sheets",
        "--no-sheets",
        dest="sheets",
        action="store_false",
        help="Skip the per-finding contact sheets. They are the review surface — "
        "boxed original + the crops the tagger saw + the proposed edit, one PNG "
        "per finding under <report_dir>/sheets/, named verdict-first",
    )
    p.add_argument("--checkpoint", default="models/sam3/sam3.pt", help="SAM3 weights")
    p.add_argument("--tagger_dir", "--tagger-dir", dest="tagger_dir", default=None)
    p.add_argument("--device", default="cuda")

    g = p.add_argument_group("detection")
    g.add_argument("--prompt", default="girl", help="SAM3 text prompt for a subject")
    g.add_argument(
        "--prompt_embed",
        default=DEFAULT_SUBJECT_PROMPT_EMBED,
        help="learned soft prompt (.safetensors) standing in for --prompt on the "
        "subject pass (part prompts stay text); default = shipped, `none` = text",
    )
    g.add_argument(
        "--score_threshold",
        type=float,
        default=0.5,
        help="Subject confidence floor. Raising it trades recall for a shorter "
        "review list; this audit is precision-sensitive since every hit is "
        "read by hand",
    )
    g.add_argument("--retry_score_threshold", type=float, default=0.35)
    g.add_argument(
        "--part_prompts",
        "--part-prompts",
        dest="part_prompts",
        default="",
        help="Comma-separated body-part prompts, tried only when 'girl' finds "
        "fewer than two subjects — recovers a sheet whose second view is a "
        'headless close-up. Off by default; try "buttocks,hips,thighs"',
    )
    g.add_argument("--part_score_threshold", type=float, default=0.5)
    g.add_argument("--part_containment_threshold", type=float, default=0.7)
    g.add_argument("--iou_threshold", type=float, default=0.65)
    g.add_argument("--containment_threshold", type=float, default=1.01)
    g.add_argument(
        "--dedupe_fill_ratio",
        type=float,
        default=2.0,
        help="Mask-quality tie-break inside an NMS-matched pair; 0 = off "
        "(score-only survivor). See docs/experimental/multiview_audit.md §5.",
    )
    g.add_argument("--min_area_frac", type=float, default=0.005)
    g.add_argument("--pad", type=float, default=0.06)
    g.add_argument("--row_tol", type=float, default=0.25)
    g.add_argument("--max_instances", type=int, default=8)
    g.add_argument("--name_confidence", type=float, default=0.5)

    v = p.add_argument_group("verdict")
    v.add_argument(
        "--multiview_threshold",
        "--multiview-threshold",
        dest="multiview_threshold",
        type=float,
        default=DEFAULT_MULTIVIEW_PROB,
        help="Whole-image P(multiple views) at which the tagger counts as a "
        "witness — and, on its own, raises an image detection saw as one box",
    )
    v.add_argument(
        "--identity_confidence",
        "--identity-confidence",
        dest="identity_confidence",
        type=float,
        default=DEFAULT_IDENTITY_CONFIDENCE,
        help="Probability an identity-group winner needs before the verdict "
        "believes it. The group heads are softmax argmaxes, so they name a hair "
        "colour for a headless crop too — lowering this lets those back in",
    )
    v.add_argument(
        "--suggest_counts",
        "--suggest-counts",
        dest="suggest_counts",
        action="store_true",
        help=f"Also propose an 'Ngirls' fix for a '{EXTRA_CHARACTER}' verdict. Off "
        "because the 'girl' prompt does not exclude males — check the "
        "people-count head in the report before trusting any of these",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src = resolve_under_home(args.src)
    dst = resolve_under_home(args.dst)
    report_dir = resolve_under_home(args.report_dir)

    detect_fn, part_detect_fn, sam_model, sam_processor = build_detect_fn(args)

    from library.captioning.anima_tagger import (
        DEFAULT_TAGGER_DIR,
        AnimaTagger,
        ensure_tagger_checkpoint,
    )

    ckpt_dir = ensure_tagger_checkpoint(
        resolve_under_home(args.tagger_dir or DEFAULT_TAGGER_DIR)
    )
    print(f"Loading Anima Tagger from {ckpt_dir}...", flush=True)
    tagger = AnimaTagger(ckpt_dir, device=args.device)
    vocabulary = load_clause_vocabulary(ckpt_dir)

    options = PositionCaptionOptions(
        prompt=args.prompt,
        score_threshold=args.score_threshold,
        retry_score_threshold=args.retry_score_threshold,
        part_prompts=tuple(
            t.strip() for t in args.part_prompts.split(",") if t.strip()
        ),
        part_score_threshold=args.part_score_threshold,
        part_containment_threshold=args.part_containment_threshold,
        iou_threshold=args.iou_threshold,
        containment_threshold=args.containment_threshold,
        dedupe_fill_ratio=args.dedupe_fill_ratio,
        min_area_frac=args.min_area_frac,
        pad=args.pad,
        row_tol=args.row_tol,
        min_instances=2,
        max_instances=args.max_instances,
        name_confidence=args.name_confidence,
    )

    def progress(index: int, total: int, rel: str) -> None:
        if index % 200 == 0 or index == total:
            print(f"  [{index}/{total}] {rel}", flush=True)

    rows, stats = run_multiview_audit(
        resized_dir=dst,
        source_dir=src,
        detect_fn=detect_fn,
        tag_fn=tagger.predict,
        vocabulary=vocabulary,
        options=options,
        path_pattern=args.path_pattern,
        crops_dir=(report_dir / "crops") if args.crops else None,
        sheets_dir=(report_dir / "sheets") if args.sheets else None,
        progress=progress,
        part_detect_fn=part_detect_fn,
        multiview_threshold=args.multiview_threshold,
        identity_confidence=args.identity_confidence,
        suggest_counts=args.suggest_counts,
    )
    del sam_processor, sam_model

    written: list[tuple[str, str, str]] = []
    if args.apply:
        written = apply_findings(
            rows,
            source_dir=src,
            verdicts=tuple(
                v.strip() for v in args.apply_verdicts.split(",") if v.strip()
            ),
            confidences=tuple(
                c.strip() for c in args.apply_confidence.split(",") if c.strip()
            ),
        )

    summary = {
        "applied": bool(args.apply),
        "seen": stats.seen,
        "audited": stats.audited,
        "findings": stats.findings,
        "verdicts": dict(sorted(stats.verdicts.items(), key=lambda kv: -kv[1])),
        "by_confidence": {
            tier: sum(1 for r in rows if r.confidence == tier)
            for tier in ("strong", "weak")
        },
        "actionable": sum(1 for r in rows if r.suggested_tag),
        "by_source": {
            "detection": sum(1 for r in rows if r.source == "detection"),
            "tagger-only": sum(1 for r in rows if r.source == "tagger-only"),
        },
        "written": len(written),
        "part_prompts": list(options.part_prompts),
        "part_recovered": sum(
            1 for r in rows if any(c.source != "subject" for c in r.crops)
        ),
        "skipped": dict(sorted(stats.skipped.items(), key=lambda kv: -kv[1])),
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "report.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "images": [asdict(r) for r in rows],
                "written": [
                    {"caption_path": rel, "before": before, "after": after}
                    for rel, before, after in written
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nreport: {report_dir / 'report.json'}")
    if args.sheets:
        print(f"sheets: {report_dir / 'sheets'} (one PNG per finding, verdict-first)")
    if args.apply:
        print(
            f"\n{len(written)} caption(s) written to the master ({src}). Run "
            "`make preprocess-te` now to re-encode. The master is gitignored — "
            "report.json holds the before-text if you need to back this out."
        )
    else:
        print("\nDry run — no captions written. Re-run with --apply to write.")


if __name__ == "__main__":
    main()
