#!/usr/bin/env python3
"""Append position-aware clauses to multi-subject captions (SAM3 + Anima Tagger).

Thin CLI over ``library.preprocess.position_captions``: loads SAM3 and the Anima
Tagger, drives the detect → order → crop+blank → tag → compose pipeline over the
resized dataset, and writes a review report.

**Dry-run is the default.** Nothing is written to any caption until ``--apply``
is passed; a dry run emits ``report.json`` (+ the mask-blanked crops with
``--crops``) so the proposals can be eyeballed first.

After an ``--apply`` run the TE caches are stale but *look* current — caption
edits do not invalidate them. Follow with ``make preprocess-te``, which
regenerates the ``.variants.txt`` sidecars first (those override the CLI dropout
rate, so a stale sidecar would keep training the pre-clause caption).

    make caption-position                      # dry run over the whole dataset
    make caption-position ARGS="--apply"       # write the clauses
    make preprocess-te                         # re-encode (required after apply)
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
from library.preprocess.position_captions import (  # noqa: E402
    Detection,
    PositionCaptionOptions,
    load_clause_vocabulary,
    run_position_captions,
)

DEFAULT_REPORT_DIR = "post_image_dataset/captions/position"
# Both tokenizers pad to this (``--qwen3_max_token_length`` / ``--t5_…``); a
# caption past it is silently truncated, and the padding invariant means the
# tail simply never reaches the model.
DEFAULT_MAX_TOKENS = 512


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
        help="Write the proposed clauses into the caption master (default: dry run)",
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
        help="Also export the mask-blanked crops next to the report (review aid)",
    )
    p.add_argument("--checkpoint", default="models/sam3/sam3.pt", help="SAM3 weights")
    p.add_argument("--tagger_dir", "--tagger-dir", dest="tagger_dir", default=None)
    p.add_argument("--device", default="cuda")

    g = p.add_argument_group("detection")
    g.add_argument("--prompt", default="girl", help="SAM3 text prompt for a subject")
    g.add_argument("--score_threshold", type=float, default=0.5)
    g.add_argument(
        "--retry_score_threshold",
        type=float,
        default=0.35,
        help="Retry threshold when detection undershoots the expected count. "
        "This is SAM3's own confidence floor, not a post-filter — see "
        "build_detect_fn",
    )
    g.add_argument("--iou_threshold", type=float, default=0.65)
    g.add_argument(
        "--containment_threshold",
        "--containment-threshold",
        dest="containment_threshold",
        type=float,
        default=1.01,
        help="Suppress a box this nested inside a kept one (intersection over "
        "the smaller box). Off by default (>1.0 disables): a real second "
        "subject is as nested as a group box — enabling it cost 32 real "
        "subjects to save 12 group boxes",
    )
    g.add_argument(
        "--min_area_frac",
        "--min-area-frac",
        dest="min_area_frac",
        type=float,
        default=0.005,
        help="Drop detections smaller than this fraction of the image — an "
        "inset (a character on a phone screen) is not a bindable subject",
    )
    g.add_argument("--pad", type=float, default=0.06, help="bbox padding fraction")
    g.add_argument(
        "--no_blank_crops",
        "--no-blank-crops",
        dest="blank_crops",
        action="store_false",
        help="Skip mask-blanking (probe B: this is what caused the hair-color misses)",
    )
    g.add_argument(
        "--row_tol",
        type=float,
        default=0.25,
        help="Row-clustering gap as a fraction of image height (grid sheets)",
    )
    g.add_argument("--min_instances", type=int, default=2)
    g.add_argument("--max_instances", type=int, default=8)
    g.add_argument(
        "--no_strict_count",
        "--no-strict-count",
        dest="strict_count",
        action="store_false",
        help="Propose clauses even when detection disagrees with the girls-count",
    )

    c = p.add_argument_group("clause composition")
    c.add_argument("--max_clause_tags", type=int, default=8)
    c.add_argument(
        "--name_confidence",
        type=float,
        default=0.5,
        help="Confidence floor for putting a character name in a clause",
    )
    c.add_argument(
        "--allow_unlisted_names",
        "--allow-unlisted-names",
        dest="allow_unlisted_names",
        action="store_true",
        help="Allow a clause name the flat caption never mentions (off: probe B "
        "scored names 4/7, so an unlisted one is most likely a crop artifact)",
    )
    c.add_argument(
        "--keep_shared_tags",
        "--keep-shared-tags",
        dest="discriminative_only",
        action="store_false",
        help="Keep tags every crop agrees on in every clause. Off by default: on "
        "a multiple-views sheet all views share the character, hair and eyes, so "
        "repeating them binds nothing and crowds out the outfit that differs "
        "(they stay in the flat bag either way — v1 never removes anything).",
    )
    c.add_argument(
        "--qwen3",
        default=None,
        help="Qwen3 tokenizer path — enables the token-budget column in the report",
    )
    c.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    return p.parse_args()


def _under_root(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def build_detect_fn(args: argparse.Namespace):
    """SAM3 text-prompt detector returning per-instance boxes + masks.

    Two things this has to get right, both of which the first version didn't:

    * **SAM3 filters before we do.** ``Sam3Processor`` carries its own
      ``confidence_threshold`` (default 0.5) and applies it inside
      ``_forward_grounding`` — boxes below it never reach the caller. Filtering
      the returned list against a *lower* retry threshold is therefore a no-op:
      the low-score boxes were already discarded. The processor is built at the
      lowest threshold we might ask for, and the score gate is applied here.
    * **The retry must not re-encode.** ``detect_subjects`` calls this twice for
      the same image, so the raw detections are memoised per image and the retry
      is a pure re-filter.
    """
    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    print("Loading SAM3...", flush=True)
    model = build_sam3_image_model(
        device=args.device,
        eval_mode=True,
        checkpoint_path=str(_under_root(args.checkpoint)),
        load_from_HF=False,
    )
    floor = min(args.score_threshold, args.retry_score_threshold)
    processor = Sam3Processor(model, confidence_threshold=floor)
    cache: dict[str, object] = {"key": None, "dets": []}

    def _detect_all(image) -> list[Detection]:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            state = processor.set_image(image)
            out = processor.set_text_prompt(prompt=args.prompt, state=state)
        masks = out.get("masks")
        dets: list[Detection] = []
        for i, (box, score) in enumerate(zip(out["boxes"], out["scores"])):
            coords = box.tolist() if torch.is_tensor(box) else list(box)
            mask = None
            if masks is not None and i < len(masks):
                m = masks[i]
                mask = m.cpu().numpy() if torch.is_tensor(m) else np.asarray(m)
            dets.append(
                Detection(
                    box=tuple(float(v) for v in coords), score=float(score), mask=mask
                )
            )
        return dets

    def detect(image, score_threshold: float) -> list[Detection]:
        if cache["key"] is not image:
            cache["key"] = image
            cache["dets"] = _detect_all(image)
        return [d for d in cache["dets"] if d.score >= score_threshold]

    return detect, model, processor


def main() -> None:
    args = parse_args()
    src = _under_root(args.src)
    dst = _under_root(args.dst)
    report_dir = _under_root(args.report_dir)

    # SAM3 first, tagger second: the DiT-free pair still costs a few GB each and
    # detection has to finish before any crop exists to tag. Unlike the probe
    # they must both stay resident here — the pipeline is per-image, not
    # two dataset-wide passes, so a proposal never outlives its crop.
    detect_fn, sam_model, sam_processor = build_detect_fn(args)

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

    token_count_fn = None
    if args.qwen3:
        from library.anima.weights import load_qwen3_tokenizer

        tokenizer = load_qwen3_tokenizer(args.qwen3)

        def token_count_fn(text: str) -> int:
            return len(tokenizer(text, add_special_tokens=True)["input_ids"])

    options = PositionCaptionOptions(
        prompt=args.prompt,
        score_threshold=args.score_threshold,
        retry_score_threshold=args.retry_score_threshold,
        iou_threshold=args.iou_threshold,
        containment_threshold=args.containment_threshold,
        min_area_frac=args.min_area_frac,
        pad=args.pad,
        blank_crops=args.blank_crops,
        row_tol=args.row_tol,
        max_clause_tags=args.max_clause_tags,
        name_confidence=args.name_confidence,
        allow_unlisted_names=args.allow_unlisted_names,
        min_instances=args.min_instances,
        max_instances=args.max_instances,
        strict_count=args.strict_count,
        discriminative_only=args.discriminative_only,
    )

    def progress(index: int, total: int, rel: str) -> None:
        if index % 200 == 0 or index == total:
            print(f"  [{index}/{total}] {rel}", flush=True)

    rows, stats = run_position_captions(
        resized_dir=dst,
        source_dir=src,
        detect_fn=detect_fn,
        tag_fn=tagger.predict,
        vocabulary=vocabulary,
        options=options,
        path_pattern=args.path_pattern,
        apply=args.apply,
        crops_dir=(report_dir / "crops") if args.crops else None,
        token_count_fn=token_count_fn,
        progress=progress,
    )
    del sam_processor, sam_model

    over_budget = [
        r for r in rows if r.tokens is not None and r.tokens > args.max_tokens
    ]
    summary = {
        "applied": bool(args.apply),
        "seen": stats.seen,
        "candidates": stats.candidates,
        "proposed": stats.proposed,
        "written": stats.written,
        "skipped": dict(sorted(stats.skipped.items(), key=lambda kv: -kv[1])),
        "max_tokens": max(
            (r.tokens for r in rows if r.tokens is not None), default=None
        ),
        "over_token_budget": [r.image for r in over_budget],
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "report.json").write_text(
        json.dumps(
            {"summary": summary, "images": [asdict(r) for r in rows]},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nreport: {report_dir / 'report.json'}")
    if over_budget:
        print(
            f"WARNING: {len(over_budget)} caption(s) exceed {args.max_tokens} tokens — "
            "the tail truncates silently at TE-cache time."
        )
    if args.apply:
        print(
            "\nCaption edits do NOT invalidate the TE caches. Run "
            "`make preprocess-te` now to regenerate the variant sidecars and "
            "re-encode."
        )
    else:
        print("\nDry run — no captions written. Re-run with --apply to write.")


if __name__ == "__main__":
    main()
