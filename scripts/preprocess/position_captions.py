#!/usr/bin/env python3
"""Rewrite multi-subject captions into position clauses (SAM3 + Anima Tagger).

Thin CLI over ``library.preprocess.position_captions``: loads SAM3 and the Anima
Tagger, drives the detect → order → crop+blank → tag → compose pipeline over the
resized dataset, and writes a review report.

An attributable tag is **moved** out of the flat bag into its clause, so each
attribute is asserted exactly once and bound to its subject (v2). ``--no_rewrite``
restores the additive v1 behaviour for the training A/B; ``--flatten`` is the
inverse pass that merges clauses back into the bag.

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
    make caption-position ARGS="--flatten --apply"   # back the rewrite out
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
    flatten_captions,
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
    p.add_argument(
        "--flatten",
        action="store_true",
        help="Inverse pass: merge every caption's clauses back into its flat bag "
        "and drop the clauses. Text-only (no SAM3, no tagger) — this is how an "
        "--apply run is backed out, and how the clause-free control corpus for a "
        "training A/B is built. Flattens hand-written clauses too.",
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
    g.add_argument(
        "--part_prompts",
        "--part-prompts",
        dest="part_prompts",
        default="",
        help="Comma-separated body-part prompts, tried only when the subject "
        "prompt undershoots — recovers headless close-up panels (a hip / "
        "backside crop next to one full body) that 'girl' cannot see at any "
        'threshold. Off by default; try "buttocks,hips,thighs"',
    )
    g.add_argument(
        "--part_score_threshold",
        "--part-score-threshold",
        dest="part_score_threshold",
        type=float,
        default=0.5,
        help="Confidence floor for a body-part box (kept separate from the "
        "subject threshold — part prompts are the looser concept)",
    )
    g.add_argument(
        "--part_containment_threshold",
        "--part-containment-threshold",
        dest="part_containment_threshold",
        type=float,
        default=0.7,
        help="Drop a part box this nested inside an already-kept box. Unlike "
        "--containment_threshold this is safe to leave on: a part inside a "
        "subject is that subject's own body, never a second subject",
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
        "--max_novel_tags",
        "--max-novel-tags",
        dest="max_novel_tags",
        type=int,
        default=1,
        help="How many tags a clause may introduce that the caption never "
        "contained. The rest of the clause is filled from the flat bag first. "
        "Only a bag tag can MOVE — a novel one is a pure addition the curated "
        "caption never made. 0 = never invent, --max_clause_tags = the old "
        "bag-blind behaviour (46%% novel; on ama_mitsuki, 1 vs 8 cut novel tags "
        "515 to 115 and the caption 40%% shorter, with the moved set unchanged)",
    )
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
        "--ungated_identity",
        "--ungated-identity",
        dest="bag_gated_identity",
        action="store_false",
        help="Let a clause carry a hair/eye color the flat caption never listed. "
        "Gated by default: the caption is the curated ground truth, the crop "
        "tagger guesses one for every crop including headless ones, and "
        "discriminative-only then promotes the guess precisely because it "
        "disagrees — 520 of 1600 identity clause tags in the first full-corpus "
        "dry run contradicted the caption",
    )
    c.add_argument(
        "--bind_view_traits",
        "--bind-view-traits",
        dest="multi_view_gate",
        action="store_false",
        help="On a repeated-subject layout (`multiple views`, comic panels), "
        "let a clause carry the character's name and traits (hair, eyes, body, "
        "anatomy). Gated by default: every view or panel is the SAME girl, so "
        "those belong to her, not to a view — 45%% of the multiple-views clause "
        "tags in the first full-corpus dry run were view-invariant, and the ones "
        "that survived shared-tag suppression did so precisely because a crop "
        "disagreed",
    )
    c.add_argument(
        "--no_rewrite",
        "--no-rewrite",
        dest="rewrite",
        action="store_false",
        help="Additive v1: append the clauses but leave the flat bag untouched, "
        "so every bound attribute is asserted twice. Default is the v2 rewrite, "
        "which moves an attributable tag out of the bag into its clause. Kept for "
        "the training A/B arm",
    )
    c.add_argument(
        "--attribution_margin",
        "--attribution-margin",
        dest="attribution_margin",
        type=float,
        default=0.25,
        help="How far the winning crop's probability must clear every other "
        "crop's, RELATIVE to its own (1 - rival/winner), before a tag may LEAVE "
        "the flat bag (the clause carries it either way). Applies on top of the "
        "hard rule that no other crop kept the tag; 0.0 trusts the tagger's "
        "per-tag thresholds alone. Guards the one thing v2 can get wrong that "
        "v1 cannot: removing an attribute the other subjects also have",
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
    * **Neither escalation may re-encode.** ``detect_subjects`` calls back into
      this for the low-threshold retry and again per body-part prompt, all on the
      same image. The image encoding (``set_image``) and every prompt's raw
      detections are memoised per image, so the retry is a pure re-filter and a
      part prompt costs only a grounding pass — ``set_text_prompt`` re-grounds
      against the cached ``backbone_out`` and erases the previous text prompt, so
      running several over one state is safe.

    Returns ``(detect, part_detect, model, processor)``. ``part_detect`` takes
    the prompt as an argument; ``detect`` is pinned to ``args.prompt``.
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
    floor = min(
        args.score_threshold, args.retry_score_threshold, args.part_score_threshold
    )
    processor = Sam3Processor(model, confidence_threshold=floor)
    cache: dict[str, object] = {"key": None, "state": None, "dets": {}}

    def _ground(image, prompt: str) -> list[Detection]:
        """Raw detections for one prompt, reusing this image's encoded state."""
        if cache["key"] is not image:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                cache["state"] = processor.set_image(image)
            cache["key"] = image
            cache["dets"] = {}
        memo: dict = cache["dets"]  # type: ignore[assignment]
        if prompt in memo:
            return memo[prompt]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = processor.set_text_prompt(prompt=prompt, state=cache["state"])
        masks = out.get("masks")
        source = "subject" if prompt == args.prompt else prompt
        dets: list[Detection] = []
        for i, (box, score) in enumerate(zip(out["boxes"], out["scores"])):
            coords = box.tolist() if torch.is_tensor(box) else list(box)
            mask = None
            if masks is not None and i < len(masks):
                m = masks[i]
                mask = m.cpu().numpy() if torch.is_tensor(m) else np.asarray(m)
            dets.append(
                Detection(
                    box=tuple(float(v) for v in coords),
                    score=float(score),
                    mask=mask,
                    source=source,
                )
            )
        memo[prompt] = dets
        return dets

    def detect(image, score_threshold: float) -> list[Detection]:
        return [d for d in _ground(image, args.prompt) if d.score >= score_threshold]

    def part_detect(image, prompt: str, score_threshold: float) -> list[Detection]:
        return [d for d in _ground(image, prompt) if d.score >= score_threshold]

    return detect, part_detect, model, processor


def _run_flatten(args, src: Path, dst: Path, report_dir: Path) -> None:
    """The inverse pass — text only, so it short-circuits before any model load."""
    rows, stats = flatten_captions(
        resized_dir=dst,
        source_dir=src,
        path_pattern=args.path_pattern,
        apply=args.apply,
    )
    summary = {
        "mode": "flatten",
        "applied": bool(args.apply),
        "seen": stats.seen,
        "with_clauses": stats.candidates,
        "flattened": stats.proposed,
        "written": stats.written,
        "skipped": dict(sorted(stats.skipped.items(), key=lambda kv: -kv[1])),
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "flatten_report.json").write_text(
        json.dumps({"summary": summary, "images": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nreport: {report_dir / 'flatten_report.json'}")
    if args.apply:
        print(
            "\nCaption edits do NOT invalidate the TE caches. Run "
            "`make preprocess-te` now to regenerate the variant sidecars and "
            "re-encode."
        )
    else:
        print("\nDry run — no captions written. Re-run with --apply to write.")


def main() -> None:
    args = parse_args()
    src = _under_root(args.src)
    dst = _under_root(args.dst)
    report_dir = _under_root(args.report_dir)

    if args.flatten:
        _run_flatten(args, src, dst, report_dir)
        return

    # SAM3 first, tagger second: the DiT-free pair still costs a few GB each and
    # detection has to finish before any crop exists to tag. Unlike the probe
    # they must both stay resident here — the pipeline is per-image, not
    # two dataset-wide passes, so a proposal never outlives its crop.
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
        part_prompts=tuple(
            t.strip() for t in args.part_prompts.split(",") if t.strip()
        ),
        part_score_threshold=args.part_score_threshold,
        part_containment_threshold=args.part_containment_threshold,
        iou_threshold=args.iou_threshold,
        containment_threshold=args.containment_threshold,
        min_area_frac=args.min_area_frac,
        pad=args.pad,
        blank_crops=args.blank_crops,
        row_tol=args.row_tol,
        max_clause_tags=args.max_clause_tags,
        max_novel_tags=args.max_novel_tags,
        name_confidence=args.name_confidence,
        allow_unlisted_names=args.allow_unlisted_names,
        min_instances=args.min_instances,
        max_instances=args.max_instances,
        strict_count=args.strict_count,
        discriminative_only=args.discriminative_only,
        bag_gated_identity=args.bag_gated_identity,
        multi_view_gate=args.multi_view_gate,
        rewrite=args.rewrite,
        attribution_margin=args.attribution_margin,
    )

    def progress(index: int, total: int, rel: str) -> None:
        if index % 200 == 0 or index == total:
            print(f"  [{index}/{total}] {rel}", flush=True)

    rows, stats = run_position_captions(
        resized_dir=dst,
        source_dir=src,
        detect_fn=detect_fn,
        part_detect_fn=part_detect_fn,
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
        "rewrite": bool(args.rewrite),
        "attribution_margin": args.attribution_margin,
        "seen": stats.seen,
        "candidates": stats.candidates,
        "proposed": stats.proposed,
        "written": stats.written,
        # v2: how much of the flat bag the clauses actually took, and which of
        # the two safety rules pinned the rest. Zero under --no_rewrite.
        "rewritten": stats.rewritten,
        "moved_tags": stats.moved_tags,
        # Clause composition. ``reuse_ratio`` is the headline for the
        # move-don't-invent rule: a bag tag is a candidate move, a novel one can
        # only ever be an addition.
        "max_novel_tags": args.max_novel_tags,
        "clause_tags": stats.clause_tags,
        "novel_tags": stats.novel_tags,
        "reuse_ratio": (
            round(1.0 - stats.novel_tags / stats.clause_tags, 3)
            if stats.clause_tags
            else None
        ),
        "pinned_tags": dict(sorted(stats.pinned_tags.items(), key=lambda kv: -kv[1])),
        "skipped": dict(sorted(stats.skipped.items(), key=lambda kv: -kv[1])),
        "part_prompts": list(options.part_prompts),
        # Images the body-part fallback actually rescued: at least one bound
        # instance whose box came from a part prompt. Zero with the feature off.
        "part_recovered": sum(
            1 for r in rows if any(i.source != "subject" for i in r.instances)
        ),
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
        if args.rewrite and stats.moved_tags:
            print(
                f"{stats.moved_tags} tag(s) moved out of the flat bag across "
                f"{stats.rewritten} caption(s). To back that out: "
                '`make caption-position ARGS="--flatten --apply"`.'
            )
    else:
        print("\nDry run — no captions written. Re-run with --apply to write.")


if __name__ == "__main__":
    main()
