#!/usr/bin/env python3
"""Write corrected captions next to resized preprocessing images."""

from __future__ import annotations

import argparse
from pathlib import Path

from library.captioning.correction import (
    CaptionCorrectionOptions,
    find_tag_csv,
    load_tag_knowledge_base,
)
from library.captioning.preprocess import write_corrected_preprocess_captions


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="Raw source image directory")
    parser.add_argument("--dst", required=True, help="Resized image directory")
    parser.add_argument(
        "--tag_csv",
        default=None,
        help="danbooru_tags_classified.csv path (default: models/ lookup)",
    )
    parser.add_argument(
        "--path_pattern",
        "--path-pattern",
        dest="path_pattern",
        default="*",
        help="Only write captions for resized images matching this relative glob",
    )
    parser.add_argument("--recursive", action="store_true", help="Walk subfolders")
    parser.add_argument(
        "--caption_insert_no_artist",
        "--caption-insert-no-artist",
        dest="caption_insert_no_artist",
        action="store_true",
        help="Insert @no-artist at the artist slot when no artist marker exists",
    )
    parser.add_argument(
        "--caption_trigger_word",
        "--caption-trigger-word",
        dest="caption_trigger_word",
        default="",
        help="Trigger tag to move into the caption order",
    )
    parser.add_argument(
        "--caption_trigger_at_front",
        "--caption-trigger-at-front",
        dest="caption_trigger_at_front",
        action="store_true",
        help="Place caption_trigger_word at the very front instead of artist slot",
    )
    parser.add_argument(
        "--no_correct",
        "--no-correct",
        dest="no_correct",
        action="store_true",
        help=(
            "Skip bucket-reordering — mirror the raw source caption verbatim as "
            "v0 (the variant-only path: shuffle sidecars without reordering)."
        ),
    )
    parser.add_argument(
        "--caption_shuffle_variants",
        "--caption-shuffle-variants",
        dest="caption_shuffle_variants",
        type=int,
        default=0,
        help=(
            "Number of caption variants to materialize as {stem}.variants.txt "
            "sidecars (0 = none). v0 is the corrected caption; v1..v{N-1} are "
            "smart-shuffled. The TE step encodes these verbatim."
        ),
    )
    parser.add_argument(
        "--caption_tag_dropout_rate",
        "--caption-tag-dropout-rate",
        dest="caption_tag_dropout_rate",
        type=float,
        default=0.0,
        help="Per-tag dropout probability for v1..v{N-1} (ignored without variants).",
    )
    parser.add_argument(
        "--caption_tag_randomize_rate",
        "--caption-tag-randomize-rate",
        dest="caption_tag_randomize_rate",
        type=float,
        default=0.0,
        help=(
            "Per-tag identity-randomize probability — emits an r-family alongside "
            "the v-family. Needs --qwen3 + --t5_tokenizer_path to build the "
            "dual-single erasure pool. Ignored without >=2 variants."
        ),
    )
    parser.add_argument(
        "--qwen3",
        type=str,
        default=None,
        help="Qwen3 tokenizer path (tokenizer-only load; required for randomize).",
    )
    parser.add_argument(
        "--t5_tokenizer_path",
        "--t5-tokenizer-path",
        dest="t5_tokenizer_path",
        type=str,
        default=None,
        help="T5 tokenizer path (default: library/anima/configs/t5_old/).",
    )
    args = parser.parse_args()

    csv_path = Path(args.tag_csv) if args.tag_csv else find_tag_csv(ROOT)
    if csv_path is None or not csv_path.exists():
        raise SystemExit(
            "danbooru_tags_classified.csv not found. Run "
            "`python tasks.py download-danbooru-tags` first."
        )

    num_variants = int(args.caption_shuffle_variants or 0)
    randomize_rate = float(args.caption_tag_randomize_rate or 0.0)
    # The erasure pool (identity-randomize only) needs both tokenizers. Loaded
    # tokenizer-only — no encoder weights — so this stays a light pass.
    qwen3_tokenizer = None
    t5_tokenizer = None
    if randomize_rate > 0.0 and num_variants >= 2:
        from library.anima.weights import load_qwen3_tokenizer, load_t5_tokenizer

        if not args.qwen3:
            raise SystemExit(
                "--caption_tag_randomize_rate > 0 requires --qwen3 (tokenizer path)."
            )
        qwen3_tokenizer = load_qwen3_tokenizer(args.qwen3)
        t5_tokenizer = load_t5_tokenizer(args.t5_tokenizer_path)

    stats = write_corrected_preprocess_captions(
        Path(args.src),
        Path(args.dst),
        load_tag_knowledge_base(csv_path),
        options=CaptionCorrectionOptions(
            insert_no_artist=bool(args.caption_insert_no_artist),
            trigger_word=str(args.caption_trigger_word or ""),
            trigger_at_front=bool(args.caption_trigger_at_front),
        ),
        recursive=bool(args.recursive),
        path_pattern=str(args.path_pattern or "*"),
        correct=not bool(args.no_correct),
        num_variants=num_variants,
        tag_dropout_rate=float(args.caption_tag_dropout_rate or 0.0),
        tag_randomize_rate=randomize_rate,
        qwen3_tokenizer=qwen3_tokenizer,
        t5_tokenizer=t5_tokenizer,
    )
    print(
        "Corrected preprocess captions: "
        f"{stats.written} written, {stats.unchanged} unchanged, "
        f"{stats.missing_source} missing source, {stats.removed_stale} stale removed, "
        f"{stats.variants_written} variant sidecars "
        f"({stats.seen} resized images)"
    )


if __name__ == "__main__":
    main()
