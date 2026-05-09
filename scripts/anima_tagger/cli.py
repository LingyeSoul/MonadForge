"""CLI entry — argparse + mode dispatcher.

External-corpus paths are resolved via the ``CAPTION_CORPUS_DIR`` env var
(typically set in ``anima_lora/.env``). The corpus directory is expected to
contain ``retrieved/`` (raw caption pool), ``selected/`` (curated subset),
``tag_rules.yaml`` (caption normalization rules), and ``.tag_cache.json``
(per-tag Booru-style category cache, indexed under ``retrieved/``). All of
these can be overridden individually by CLI flags.

Modes (selected by ``--mode``):

* ``build_vocab``    — scan caption sources, intersect with the tag-taxonomy
                       cache, snapshot ``tag_rules.yaml``, emit
                       ``vocab.json`` plus a fixed train/val split and a
                       per-stem ``dataset.json`` manifest.
* ``build_features`` — encode every manifest image through frozen PE-Core,
                       mean-pool over patch tokens, write per-stem cache.
* ``build_resized``  — LANCZOS-resize every manifest image to its PE bucket,
                       cache as ``uint8 [C, H, W]`` for end-to-end PE-LoRA.
* ``train``          — train the multi-label head + 3-class rating head.
                       Dispatches to the cached path or PE-LoRA path based
                       on ``--pe_lora_rank``.
* ``calibrate``      — sweep per-tag F1-optimal thresholds on the val split.
* ``predict``        — single-image debug entry.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Make ``anima_lora/`` importable when this module is loaded via the
# ``scripts/train_anima_tagger.py`` shim — the shim adds parents[1] but
# direct ``python -m scripts.anima_tagger.cli`` invocation also lands here.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from library.env import load_dotenv  # noqa: E402
from library.log import setup_logging  # noqa: E402

# Pull CAPTION_CORPUS_DIR (and any other overrides) from anima_lora/.env
# before argparse builds defaults. CLI flags still win over env values.
load_dotenv()

setup_logging()
logger = logging.getLogger(__name__)


def _corpus_default(rel: str):
    """Resolve ``$CAPTION_CORPUS_DIR/<rel>`` for argparse defaults.

    Returns ``None`` when the env var is unset so argparse renders an
    explicit '(unset)' marker in --help instead of a misleading empty path.
    """
    root = os.environ.get("CAPTION_CORPUS_DIR")
    if not root:
        return None
    return str(Path(root) / rel)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Anima tagger trainer")
    p.add_argument(
        "--mode",
        choices=[
            "build_vocab",
            "build_features",
            "build_resized",
            "train",
            "calibrate",
            "predict",
        ],
        default="build_vocab",
    )
    p.add_argument(
        "--encoder",
        default="pe",
        help="Vision encoder registry name (passed to load_pe_encoder). "
        "Default: pe (PE-Core-L14-336).",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Torch device for build_features / train (default: cuda if available).",
    )
    p.add_argument(
        "--feature_cache_workers",
        type=int,
        default=4,
        help="DataLoader workers for build_features CPU-side decode + LANCZOS "
        "resize (default: 4). Set to 0 to run inline on the main process.",
    )

    # Vocab-build inputs. All three default to subpaths of
    # ``$CAPTION_CORPUS_DIR``; pass --caption_roots / --tag_cache / --rules
    # explicitly to override.
    raw_default = _corpus_default("retrieved")
    curated_default = _corpus_default("selected")
    p.add_argument(
        "--caption_roots",
        nargs="+",
        default=[d for d in (raw_default, curated_default, "image_dataset") if d],
        help="Directories to scan recursively for *.txt caption files. "
        "First-match-wins by stem when a duplicate appears across roots. "
        "Defaults: $CAPTION_CORPUS_DIR/retrieved + "
        "$CAPTION_CORPUS_DIR/selected + image_dataset/.",
    )
    p.add_argument(
        "--tag_cache",
        default=_corpus_default("retrieved/.tag_cache.json"),
        help="Tag-taxonomy JSON (tag → integer type ID). "
        "Default: $CAPTION_CORPUS_DIR/retrieved/.tag_cache.json.",
    )
    p.add_argument(
        "--rules",
        default=_corpus_default("tag_rules.yaml"),
        help="Caption-normalization rules (snapshotted into out_dir at "
        "build time). Default: $CAPTION_CORPUS_DIR/tag_rules.yaml.",
    )
    p.add_argument(
        "--groups",
        default=_corpus_default("tag_groups.yaml"),
        help="Tag-groups YAML (typed groupings — eye_color, hair_color, "
        "rating, …). Resolved against the kept vocab and embedded into "
        "vocab.json[groups]; the YAML is snapshotted to out_dir/groups.yaml. "
        "Optional — pass empty / unset to build a flat-vocab checkpoint. "
        "Default: $CAPTION_CORPUS_DIR/tag_groups.yaml.",
    )
    p.add_argument("--min_freq", type=int, default=5)
    p.add_argument("--val_frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)

    # Train-mode knobs.
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--d_hidden", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument(
        "--lambda_rating",
        type=float,
        default=0.1,
        help="Weight on the rating CE loss relative to multi-label BCE.",
    )

    # PE-LoRA knobs (end-to-end PE-Core fine-tuning on the trailing N blocks).
    # When --pe_lora_rank > 0, the trainer ignores the pre-pooled feature
    # cache and reads pre-resized images from .cache/resized-<encoder>/
    # (build via `--mode build_resized`). The frozen PE encoder runs each
    # step with LoRA active on the last `--pe_lora_layers` resblocks.
    p.add_argument(
        "--pe_lora_rank",
        type=int,
        default=0,
        help="LoRA rank on PE-Core's trailing blocks. 0 (default) → encoder "
        "stays frozen and trainer reads pre-pooled features from cache. "
        ">0 → end-to-end PE-LoRA training; reads pre-resized images from "
        ".cache/resized-<encoder>/ (build via --mode build_resized).",
    )
    p.add_argument(
        "--pe_lora_alpha",
        type=float,
        default=16.0,
        help="LoRA scale = alpha / rank.",
    )
    p.add_argument(
        "--pe_lora_layers",
        type=int,
        default=4,
        help="Number of trailing PE resblocks to adapt with LoRA. Mapped to "
        "inject_pe_lora's layer_from arg.",
    )
    p.add_argument(
        "--pe_lora_lr",
        type=float,
        default=1e-4,
        help="Learning rate for PE-LoRA params (head/trunk keeps --lr).",
    )
    p.add_argument(
        "--pe_lora_qkv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Adapt the QKV in_proj path (default: on).",
    )
    p.add_argument(
        "--pe_lora_attn_out",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Adapt attn.out_proj (default: on).",
    )
    p.add_argument(
        "--pe_lora_mlp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Adapt MLP c_fc / c_proj (default: on).",
    )

    # Predict mode: single-image debug entry.
    p.add_argument(
        "--image",
        default=None,
        help="Image path for --mode predict.",
    )
    p.add_argument(
        "--show_scores",
        action="store_true",
        help="Predict mode: also print rating distribution + top-K kept tags.",
    )
    p.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Predict mode: number of top kept tags to show with --show_scores.",
    )

    # Output.
    p.add_argument(
        "--out_dir",
        default="models/captioners/anima-tagger-v1",
    )

    args = p.parse_args()

    if args.mode == "build_vocab":
        missing = [
            name
            for name, val in (
                ("--tag_cache", args.tag_cache),
                ("--rules", args.rules),
            )
            if not val
        ]
        if missing or not args.caption_roots:
            raise SystemExit(
                "build_vocab needs CAPTION_CORPUS_DIR set in anima_lora/.env "
                f"(or {', '.join(missing) or '--caption_roots'} passed "
                "explicitly). Add a line like\n"
                "    CAPTION_CORPUS_DIR=/path/to/corpus\n"
                "to anima_lora/.env, or pass the paths via CLI flags."
            )

    return args


def main() -> None:
    args = parse_args()
    if args.mode == "build_vocab":
        from .vocab import cmd_build_vocab

        cmd_build_vocab(args)
    elif args.mode == "build_features":
        from .caches import cmd_build_features

        cmd_build_features(args)
    elif args.mode == "build_resized":
        from .caches import cmd_build_resized

        cmd_build_resized(args)
    elif args.mode == "train":
        if args.pe_lora_rank > 0:
            from .train_pe_lora import cmd_train_pe_lora

            cmd_train_pe_lora(args)
        else:
            from .train_cached import cmd_train_cached

            cmd_train_cached(args)
    elif args.mode == "calibrate":
        from .calibrate import cmd_calibrate

        cmd_calibrate(args)
    elif args.mode == "predict":
        from .predict import cmd_predict

        cmd_predict(args)
    else:
        raise SystemExit(f"unknown --mode={args.mode!r}")


if __name__ == "__main__":
    main()
