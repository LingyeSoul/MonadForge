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
* ``build_features`` — encode every manifest image through frozen PE-Core +
                       PE-Spatial and write per-stem caches. Each side's
                       layout follows ``--pool_kind`` / ``--pool_kind_aux``
                       (``map`` = full token sequence, ``mean`` = pooled vector).
* ``train``          — train the dual-encoder hard-routed head: multi-label
                       tags + 3-class rating + 8-class people-count.
* ``calibrate``      — sweep per-tag F1-optimal thresholds on the val split.
* ``predict``        — single-image debug entry.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from library.env import load_dotenv  # noqa: E402
from library.log import setup_logging  # noqa: E402

# Pull CAPTION_CORPUS_DIR from anima_lora/.env before argparse builds defaults;
# CLI flags still win over env values.
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
            "train",
            "calibrate",
            "predict",
            "scan_role_markers",
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
        "--aux_encoder",
        default="pe_spatial",
        help="Spatial vision encoder for the dual-encoder head (default: "
        "'pe_spatial' for PE-Spatial-B16-512). build_features builds a "
        "parallel cache; train routes localized tags through this encoder's "
        "trunk. Dual encoder is mandatory — this must name a real encoder "
        "different from --encoder.",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Torch device for build_features / train (default: cuda if available).",
    )
    p.add_argument(
        "--feature_cache_workers",
        type=int,
        default=5,
        help="DataLoader workers for build_features CPU-side decode + LANCZOS "
        "resize (default: 4). Set to 0 to run inline on the main process.",
    )
    p.add_argument(
        "--feature_cache_batch_size",
        type=int,
        default=8,
        help="Images per encoder forward during build_features. Stems are "
        "grouped by aspect bucket so each batch is shape-homogeneous; raise "
        "for more GPU throughput / lower for less VRAM (default: 8).",
    )

    # Vocab-build inputs default to subpaths of ``$CAPTION_CORPUS_DIR``.
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
    p.add_argument("--min_freq", type=int, default=50)
    p.add_argument("--val_frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=96)
    p.add_argument(
        "--ram_resident",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load the whole packed feature set into RAM once at startup and "
        "serve batches from memory (no per-epoch disk IO; runs the loader inline "
        "with a free global shuffle). Needs ~feature-set-sized RAM (~40 GB here). "
        "Use --no-ram_resident to fall back to the mmap-shard path with chunked "
        "shuffle + prefetch workers (default: on).",
    )
    p.add_argument(
        "--shuffle_chunk_size",
        type=int,
        default=2048,
        help="IO-locality knob for the cached-feature loader. Each epoch "
        "shuffles within contiguous chunks of this many samples (snapped to a "
        "multiple of --batch_size) and shuffles chunk order, instead of a global "
        "shuffle — keeps packed-shard reads inside a cache-resident window so the "
        "~40 GB token set doesn't thrash on a RAM-bound box. Larger = closer to a "
        "full global shuffle (more random IO); smaller = more sequential IO, "
        "slightly more correlated batch composition (default: 2048).",
    )
    p.add_argument(
        "--postfix_every",
        type=int,
        default=8,
        help="PE-LoRA training: refresh the tqdm postfix (and force a "
        "host-device sync) every N steps. Higher = fewer syncs / faster "
        "training; lower = more responsive progress bar (default: 10).",
    )
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument(
        "--warmup_steps",
        type=int,
        default=50,
        help="Linear lr warmup over the first N optimizer steps before cosine "
        "decay takes over. 0 (default) disables warmup and runs pure cosine "
        "on a per-step schedule. Typical values: 200-1000 for fresh-head "
        "training on this scale.",
    )
    p.add_argument("--d_hidden", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument(
        "--label_smooth",
        type=float,
        default=0.0,
        help="Label-smoothing ε for the tag head (train only). Softens the "
        "multi-label BCE targets to [ε/2, 1−ε/2] and feeds the same ε to the "
        "softmax-group cross-entropy, regularizing against the overconfidence "
        "that drives the train/val tag-loss gap. 0.0 (default) is inert; "
        "0.05–0.1 is the usual range. Val loss is always reported unsmoothed.",
    )
    p.add_argument(
        "--drop_sidecars_after_pack",
        action="store_true",
        help="After the per-bucket mmap shards are built (and verified), delete "
        "the original per-stem token sidecars to reclaim disk (~the full "
        "tokens-<encoder>/ trees). DESTRUCTIVE: repacking a different split "
        "later then requires re-running `--mode build_features` (GPU "
        "re-encode). Off by default. Only deletes once BOTH train and val "
        "shards are present.",
    )

    # build_features / train / calibrate all read --pool_kind to pick the cache
    # subdir and head shape — they must agree.
    p.add_argument(
        "--pool_kind",
        choices=["map", "mean"],
        default="map",
        help="Pool head over the PE-Core encoder's tokens. 'map' (default): "
        "K-query attention pool + CLS + mean concat → trunk. 'mean': "
        "single-vector mean-pool. Selects cache subdir "
        "(tokens-<encoder>/ vs pooled-<encoder>/ under --feature_cache_dir) "
        "and head arch.",
    )
    p.add_argument(
        "--pool_kind_aux",
        choices=["map", "mean"],
        default="map",
        help="Pool kind for the auxiliary encoder. Default 'map' pairs with "
        "PE-Spatial's full attention pool. Set 'mean' to swap for a cheap "
        "mean-pool on the aux side (rare — defeats the point of PE-Spatial).",
    )
    p.add_argument(
        "--pool_n_queries",
        type=int,
        default=4,
        help="MAP pool: number of learnable queries (default 4). Each query "
        "produces one [d_enc] vector; trunk input is "
        "(K + use_cls + use_mean) * d_enc.",
    )
    p.add_argument(
        "--pool_n_heads",
        type=int,
        default=8,
        help="MAP pool: number of attention heads (default 8). Must divide "
        "the encoder dim (d_enc=1024 for PE-Core).",
    )
    p.add_argument(
        "--pool_use_cls",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="MAP pool: concat the encoder's CLS token as an aux channel (default on).",
    )
    p.add_argument(
        "--pool_use_mean",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="MAP pool: concat the patch-token mean as an aux channel "
        "(default on — gives the legacy baseline as a residual).",
    )

    # Aux encoder MAP-pool knobs (PE-Spatial's d=768 admits more head divisors,
    # so a bigger n_heads_aux is fine).
    p.add_argument(
        "--pool_n_queries_aux",
        type=int,
        default=16,
        help="Aux MAP pool: number of learnable queries (default 4). Each "
        "query produces one [d_in_aux] vector.",
    )
    p.add_argument(
        "--pool_n_heads_aux",
        type=int,
        default=16,
        help="Aux MAP pool: attention heads (default 8). Must divide d_in_aux "
        "(768 for PE-Spatial-B16-512 — divisors include 8, 12, 16, 24).",
    )
    p.add_argument(
        "--pool_use_cls_aux",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Aux MAP pool: concat the encoder's CLS token (default on).",
    )
    p.add_argument(
        "--pool_use_mean_aux",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Aux MAP pool: concat the patch-token mean (default on).",
    )
    p.add_argument(
        "--lambda_rating",
        type=float,
        default=0.1,
        help="Weight on the rating CE loss relative to multi-label BCE.",
    )
    p.add_argument(
        "--lambda_people",
        type=float,
        default=0.1,
        help="Weight on the people-count CE loss relative to multi-label BCE. "
        "0 disables the head's gradient contribution (still runs forward "
        "if the manifest carries labels).",
    )

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

    # scan_role_markers: high solo co-occurrence ratio → likely a class marker
    # mis-typed as character.
    p.add_argument(
        "--min_solo",
        type=int,
        default=5,
        help="scan_role_markers: drop tags with fewer than this many solo "
        "occurrences (default: 5).",
    )
    p.add_argument(
        "--min_ratio",
        type=float,
        default=0.5,
        help="scan_role_markers: drop tags whose conditional co-occurrence "
        "ratio with another character on solo images is below this (default: 0.5).",
    )
    p.add_argument(
        "--top_partners",
        type=int,
        default=3,
        help="scan_role_markers: how many top co-occurring partners to print "
        "per row (default: 3).",
    )
    p.add_argument(
        "--min_role_partners",
        type=int,
        default=5,
        help="scan_role_markers: a candidate with at least this many distinct "
        "co-occurrence partners is classified D_role (broad pool → "
        "affiliation marker). Default: 5.",
    )
    p.add_argument(
        "--pair_dominance",
        type=float,
        default=0.6,
        help="scan_role_markers: a candidate whose top-1 partner accounts for "
        "at least this fraction of co-occurrences is classified C_pair "
        "(narrow pool → genuine couple/sibling). Default: 0.6.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=200,
        help="scan_role_markers: cap rows printed in the table (default: 200).",
    )
    p.add_argument(
        "--out_yaml",
        default=None,
        help="scan_role_markers: optional path for a YAML stub of candidates, "
        "ready to paste into tag_rules.yaml.",
    )

    # --out_dir holds the checkpoint + vocab; bulky feature caches are decoupled
    # into --feature_cache_dir under post_image_dataset/.
    p.add_argument(
        "--out_dir",
        default="models/captioners/anima-tagger-v1",
    )
    p.add_argument(
        "--feature_cache_dir",
        default=None,
        help="Root dir for build_features caches (per-stem token sidecars + "
        "packed mmap shards). Decoupled from --out_dir so these bulky "
        "dataset-derived caches live under post_image_dataset/. Default "
        "(unset): post_image_dataset/anima_tagger/. "
        "Read by build_features / train / calibrate — they must all agree, "
        "so pass the same value (or none) to every mode.",
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

    # Dual encoder is mandatory here; calibrate / predict read it from
    # config.json so they skip this check.
    if args.mode in ("train", "build_features"):
        if not args.aux_encoder:
            raise SystemExit(
                "--aux_encoder is required (dual encoder is the only mode). "
                "Pass e.g. --aux_encoder pe_spatial."
            )
        if args.aux_encoder == args.encoder:
            raise SystemExit(
                f"--aux_encoder={args.aux_encoder!r} matches --encoder; aux must "
                f"be a different encoder (e.g. --encoder pe --aux_encoder pe_spatial)."
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
    elif args.mode == "train":
        from .train_cached import cmd_train_cached

        cmd_train_cached(args)
    elif args.mode == "calibrate":
        from .calibrate import cmd_calibrate

        cmd_calibrate(args)
    elif args.mode == "predict":
        from .predict import cmd_predict

        cmd_predict(args)
    elif args.mode == "scan_role_markers":
        from .role_markers import cmd_scan_role_markers

        cmd_scan_role_markers(args)
    else:
        raise SystemExit(f"unknown --mode={args.mode!r}")


if __name__ == "__main__":
    main()
