"""Anima Tagger task entry-points: preprocess (vocab + dual feature cache),
train (dual-encoder hard-routed head), predict (single-image debug).

All three invoke ``python -m scripts.anima_tagger.cli`` with the appropriate
``--mode`` flag. Extra args are forwarded verbatim, so per-mode knobs
(``--epochs``, ``--image``, ``--show_scores``, …) work as documented in
``scripts/anima_tagger/cli.py``.
"""

from __future__ import annotations

from ._common import PY, run


def _tagger(mode: str, extra):
    run([PY, "-m", "scripts.anima_tagger.cli", "--mode", mode, *extra])


def cmd_preprocess_tagger(extra):
    """Build the tagger vocab/manifest + cache both encoders' PE features.

    Two idempotent stages:

    1. ``--mode build_vocab`` — scans caption sources, emits ``vocab.json`` +
       ``dataset.json``.
    2. ``--mode build_features`` — encodes each manifest image through both
       PE-Core (``--encoder`` / ``--pool_kind``) and PE-Spatial
       (``--aux_encoder`` / ``--pool_kind_aux``), writing per-stem safetensors
       (token sequence for ``map`` / pooled vector for ``mean``).

    Requires ``CAPTION_CORPUS_DIR`` set in ``anima_lora/.env`` (or the relevant
    paths passed via flags). Extra args are forwarded to both stages — pass
    only flags they share (e.g. ``--out_dir``, ``--encoder``, ``--device``).
    """
    _tagger("build_vocab", extra)
    _tagger("build_features", extra)


def cmd_tagger(extra):
    """Train the dual-encoder, hard-routed Anima Tagger head on cached features.

    PE-Core drives rating / people-count / identity tags; PE-Spatial drives
    localized tags (both pooled per ``--pool_kind`` / ``--pool_kind_aux``).
    Encoders are frozen — this reads the per-stem caches built by
    ``make preprocess-tagger`` and saves the head to
    ``<out_dir>/model.safetensors``.

    Tunable defaults (epochs, batch_size, lr, pool kinds) are applied first;
    ``extra`` flags follow so they override (argparse last-wins).
    """
    defaults = [
        "--epochs",
        "32",
        "--batch_size",
        "64",
        "--lr",
        "2e-5",
        "--pool_kind",
        "map",
        "--pool_kind_aux",
        "map",
    ]
    _tagger("train", [*defaults, *extra])


def cmd_test_tagger(extra):
    """Single-image debug entry — runs the trained head and prints the caption.

    Without ``--image``, samples a random stem from the val split for a
    side-by-side comparison against ground-truth tags. Pass ``--show_scores``
    to also print rating distribution + top-K kept tags.
    """
    _tagger("predict", extra)
