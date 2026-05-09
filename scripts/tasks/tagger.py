"""Anima Tagger task entry-points: preprocess (vocab + feature cache),
train (cached-encoder fast path), predict (single-image debug).

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
    """Build the tagger vocab/manifest + cache pooled PE features.

    Runs ``--mode build_vocab`` (scans caption sources, emits vocab.json +
    dataset.json) followed by ``--mode build_features`` (encodes each manifest
    image through frozen PE-Core, mean-pools patches, writes per-stem
    safetensors). Both stages are idempotent.

    Requires ``CAPTION_CORPUS_DIR`` set in ``anima_lora/.env`` (or the relevant
    paths passed via flags). Extra args are forwarded to BOTH stages — pass
    only flags they share (e.g. ``--out_dir``, ``--encoder``, ``--device``).
    """
    _tagger("build_vocab", extra)
    _tagger("build_features", extra)


def cmd_tagger(extra):
    """Train the Anima Tagger head on cached PE features (fast path).

    Default ``--pe_lora_rank=0`` keeps the encoder frozen and reads pre-pooled
    features from ``<out_dir>/.cache/pooled-pe/`` (built by
    ``make preprocess-tagger``). Override knobs via extra args, e.g.
    ``python tasks.py tagger --epochs 30 --batch_size 512``.
    """
    _tagger("train", extra)


def cmd_test_tagger(extra):
    """Single-image debug entry — runs the trained head and prints the caption.

    Without ``--image``, samples a random stem from the val split for a
    side-by-side comparison against ground-truth tags. Pass ``--show_scores``
    to also print rating distribution + top-K kept tags.
    """
    _tagger("predict", extra)
