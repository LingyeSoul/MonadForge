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


def _mode_in(extra):
    """Return ``(mode, normalized_extra)`` — the ``--mode`` value in ``extra``
    (hyphens→underscores in place), or ``(None, extra)`` when absent.

    Lets ``make tagger ARGS="--mode build-vocab"`` dispatch to any tagger mode
    through the one target, accepting the hyphenated spelling users reach for.
    """
    extra = list(extra)
    for i, a in enumerate(extra):
        if a == "--mode" and i + 1 < len(extra):
            extra[i + 1] = extra[i + 1].replace("-", "_")
            return extra[i + 1], extra
    return None, extra


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
    """Run the Anima Tagger CLI — trains by default, or any mode via ``--mode``.

    ``make tagger`` (no ``--mode``) trains the dual-encoder hard-routed head:
    PE-Core drives rating / people-count / identity tags, PE-Spatial drives
    localized tags (both pooled per ``--pool_kind`` / ``--pool_kind_aux``);
    encoders are frozen, reading the caches from ``make preprocess-tagger`` and
    saving to ``<out_dir>/model.safetensors``. The training defaults (epochs,
    batch_size, lr, pool kinds) are applied first so ``extra`` flags override.

    ``make tagger ARGS="--mode build_vocab"`` (or any other ``--mode``) forwards
    straight to that mode with the CLI's own defaults — no training knobs
    injected. ``build_vocab`` derives + bakes tag-groups by default
    (``--no-derive_groups`` to opt out). Hyphenated modes (``build-vocab``) are
    accepted.
    """
    mode, extra = _mode_in(extra)
    if mode is not None and mode != "train":
        run([PY, "-m", "scripts.anima_tagger.cli", *extra])
        return
    defaults = [
        "--epochs",
        "32",
        "--batch_size",
        "64",
        "--lr",
        "1.5e-4",
        "--label_smooth",
        "0.0",
        "--pool_kind",
        "map",
        "--pool_kind_aux",
        "map",
    ]
    # mode is None (→ train) or explicitly "train" (already in extra).
    if mode is None:
        _tagger("train", [*defaults, *extra])
    else:
        run([PY, "-m", "scripts.anima_tagger.cli", *defaults, *extra])


def cmd_test_tagger(extra):
    """Single-image debug entry — runs the trained head and prints the caption.

    Without ``--image``, samples a random stem from the val split for a
    side-by-side comparison against ground-truth tags. Pass ``--show_scores``
    to also print rating distribution + top-K kept tags.
    """
    _tagger("predict", extra)


def cmd_autotag(extra):
    """Autotag a single image (CLI one-shot).

    Thin wrapper over ``scripts.anima_tagger.autotag``: auto-downloads the
    tagger checkpoint on first use, runs it on ``--image``, and prints the
    predicted caption on one sentinel-prefixed stdout line. Handy for smoke-
    testing the tagger without the GUI (which runs a resident worker —
    ``scripts.anima_tagger.autotag_server`` — for fast consecutive tagging).
    Extra args (``--image``, ``--tagger_dir``, ``--device``) forwarded verbatim.
    """
    run([PY, "-m", "scripts.anima_tagger.autotag", *extra])
