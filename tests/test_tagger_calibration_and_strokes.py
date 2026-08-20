"""Tests for the two tagger over-sensitivity guards added 2026-08-20:

* ``calibrate_thresholds(min_support=…)`` — a tag with too few val positives
  keeps the 0.5 default instead of trusting a degenerate F1 sweep (62% of the
  vocab has <5 val positives on a ~800-image val split, and ~300 of those
  swept to hair-trigger thresholds ≤0.3).
* white brush-stroke augmentation (``--stroke_frac``) — domain alignment for
  the mask-blanked position-caption crops; deterministic, train-split only.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from PIL import Image

from scripts.anima_tagger.calibrate import calibrate_thresholds


def _sweep():
    return torch.linspace(0.05, 0.95, 19)


def test_calibrate_min_support_keeps_default_for_thin_tags():
    # Tag 0: 2 positives, perfectly separable at a low threshold.
    # Tag 1: 8 positives, separable at a high threshold.
    scores = torch.zeros(20, 2)
    targets = torch.zeros(20, 2)
    scores[:2, 0] = 0.30
    targets[:2, 0] = 1.0
    scores[2:, 0] = 0.05
    scores[:8, 1] = 0.90
    targets[:8, 1] = 1.0
    scores[8:, 1] = 0.10

    thresh, f1 = calibrate_thresholds(scores, targets, _sweep(), min_support=5)
    assert thresh[0].item() == 0.5  # 2 < 5 positives → default, sweep untrusted
    assert f1[0].item() == 0.0
    assert 0.10 < thresh[1].item() < 0.90  # 8 ≥ 5 → swept normally
    assert f1[1].item() > 0.99

    # min_support=1 restores the old trust-any-positive behaviour.
    thresh_old, _ = calibrate_thresholds(scores, targets, _sweep(), min_support=1)
    assert thresh_old[0].item() < 0.30


def test_paint_white_strokes_is_deterministic_and_paints_white():
    from library.captioning.anima_tagger_data import paint_white_strokes

    def painted(seed):
        im = Image.new("RGB", (256, 256), (40, 80, 120))
        return np.array(paint_white_strokes(im, seed))

    a, b, c = painted(7), painted(7), painted(8)
    assert np.array_equal(a, b)  # same seed → bit-identical
    assert not np.array_equal(a, c)  # different seed → different strokes
    white = (a == 255).all(axis=-1).mean()
    assert 0.005 < white < 0.5  # visible but far from destroying the image


def test_stroke_stems_selects_train_split_only_at_roughly_frac():
    from scripts.anima_tagger.caches import _stroke_stems_for

    class Manifest:
        train_stems = [f"t{i}" for i in range(2000)]
        val_stems = [f"v{i}" for i in range(200)]

    args = argparse.Namespace(stroke_frac=0.25, stroke_seed=1234)
    picked = _stroke_stems_for(args, Manifest())
    assert picked == _stroke_stems_for(args, Manifest())  # stable across calls
    assert all(s.startswith("t") for s in picked)  # never a val stem
    assert 0.20 < len(picked) / 2000 < 0.30

    off = _stroke_stems_for(argparse.Namespace(stroke_frac=0.0), Manifest())
    assert off == frozenset()
