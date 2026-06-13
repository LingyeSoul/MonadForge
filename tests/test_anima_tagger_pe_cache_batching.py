"""Unit tests for the bucket-batched PE cache path (tagger).

Covers the pure grouping / collate logic that replaced the old
``batch_size=1`` per-image loop in ``FeatureCacheBuilder`` /
``TokenCacheBuilder``. The encoder forward itself is unchanged (a ViT has
no cross-image interaction, so a shape-homogeneous batch is bit-identical
to one-at-a-time), so the only new failure surface is the bucketing — does
every batch share an aspect bucket, are all stems covered exactly once, and
does the collate filter decode failures without breaking the stack.

No PE checkpoint needed — uses the static PE-Core bucket spec + synthetic
PNGs on disk.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from library.captioning.anima_tagger_data import _bucket_batches, _bucket_collate
from library.vision.buckets import PE_CORE_L14_336_SPEC, pick_bucket


def _write_png(path, w, h):
    Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8), mode="RGB").save(path)


def test_bucket_batches_groups_by_aspect_and_covers_all(tmp_path):
    spec = PE_CORE_L14_336_SPEC
    # A mix of aspect ratios; several share a bucket, one is unreadable.
    sizes = [(640, 480), (480, 640), (512, 512), (660, 500), (1024, 1024)]
    paths = []
    for i, (w, h) in enumerate(sizes):
        p = tmp_path / f"img_{i}.png"
        _write_png(p, w, h)
        paths.append(p)
    bad = tmp_path / "corrupt.png"
    bad.write_bytes(b"not an image")
    paths.append(bad)

    batches, header_errs = _bucket_batches(paths, spec, batch_size=2)

    # The corrupt file is surfaced as a header error (by index), not batched.
    assert [i for i, _ in header_errs] == [len(paths) - 1]

    # Every readable index appears exactly once across all batches.
    flat = [i for b in batches for i in b]
    assert sorted(flat) == list(range(len(sizes)))

    # No batch exceeds batch_size and every batch is shape-homogeneous.
    for b in batches:
        assert 1 <= len(b) <= 2
        keys = {pick_bucket(*Image.open(paths[i]).size[::-1], spec) for i in b}
        assert len(keys) == 1, "batch mixes aspect buckets"


def test_bucket_batches_chunks_large_bucket(tmp_path):
    spec = PE_CORE_L14_336_SPEC
    paths = []
    for i in range(5):
        p = tmp_path / f"sq_{i}.png"
        _write_png(p, 512, 512)  # all identical → one bucket
        paths.append(p)

    batches, header_errs = _bucket_batches(paths, spec, batch_size=2)
    assert header_errs == []
    # 5 same-bucket images @ bs=2 → [2, 2, 1]
    assert sorted(len(b) for b in batches) == [1, 2, 2]


def test_bucket_collate_filters_failures_and_stacks():
    import torch

    good_a = ("a", torch.zeros(3, 8, 8), "")
    bad = ("b", None, "OSError: boom")
    good_c = ("c", torch.ones(3, 8, 8), "")

    stems_ok, stacked, errs = _bucket_collate([good_a, bad, good_c])
    assert stems_ok == ["a", "c"]
    assert stacked.shape == (2, 3, 8, 8)
    assert errs == [("b", "OSError: boom")]


def test_bucket_collate_all_failed_returns_none():
    stems_ok, stacked, errs = _bucket_collate([("x", None, "err")])
    assert stems_ok == []
    assert stacked is None
    assert errs == [("x", "err")]
