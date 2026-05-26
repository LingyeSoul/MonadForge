"""Back-compat re-export.

The general cached-pair train dataset (``CachedDataset`` + its
``BucketBatchSampler``) moved to :mod:`library.datasets.cache` — it was never
distill-specific. This shim keeps ``from library.datasets.distill import
CachedDataset`` working for the distill scripts and SPD probes that predate the
move.
"""

from __future__ import annotations

from library.datasets.cache import BucketBatchSampler, CachedDataset

__all__ = ["BucketBatchSampler", "CachedDataset"]
