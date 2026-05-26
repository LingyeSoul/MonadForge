"""General cached-pair train dataset (VAE latents + text encoder outputs).

Loads pre-cached VAE latents + text encoder outputs from disk, grouped by
latent resolution so that each batch has uniform spatial dimensions (matching
the bucket-based batching used in LoRA training). Despite the name it is not
distill-specific — it is the general train-cache reader, used by the distill
scripts (``scripts/distill_mod/distill.py``, ``scripts/distill_turbo.py``,
``scripts/distill_spd.py``) and the SPD probes. ``library.datasets.distill``
re-exports it for back-compat.
"""

from __future__ import annotations

import glob
import logging
import os
import random

import torch

from library.io.cache import (
    LATENT_CACHE_SUFFIX,
    discover_cached_pairs,
    get_latent_resolution,
    load_cached_latents,
    load_cached_text_features,
)

logger = logging.getLogger(__name__)


class BucketBatchSampler(torch.utils.data.Sampler):
    """Yields batches of sample indices grouped by resolution bucket.

    Every batch contains only same-resolution samples, so the default
    tensor-stacking collate still works at ``batch_size > 1`` (a plain
    ``DataLoader(shuffle=True)`` would mix resolutions into one batch and crash
    the stack). When ``shuffle`` is set, the *order of batches* is reshuffled
    each time iteration restarts (i.e. once per epoch, since the training loop
    rebuilds the iterator on ``StopIteration``) — but the largest-token-count
    bucket's first batch is always pinned to step 0 so ``torch.compile``'s
    biggest block graph + peak VRAM allocation land up front (fail-fast).
    ``shuffle=False`` preserves the deterministic largest-first bucket order.
    """

    def __init__(self, batches, largest_idx, *, shuffle=True, seed=0):
        self._batches = batches  # list[list[int]]
        self._largest_idx = largest_idx  # index into _batches, or None
        self._shuffle = shuffle
        self._seed = seed
        self._epoch = 0

    def __len__(self):
        return len(self._batches)

    def __iter__(self):
        order = list(range(len(self._batches)))
        if self._shuffle:
            random.Random(self._seed + self._epoch).shuffle(order)
            self._epoch += 1
            if self._largest_idx is not None:
                order.remove(self._largest_idx)
                order.insert(0, self._largest_idx)
        for bi in order:
            yield self._batches[bi]


class CachedDataset(torch.utils.data.Dataset):
    """Loads pre-cached latents and text encoder outputs for distillation.

    Samples are grouped by latent resolution so that each batch has uniform
    spatial dimensions (matching the bucket-based batching used in training).
    A deterministic per-bucket split (seeded by ``validation_seed``) carves off
    the last ``validation_split`` fraction for the val set, mirroring the
    LoRA training convention.
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 1,
        *,
        split: str = "train",
        validation_split: float = 0.0,
        validation_seed: int = 42,
        sample_ratio: float = 1.0,
        synth_data_dir: str | None = None,
    ):
        assert split in ("train", "val")
        self.data_dir = data_dir
        self.synth_data_dir = synth_data_dir
        cached = discover_cached_pairs(data_dir)

        # When --synth_data_dir is set, rewrite each sample's latent path to the
        # synthetic NPZ for the same stem. Samples without a synthetic
        # counterpart are dropped — the teacher pool is bounded by what was
        # generated via `make distill-prep`. TE paths remain in data_dir.
        # Lookup is stem-keyed (not basename-keyed) because the lora cache uses
        # WxH pixel dims zero-padded to 4 (e.g. `0896x1152`) while the synth
        # writer uses HxW latent dims (e.g. `144x112`) — the same logical pair
        # has different basenames in the two dirs.
        n_dropped_no_synth = 0
        if synth_data_dir is not None:
            synth_by_stem: dict[str, str] = {}
            for path in glob.glob(
                os.path.join(synth_data_dir, "**", f"*{LATENT_CACHE_SUFFIX}"),
                recursive=True,
            ):
                # `{stem}_{HxW}_anima.npz` → strip suffix, drop trailing `_HxW`
                without_suffix = os.path.basename(path).removesuffix(LATENT_CACHE_SUFFIX)
                stem = without_suffix.rsplit("_", 1)[0]
                synth_by_stem.setdefault(stem, path)
            remapped: list = []
            for img in cached:
                if img.te_path is None:
                    continue
                synth_path = synth_by_stem.get(img.stem)
                if synth_path is None:
                    n_dropped_no_synth += 1
                    continue
                # Reuse CachedImage shape so the downstream code is unchanged.
                remapped.append(img._replace(npz_path=synth_path))
            cached = remapped
            if n_dropped_no_synth:
                logger.warning(
                    f"[{split}] {n_dropped_no_synth} samples have no synthetic "
                    f"latent under {synth_data_dir}; dropped."
                )

        # Group samples by latent resolution
        buckets: dict[str, list[tuple[str, str]]] = {}
        for img in cached:
            if img.te_path is None:
                continue
            res = get_latent_resolution(img.npz_path)
            buckets.setdefault(res, []).append((img.npz_path, img.te_path))

        # Per-bucket deterministic shuffle, then carve last `validation_split`
        # off as val so train/val never overlap and remain bucket-grouped.
        # Apply sample_ratio per-bucket (mirrors the LoRA pipeline's per-subset
        # subsampling), keeping at least one sample per non-empty bucket so
        # debug/half presets don't silently drop entire resolutions.
        # Drop per-bucket remainders for whichever side we're emitting.
        #
        # Emit buckets largest-token-count first. The DataLoader runs
        # shuffle=False, so iteration order == this bucket order; front-loading
        # the biggest resolution means torch.compile traces the largest block
        # graph and allocates peak activations on step 0. With native-shape
        # buckets (4032 + 4200 token families), the 4200 bucket would otherwise
        # only get hit once iteration reached it (~step 100), spiking VRAM
        # mid-run — front-loading turns a mid-run OOM into a fail-fast at start.
        def _tok_count(res: str) -> int:
            a, b = res.split("x")
            return int(a) * int(b)

        rng = random.Random(validation_seed)
        self.batch_size = batch_size
        self.samples: list[tuple[str, str]] = []
        # Same-resolution batches of sample indices, built as samples are
        # emitted bucket-by-bucket (each bucket's remainder is dropped to a
        # multiple of batch_size, so a contiguous chunk is always one bucket).
        self._batches: list[list[int]] = []
        self._batch_tok: list[int] = []
        n_train = n_val = 0
        for _res, items in sorted(
            buckets.items(), key=lambda kv: _tok_count(kv[0]), reverse=True
        ):
            items = list(items)
            rng.shuffle(items)
            n = len(items)
            n_v = int(round(n * validation_split)) if validation_split > 0.0 else 0
            n_t = n - n_v
            train_items = items[:n_t]
            val_items = items[n_t:]
            n_train += n_t
            n_val += n_v
            picked = train_items if split == "train" else val_items
            if sample_ratio < 1.0 and picked:
                n_keep = max(1, int(round(len(picked) * sample_ratio)))
                picked = picked[:n_keep]
            full = (len(picked) // batch_size) * batch_size
            start = len(self.samples)
            self.samples.extend(picked[:full])
            tok = _tok_count(_res)
            for j in range(start, start + full, batch_size):
                self._batches.append(list(range(j, j + batch_size)))
                self._batch_tok.append(tok)

        sr_note = f", sample_ratio={sample_ratio}" if sample_ratio < 1.0 else ""
        source = (
            f"latents={synth_data_dir} (synth), te={data_dir}"
            if synth_data_dir is not None
            else data_dir
        )
        logger.info(
            f"[{split}] {len(self.samples)} samples from {source} "
            f"({len(buckets)} buckets; pre-drop train={n_train}, val={n_val}{sr_note})"
        )

    def __len__(self):
        return len(self.samples)

    def make_batch_sampler(
        self, *, shuffle: bool = True, seed: int = 0
    ) -> BucketBatchSampler:
        """Build a bucket-grouped batch sampler over this dataset.

        Pass to ``DataLoader(batch_sampler=...)`` (not ``batch_size=``). Each
        batch is one resolution; ``shuffle`` reshuffles batch order per epoch
        while keeping the largest-token bucket first. See ``BucketBatchSampler``.
        """
        largest = (
            max(range(len(self._batches)), key=lambda i: self._batch_tok[i])
            if self._batches
            else None
        )
        return BucketBatchSampler(
            self._batches, largest, shuffle=shuffle, seed=seed
        )

    def __getitem__(self, idx):
        latent_path, te_path = self.samples[idx]
        latents, _res, _h, _w = load_cached_latents(latent_path)  # (16, H, W)
        # Fixed variant=0: distill-mod targets a deterministic teacher mapping,
        # and the teacher cache keys on (sample_idx, sigma_idx) only — drawing
        # a random variant per visit would let cache hits return a teacher pred
        # computed under a different caption than the student is conditioned on.
        crossattn_emb, pooled_text = load_cached_text_features(te_path, variant=0)
        return idx, latents, crossattn_emb, pooled_text
