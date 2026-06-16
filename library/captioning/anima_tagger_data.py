"""Dataset plumbing for the Anima tagger (dual-encoder, hard-routed).

* :class:`TaggerManifest` — loads ``dataset.json`` (the per-stem
  image-path + multi-hot tag indices + rating-class index + people-count
  class index emitted by ``python -m scripts.anima_tagger.cli --mode
  build_vocab``).
* :class:`FeatureCacheBuilder` — mean-pool cache (``pool_kind="mean"``).
  Encodes each manifest image through a frozen PE trunk, mean-pools over
  patch tokens, writes a per-stem ``[d_enc] fp32`` safetensors. Idempotent.
* :class:`TokenCacheBuilder` — full token cache (``pool_kind="map"``).
  Encodes each manifest image through a frozen PE trunk, writes a per-stem
  ``[T, d_enc] bf16`` safetensors (CLS at row 0). Storage is ~1.2 MB /
  stem at PE-Core-L14-336; the win is that swapping pool architectures no
  longer requires re-encoding.
* :class:`CachedDualDataset` + :func:`collate_dual_token_batch` — the lazy
  bucket-grouped dataset feeding both encoders. Each side independently
  loads a pooled ``[d_enc]`` feature (mean) or ``[T, d_enc]`` tokens (map);
  T is constant within a bucket so the collate just stacks per side.
* :class:`BucketBatchSampler` — groups dataset indices into
  shape-homogeneous batches per (main, aux) bucket pair.

Cache layout is locked into each builder's file format — swap pooling /
storage layout → invalidate the cache dir → rebuild.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file as st_load
from safetensors.torch import save_file as st_save
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

from library.datasets.image_utils import IMAGE_TRANSFORMS
from library.vision.buckets import (
    BucketSpec,
    bucket_pixel_size,
    pick_bucket,
)
from library.vision.encoder import (
    VisionEncoderBundle,
    encode_pe_from_imageminus1to1,
    load_pe_encoder,
)

logger = logging.getLogger(__name__)


def pil_resize_to_bucket(img: Image.Image, spec: BucketSpec) -> Image.Image:
    """LANCZOS-resize a PIL image to its closest bucket size for ``spec``.

    Pre-resizing on the PIL side (high quality LANCZOS) avoids decoding
    multi-megapixel source images into a tensor only to bilinear-resize
    them down inside the encoder. Speeds up cache builds 5–10× on
    high-resolution corpora and removes a quality penalty (LANCZOS >
    bilinear for severe downscales).
    """
    w, h = img.size
    h_p, w_p = pick_bucket(h, w, spec)
    target_h, target_w = bucket_pixel_size((h_p, w_p), spec)
    if (h, w) != (target_h, target_w):
        img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    return img


class _ResizeDataset(Dataset):
    """CPU-side decode + bucket-resize + IMAGE_TRANSFORMS for one stem.

    Returns ``(stem, tensor[C,H,W] | None, err)``. Errors are surfaced as
    a non-empty string so the consumer can log and continue.
    """

    def __init__(
        self,
        stems: Sequence[str],
        image_paths: Sequence[Path],
        spec: BucketSpec,
    ):
        self._stems = list(stems)
        self._paths = list(image_paths)
        self._spec = spec

    def __len__(self) -> int:
        return len(self._stems)

    def __getitem__(self, k: int):
        stem = self._stems[k]
        path = self._paths[k]
        try:
            with Image.open(path) as im:
                im = pil_resize_to_bucket(im.convert("RGB"), self._spec)
                arr = np.array(im)
            tensor = IMAGE_TRANSFORMS(arr)
            return stem, tensor, ""
        except Exception as e:
            return stem, None, f"{type(e).__name__}: {e}"


def _bucket_collate(batch):
    """Stack a shape-homogeneous batch, filtering decode failures.

    The :func:`_bucket_batches` sampler guarantees every item in a batch
    shares an aspect bucket, so the surviving tensors stack cleanly into
    ``[B, C, H, W]``. Returns ``(stems_ok, stacked | None, errs)`` where
    ``errs`` carries ``(stem, msg)`` for items that failed to decode.
    """
    stems_ok: list[str] = []
    tensors: list[torch.Tensor] = []
    errs: list[tuple[str, str]] = []
    for stem, tensor, err in batch:
        if tensor is None:
            errs.append((stem, err))
        else:
            stems_ok.append(stem)
            tensors.append(tensor)
    stacked = torch.stack(tensors, dim=0) if tensors else None
    return stems_ok, stacked, errs


def _bucket_batches(
    image_paths: Sequence[Path], spec: BucketSpec, batch_size: int
) -> tuple[list[list[int]], list[tuple[int, str]]]:
    """Group image indices by aspect bucket, then chunk into batches.

    Reads only each image's header (``PIL.Image.size`` — no pixel decode) to
    pick its bucket, so images that resize to the same pixel grid land in the
    same batch and can share one encoder forward. Returns the batch list (each
    a list of dataset indices) plus ``(index, msg)`` for unreadable headers,
    which the caller logs and skips.
    """
    by_bucket: Dict[tuple, List[int]] = defaultdict(list)
    header_errs: list[tuple[int, str]] = []
    for i, path in enumerate(image_paths):
        try:
            with Image.open(path) as im:
                w, h = im.size
            key = pick_bucket(h, w, spec)
        except Exception as e:
            header_errs.append((i, f"{type(e).__name__}: {e}"))
            continue
        by_bucket[key].append(i)

    batches: list[list[int]] = []
    for idxs in by_bucket.values():
        for j in range(0, len(idxs), batch_size):
            batches.append(idxs[j : j + batch_size])
    return batches, header_errs


def _run_pe_cache(
    *,
    stems: Sequence[str],
    image_paths: Sequence[Path],
    spec: BucketSpec,
    bundle: VisionEncoderBundle,
    num_workers: int,
    batch_size: int,
    save_one,
    desc: str,
) -> int:
    """Bucket-batched encode loop shared by the feature / token builders.

    Groups the (already-filtered-to-missing) ``stems`` by aspect bucket, runs
    one batched ``encode_pe_from_imageminus1to1(..., same_bucket=True)`` forward
    per batch, and hands each ``[T, d_enc]`` result to ``save_one(stem, feats)``.
    Per-image decode / save failures are logged and skipped. Returns the count
    of entries written.
    """
    batches, header_errs = _bucket_batches(image_paths, spec, batch_size)
    for i, err in header_errs:
        logger.warning("failed to read %s: %s", stems[i], err)

    ds = _ResizeDataset(stems=stems, image_paths=image_paths, spec=spec)
    loader = DataLoader(
        ds,
        batch_sampler=batches,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=_bucket_collate,
        pin_memory=False,
        persistent_workers=(num_workers > 0 and len(batches) > 1),
    )

    n_done = 0
    for stems_ok, img_batch, errs in tqdm(
        loader, desc=desc, unit="batch", total=len(batches)
    ):
        for stem, err in errs:
            logger.warning("failed to decode %s: %s", stem, err)
        if img_batch is None:
            continue
        try:
            feats_list = encode_pe_from_imageminus1to1(
                bundle, img_batch, same_bucket=True
            )
        except Exception as e:
            logger.warning("failed to encode batch of %d: %s", len(stems_ok), e)
            continue
        for stem, feats in zip(stems_ok, feats_list):
            try:
                save_one(stem, feats)
                n_done += 1
            except Exception as e:
                logger.warning("failed to save %s: %s", stem, e)
    return n_done


@dataclass
class TaggerManifest:
    """Trainable-sample manifest emitted by ``--mode build_vocab``."""

    stems: List[str]
    image_paths: List[Path]
    tag_indices: List[List[int]]
    rating_indices: List[int]
    people_count_indices: List[int]
    train_stems: List[str]
    val_stems: List[str]
    n_tags: int
    n_ratings: int
    n_people_counts: int

    @classmethod
    def from_path(cls, path: Path) -> "TaggerManifest":
        with open(path) as f:
            d = json.load(f)
        # ``people_count_indices`` / ``n_people_counts`` were added late; default
        # to a zero-length head so old manifests signal "no people supervision"
        # instead of KeyError-ing (they rebuild on next ``build_vocab``).
        people_idx = list(d.get("people_count_indices") or [])
        n_people = int(d.get("n_people_counts", 0))
        if people_idx and not n_people:
            n_people = max(people_idx) + 1
        return cls(
            stems=list(d["stems"]),
            image_paths=[Path(p) for p in d["image_paths"]],
            tag_indices=[list(idxs) for idxs in d["tag_indices"]],
            rating_indices=list(d["rating_indices"]),
            people_count_indices=people_idx,
            train_stems=list(d["split"]["train"]),
            val_stems=list(d["split"]["val"]),
            n_tags=int(d["n_tags"]),
            n_ratings=int(d["n_ratings"]),
            n_people_counts=n_people,
        )

    def stem_index(self) -> Dict[str, int]:
        return {s: i for i, s in enumerate(self.stems)}


def _cache_path(cache_dir: Path, stem: str) -> Path:
    return cache_dir / f"{stem}.safetensors"


class FeatureCacheBuilder:
    """Build per-stem mean-pooled PE-Core features into ``cache_dir``.

    Groups missing stems by aspect bucket and runs one batched encoder
    forward per ``batch_size`` shape-homogeneous images (``_run_pe_cache``),
    keeping the GPU fed instead of one image per forward. Idempotent.
    """

    def __init__(
        self,
        manifest: TaggerManifest,
        cache_dir: Path,
        device: torch.device,
        encoder_name: str = "pe",
        dtype: torch.dtype = torch.bfloat16,
        num_workers: int = 4,
        batch_size: int = 8,
    ):
        self.manifest = manifest
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.encoder_name = encoder_name
        self.dtype = dtype
        self.num_workers = num_workers
        self.batch_size = batch_size
        self._bundle: Optional[VisionEncoderBundle] = None

    def _bundle_lazy(self) -> VisionEncoderBundle:
        if self._bundle is None:
            self._bundle = load_pe_encoder(
                self.device, name=self.encoder_name, dtype=self.dtype
            )
        return self._bundle

    def missing_stems(self) -> List[int]:
        return [
            i
            for i, stem in enumerate(self.manifest.stems)
            if not _cache_path(self.cache_dir, stem).exists()
        ]

    @torch.no_grad()
    def build(self) -> int:
        """Encode + cache every stem missing from ``cache_dir``.

        Returns the count of newly cached entries (0 if everything was
        already cached). Errors on individual images are logged and the
        loop continues — a single corrupt image shouldn't tank the run.
        """
        missing = self.missing_stems()
        if not missing:
            logger.info(
                "feature cache: all %d entries present", len(self.manifest.stems)
            )
            return 0

        logger.info(
            "feature cache: encoding %d missing entries (out of %d total)",
            len(missing),
            len(self.manifest.stems),
        )
        bundle = self._bundle_lazy()
        spec = bundle.bucket_spec
        d_enc = bundle.d_enc

        def save_one(stem: str, feats: torch.Tensor) -> None:
            pooled = feats.mean(dim=0).to(torch.float32).cpu()
            assert pooled.shape == (d_enc,), pooled.shape
            st_save({"feature": pooled}, str(_cache_path(self.cache_dir, stem)))

        n_done = _run_pe_cache(
            stems=[self.manifest.stems[i] for i in missing],
            image_paths=[self.manifest.image_paths[i] for i in missing],
            spec=spec,
            bundle=bundle,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            save_one=save_one,
            desc="pooled-pe",
        )
        logger.info("feature cache: wrote %d new entries", n_done)
        return n_done


def _token_cache_path(cache_dir: Path, stem: str) -> Path:
    return cache_dir / f"{stem}.safetensors"


class TokenCacheBuilder:
    """Build per-stem full-token PE-Core caches into ``cache_dir``.

    Writes each stem as ``{"tokens": bf16 [T, d_enc]}`` with the encoder's
    native CLS token at row 0 (use_cls=True). T varies per aspect-bucket
    (~576–588 for PE-Core-L14-336).

    Storage per stem ≈ ``T * d_enc * 2`` bytes; ~1.2 MB at PE-Core defaults.
    At 12K stems that's ~14 GB total — pay once, iterate on pool design
    freely. Missing stems are grouped by aspect bucket and encoded in
    batches of ``batch_size`` (``_run_pe_cache``) to keep the GPU saturated.
    """

    def __init__(
        self,
        manifest: TaggerManifest,
        cache_dir: Path,
        device: torch.device,
        encoder_name: str = "pe",
        dtype: torch.dtype = torch.bfloat16,
        num_workers: int = 4,
        batch_size: int = 8,
    ):
        self.manifest = manifest
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.encoder_name = encoder_name
        self.dtype = dtype
        self.num_workers = num_workers
        self.batch_size = batch_size
        self._bundle: Optional[VisionEncoderBundle] = None

    def _bundle_lazy(self) -> VisionEncoderBundle:
        if self._bundle is None:
            self._bundle = load_pe_encoder(
                self.device, name=self.encoder_name, dtype=self.dtype
            )
        return self._bundle

    def missing_stems(self) -> List[int]:
        return [
            i
            for i, stem in enumerate(self.manifest.stems)
            if not _token_cache_path(self.cache_dir, stem).exists()
        ]

    @torch.no_grad()
    def build(self) -> int:
        """Encode + cache every stem missing from ``cache_dir``.

        Returns the count of newly cached entries (0 if everything was
        already cached). Errors on individual images are logged and the
        loop continues — a single corrupt image shouldn't tank the run.
        """
        missing = self.missing_stems()
        if not missing:
            logger.info("token cache: all %d entries present", len(self.manifest.stems))
            return 0

        logger.info(
            "token cache: encoding %d missing entries (out of %d total)",
            len(missing),
            len(self.manifest.stems),
        )
        bundle = self._bundle_lazy()
        spec = bundle.bucket_spec
        d_enc = bundle.d_enc

        def save_one(stem: str, feats: torch.Tensor) -> None:
            assert feats.shape[-1] == d_enc, feats.shape
            # bf16 = encoder's native dtype, halves cache size with no quality loss
            # (MAP head's LayerNorm + Linear projects back into fp32/autocast-bf16).
            tokens = feats.to(torch.bfloat16).cpu().contiguous()
            st_save({"tokens": tokens}, str(_token_cache_path(self.cache_dir, stem)))

        n_done = _run_pe_cache(
            stems=[self.manifest.stems[i] for i in missing],
            image_paths=[self.manifest.image_paths[i] for i in missing],
            spec=spec,
            bundle=bundle,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            save_one=save_one,
            desc="tokens-pe",
        )
        logger.info("token cache: wrote %d new entries", n_done)
        return n_done


class CachedDualDataset(Dataset):
    """Lazy per-stem ``(feat_main, feat_aux, multi_hot, rating, people, bucket_pair)``.

    Each side independently picks ``"mean"`` or ``"map"``:
      * ``"mean"`` → load pooled cache (``{stem}.safetensors`` with key
        ``"feature"``), shape ``[D]``. Bucket key is ``None`` for that side
        (no shape variation → no batch-bucketing constraint).
      * ``"map"`` → load token cache (``{stem}.safetensors`` with key
        ``"tokens"``), shape ``[T, D]``. Bucket key is ``(h_p, w_p)`` derived
        from T via the encoder's :class:`BucketSpec`.

    The compound bucket key is ``(main_bucket | None, aux_bucket | None)``;
    :class:`BucketBatchSampler` groups by it, so within-batch shape
    homogeneity holds for whichever side(s) need it. When PE-Core uses
    mean and PE-Spatial uses map (the most common asymmetric setup),
    batches are grouped by aux bucket only.

    ``cache_dir`` / ``cache_dir_aux`` should be the per-side outputs of
    :func:`scripts.anima_tagger.caches.cache_dir_for` (i.e.
    ``.cache/pooled-<encoder>/`` for mean, ``.cache/tokens-<encoder>/`` for
    map). Spec is only used when the side is map (to map T → bucket); pass
    ``None`` for the spec on a mean side.

    Stems present in only one cache are skipped (logged); typical cause is
    an asymmetric incremental cache build.
    """

    def __init__(
        self,
        manifest: TaggerManifest,
        cache_dir: Path,
        pool_kind: str,
        spec: Optional[BucketSpec],
        cache_dir_aux: Path,
        pool_kind_aux: str,
        spec_aux: Optional[BucketSpec],
        stems_subset: Optional[Sequence[str]] = None,
        pack_root: Optional[Path] = None,
        ram_resident: bool = False,
    ):
        self._ram_resident_req = bool(ram_resident)
        self._ram_resident = False
        self._ram: Dict[tuple, torch.Tensor] = {}
        if pool_kind not in ("mean", "map"):
            raise ValueError(f"pool_kind must be 'mean' or 'map', got {pool_kind!r}")
        if pool_kind_aux not in ("mean", "map"):
            raise ValueError(
                f"pool_kind_aux must be 'mean' or 'map', got {pool_kind_aux!r}"
            )
        if pool_kind == "map" and spec is None:
            raise ValueError("pool_kind='map' requires a BucketSpec for the main side")
        if pool_kind_aux == "map" and spec_aux is None:
            raise ValueError(
                "pool_kind_aux='map' requires a BucketSpec for the aux side"
            )

        idx_of = manifest.stem_index()
        if stems_subset is None:
            stems_subset = manifest.stems
        self._stems_requested = list(stems_subset)

        # Recovery path: once `--drop_sidecars_after_pack` removes the per-stem
        # caches, rebuild from the mmap shards + index instead. The index is keyed
        # by the REQUESTED subset, unlike shard dirs which key on the KEPT stems.
        if pack_root is not None:
            pack_index = self._try_load_pack_index(Path(pack_root), stems_subset)
            if pack_index is not None and not self._sidecars_present(
                cache_dir, stems_subset, idx_of
            ):
                self._init_from_pack_index(
                    manifest,
                    cache_dir,
                    cache_dir_aux,
                    pool_kind,
                    pool_kind_aux,
                    spec,
                    spec_aux,
                    Path(pack_root),
                    pack_index,
                )
                return

        # Per-side T → bucket map (only used on map sides; empty for mean sides).
        def _bucket_map(spec: Optional[BucketSpec]) -> Dict[int, tuple[int, int]]:
            if spec is None:
                return {}
            return {h * w + (1 if spec.use_cls else 0): (h, w) for h, w in spec.buckets}

        token_to_bucket = _bucket_map(spec)
        token_to_bucket_aux = _bucket_map(spec_aux)

        kept_stems: List[str] = []
        kept_paths: List[Path] = []
        kept_paths_aux: List[Path] = []
        kept_tag_idx: List[List[int]] = []
        kept_rating_idx: List[int] = []
        kept_people_idx: List[int] = []
        kept_buckets: List[
            tuple[Optional[tuple[int, int]], Optional[tuple[int, int]]]
        ] = []
        has_people = bool(manifest.people_count_indices)
        n_missing_main = 0
        n_missing_aux = 0
        n_unknown_bucket = 0
        for stem in stems_subset:
            i = idx_of.get(stem)
            if i is None:
                n_missing_main += 1
                continue
            # Pooled + token caches share the per-stem filename; differentiator is
            # the cache_dir (.cache/pooled-X/ vs .cache/tokens-X/) + safetensors key.
            cache_file = cache_dir / f"{stem}.safetensors"
            cache_file_aux = cache_dir_aux / f"{stem}.safetensors"
            if not cache_file.exists():
                n_missing_main += 1
                continue
            if not cache_file_aux.exists():
                n_missing_aux += 1
                continue
            bucket_main: Optional[tuple[int, int]] = None
            if pool_kind == "map":
                with safe_open(str(cache_file), framework="pt") as f:
                    T_main = int(f.get_slice("tokens").get_shape()[0])
                bucket_main = token_to_bucket.get(T_main)
                if bucket_main is None:
                    n_unknown_bucket += 1
                    continue
            bucket_aux: Optional[tuple[int, int]] = None
            if pool_kind_aux == "map":
                with safe_open(str(cache_file_aux), framework="pt") as f:
                    T_aux = int(f.get_slice("tokens").get_shape()[0])
                bucket_aux = token_to_bucket_aux.get(T_aux)
                if bucket_aux is None:
                    n_unknown_bucket += 1
                    continue
            kept_stems.append(stem)
            kept_paths.append(cache_file)
            kept_paths_aux.append(cache_file_aux)
            kept_tag_idx.append(manifest.tag_indices[i])
            kept_rating_idx.append(manifest.rating_indices[i])
            kept_people_idx.append(
                manifest.people_count_indices[i] if has_people else 0
            )
            kept_buckets.append((bucket_main, bucket_aux))
        if not kept_stems:
            raise RuntimeError(
                f"no paired sidecars in {cache_dir} + {cache_dir_aux} for the "
                f"requested stems (n_requested={len(stems_subset)}, "
                f"n_missing_main={n_missing_main}, n_missing_aux={n_missing_aux}, "
                f"n_unknown_bucket={n_unknown_bucket})"
            )
        if n_missing_main or n_missing_aux:
            logger.warning(
                "CachedDualDataset: missing main=%d aux=%d (out of %d "
                "requested) - those stems are skipped",
                n_missing_main,
                n_missing_aux,
                len(stems_subset),
            )
        if n_unknown_bucket:
            logger.warning(
                "CachedDualDataset: %d stems had unexpected token counts "
                "(not in spec.buckets) and were skipped",
                n_unknown_bucket,
            )
        self.stems = kept_stems
        self.paths = kept_paths
        self.paths_aux = kept_paths_aux
        self.buckets = kept_buckets
        self.pool_kind = pool_kind
        self.pool_kind_aux = pool_kind_aux
        self.multi_hot = torch.zeros(len(kept_stems), manifest.n_tags)
        for row, idxs in enumerate(kept_tag_idx):
            self.multi_hot[row, idxs] = 1.0
        self.rating_idx = torch.tensor(kept_rating_idx, dtype=torch.long)
        self.people_idx = torch.tensor(kept_people_idx, dtype=torch.long)
        self.n_tags = manifest.n_tags
        self.n_ratings = manifest.n_ratings
        self.n_people_counts = manifest.n_people_counts
        self.spec = spec
        self.spec_aux = spec_aux
        self.d_in = self._peek_d(kept_paths[0], pool_kind)
        self.d_in_aux = self._peek_d(kept_paths_aux[0], pool_kind_aux)

        # Optional packed-cache layer: loading two sidecars per __getitem__ is ~30k
        # file opens/epoch and the loader (not the GPU) is the wall. pack_root
        # consolidates each (side, bucket) into one [N, T, D] mmap shard served as
        # zero-copy slices (~6x faster/batch). Bit-identical to the per-file path.
        self._packed = False
        if pack_root is not None:
            self._init_packed(Path(pack_root))

    @staticmethod
    def _peek_d(path: Path, kind: str) -> int:
        key = "feature" if kind == "mean" else "tokens"
        with safe_open(str(path), framework="pt") as f:
            return int(f.get_slice(key).get_shape()[-1])

    def __len__(self) -> int:
        return len(self.stems)

    @staticmethod
    def _load_one(path: Path, kind: str) -> torch.Tensor:
        key = "feature" if kind == "mean" else "tokens"
        return st_load(str(path))[key]

    @staticmethod
    def _bucket_key(bucket, kind: str) -> str:
        # mean sides have no spatial bucket — one shard holds every row.
        if kind == "mean" or bucket is None:
            return "mean"
        h, w = bucket
        return f"{h}x{w}"

    def _init_packed(self, pack_root: Path) -> None:
        """Build (if missing) per-(side, bucket) shards and per-item row maps.

        Shards are keyed by a hash of the kept-stem list so train and val get
        distinct shard sets and an unchanged split reuses shards across runs.
        Handles are opened lazily inside __getitem__ so each DataLoader worker
        mmaps its own (nothing is opened in the parent before fork).
        """
        self._pack_dirs: Dict[str, Path] = {}
        self._pack_main = self._pack_side(
            "main", self.paths, self.pool_kind, [b[0] for b in self.buckets], pack_root
        )
        self._pack_aux = self._pack_side(
            "aux",
            self.paths_aux,
            self.pool_kind_aux,
            [b[1] for b in self.buckets],
            pack_root,
        )
        self._handles: Dict[tuple, object] = {}
        self._packed = True
        self._write_pack_index(pack_root)
        if self._ram_resident_req:
            self._load_ram_from_shards()

    @staticmethod
    def _subset_sig(stems: Sequence[str]) -> str:
        return hashlib.sha1("\n".join(stems).encode()).hexdigest()[:12]

    def _write_pack_index(self, pack_root: Path) -> None:
        """Persist a sidecar-independent index so the dataset can be rebuilt
        from the shards alone after the per-stem sidecars are dropped. Keyed by
        a hash of the REQUESTED subset (available before the sidecar-dependent
        keep-loop) so recovery can find it; records the kept stems, per-side
        (bucket_key, row), shard dir names, and feature dims — everything
        __init__ otherwise derives from the sidecars. Written atomically.
        """
        sig = self._subset_sig(self._stems_requested)
        index = {
            "version": 1,
            "stems": self.stems,
            "pool_kind": self.pool_kind,
            "pool_kind_aux": self.pool_kind_aux,
            "d_in": int(self.d_in),
            "d_in_aux": int(self.d_in_aux),
            "dirs": {k: v.name for k, v in self._pack_dirs.items()},
            "rows": {
                "main": [[bk, row] for bk, row in self._pack_main],
                "aux": [[bk, row] for bk, row in self._pack_aux],
            },
        }
        tmp = pack_root / f"index-{sig}.json.tmp"
        tmp.write_text(json.dumps(index))
        os.replace(tmp, pack_root / f"index-{sig}.json")

    def _try_load_pack_index(self, pack_root: Path, stems_subset) -> Optional[dict]:
        """Return the packed index for this requested subset iff it AND every
        shard it references exist; else None (so a pruned/corrupt/stale pack
        falls back to the per-stem path rather than half-loading)."""
        index_path = pack_root / f"index-{self._subset_sig(stems_subset)}.json"
        if not index_path.exists():
            return None
        try:
            index = json.loads(index_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        dirs = index.get("dirs", {})
        rows = index.get("rows", {})
        for side in ("main", "aux"):
            d = pack_root / dirs.get(side, "")
            if not d.is_dir():
                return None
            for bk in {bk for bk, _row in rows.get(side, [])}:
                if not (d / f"{bk}.safetensors").exists():
                    logger.warning(
                        "packed index %s references missing shard %s — falling "
                        "back to per-stem sidecars",
                        index_path.name,
                        d / f"{bk}.safetensors",
                    )
                    return None
        return index

    @staticmethod
    def _sidecars_present(cache_dir: Path, stems_subset, idx_of) -> bool:
        """True while the per-stem caches still back this subset (normal path);
        False once dropped (recover from shards). The drop is all-or-nothing, so
        the first manifest-known stem's sidecar settles it."""
        for stem in stems_subset:
            if stem in idx_of:
                return (cache_dir / f"{stem}.safetensors").exists()
        return True

    def _init_from_pack_index(
        self,
        manifest,
        cache_dir: Path,
        cache_dir_aux: Path,
        pool_kind: str,
        pool_kind_aux: str,
        spec: Optional[BucketSpec],
        spec_aux: Optional[BucketSpec],
        pack_root: Path,
        index: dict,
    ) -> None:
        """Populate the dataset from a packed index (sidecars absent). Mirrors
        the keep-loop tail, sourcing stems / buckets / dims / row maps from the
        index and the per-row labels from the manifest."""
        kept_stems = list(index["stems"])
        idx_of = manifest.stem_index()
        has_people = bool(manifest.people_count_indices)
        kept_tag_idx: List[List[int]] = []
        kept_rating: List[int] = []
        kept_people: List[int] = []
        for stem in kept_stems:
            i = idx_of.get(stem)
            if i is None:
                raise RuntimeError(
                    f"packed index lists stem {stem!r} absent from the manifest "
                    f"— manifest changed under a stale pack; re-run `--mode "
                    f"build_features` to repack."
                )
            kept_tag_idx.append(manifest.tag_indices[i])
            kept_rating.append(manifest.rating_indices[i])
            kept_people.append(manifest.people_count_indices[i] if has_people else 0)

        def _parse_bk(bk: str) -> Optional[tuple[int, int]]:
            if bk == "mean":
                return None
            h, w = bk.split("x")
            return (int(h), int(w))

        rows_main = [(bk, int(r)) for bk, r in index["rows"]["main"]]
        rows_aux = [(bk, int(r)) for bk, r in index["rows"]["aux"]]
        self.stems = kept_stems
        self.paths = [cache_dir / f"{s}.safetensors" for s in kept_stems]
        self.paths_aux = [cache_dir_aux / f"{s}.safetensors" for s in kept_stems]
        self.buckets = [
            (_parse_bk(rows_main[i][0]), _parse_bk(rows_aux[i][0]))
            for i in range(len(kept_stems))
        ]
        self.pool_kind = pool_kind
        self.pool_kind_aux = pool_kind_aux
        self.multi_hot = torch.zeros(len(kept_stems), manifest.n_tags)
        for row, idxs in enumerate(kept_tag_idx):
            self.multi_hot[row, idxs] = 1.0
        self.rating_idx = torch.tensor(kept_rating, dtype=torch.long)
        self.people_idx = torch.tensor(kept_people, dtype=torch.long)
        self.n_tags = manifest.n_tags
        self.n_ratings = manifest.n_ratings
        self.n_people_counts = manifest.n_people_counts
        self.spec = spec
        self.spec_aux = spec_aux
        self.d_in = int(index["d_in"])
        self.d_in_aux = int(index["d_in_aux"])
        self._pack_dirs = {
            "main": pack_root / index["dirs"]["main"],
            "aux": pack_root / index["dirs"]["aux"],
        }
        self._pack_main = rows_main
        self._pack_aux = rows_aux
        self._handles: Dict[tuple, object] = {}
        self._packed = True
        logger.info(
            "CachedDualDataset: recovered %d stems from packed shards "
            "(per-stem sidecars absent)",
            len(kept_stems),
        )
        if self._ram_resident_req:
            self._load_ram_from_shards()

    def _pack_side(self, side, paths, kind, buckets, pack_root):
        groups: Dict[str, List[int]] = defaultdict(list)
        for i, b in enumerate(buckets):
            groups[self._bucket_key(b, kind)].append(i)
        stem_sig = hashlib.sha1("\n".join(self.stems).encode()).hexdigest()[:12]
        pack_dir = Path(pack_root) / f"{side}-{kind}-{stem_sig}"
        pack_dir.mkdir(parents=True, exist_ok=True)
        self._pack_dirs[side] = pack_dir
        row_of: List[tuple] = [(None, -1)] * len(paths)
        for bk, idxs in groups.items():
            for row, i in enumerate(idxs):
                row_of[i] = (bk, row)
            shard = pack_dir / f"{bk}.safetensors"
            if shard.exists():
                continue
            first = self._load_one(paths[idxs[0]], kind)
            out = torch.empty((len(idxs),) + tuple(first.shape), dtype=first.dtype)
            out[0] = first
            for row, i in enumerate(
                tqdm(idxs[1:], desc=f"pack {side} {bk}", leave=False), start=1
            ):
                out[row] = self._load_one(paths[i], kind)
            # tmp + rename so an interrupted build never leaves a partial shard
            # that exists() would treat as complete.
            tmp = shard.with_suffix(".safetensors.tmp")
            st_save({"data": out.contiguous()}, str(tmp))
            os.replace(tmp, shard)
            del out
        return row_of

    def _load_ram_from_shards(self) -> None:
        """Pull every per-(side, bucket) shard fully into a process-resident CPU
        tensor, so ``__getitem__`` indexes RAM instead of mmap'ing the shard.

        The on-disk shards are kept only as the load source (one big sequential
        read per bucket beats 30k sidecar opens); once here, the training loop
        touches zero disk. ``get_tensor`` copies into anonymous RAM (unlike
        ``get_slice``, which stays lazily mmap-backed), so a shuffled epoch never
        page-faults. Run single-process (``num_workers=0``) — the resident set is
        large and there's no IO left for workers to hide.
        """
        keys = {("main", bk) for bk, _ in self._pack_main}
        keys |= {("aux", bk) for bk, _ in self._pack_aux}
        total = 0
        for side, bk in sorted(keys):
            shard = self._pack_dirs[side] / f"{bk}.safetensors"
            with safe_open(str(shard), framework="pt") as h:
                t = h.get_tensor("data")  # copied into RAM, not mmap-backed
            self._ram[(side, bk)] = t
            total += t.numel() * t.element_size()
        self._handles = {}
        self._ram_resident = True
        logger.info(
            "CachedDualDataset: %d shards resident in RAM (%.1f GB) — "
            "loader is now disk-free",
            len(keys),
            total / 1024**3,
        )

    def _handle(self, side: str, bucket_key: str):
        h = self._handles.get((side, bucket_key))
        if h is None:
            shard = self._pack_dirs[side] / f"{bucket_key}.safetensors"
            h = safe_open(str(shard), framework="pt")
            self._handles[(side, bucket_key)] = h
        return h

    def __getitem__(self, idx: int):
        if self._ram_resident:
            bk_m, row_m = self._pack_main[idx]
            bk_a, row_a = self._pack_aux[idx]
            feat = self._ram[("main", bk_m)][row_m]
            feat_aux = self._ram[("aux", bk_a)][row_a]
        elif self._packed:
            bk_m, row_m = self._pack_main[idx]
            bk_a, row_a = self._pack_aux[idx]
            feat = self._handle("main", bk_m).get_slice("data")[row_m]
            feat_aux = self._handle("aux", bk_a).get_slice("data")[row_a]
        else:
            feat = self._load_one(self.paths[idx], self.pool_kind)
            feat_aux = self._load_one(self.paths_aux[idx], self.pool_kind_aux)
        return (
            feat,
            feat_aux,
            self.multi_hot[idx],
            self.rating_idx[idx],
            self.people_idx[idx],
            self.buckets[idx],
        )


# Back-compat alias — callers (and smoke tests) that imported the old name keep
# working.
CachedDualTokenDataset = CachedDualDataset


def collate_dual_token_batch(batch):
    """Stack a same-bucket-pair batch into
    ``(feat_main, feat_aux, multi_hot, rating, people, bucket_pair)``.

    BucketBatchSampler guarantees both shapes are constant within a batch
    (sampler groups by the compound ``(main_bucket | None, aux_bucket | None)``
    key). torch.stack works whether each side is rank-2 (mean-pool) or
    rank-3 (token sequence).
    """
    feat = torch.stack([b[0] for b in batch], dim=0)
    feat_aux = torch.stack([b[1] for b in batch], dim=0)
    multi_hot = torch.stack([b[2] for b in batch], dim=0)
    rating_idx = torch.stack([b[3] for b in batch], dim=0)
    people_idx = torch.stack([b[4] for b in batch], dim=0)
    bucket_pair = batch[0][5]
    return feat, feat_aux, multi_hot, rating_idx, people_idx, bucket_pair


class BucketBatchSampler(Sampler[List[int]]):
    """Yields batches of indices that share a single bucket — with IO-locality
    via **chunked shuffle**.

    A dataset index maps monotonically to a row in its packed shard (``_pack_*``
    assigns rows in ascending-index order within each bucket), so a *globally*
    shuffled epoch reads ~40 GB of token maps in random shard order — on a box
    whose RAM ≈ the working-set size, that thrashes the page cache every epoch.

    Chunked shuffle keeps reads local: within each bucket the (already ascending)
    index pool is sliced into contiguous ``chunk_size``-sample **chunks** (each a
    bounded, monotone row window on both shards — small enough to stay cache-
    resident while it's the active chunk). We shuffle the *order of chunks*
    (pooled across buckets, so aspect ratios still interleave and the order
    varies per epoch) and shuffle *within* each chunk (full gradient mixing, but
    the random access stays inside the resident window). Batches never straddle a
    chunk, and we do **not** globally shuffle batches across chunks — that would
    re-randomize the read order and defeat the locality.

    ``chunk_size`` is snapped down to a multiple of ``batch_size`` so only the
    final (partial) chunk of each bucket can yield a partial batch. A chunk large
    enough to cover a whole bucket reduces to the old per-bucket full shuffle.
    ``drop_last=False`` — partial trailing batches are kept.
    """

    def __init__(
        self,
        buckets: Sequence[tuple[int, int]],
        batch_size: int,
        seed: int = 42,
        shuffle: bool = True,
        chunk_size: int = 2048,
    ):
        self.buckets = list(buckets)
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        # chunk_size <= 0 → global shuffle (no IO locality constraint, e.g. the
        # RAM-resident loader); else snap to a whole number of batches (≥1).
        self.chunk_size = (
            0 if chunk_size <= 0 else max(1, chunk_size // batch_size) * batch_size
        )
        self._epoch = 0
        # Insertion order is ascending index == ascending shard row, which the
        # chunking below relies on for locality.
        self._by_bucket: Dict[tuple[int, int], List[int]] = defaultdict(list)
        for i, b in enumerate(self.buckets):
            self._by_bucket[b].append(i)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _chunks(self) -> List[List[int]]:
        """Contiguous (monotone-row) ``chunk_size`` slices, pooled over buckets."""
        chunks: List[List[int]] = []
        for _, idxs in sorted(self._by_bucket.items()):
            for s in range(0, len(idxs), self.chunk_size):
                chunks.append(idxs[s : s + self.chunk_size])
        return chunks

    def __iter__(self) -> Iterator[List[int]]:
        rng = torch.Generator().manual_seed(self.seed + self._epoch)
        if not self.chunk_size:
            # Global shuffle: shuffle each bucket, batch, then shuffle batch order
            # across buckets so aspect ratios interleave.
            all_batches: List[List[int]] = []
            for _, idxs in sorted(self._by_bucket.items()):
                order = idxs[:]
                if self.shuffle:
                    perm = torch.randperm(len(order), generator=rng).tolist()
                    order = [order[k] for k in perm]
                for s in range(0, len(order), self.batch_size):
                    all_batches.append(order[s : s + self.batch_size])
            if self.shuffle:
                perm = torch.randperm(len(all_batches), generator=rng).tolist()
                all_batches = [all_batches[k] for k in perm]
            yield from all_batches
            return
        chunks = self._chunks()
        if self.shuffle:
            cperm = torch.randperm(len(chunks), generator=rng).tolist()
            chunks = [chunks[k] for k in cperm]
        for chunk in chunks:
            order = chunk
            if self.shuffle:
                perm = torch.randperm(len(order), generator=rng).tolist()
                order = [order[k] for k in perm]
            for s in range(0, len(order), self.batch_size):
                yield order[s : s + self.batch_size]

    def __len__(self) -> int:
        n = 0
        if not self.chunk_size:
            for idxs in self._by_bucket.values():
                n += (len(idxs) + self.batch_size - 1) // self.batch_size
            return n
        for chunk in self._chunks():
            n += (len(chunk) + self.batch_size - 1) // self.batch_size
        return n
