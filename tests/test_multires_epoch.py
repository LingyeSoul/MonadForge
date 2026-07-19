"""Epoch-boundary coverage for mixed-resolution bucket datasets."""

from __future__ import annotations

from dataclasses import asdict
from multiprocessing import Value
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from library.datasets.base import BaseDataset
from library.datasets.buckets import BucketBatchIndex
from library.datasets.collator import collator_class
from library.datasets.group import DatasetGroup


def _write_latent_cache(path: Path, width: int, height: int) -> None:
    suffix = f"_{height // 8}x{width // 8}"
    np.savez(
        path,
        **{
            f"latents{suffix}": np.zeros(
                (2, height // 8, width // 8), dtype=np.float32
            ),
            f"original_size{suffix}": np.array([width, height]),
            f"crop_ltrb{suffix}": np.array([0, 0, width, height]),
        },
    )


class _FakeCpuVae:
    device = torch.device("cpu")
    dtype = torch.float32

    def encode_pixels_to_latents(self, images: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = images.shape
        return torch.zeros(
            (batch, 2, height // 8, width // 8),
            dtype=self.dtype,
            device=self.device,
        )


class _EpochBucketDataset(BaseDataset):
    def __init__(self) -> None:
        super().__init__(network_multiplier=1.0, debug_dataset=False)
        self.bucket_manager = SimpleNamespace(
            buckets=[["low-resolution"], ["high-resolution"]]
        )
        self.buckets_indices = [
            BucketBatchIndex(0, 1, 0),
            BucketBatchIndex(1, 1, 0),
        ]
        self.caching_mode = "test"
        self.image_data = {}
        self.num_train_images = 2
        self.num_reg_images = 0

    def __len__(self) -> int:
        return len(self.buckets_indices)

    def shuffle_buckets(self) -> None:
        self.buckets_indices.reverse()

    def get_item_for_caching(self, bucket, bucket_batch_size, image_index):
        return bucket[image_index]


def test_shared_epoch_is_applied_before_first_bucket_lookup():
    """Changing epoch after the first lookup duplicates one batch and drops one."""
    dataset = _EpochBucketDataset()
    current_epoch = SimpleNamespace(value=1)
    current_step = SimpleNamespace(value=0)
    dataset._shared_epoch = current_epoch

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collator_class(current_epoch, current_step, dataset),
    )

    assert list(loader) == ["high-resolution", "low-resolution"]

    current_epoch.value = 2
    assert list(loader) == ["low-resolution", "high-resolution"]


def test_dataset_group_propagates_shared_epoch_to_members():
    members = [_EpochBucketDataset(), _EpochBucketDataset()]
    group = DatasetGroup(members)
    current_epoch = SimpleNamespace(value=3)

    group.set_shared_epoch(current_epoch)

    assert all(member._shared_epoch is current_epoch for member in members)


def test_persistent_worker_keeps_all_resolutions_across_epochs():
    dataset = _EpochBucketDataset()
    current_epoch = Value("i", 1)
    current_step = Value("i", 0)
    dataset.set_shared_epoch(current_epoch)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collator_class(current_epoch, current_step, None),
        num_workers=1,
        persistent_workers=True,
    )

    try:
        assert list(loader) == ["high-resolution", "low-resolution"]
        current_epoch.value = 2
        assert list(loader) == ["low-resolution", "high-resolution"]
    finally:
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()


def test_one_physical_image_expands_to_all_cached_resolutions(
    tmp_path: Path, monkeypatch
):
    from PIL import Image

    from library.anima.strategy import AnimaLatentsCachingStrategy
    from library.config.loader import DreamBoothSubsetParams
    from library.datasets.dreambooth import DreamBoothDataset
    from library.datasets.subsets import DreamBoothSubset
    from library.anima.text_strategies import LatentsCachingStrategy

    image_dir = tmp_path / "resized"
    cache_dir = tmp_path / "cache"
    image_dir.mkdir()
    cache_dir.mkdir()
    image_path = image_dir / "sample.png"
    Image.new("RGB", (512, 512)).save(image_path)
    cache_paths = [
        cache_dir / "sample_0512x0512_anima.npz",
        cache_dir / "sample_1024x1024_anima.npz",
    ]
    for path, edge in zip(cache_paths, (512, 1024)):
        _write_latent_cache(path, edge, edge)

    subset = DreamBoothSubset(
        **asdict(
            DreamBoothSubsetParams(
                image_dir=str(image_dir),
                cache_dir=str(cache_dir),
            )
        )
    )
    monkeypatch.setattr(
        LatentsCachingStrategy,
        "_strategy",
        AnimaLatentsCachingStrategy(True, 1, False),
    )
    dataset = DreamBoothDataset(
        subsets=[subset],
        is_training_dataset=True,
        batch_size=1,
        network_multiplier=1.0,
        prior_loss_weight=1.0,
        debug_dataset=False,
        validation_split=0.0,
        validation_seed=0,
        resize_interpolation=None,
        multires_per_image=True,
        target_res=[512, 1024],
    )
    dataset.make_buckets()

    infos = list(dataset.image_data.values())
    assert len(infos) == 2
    assert {info.absolute_path for info in infos} == {str(image_path)}
    assert {info.image_size for info in infos} == {(512, 512), (1024, 1024)}
    assert {Path(info.latents_npz) for info in infos} == set(cache_paths)
    assert len(dataset) == 2


def test_multires_resize_cache_and_epoch_expansion_end_to_end(
    tmp_path: Path, monkeypatch
):
    from PIL import Image

    from library.anima.strategy import AnimaLatentsCachingStrategy
    from library.anima.text_strategies import LatentsCachingStrategy
    from library.config.loader import DreamBoothSubsetParams
    from library.datasets.dreambooth import DreamBoothDataset
    from library.datasets.subsets import DreamBoothSubset
    from library.preprocess import resize_to_buckets
    from library.preprocess.latents import cache_latents

    source_dir = tmp_path / "source"
    resized_dir = tmp_path / "resized"
    multires_dir = tmp_path / "multires"
    cache_dir = tmp_path / "cache"
    source_dir.mkdir()
    Image.new("RGB", (640, 768), "white").save(source_dir / "sample.png")
    (source_dir / "sample.txt").write_text("sample caption", encoding="utf-8")

    resize_to_buckets(
        source_dir,
        resized_dir,
        target_res=[512, 768],
        multires_per_image=True,
        multires_dir=multires_dir,
        min_pixels=0,
        workers=1,
        verbose=False,
    )
    for edge in (512, 768):
        stats = cache_latents(
            multires_dir / str(edge),
            _FakeCpuVae(),
            cache_dir=cache_dir,
            recursive=True,
            batch_size=1,
            io_workers=1,
        )
        assert stats.written == 1

    strategy = AnimaLatentsCachingStrategy(True, 1, False)
    monkeypatch.setattr(LatentsCachingStrategy, "_strategy", strategy)
    subset = DreamBoothSubset(
        **asdict(
            DreamBoothSubsetParams(
                image_dir=str(resized_dir),
                cache_dir=str(cache_dir),
            )
        )
    )
    dataset = DreamBoothDataset(
        subsets=[subset],
        is_training_dataset=True,
        # A batch larger than each one-item resolution bucket must not drop the
        # tail: every selected tier still appears in this epoch.
        batch_size=2,
        network_multiplier=1.0,
        prior_loss_weight=1.0,
        debug_dataset=False,
        validation_split=0.0,
        validation_seed=0,
        resize_interpolation=None,
        multires_per_image=True,
        target_res=[512, 768],
    )
    dataset.make_buckets()
    dataset.set_current_strategies()

    infos = list(dataset.image_data.values())
    assert len(infos) == 2
    assert len(dataset) == 2
    epoch_keys = []
    for batch_index in dataset.buckets_indices:
        bucket = dataset.bucket_manager.buckets[batch_index.bucket_index]
        start = batch_index.batch_index * batch_index.bucket_batch_size
        epoch_keys.extend(bucket[start : start + batch_index.bucket_batch_size])
    assert set(epoch_keys) == set(dataset.image_data)

    for info in infos:
        latents, original_size, crop_ltrb, _, _ = strategy.load_latents_from_disk(
            info.latents_npz, info.bucket_reso
        )
        assert latents.shape[-2:] == (
            info.image_size[1] // 8,
            info.image_size[0] // 8,
        )
        assert original_size == list(info.image_size)
        assert crop_ltrb == [0, 0, *info.image_size]


def test_multires_dataset_fails_when_selected_tier_cache_is_missing(
    tmp_path: Path, monkeypatch
):
    from PIL import Image

    from library.anima.strategy import AnimaLatentsCachingStrategy
    from library.anima.text_strategies import LatentsCachingStrategy
    from library.config.loader import DreamBoothSubsetParams
    from library.datasets.dreambooth import DreamBoothDataset
    from library.datasets.subsets import DreamBoothSubset

    image_dir = tmp_path / "resized"
    cache_dir = tmp_path / "cache"
    image_dir.mkdir()
    cache_dir.mkdir()
    Image.new("RGB", (512, 512)).save(image_dir / "sample.png")
    _write_latent_cache(cache_dir / "sample_0512x0512_anima.npz", 512, 512)
    monkeypatch.setattr(
        LatentsCachingStrategy,
        "_strategy",
        AnimaLatentsCachingStrategy(True, 1, False),
    )
    subset = DreamBoothSubset(
        **asdict(
            DreamBoothSubsetParams(
                image_dir=str(image_dir),
                cache_dir=str(cache_dir),
            )
        )
    )

    with pytest.raises(FileNotFoundError, match=r"missing VAE cache tier.*1024"):
        DreamBoothDataset(
            subsets=[subset],
            is_training_dataset=True,
            batch_size=1,
            network_multiplier=1.0,
            prior_loss_weight=1.0,
            debug_dataset=False,
            validation_split=0.0,
            validation_seed=0,
            resize_interpolation=None,
            multires_per_image=True,
            target_res=[512, 1024],
        )


def test_multires_dataset_rejects_cache_without_expected_latent_keys(
    tmp_path: Path, monkeypatch
):
    from PIL import Image

    from library.anima.strategy import AnimaLatentsCachingStrategy
    from library.anima.text_strategies import LatentsCachingStrategy
    from library.config.loader import DreamBoothSubsetParams
    from library.datasets.dreambooth import DreamBoothDataset
    from library.datasets.subsets import DreamBoothSubset

    image_dir = tmp_path / "resized"
    cache_dir = tmp_path / "cache"
    image_dir.mkdir()
    cache_dir.mkdir()
    Image.new("RGB", (512, 512)).save(image_dir / "sample.png")
    _write_latent_cache(cache_dir / "sample_0512x0512_anima.npz", 512, 512)
    np.savez(cache_dir / "sample_1024x1024_anima.npz", unrelated=np.array([1]))
    monkeypatch.setattr(
        LatentsCachingStrategy,
        "_strategy",
        AnimaLatentsCachingStrategy(True, 1, False),
    )
    subset = DreamBoothSubset(
        **asdict(
            DreamBoothSubsetParams(
                image_dir=str(image_dir),
                cache_dir=str(cache_dir),
            )
        )
    )

    with pytest.raises(ValueError, match=r"no usable VAE cache.*1024"):
        DreamBoothDataset(
            subsets=[subset],
            is_training_dataset=True,
            batch_size=1,
            network_multiplier=1.0,
            prior_loss_weight=1.0,
            debug_dataset=False,
            validation_split=0.0,
            validation_seed=0,
            resize_interpolation=None,
            multires_per_image=True,
            target_res=[512, 1024],
        )


def test_multires_dataset_rejects_ambiguous_same_tier_caches(
    tmp_path: Path, monkeypatch
):
    from PIL import Image

    from library.anima.strategy import AnimaLatentsCachingStrategy
    from library.anima.text_strategies import LatentsCachingStrategy
    from library.config.loader import DreamBoothSubsetParams
    from library.datasets.dreambooth import DreamBoothDataset
    from library.datasets.subsets import DreamBoothSubset

    image_dir = tmp_path / "resized"
    cache_dir = tmp_path / "cache"
    image_dir.mkdir()
    cache_dir.mkdir()
    Image.new("RGB", (512, 512)).save(image_dir / "sample.png")
    _write_latent_cache(cache_dir / "sample_0512x0512_anima.npz", 512, 512)
    _write_latent_cache(cache_dir / "sample_0480x0544_anima.npz", 480, 544)
    _write_latent_cache(cache_dir / "sample_1024x1024_anima.npz", 1024, 1024)
    monkeypatch.setattr(
        LatentsCachingStrategy,
        "_strategy",
        AnimaLatentsCachingStrategy(True, 1, False),
    )
    subset = DreamBoothSubset(
        **asdict(
            DreamBoothSubsetParams(
                image_dir=str(image_dir),
                cache_dir=str(cache_dir),
            )
        )
    )

    with pytest.raises(ValueError, match=r"multiple usable VAE caches.*tier 512"):
        DreamBoothDataset(
            subsets=[subset],
            is_training_dataset=True,
            batch_size=1,
            network_multiplier=1.0,
            prior_loss_weight=1.0,
            debug_dataset=False,
            validation_split=0.0,
            validation_seed=0,
            resize_interpolation=None,
            multires_per_image=True,
            target_res=[512, 1024],
        )
