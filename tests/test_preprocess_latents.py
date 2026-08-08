from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from library.preprocess.latents import cache_latents, get_latents_npz_path


class _FakeVae:
    device = torch.device("cpu")
    dtype = torch.float32

    def __init__(
        self,
        *,
        fail_above: int | None = None,
        error: Exception | None = None,
    ) -> None:
        self.fail_above = fail_above
        self.error = error
        self.batch_calls: list[int] = []
        self.clear_cache_calls = 0

    def clear_cache(self) -> None:
        self.clear_cache_calls += 1

    def encode_pixels_to_latents(self, images: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = images.shape
        self.batch_calls.append(batch)
        if self.error is not None:
            raise self.error
        if self.fail_above is not None and batch > self.fail_above:
            raise torch.cuda.OutOfMemoryError(f"synthetic OOM for batch {batch}")

        values = images.mean(dim=(1, 2, 3), keepdim=True)
        return values.expand(batch, 2, height // 8, width // 8).clone()


def _write_images(data_dir: Path, values: list[int]) -> list[Path]:
    data_dir.mkdir()
    paths = []
    for index, value in enumerate(values):
        path = data_dir / f"{index:02d}.png"
        Image.new("RGB", (64, 64), (value, value, value)).save(path)
        paths.append(path)
    return paths


def _cached_mean(data_dir: Path, cache_dir: Path, image_path: Path) -> float:
    cache_path = get_latents_npz_path(
        image_path,
        (64, 64),
        cache_dir=cache_dir,
        image_dir=data_dir,
    )
    with np.load(cache_path) as cached:
        return float(cached["latents_8x8"].mean())


def test_cache_latents_keeps_normal_batch_intact(tmp_path: Path) -> None:
    data_dir = tmp_path / "images"
    cache_dir = tmp_path / "cache"
    paths = _write_images(data_dir, [16, 64, 128, 240])
    vae = _FakeVae()

    stats = cache_latents(
        data_dir,
        vae,
        cache_dir=cache_dir,
        batch_size=4,
        io_workers=1,
    )

    assert (stats.seen, stats.written, stats.skipped) == (4, 4, 0)
    assert vae.batch_calls == [4]
    assert vae.clear_cache_calls == 0
    means = [_cached_mean(data_dir, cache_dir, path) for path in paths]
    assert means == sorted(means)
    assert len(set(means)) == 4


def test_cache_latents_recursively_bisects_cuda_oom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    data_dir = tmp_path / "images"
    cache_dir = tmp_path / "cache"
    paths = _write_images(data_dir, [16, 64, 128, 240])
    vae = _FakeVae(fail_above=1)
    empty_cache_calls = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: empty_cache_calls.append(1))

    stats = cache_latents(
        data_dir,
        vae,
        cache_dir=cache_dir,
        batch_size=4,
        io_workers=1,
    )

    assert (stats.seen, stats.written, stats.skipped) == (4, 4, 0)
    assert vae.batch_calls == [4, 2, 1, 1, 2, 1, 1]
    assert vae.clear_cache_calls == 3
    assert len(empty_cache_calls) == 3
    output = capsys.readouterr().out
    assert "batch_size=4 for 64x64" in output
    assert "retrying as 2+2" in output
    assert "batch_size=2 for 64x64" in output
    assert "retrying as 1+1" in output
    means = [_cached_mean(data_dir, cache_dir, path) for path in paths]
    assert means == sorted(means)
    assert len(set(means)) == 4


def test_cache_latents_reraises_single_image_cuda_oom(tmp_path: Path) -> None:
    data_dir = tmp_path / "images"
    cache_dir = tmp_path / "cache"
    _write_images(data_dir, [128])
    vae = _FakeVae(fail_above=0)

    with pytest.raises(torch.cuda.OutOfMemoryError, match="synthetic OOM"):
        cache_latents(
            data_dir,
            vae,
            cache_dir=cache_dir,
            batch_size=1,
            io_workers=1,
        )

    assert vae.batch_calls == [1]
    assert vae.clear_cache_calls == 0
    assert not list(cache_dir.rglob("*.npz"))


def test_cache_latents_does_not_intercept_other_errors(tmp_path: Path) -> None:
    data_dir = tmp_path / "images"
    cache_dir = tmp_path / "cache"
    _write_images(data_dir, [128, 192])
    vae = _FakeVae(error=RuntimeError("encoder failed"))

    with pytest.raises(RuntimeError, match="encoder failed"):
        cache_latents(
            data_dir,
            vae,
            cache_dir=cache_dir,
            batch_size=2,
            io_workers=1,
        )

    assert vae.batch_calls == [2]
    assert vae.clear_cache_calls == 0
    assert not list(cache_dir.rglob("*.npz"))
