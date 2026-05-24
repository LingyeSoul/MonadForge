"""Tests for ``library.preprocess._dataset`` — the shared walk/group/skip loop
extracted from the ``preprocess/cache_*.py`` scripts (Phase 1 of
``docs/proposal/tooling_architecture.md``).

These exercise the orchestration helpers without any model/encoder, so they
run in the unit suite. End-to-end content parity for the PE cache is gated
separately on ``make preprocess-pe`` (needs the encoder weights).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _write_image(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def test_walk_images_flat(tmp_path: Path) -> None:
    from library.preprocess import walk_images

    _write_image(tmp_path / "b.png", (8, 8))
    _write_image(tmp_path / "a.png", (8, 8))
    (tmp_path / "caption.txt").write_text("not an image")

    paths = walk_images(tmp_path, recursive=False)
    assert [p.name for p in paths] == ["a.png", "b.png"]  # sorted, txt excluded


def test_walk_images_recursive_same_stem_across_folders_ok(tmp_path: Path) -> None:
    from library.preprocess import walk_images

    _write_image(tmp_path / "charA" / "cover.png", (8, 8))
    _write_image(tmp_path / "charB" / "cover.png", (8, 8))

    paths = walk_images(tmp_path, recursive=True)
    assert len(paths) == 2  # same stem in different folders is fine


def test_walk_images_collision_within_folder_raises(tmp_path: Path) -> None:
    from library.preprocess import walk_images

    _write_image(tmp_path / "cover.png", (8, 8))
    _write_image(tmp_path / "cover.jpg", (8, 8))

    with pytest.raises(ValueError, match="Duplicate image stems"):
        walk_images(tmp_path, recursive=False)


def test_group_by_shape(tmp_path: Path) -> None:
    from library.preprocess import group_by_shape

    _write_image(tmp_path / "a.png", (8, 16))
    _write_image(tmp_path / "b.png", (8, 16))
    _write_image(tmp_path / "c.png", (16, 8))

    groups = group_by_shape(
        [tmp_path / "a.png", tmp_path / "b.png", tmp_path / "c.png"]
    )
    assert {k: sorted(p.name for p in v) for k, v in groups.items()} == {
        (8, 16): ["a.png", "b.png"],
        (16, 8): ["c.png"],
    }


def test_partition_cached(tmp_path: Path) -> None:
    from library.preprocess import partition_cached

    imgs = [tmp_path / f"img{i}.png" for i in range(3)]
    for p in imgs:
        _write_image(p, (8, 8))
    # Pretend img1 is already cached.
    (tmp_path / "img1.cached").touch()

    pending, skipped = partition_cached(imgs, lambda p: tmp_path / f"{p.stem}.cached")
    assert skipped == 1
    assert [p.name for p in pending] == ["img0.png", "img2.png"]


# ---------------------------------------------------------------------------
# Model-free end-to-end coverage for the loops moved into library/preprocess/
# (item A of the proposal). cache_pe_features / cache_latents /
# cache_text_embeddings need real encoders, so they stay gated on make
# preprocess-*; these two need no model.
# ---------------------------------------------------------------------------


def _write_te_cache(path: Path, crossattn: "object") -> None:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"crossattn_emb": crossattn}, str(path))


def test_cache_pooled_text_pools_and_is_idempotent(tmp_path: Path) -> None:
    import torch
    from safetensors.torch import load_file

    from library.io.cache import POOLED_CACHE_SUFFIX, TE_CACHE_SUFFIX
    from library.preprocess import cache_pooled_text

    crossattn = torch.randn(5, 4)
    te_path = tmp_path / f"img1{TE_CACHE_SUFFIX}"
    _write_te_cache(te_path, crossattn)
    # A TE cache with no crossattn key -> counted as failed.
    from safetensors.torch import save_file

    bad = tmp_path / f"img2{TE_CACHE_SUFFIX}"
    save_file({"prompt_embeds": torch.zeros(2, 2)}, str(bad))

    stats = cache_pooled_text(tmp_path)
    assert stats.seen == 2
    assert stats.written == 1
    assert stats.failed == 1

    pooled_path = tmp_path / f"img1{POOLED_CACHE_SUFFIX}"
    pooled = load_file(str(pooled_path))["pooled"]
    assert torch.allclose(pooled, crossattn.amax(dim=0))

    # Re-run: the written sidecar is skipped (idempotent).
    stats2 = cache_pooled_text(tmp_path)
    assert stats2.written == 0
    assert stats2.skipped == 1


def test_resize_to_buckets_writes_and_mirrors_layout(tmp_path: Path) -> None:
    from library.preprocess import resize_to_buckets

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    # Two images >= 0.5MP (so min_pixels keeps them); one nested.
    _write_image(src / "a.png", (900, 900))
    (src / "a.txt").write_text("caption a")
    _write_image(src / "charB" / "b.png", (900, 900))

    stats, bucket_counts = resize_to_buckets(
        src, dst, recursive=True, workers=1, verbose=False
    )
    assert stats.seen == 2
    assert stats.written == 2
    assert sum(bucket_counts.values()) == 2

    out_a = dst / "a.png"
    out_b = dst / "charB" / "b.png"
    assert out_a.exists() and out_b.exists()  # nested layout mirrored
    assert (dst / "a.txt").read_text() == "caption a"  # caption copied
    # Output matches a real bucket resolution.
    with Image.open(out_a) as im:
        assert (im.width, im.height) in bucket_counts


def test_resize_to_buckets_min_pixels_filter(tmp_path: Path) -> None:
    from library.preprocess import resize_to_buckets

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write_image(src / "tiny.png", (64, 64))  # 4096 px, below default 0.5MP

    stats, _ = resize_to_buckets(src, dst, workers=1, verbose=False)
    assert stats.seen == 1
    assert stats.skipped == 1
    assert stats.written == 0
    assert not (dst / "tiny.png").exists()
