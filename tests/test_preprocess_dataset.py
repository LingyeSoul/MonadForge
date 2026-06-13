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

from library.datasets.buckets import CONSTANT_TOKEN_BUCKETS_768


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


def test_walk_images_path_pattern_filters_relative_paths(tmp_path: Path) -> None:
    from library.preprocess import walk_images

    _write_image(tmp_path / "charA" / "cover.png", (8, 8))
    _write_image(tmp_path / "charB" / "cover.png", (8, 8))

    paths = walk_images(tmp_path, recursive=True, pattern="charA/*")
    assert [p.relative_to(tmp_path).as_posix() for p in paths] == [
        "charA/cover.png"
    ]


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


def test_resize_to_buckets_path_pattern_preserves_filtered_layout(
    tmp_path: Path,
) -> None:
    from library.preprocess import resize_to_buckets

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write_image(src / "charA" / "a.png", (900, 900))
    _write_image(src / "charB" / "b.png", (900, 900))

    stats, bucket_counts = resize_to_buckets(
        src,
        dst,
        recursive=True,
        path_pattern="charA/*",
        min_pixels=0,
        workers=1,
        verbose=False,
    )
    assert stats.seen == 1
    assert stats.written == 1
    assert sum(bucket_counts.values()) == 1
    assert (dst / "charA" / "a.png").exists()
    assert not (dst / "charB" / "b.png").exists()


def test_resize_to_buckets_default_tier_does_not_upscale_to_multitier(
    tmp_path: Path,
) -> None:
    """Regression: target_res=None (no preprocess.toml / no flag, and the bare
    [1024] that tasks.py strips to None) must resize against the single 1024
    tier, NOT the full multi-tier catalog. The old else-branch fell back to
    all_constant_token_buckets(), whose aspect-only match shoved a 0.73MP
    portrait into the 1536-tier (1024, 2160) bucket — a 3x upscale."""
    from library.datasets.buckets import buckets_for_edges
    from library.preprocess import resize_to_buckets

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write_image(src / "portrait.png", (589, 1233))  # 0.73MP, ar 0.478

    one_tier = set(buckets_for_edges([1024]))
    for target_res in (None, [1024]):
        stats, _ = resize_to_buckets(
            src,
            dst,
            target_res=target_res,
            min_pixels=0,
            workers=1,
            verbose=False,
            overwrite=True,
        )
        assert stats.written == 1
        with Image.open(dst / "portrait.png") as im:
            reso = (im.width, im.height)
        assert reso in one_tier, f"{target_res}: {reso} escaped the 1024 tier"
        assert reso != (1024, 2160), f"{target_res}: reproduced the upscale bug"


def test_resize_to_buckets_skips_up_to_date_and_rebuckets_on_tier_change(
    tmp_path: Path,
) -> None:
    """Idempotent skip: a re-run touches nothing; adding a 768 tier re-resizes
    only the image whose target bucket actually moves."""
    from library.preprocess import resize_to_buckets

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write_image(src / "small.png", (700, 860))  # ~0.6MP → flips to 768 tier
    _write_image(src / "big.png", (1400, 1050))  # ~1.5MP → stays 1024 tier

    # First pass at the single 1024 tier writes both.
    stats, _ = resize_to_buckets(
        src, dst, target_res=[1024], min_pixels=0, workers=1, verbose=False
    )
    assert (stats.written, stats.skipped) == (2, 0)

    # Re-run, same tiers: both already at their bucket → all skipped.
    stats, _ = resize_to_buckets(
        src, dst, target_res=[1024], min_pixels=0, workers=1, verbose=False
    )
    assert (stats.written, stats.skipped) == (0, 2)

    # Add the 768 tier: only `small` moves bucket → exactly one re-resize.
    stats, counts = resize_to_buckets(
        src, dst, target_res=[768, 1024], min_pixels=0, workers=1, verbose=False
    )
    assert (stats.written, stats.skipped) == (1, 1)
    with Image.open(dst / "small.png") as im:
        assert (im.width, im.height) in CONSTANT_TOKEN_BUCKETS_768

    # overwrite=True forces both even when up to date.
    stats, _ = resize_to_buckets(
        src,
        dst,
        target_res=[768, 1024],
        min_pixels=0,
        workers=1,
        verbose=False,
        overwrite=True,
    )
    assert (stats.written, stats.skipped) == (2, 0)


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


def test_reconcile_caches_removes_only_wrong_bucket(tmp_path: Path) -> None:
    """Under [768, 1024], the small image's old 1024-tier caches are stale; the
    big image's correct caches and every TE sidecar are left untouched."""
    from library.preprocess import find_stale_caches, delete_stale

    image_dir = tmp_path / "image_dataset"
    resized = tmp_path / "post" / "resized"
    lora = tmp_path / "post" / "lora"
    masks = tmp_path / "post" / "masks"
    target_res = [768, 1024]

    # small: 0.6MP → flips to the 768 tier (640x864); caches still at 1024 (896x1152).
    _write_image(image_dir / "charA" / "small.png", (700, 860))
    _write_image(resized / "charA" / "small.png", (896, 1152))  # wrong size
    (lora / "charA").mkdir(parents=True)
    small_npz = lora / "charA" / "small_0896x1152_anima.npz"
    small_pe = lora / "charA" / "small_anima_pe.safetensors"
    small_te = lora / "charA" / "small_anima_te.safetensors"  # must survive
    small_npz.touch()
    small_pe.touch()
    small_te.touch()
    (masks / "charA").mkdir(parents=True)
    small_mask = masks / "charA" / "small_mask.png"
    small_mask.touch()

    # big: 1.47MP → stays 1024 tier (1200x896); caches already correct.
    _write_image(image_dir / "big.png", (1400, 1050))
    _write_image(resized / "big.png", (1200, 896))  # correct size
    big_npz = lora / "big_1200x0896_anima.npz"
    big_npz.touch()

    stale = find_stale_caches(image_dir, resized, lora, masks, target_res)
    assert stale.n_images == 1
    assert stale.npz == [small_npz]
    assert stale.png == [resized / "charA" / "small.png"]
    assert stale.pe == [small_pe]
    assert stale.mask == [small_mask]
    assert big_npz not in stale.npz  # consistent image untouched

    removed = delete_stale(stale)
    assert removed == {"npz": 1, "png": 1, "pe": 1, "mask": 1}
    assert not small_npz.exists() and not small_pe.exists() and not small_mask.exists()
    assert small_te.exists()  # TE is text-only — never reconciled
    assert big_npz.exists()
