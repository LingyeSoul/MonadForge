"""Invariant test for CachedDualDataset's RAM-resident layer.

``ram_resident`` stacks the per-stem token sidecars into one CPU tensor per
(side, bucket) and serves rows as zero-copy slices — a pure storage relayout,
so it MUST be bit-identical to the lazy per-file ``load_file`` path. This builds
synthetic caches in a tmp dir (no PE checkpoints, no real model dir) and asserts
equality + full-coverage shuffled iteration.
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file as st_save

from library.captioning.anima_tagger_data import CachedDualDataset, TaggerManifest
from library.vision.buckets import BucketSpec


def _spec(encoder: str) -> BucketSpec:
    # Two buckets with DISTINCT token counts (buckets key on h*w+cls, so
    # (2,3) and (3,2) would collide at 7 — pick distinct areas instead).
    return BucketSpec(encoder=encoder, patch=16, use_cls=True, buckets=[(2, 3), (2, 4)])


def _write_caches(cache_dir: Path, stems, d: int, hw_of_stem, key: str):
    cache_dir.mkdir(parents=True, exist_ok=True)
    feats = {}
    for s in stems:
        h, w = hw_of_stem(s)
        t = h * w + 1  # +CLS
        x = torch.randn(t, d, dtype=torch.bfloat16)
        st_save({key: x}, str(cache_dir / f"{s}.safetensors"))
        feats[s] = x
    return feats


def _manifest(stems, n_tags):
    return TaggerManifest(
        stems=list(stems),
        image_paths=[f"{s}.png" for s in stems],
        tag_indices=[[i % n_tags] for i, _ in enumerate(stems)],
        rating_indices=[i % 3 for i, _ in enumerate(stems)],
        people_count_indices=[0 for _ in stems],
        train_stems=list(stems),
        val_stems=[],
        n_tags=n_tags,
        n_ratings=3,
        n_people_counts=1,
    )


def _build(manifest, cd, cda, spec, spec_aux, ram_resident=False, subset=None):
    return CachedDualDataset(
        manifest,
        cd,
        "map",
        spec,
        cda,
        "map",
        spec_aux,
        stems_subset=subset if subset is not None else manifest.stems,
        ram_resident=ram_resident,
    )


def test_ram_resident_bit_identical(tmp_path: Path):
    """RAM-resident serving matches the lazy per-file path row-for-row."""
    stems = [f"s{i:03d}" for i in range(24)]
    n_tags = 6
    spec, spec_aux = _spec("pe"), _spec("pe_spatial")
    cd, cda = tmp_path / "tokens-main", tmp_path / "tokens-aux"
    # Alternate stems across the two buckets so each group gets several rows.
    hw = lambda s: (2, 3) if int(s[1:]) % 2 == 0 else (2, 4)  # noqa: E731
    _write_caches(cd, stems, 16, hw, "tokens")
    _write_caches(cda, stems, 8, hw, "tokens")
    manifest = _manifest(stems, n_tags)

    lazy = _build(manifest, cd, cda, spec, spec_aux)
    ram = _build(manifest, cd, cda, spec, spec_aux, ram_resident=True)
    assert ram._ram_resident and not lazy._ram_resident
    assert len(ram) == len(lazy) == len(stems)

    for i in range(len(ram)):
        a, b = lazy[i], ram[i]
        assert torch.equal(a[0], b[0]), f"main feature mismatch @ {i}"
        assert torch.equal(a[1], b[1]), f"aux feature mismatch @ {i}"
        assert torch.equal(a[2], b[2]) and a[3] == b[3] and a[4] == b[4]
        assert a[5] == b[5], f"bucket mismatch @ {i}"


def test_ram_resident_is_disk_free(tmp_path: Path):
    """Once resident, deleting every sidecar must not break serving — the rows
    live in RAM (one tensor per (side, bucket)), not behind lazy file handles."""
    stems = [f"s{i:03d}" for i in range(24)]
    spec, spec_aux = _spec("pe"), _spec("pe_spatial")
    cd, cda = tmp_path / "tokens-main", tmp_path / "tokens-aux"
    hw = lambda s: (2, 3) if int(s[1:]) % 2 == 0 else (2, 4)  # noqa: E731
    _write_caches(cd, stems, 16, hw, "tokens")
    _write_caches(cda, stems, 8, hw, "tokens")
    manifest = _manifest(stems, 6)

    ram = _build(manifest, cd, cda, spec, spec_aux, ram_resident=True)
    ref = [ram[i] for i in range(len(ram))]
    # 2 buckets × 2 sides = 4 resident tensors.
    assert len(ram._ram) == 4

    # Nuke the on-disk sidecars; resident serving must be unaffected.
    for p in list(ram.paths) + list(ram.paths_aux):
        Path(p).unlink()
    for i in range(len(ram)):
        a, b = ref[i], ram[i]
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
        assert a[5] == b[5]


def test_ram_resident_loader_full_coverage(tmp_path: Path):
    """A chunk_size=0 global-shuffle loader at num_workers=0 over the resident
    set covers every sample each epoch."""
    from torch.utils.data import DataLoader

    from library.captioning.anima_tagger_data import (
        BucketBatchSampler,
        collate_dual_token_batch,
    )

    stems = [f"s{i:03d}" for i in range(40)]
    spec, spec_aux = _spec("pe"), _spec("pe_spatial")
    cd, cda = tmp_path / "tokens-main", tmp_path / "tokens-aux"
    hw = lambda s: (2, 3) if int(s[1:]) % 2 == 0 else (2, 4)  # noqa: E731
    _write_caches(cd, stems, 16, hw, "tokens")
    _write_caches(cda, stems, 8, hw, "tokens")
    manifest = _manifest(stems, 6)

    ram = _build(manifest, cd, cda, spec, spec_aux, ram_resident=True)
    samp = BucketBatchSampler(
        ram.buckets, batch_size=8, seed=0, shuffle=True, chunk_size=0
    )
    assert len(list(samp)) == len(samp)
    dl = DataLoader(
        ram,
        batch_sampler=samp,
        num_workers=0,
        collate_fn=collate_dual_token_batch,
    )
    seen = 0
    for _ in range(2):
        for batch in dl:
            assert batch[0].dim() == 3 and batch[1].dim() == 3
            seen += batch[0].shape[0]
    assert seen == 2 * len(stems)


def test_lazy_dataloader_workers(tmp_path: Path):
    """The --no-ram_resident lazy per-stem path iterates cleanly under forked
    persistent workers across epochs."""
    from torch.utils.data import DataLoader

    from library.captioning.anima_tagger_data import (
        BucketBatchSampler,
        collate_dual_token_batch,
    )

    stems = [f"s{i:03d}" for i in range(20)]
    spec, spec_aux = _spec("pe"), _spec("pe_spatial")
    cd, cda = tmp_path / "tokens-main", tmp_path / "tokens-aux"
    hw = lambda s: (2, 3) if int(s[1:]) % 2 == 0 else (2, 4)  # noqa: E731
    _write_caches(cd, stems, 16, hw, "tokens")
    _write_caches(cda, stems, 8, hw, "tokens")
    manifest = _manifest(stems, 6)
    ds = _build(manifest, cd, cda, spec, spec_aux)  # lazy (ram_resident=False)

    samp = BucketBatchSampler(ds.buckets, batch_size=4, seed=0, shuffle=True)
    dl = DataLoader(
        ds,
        batch_sampler=samp,
        num_workers=2,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=collate_dual_token_batch,
    )
    seen = 0
    for _ in range(2):  # two epochs exercises persistent workers
        for batch in dl:
            tok, tok_aux = batch[0], batch[1]
            assert tok.dtype == torch.bfloat16 and tok.dim() == 3
            assert tok_aux.dim() == 3
            seen += tok.shape[0]
    assert seen == 2 * len(stems)
