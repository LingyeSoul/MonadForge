"""Invariant test for CachedDualDataset's packed-cache layer.

The packed path consolidates the per-stem token sidecars into one mmap shard
per (side, bucket) and serves rows as zero-copy slices — a pure storage
relayout, so it MUST be bit-identical to the per-file ``load_file`` path.
This builds synthetic caches in a tmp dir (no PE checkpoints, no real model
dir) and asserts equality + shard reuse.
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


def _build(manifest, cd, cda, spec, spec_aux, pack_root):
    return CachedDualDataset(
        manifest,
        cd,
        "map",
        spec,
        cda,
        "map",
        spec_aux,
        stems_subset=manifest.stems,
        pack_root=pack_root,
    )


def test_packed_cache_bit_identical_and_reused(tmp_path: Path):
    stems = [f"s{i:03d}" for i in range(24)]
    n_tags = 6
    spec, spec_aux = _spec("pe"), _spec("pe_spatial")
    cd, cda = tmp_path / "tokens-main", tmp_path / "tokens-aux"
    # Alternate stems across the two buckets so each shard gets several rows.
    hw = lambda s: (2, 3) if int(s[1:]) % 2 == 0 else (2, 4)  # noqa: E731
    _write_caches(cd, stems, 16, hw, "tokens")
    _write_caches(cda, stems, 8, hw, "tokens")
    manifest = _manifest(stems, n_tags)

    unp = _build(manifest, cd, cda, spec, spec_aux, None)
    pk = _build(manifest, cd, cda, spec, spec_aux, tmp_path / "packed")
    assert pk._packed and not unp._packed
    assert len(pk) == len(unp) == len(stems)

    for i in range(len(pk)):
        a, b = unp[i], pk[i]
        assert torch.equal(a[0], b[0]), f"main feature mismatch @ {i}"
        assert torch.equal(a[1], b[1]), f"aux feature mismatch @ {i}"
        assert torch.equal(a[2], b[2]) and a[3] == b[3] and a[4] == b[4]
        assert a[5] == b[5], f"bucket mismatch @ {i}"

    # One shard per (side, bucket); 2 buckets × 2 sides = 4 shards.
    shards = sorted((tmp_path / "packed").glob("*/*.safetensors"))
    assert len(shards) == 4, [s.name for s in shards]

    # Rebuild with the same split must REUSE shards (same stem-hash dir, files
    # already present) — assert no shard mtime changes.
    mtimes = {s: s.stat().st_mtime_ns for s in shards}
    _build(manifest, cd, cda, spec, spec_aux, tmp_path / "packed")
    for s, mt in mtimes.items():
        assert s.stat().st_mtime_ns == mt, f"shard rebuilt unexpectedly: {s.name}"


def test_changed_split_orphans_then_prune_keeps_live(tmp_path: Path):
    """A changed stem list builds a new hash dir (orphaning the old); the
    trainer's prune (keep only live train+val dirs) reclaims the orphan and
    spares the live ones. Mirrors run_cached_dual's cleanup."""
    import shutil

    spec, spec_aux = _spec("pe"), _spec("pe_spatial")
    cd, cda = tmp_path / "tokens-main", tmp_path / "tokens-aux"
    hw = lambda s: (2, 3) if int(s[1:]) % 2 == 0 else (2, 4)  # noqa: E731
    pack_root = tmp_path / "packed"

    all_stems = [f"s{i:03d}" for i in range(24)]
    _write_caches(cd, all_stems, 16, hw, "tokens")
    _write_caches(cda, all_stems, 8, hw, "tokens")

    # "Old" split, then a "new" split (one fewer stem) — distinct hashes.
    old = _build(_manifest(all_stems, 6), cd, cda, spec, spec_aux, pack_root)
    new = _build(_manifest(all_stems[:-1], 6), cd, cda, spec, spec_aux, pack_root)
    old_dirs = set(old._pack_dirs.values())
    new_dirs = set(new._pack_dirs.values())
    assert old_dirs != new_dirs, "changed split must produce new shard dirs"

    # Prune: keep only the live (new) dirs.
    live = new_dirs
    for child in pack_root.iterdir():
        if child.is_dir() and child not in live:
            shutil.rmtree(child, ignore_errors=True)

    remaining = {p for p in pack_root.iterdir() if p.is_dir()}
    assert remaining == live
    assert not (old_dirs & remaining), "orphaned old-split dirs must be gone"


def test_drop_sidecars_after_pack(tmp_path: Path):
    """--drop_sidecars_after_pack deletes the per-stem sidecars once both
    splits' shards exist; packed reads still work afterward (served from the
    shards). And the guard refuses to delete if a shard is missing."""
    import logging

    from scripts.anima_tagger.train_cached import _drop_packed_sidecars

    spec, spec_aux = _spec("pe"), _spec("pe_spatial")
    cd, cda = tmp_path / "tokens-main", tmp_path / "tokens-aux"
    hw = lambda s: (2, 3) if int(s[1:]) % 2 == 0 else (2, 4)  # noqa: E731
    stems = [f"s{i:03d}" for i in range(24)]
    _write_caches(cd, stems, 16, hw, "tokens")
    _write_caches(cda, stems, 8, hw, "tokens")
    manifest = _manifest(stems, 6)
    pack_root = tmp_path / "packed"

    def build(subset):
        return CachedDualDataset(
            manifest,
            cd,
            "map",
            spec,
            cda,
            "map",
            spec_aux,
            stems_subset=subset,
            pack_root=pack_root,
        )

    train_ds = build(stems[:18])
    val_ds = build(stems[18:])
    logger = logging.getLogger("test")

    # Guard: a missing shard must abort the drop (sidecars survive).
    a_shard = next(iter(train_ds._pack_dirs.values())).glob("*.safetensors").__next__()
    moved = a_shard.with_suffix(".bak")
    a_shard.rename(moved)
    _drop_packed_sidecars((train_ds, val_ds), logger)
    assert all(Path(p).exists() for p in train_ds.paths), "guard must keep sidecars"
    moved.rename(a_shard)  # restore

    # Now the real drop: all shards present.
    _drop_packed_sidecars((train_ds, val_ds), logger)
    assert not any(Path(p).exists() for p in train_ds.paths)
    assert not any(Path(p).exists() for p in val_ds.paths)
    assert not any(Path(p).exists() for p in train_ds.paths_aux)
    # Shards untouched; packed getitem still serves from mmap.
    feat, feat_aux = train_ds[0][0], train_ds[0][1]
    assert feat.dim() == 2 and feat_aux.dim() == 2


def test_reconstruct_from_index_after_sidecar_drop(tmp_path: Path):
    """Once --drop_sidecars_after_pack removes the per-stem caches, a freshly
    constructed dataset must rebuild purely from the packed shards + index and
    serve bit-identical rows (the keep-loop can't run without the sidecars)."""
    import logging

    from scripts.anima_tagger.train_cached import _drop_packed_sidecars

    stems = [f"s{i:03d}" for i in range(24)]
    spec, spec_aux = _spec("pe"), _spec("pe_spatial")
    cd, cda = tmp_path / "tokens-main", tmp_path / "tokens-aux"
    hw = lambda s: (2, 3) if int(s[1:]) % 2 == 0 else (2, 4)  # noqa: E731
    _write_caches(cd, stems, 16, hw, "tokens")
    _write_caches(cda, stems, 8, hw, "tokens")
    manifest = _manifest(stems, 6)
    pack_root = tmp_path / "packed"

    # Pack, capture reference rows, then drop every per-stem sidecar.
    ds = _build(manifest, cd, cda, spec, spec_aux, pack_root)
    ref = [ds[i] for i in range(len(ds))]
    _drop_packed_sidecars((ds,), logging.getLogger("test"))
    assert not any(Path(p).exists() for p in ds.paths)
    assert not any(Path(p).exists() for p in ds.paths_aux)

    # Fresh construct with no sidecars must recover from the index.
    rec = _build(manifest, cd, cda, spec, spec_aux, pack_root)
    assert rec._packed
    assert len(rec) == len(stems)
    for i in range(len(rec)):
        a, b = ref[i], rec[i]
        assert torch.equal(a[0], b[0]), f"main feature mismatch @ {i}"
        assert torch.equal(a[1], b[1]), f"aux feature mismatch @ {i}"
        assert torch.equal(a[2], b[2]) and a[3] == b[3] and a[4] == b[4]
        assert a[5] == b[5], f"bucket mismatch @ {i}"

    # A genuinely missing shard (not just sidecars) must NOT silently recover —
    # the index verifier rejects it, and with sidecars gone there's nothing to
    # fall back to, so construction fails loudly.
    import shutil as _sh

    _sh.rmtree(next(iter(rec._pack_dirs.values())))
    try:
        _build(manifest, cd, cda, spec, spec_aux, pack_root)
    except (RuntimeError, FileNotFoundError):
        pass
    else:
        raise AssertionError("expected failure when shards AND sidecars are gone")


def test_packed_dataloader_workers(tmp_path: Path):
    """Fork + lazy per-worker mmap handle: persistent workers iterate cleanly."""
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
    ds = _build(manifest, cd, cda, spec, spec_aux, tmp_path / "packed")

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
