"""Focused filesystem tests for isolated preprocessing runs."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from library.preprocess.runs import (
    PreprocessRunError,
    atomic_write_manifest,
    canonical_preprocess_config,
    config_fingerprint,
    load_manifest,
    migrate_legacy_cache,
    resolve_preprocess_run,
    run_from_manifest,
    safe_source_name,
    source_path_hash,
    validate_legacy_cache,
    validate_manifest,
)


def _config(**overrides):
    value = {
        "target_res": [1024, 896],
        "multires_per_image": False,
        "crop_anchor": "center",
        "freefit_max_ratio": 4.0,
        "drop_lowres_images": True,
        "min_pixels": 250000,
        "caption_shuffle_variants": 4,
        "caption_tag_dropout_rate": 0.1,
        "cache_schema_version": 1,
    }
    value.update(overrides)
    return value


def test_config_fingerprint_is_stable_and_covers_preprocess_axes():
    config = _config(source_image_dir="one", optimizer_type="AdamW")
    reordered = dict(reversed(list(config.items())))
    assert config_fingerprint(config) == config_fingerprint(reordered)
    canonical = canonical_preprocess_config(config)
    assert canonical["target_res"] == [896, 1024]
    assert canonical["filter"]["min_pixels"] == 250000
    assert canonical["caption"]["caption_shuffle_variants"] == 4

    for field, value in (
        ("target_res", [1536]),
        ("multires_per_image", True),
        ("crop_anchor", "top"),
        ("min_pixels", 1),
        ("caption_tag_dropout_rate", 0.2),
        ("cache_schema_version", 2),
    ):
        changed = _config(**{field: value})
        assert config_fingerprint(changed) != config_fingerprint(config), field

    # Source identity is a path concern, not a config concern.  Training-only
    # knobs likewise must not split a preprocess cache run.
    assert config_fingerprint(_config(source_image_dir="two")) == config_fingerprint(config)
    assert config_fingerprint(_config(optimizer_type="SGD")) == config_fingerprint(config)


def test_resolve_run_layout_manifest_and_idempotent_reuse(tmp_path: Path, monkeypatch):
    source = tmp_path / "nested source" / "charA"
    source.mkdir(parents=True)
    run = resolve_preprocess_run(source, _config(), post_image_dataset=tmp_path / "post")

    assert run.root == tmp_path / "post" / "runs" / f"charA-{source_path_hash(source)}" / run.config_hash
    assert run.manifest_path.is_file()
    for name in ("resized", "lora", "masks", "multires", "conditioning"):
        assert (run.root / name).is_dir()
    assert run.conditioning_data_dir.is_dir()
    assert run.conditioning_resized_dir.is_dir()

    manifest_before = run.manifest_path.read_bytes()
    # A second resolve must reuse the complete manifest, without publishing a
    # new temporary file or changing its bytes.
    def fail_write(*args, **kwargs):
        raise AssertionError("idempotent resolve rewrote manifest")

    monkeypatch.setattr("library.preprocess.runs.atomic_write_manifest", fail_write)
    reused = resolve_preprocess_run(source, _config(), post_image_dataset=tmp_path / "post")
    assert reused.root == run.root
    assert run.manifest_path.read_bytes() == manifest_before


def test_same_source_different_configs_coexist_and_nested_sources_do_not_collide(tmp_path: Path):
    post = tmp_path / "post_image_dataset"
    source_a = tmp_path / "a" / "images"
    source_b = tmp_path / "b" / "images"
    source_a.mkdir(parents=True)
    source_b.mkdir(parents=True)

    first = resolve_preprocess_run(source_a, _config(), post_image_dataset=post)
    second = resolve_preprocess_run(
        source_a, _config(target_res=[1536]), post_image_dataset=post
    )
    other_source = resolve_preprocess_run(source_b, _config(), post_image_dataset=post)

    assert first.root != second.root
    assert first.source_group != other_source.source_group
    assert first.root.parent == second.root.parent
    assert first.root.parent.parent == other_source.root.parent.parent


def test_manifest_is_atomic_and_incomplete_manifest_is_rejected(tmp_path: Path):
    path = tmp_path / "run" / "manifest.json"
    atomic_write_manifest(path, {"kind": "preprocess_run", "complete": True, "value": 3})
    assert json.loads(path.read_text(encoding="utf-8"))["value"] == 3
    assert not list(path.parent.glob("*.tmp"))
    assert load_manifest(path)["complete"] is True
    assert validate_manifest(path)

    atomic_write_manifest(path, {"kind": "preprocess_run", "status": "running", "complete": False})
    with pytest.raises(PreprocessRunError, match="Incomplete"):
        load_manifest(path)
    assert not validate_manifest(path)


def test_run_from_manifest_round_trip(tmp_path: Path):
    source = tmp_path / "images"
    source.mkdir()
    run = resolve_preprocess_run(source, _config(), post_image_dataset=tmp_path / "post")
    loaded = run_from_manifest(run.manifest_path)
    assert loaded.root == run.root
    assert loaded.run_id == run.run_id
    assert loaded.config_hash == run.config_hash


def test_legacy_validation_reports_missing_and_mismatched_metadata(tmp_path: Path):
    legacy = tmp_path / "post"
    (legacy / "resized" / "nested").mkdir(parents=True)
    (legacy / "resized" / "nested" / "a.png").write_bytes(b"png")

    compatible = validate_legacy_cache(legacy, require_metadata=False)
    assert compatible.valid and not compatible.metadata_present
    strict = validate_legacy_cache(legacy, require_metadata=True)
    assert not strict.valid

    source = tmp_path / "source"
    source.mkdir()
    run = resolve_preprocess_run(source, _config(), post_image_dataset=tmp_path / "new")
    atomic_write_manifest(
        legacy / "manifest.json",
        run.manifest_payload(source_hash="wrong", config_hash=run.config_hash),
    )
    mismatch = validate_legacy_cache(legacy, source_dir=source, config=_config())
    assert not mismatch.valid
    assert "source" in (mismatch.reason or "")


def test_legacy_migration_preserves_nested_layout_and_prefers_hardlinks(tmp_path: Path):
    legacy = tmp_path / "post"
    source = tmp_path / "source"
    source.mkdir()
    payload = b"latent-data"
    old_file = legacy / "lora" / "charA" / "cover_anima.npz"
    old_file.parent.mkdir(parents=True)
    old_file.write_bytes(payload)
    old_mask = legacy / "masks" / "charA" / "cover.png"
    old_mask.parent.mkdir(parents=True)
    old_mask.write_bytes(b"mask")

    run = resolve_preprocess_run(source, _config(), post_image_dataset=tmp_path / "new")
    result = migrate_legacy_cache(legacy, run, kinds=("lora", "masks"))
    assert result.ok
    assert len(result.migrated) == 2
    new_file = run.lora_dir / "charA" / "cover_anima.npz"
    assert new_file.read_bytes() == payload
    assert old_file.exists() and old_mask.exists()
    if os.name != "nt":
        assert os.stat(old_file).st_ino == os.stat(new_file).st_ino

    again = migrate_legacy_cache(legacy, run, kinds=("lora", "masks"))
    assert again.ok
    assert len(again.skipped) == 2


def test_legacy_migration_falls_back_to_atomic_copy(tmp_path: Path, monkeypatch):
    legacy = tmp_path / "post"
    source = tmp_path / "source"
    source.mkdir()
    old_file = legacy / "resized" / "x.png"
    old_file.parent.mkdir(parents=True)
    old_file.write_bytes(b"image")
    run = resolve_preprocess_run(source, _config(), post_image_dataset=tmp_path / "new")

    def no_hardlink(*args, **kwargs):
        raise OSError("cross-device link")

    monkeypatch.setattr("library.preprocess.runs.os.link", no_hardlink)
    result = migrate_legacy_cache(legacy, run, kinds=("resized",))
    new_file = run.resized_dir / "x.png"
    assert result.ok and new_file.read_bytes() == b"image"
    assert old_file.exists()
    assert not list(new_file.parent.glob("*.tmp"))


def test_migration_failure_keeps_legacy_source_and_reports_reason(tmp_path: Path, monkeypatch):
    legacy = tmp_path / "post"
    source = tmp_path / "source"
    source.mkdir()
    old_file = legacy / "lora" / "x.npz"
    old_file.parent.mkdir(parents=True)
    old_file.write_bytes(b"x")
    run = resolve_preprocess_run(source, _config(), post_image_dataset=tmp_path / "new")

    def fail(*args, **kwargs):
        raise OSError("locked by another process")

    monkeypatch.setattr("library.preprocess.runs._copy_or_link_atomic", fail)
    result = migrate_legacy_cache(legacy, run, kinds=("lora",))
    assert not result.ok
    assert result.failed and "locked" in result.failed[0][1]
    assert old_file.exists()


def test_safe_source_name_handles_windows_reserved_and_invalid_characters():
    assert safe_source_name("CON.txt") == "_CON.txt"
    assert "/" not in safe_source_name("a/b\\c")
    assert safe_source_name("   ") == "dataset"
