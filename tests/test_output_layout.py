from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

from library.io.output_layout import (
    atomic_replace_dir,
    discover_weights,
    latest_weight,
    layout_from_args,
    resolve_manifest_path,
    safe_output_name,
    write_run_manifest,
)


def test_safe_name_and_no_double_nesting(tmp_path):
    args = SimpleNamespace(output_dir=str(tmp_path / "output" / "ckpt"), output_name="A/B")
    layout = layout_from_args(args)
    assert layout.name == "A_B"
    assert layout.root == tmp_path / "output" / "ckpt" / "A_B"
    args2 = SimpleNamespace(output_dir=str(layout.root), output_name="A_B")
    assert layout_from_args(args2).root == layout.root
    assert safe_output_name("..") == "last"


def test_manifest_final_wins_over_trajectory(tmp_path):
    run = tmp_path / "ckpt" / "artist"
    run.mkdir(parents=True)
    final = run / "artist.safetensors"
    step = run / "artist-step00000001.safetensors"
    final.write_bytes(b"final")
    step.write_bytes(b"step")
    os.utime(step, (final.stat().st_atime + 10, final.stat().st_mtime + 10))
    layout = type("L", (), {"manifest": run / "run_manifest.json", "name": "artist", "root": run, "final": final})()
    write_run_manifest(layout, {"status": "done"})
    assert discover_weights(tmp_path / "ckpt") == [final]
    assert latest_weight(tmp_path / "ckpt") == final


def test_legacy_root_is_still_discoverable(tmp_path):
    root = tmp_path / "ckpt"
    root.mkdir()
    final = root / "legacy.safetensors"
    final.write_bytes(b"x")
    assert discover_weights(root, name="legacy") == [final]


def test_manifest_relative_path_resolution(tmp_path):
    manifest = tmp_path / "run" / "run_manifest.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({"final_weight": "model.safetensors"}), encoding="utf-8")
    assert resolve_manifest_path(manifest, "model.safetensors") == manifest.parent / "model.safetensors"


def test_manifest_and_discovery_support_selected_output_extensions(tmp_path):
    for extension in (".ckpt", ".pt"):
        run = tmp_path / extension[1:] / "artist"
        run.mkdir(parents=True)
        final = run / f"artist{extension}"
        final.write_bytes(b"final")
        layout = type(
            "L",
            (),
            {
                "manifest": run / "run_manifest.json",
                "name": "artist",
                "root": run,
                "final": run / "artist.safetensors",
            },
        )()
        write_run_manifest(layout, {"status": "done"})
        manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["final_weight"] == final.name
        assert discover_weights(tmp_path / extension[1:]) == [final]


def test_atomic_replace_dir_keeps_backup_across_publish_retries(tmp_path, monkeypatch):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    (source / "state.txt").write_text("new", encoding="utf-8")
    (target / "state.txt").write_text("old", encoding="utf-8")

    real_replace = os.replace
    publish_attempts = 0

    def fail_first_publish(src, dst):
        nonlocal publish_attempts
        if Path(src) == source and Path(dst) == target:
            publish_attempts += 1
            if publish_attempts == 1:
                raise PermissionError("transient scanner lock")
            backups = list(tmp_path.glob(".target.old-*"))
            assert len(backups) == 1
            assert (backups[0] / "state.txt").read_text(encoding="utf-8") == "old"
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", fail_first_publish)
    monkeypatch.setattr("library.io.output_layout.time.sleep", lambda _delay: None)

    atomic_replace_dir(source, target, retries=2)

    assert publish_attempts == 2
    assert (target / "state.txt").read_text(encoding="utf-8") == "new"
    assert not source.exists()
    assert not list(tmp_path.glob(".target.old-*"))


def test_atomic_replace_dir_restores_previous_target_after_publish_failure(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    (source / "state.txt").write_text("new", encoding="utf-8")
    (target / "state.txt").write_text("old", encoding="utf-8")

    real_replace = os.replace

    def fail_publish(src, dst):
        if Path(src) == source and Path(dst) == target:
            raise PermissionError("persistent scanner lock")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", fail_publish)
    monkeypatch.setattr("library.io.output_layout.time.sleep", lambda _delay: None)

    try:
        atomic_replace_dir(source, target, retries=2)
    except PermissionError as exc:
        assert str(exc) == "persistent scanner lock"
    else:
        raise AssertionError("publication failure was not propagated")

    assert (target / "state.txt").read_text(encoding="utf-8") == "old"
    assert (source / "state.txt").read_text(encoding="utf-8") == "new"
    assert not list(tmp_path.glob(".target.old-*"))


def test_atomic_replace_dir_retries_backup_restore(tmp_path, monkeypatch):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    (source / "state.txt").write_text("new", encoding="utf-8")
    (target / "state.txt").write_text("old", encoding="utf-8")

    real_replace = os.replace
    restore_attempts = 0

    def fail_publish_and_first_restore(src, dst):
        nonlocal restore_attempts
        src_path = Path(src)
        dst_path = Path(dst)
        if src_path == source and dst_path == target:
            raise PermissionError("persistent publish lock")
        if src_path.name.startswith(".target.old-") and dst_path == target:
            restore_attempts += 1
            if restore_attempts == 1:
                raise PermissionError("transient restore lock")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", fail_publish_and_first_restore)
    monkeypatch.setattr("library.io.output_layout.time.sleep", lambda _delay: None)

    try:
        atomic_replace_dir(source, target, retries=2)
    except PermissionError as exc:
        assert str(exc) == "persistent publish lock"
    else:
        raise AssertionError("publication failure was not propagated")

    assert restore_attempts == 2
    assert (target / "state.txt").read_text(encoding="utf-8") == "old"
    assert not list(tmp_path.glob(".target.old-*"))
