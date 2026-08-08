from __future__ import annotations

from pathlib import Path

from library.preprocess.runs import load_manifest, resolve_preprocess_run


def _completed_run(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    return resolve_preprocess_run(
        source,
        {"target_res": [1024], "path_pattern": "*"},
        post_image_dataset=tmp_path / "post",
    )


def test_mask_command_uses_selected_run_and_updates_manifest(monkeypatch, tmp_path):
    from scripts.tasks import masking

    run = _completed_run(tmp_path)
    calls: dict[str, list] = {"sam": [], "merge": []}

    monkeypatch.delenv("PREPROCESS_RUN", raising=False)
    monkeypatch.setenv("RUN_SAM_MASK", "1")
    monkeypatch.setenv("RUN_MIT_MASK", "0")
    monkeypatch.setattr(
        masking,
        "_run_sam",
        lambda image_dir, out_dir, extra, config_path: calls["sam"].append(
            (image_dir, out_dir, extra, config_path)
        ),
    )

    def fake_run(command):
        calls["merge"].append(command)
        output_dir = Path(command[command.index("--output-dir") + 1])
        (output_dir / "nested").mkdir(parents=True, exist_ok=True)
        (output_dir / "nested" / "frame_mask.png").write_bytes(b"mask")

    monkeypatch.setattr(masking, "run", fake_run)

    masking.cmd_mask(["--preprocess_run", str(run.manifest_path)])

    assert calls["sam"][0][0] == run.resized_dir
    merge_command = calls["merge"][0]
    assert merge_command[merge_command.index("--output-dir") + 1] == str(run.masks_dir)
    assert "--preprocess_run" not in merge_command
    assert not (tmp_path / "post" / "masks").exists()
    assert load_manifest(run.manifest_path)["artifacts"]["masks"] == 1


def test_mask_clean_only_removes_selected_run(monkeypatch, tmp_path):
    from scripts.tasks import masking

    run = _completed_run(tmp_path)
    (run.masks_dir / "run_mask.png").write_bytes(b"run")
    legacy_mask = tmp_path / "post" / "masks" / "legacy_mask.png"
    legacy_mask.parent.mkdir(parents=True)
    legacy_mask.write_bytes(b"legacy")
    monkeypatch.setenv("PREPROCESS_RUN", str(run.manifest_path))

    masking.cmd_mask_clean([])

    assert not run.masks_dir.exists()
    assert legacy_mask.exists()
    assert load_manifest(run.manifest_path)["artifacts"]["masks"] == 0


def test_mask_without_run_keeps_legacy_path(monkeypatch, tmp_path):
    from scripts.tasks import masking

    legacy_resized = tmp_path / "post" / "resized"
    monkeypatch.delenv("PREPROCESS_RUN", raising=False)
    monkeypatch.delenv("ANIMA_PREPROCESS_RUN", raising=False)
    monkeypatch.setattr(
        masking,
        "_path",
        lambda key, default: str(legacy_resized) if key == "resized_image_dir" else default,
    )

    run_obj, image_dir, mask_dir, cleaned = masking._mask_paths([])

    assert run_obj is None
    assert image_dir == legacy_resized
    assert mask_dir == masking.MASK_OUTPUT_DIR
    assert cleaned == []
