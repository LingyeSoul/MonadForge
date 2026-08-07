from __future__ import annotations


def test_preprocess_te_uses_corrected_resized_captions(monkeypatch):
    from scripts.tasks import preprocess

    calls: list[list[str]] = []

    def fake_path(key: str, default: str) -> str:
        values = {
            "source_image_dir": "image_dataset",
            "resized_image_dir": "post_image_dataset/resized",
            "lora_cache_dir": "post_image_dataset/lora",
        }
        return values.get(key, default)

    monkeypatch.setattr(preprocess, "run", lambda cmd: calls.append(cmd))
    monkeypatch.setattr(preprocess, "_path", fake_path)

    preprocess.cmd_preprocess_te(
        ["--min_pixels", "500000", "--path_pattern", "group/*"],
        caption_config={
            "correct_order": True,
            "insert_no_artist": True,
            "trigger_word": "@dataset-trigger",
            "trigger_at_front": False,
        },
    )

    assert len(calls) == 2
    caption_cmd, te_cmd = calls

    assert caption_cmd[:2] == [
        preprocess.PY,
        "scripts/preprocess/correct_captions.py",
    ]
    assert caption_cmd[caption_cmd.index("--src") + 1] == "image_dataset"
    assert caption_cmd[caption_cmd.index("--dst") + 1] == "post_image_dataset/resized"
    assert caption_cmd[caption_cmd.index("--path_pattern") + 1] == "group/*"
    assert "--caption_insert_no_artist" in caption_cmd
    assert caption_cmd[caption_cmd.index("--caption_trigger_word") + 1] == (
        "@dataset-trigger"
    )

    assert te_cmd[:2] == [
        preprocess.PY,
        "scripts/preprocess/cache_text_embeddings.py",
    ]
    assert te_cmd[te_cmd.index("--dir") + 1] == "post_image_dataset/resized"
    assert "--match_images_from" not in te_cmd
    assert te_cmd[te_cmd.index("--cache_dir") + 1] == "post_image_dataset/lora"
    assert te_cmd[te_cmd.index("--path_pattern") + 1] == "group/*"
    assert [i for i, arg in enumerate(te_cmd) if arg == "--min_pixels"] == [
        te_cmd.index("--min_pixels")
    ]
    assert te_cmd[te_cmd.index("--min_pixels") + 1] == "0"


def test_caption_correction_enabled_when_only_trigger_or_no_artist_set():
    from scripts.tasks.preprocess import _caption_correction_enabled

    base = {
        "correct_order": False,
        "insert_no_artist": False,
        "trigger_word": "",
        "trigger_at_front": False,
    }
    # Nothing set → no rewrite pass.
    assert not _caption_correction_enabled(base)
    # Trigger word alone must still run the pass (the bug: it was being dropped).
    assert _caption_correction_enabled({**base, "trigger_word": "@trig"})
    # Whitespace-only trigger does not count.
    assert not _caption_correction_enabled({**base, "trigger_word": "   "})
    # Insert-no-artist alone also requires the pass.
    assert _caption_correction_enabled({**base, "insert_no_artist": True})


def test_preprocess_te_runs_correction_for_trigger_word_without_correct_order(
    monkeypatch,
):
    from scripts.tasks import preprocess

    calls: list[list[str]] = []

    def fake_path(key: str, default: str) -> str:
        values = {
            "source_image_dir": "image_dataset",
            "resized_image_dir": "post_image_dataset/resized",
            "lora_cache_dir": "post_image_dataset/lora",
        }
        return values.get(key, default)

    monkeypatch.setattr(preprocess, "run", lambda cmd: calls.append(cmd))
    monkeypatch.setattr(preprocess, "_path", fake_path)

    preprocess.cmd_preprocess_te(
        [],
        caption_config={
            "correct_order": False,
            "insert_no_artist": False,
            "trigger_word": "@dataset-trigger",
            "trigger_at_front": False,
        },
    )

    # Correction pass runs (trigger injected) even though correct_order is off,
    # and the TE cache then reads the corrected captions from the resized dir.
    assert len(calls) == 2
    caption_cmd, te_cmd = calls
    assert caption_cmd[1] == "scripts/preprocess/correct_captions.py"
    assert caption_cmd[caption_cmd.index("--caption_trigger_word") + 1] == (
        "@dataset-trigger"
    )
    assert te_cmd[te_cmd.index("--dir") + 1] == "post_image_dataset/resized"


def _stub_overrides(monkeypatch, overrides: dict) -> None:
    """Pin the merged-config read both builders fall back to when env is unset."""
    from scripts.tasks import _common

    monkeypatch.setattr(_common, "_path_overrides", lambda: dict(overrides))


def test_min_pixels_args_env_drop_false_keeps_every_image(monkeypatch):
    """GUI auto-chain unchecks low-res → DROP_LOWRES_IMAGES=0 forces --min_pixels 0,
    overriding a merged config that still says drop=true (the snapshot strips it)."""
    from scripts.tasks.preprocess import _min_pixels_args

    _stub_overrides(monkeypatch, {"drop_lowres_images": True, "min_pixels": 250_000})
    monkeypatch.setenv("DROP_LOWRES_IMAGES", "0")
    monkeypatch.setenv("MIN_PIXELS", "250000")

    assert _min_pixels_args() == ["--min_pixels", "0"]


def test_min_pixels_args_env_drop_true_uses_env_threshold(monkeypatch):
    from scripts.tasks.preprocess import _min_pixels_args

    _stub_overrides(monkeypatch, {})
    monkeypatch.setenv("DROP_LOWRES_IMAGES", "1")
    monkeypatch.setenv("MIN_PIXELS", "250000")

    assert _min_pixels_args() == ["--min_pixels", "250000"]


def test_min_pixels_args_no_env_falls_back_to_config(monkeypatch):
    from scripts.tasks.preprocess import _min_pixels_args

    _stub_overrides(monkeypatch, {"drop_lowres_images": False, "min_pixels": 250_000})
    monkeypatch.delenv("DROP_LOWRES_IMAGES", raising=False)
    monkeypatch.delenv("MIN_PIXELS", raising=False)

    assert _min_pixels_args() == ["--min_pixels", "0"]


def test_target_res_args_env_wins_over_config(monkeypatch):
    from scripts.tasks.preprocess import _target_res_args

    _stub_overrides(monkeypatch, {"target_res": [1024]})
    monkeypatch.setenv("TARGET_RES", "1024 896")

    assert _target_res_args([]) == ["--target_res", "1024", "896"]
    # An explicit CLI --target_res still wins over both env and config.
    assert _target_res_args(["--target_res", "768"]) == []


def test_multires_resize_and_vae_commands_cover_every_tier(monkeypatch):
    from scripts.tasks import preprocess

    calls: list[list[str]] = []
    values = {
        "source_image_dir": "image_dataset",
        "resized_image_dir": "post/resized",
        "multires_image_dir": "post/multires",
        "lora_cache_dir": "post/lora",
        "vae": "models/vae.safetensors",
    }
    _stub_overrides(
        monkeypatch,
        {"target_res": [512, 1024], "multires_per_image": True},
    )
    monkeypatch.delenv("TARGET_RES", raising=False)
    monkeypatch.delenv("MULTIRES_PER_IMAGE", raising=False)
    monkeypatch.setattr(preprocess, "run", lambda command: calls.append(command))
    monkeypatch.setattr(
        preprocess, "_path", lambda key, default: values.get(key, default)
    )

    preprocess.cmd_preprocess_resize([])
    resize_command = calls.pop()
    assert "--multires_per_image" in resize_command
    assert resize_command[resize_command.index("--multires_dir") + 1] == "post/multires"
    target_index = resize_command.index("--target_res")
    assert resize_command[target_index + 1 : target_index + 3] == [
        "512",
        "1024",
    ]

    preprocess.cmd_preprocess_vae([])
    vae_dirs = [
        command[command.index("--dir") + 1].replace("\\", "/") for command in calls
    ]
    assert vae_dirs == [
        "post/multires/512",
        "post/multires/1024",
    ]
    assert all("--multires_per_image" not in command for command in calls)


def test_explicit_multires_dir_is_shared_by_resize_and_vae(monkeypatch):
    from scripts.tasks import preprocess

    calls: list[list[str]] = []
    values = {
        "source_image_dir": "image_dataset",
        "resized_image_dir": "post/resized",
        "lora_cache_dir": "post/lora",
        "vae": "models/vae.safetensors",
    }
    _stub_overrides(
        monkeypatch,
        {"target_res": [512, 1024], "multires_per_image": True},
    )
    monkeypatch.delenv("TARGET_RES", raising=False)
    monkeypatch.delenv("MULTIRES_PER_IMAGE", raising=False)
    monkeypatch.setattr(preprocess, "run", lambda command: calls.append(command))
    monkeypatch.setattr(
        preprocess, "_path", lambda key, default: values.get(key, default)
    )

    extra = [
        "--multires_per_image",
        "--target_res",
        "512",
        "1024",
        "--multires_dir",
        "custom/mr",
    ]
    preprocess.cmd_preprocess_resize(extra)
    resize_command = calls.pop()
    assert resize_command[resize_command.index("--multires_dir") + 1] == "custom/mr"

    preprocess.cmd_preprocess_vae(extra)
    vae_dirs = [
        command[command.index("--dir") + 1].replace("\\", "/") for command in calls
    ]
    assert vae_dirs == ["custom/mr/512", "custom/mr/1024"]
    assert all("--multires_dir" not in command for command in calls)

    calls.clear()
    preprocess.cmd_preprocess_cond_resize(extra)
    cond_resize_command = calls.pop()
    assert (
        cond_resize_command[cond_resize_command.index("--multires_dir") + 1]
        == "custom/mr"
    )

    preprocess.cmd_preprocess_cond_vae(extra)
    cond_vae_dirs = [
        command[command.index("--dir") + 1].replace("\\", "/") for command in calls
    ]
    assert cond_vae_dirs == ["custom/mr/512", "custom/mr/1024"]


def test_pop_resize_only_args_keeps_following_downstream_flag():
    from scripts.tasks.preprocess import _pop_resize_only_args

    assert _pop_resize_only_args(
        [
            "--target_res",
            "512",
            "1024",
            "--multires_per_image",
            "--batch_size",
            "2",
        ]
    ) == ["--batch_size", "2"]


def test_multires_vae_requires_at_least_two_target_tiers(monkeypatch):
    import pytest

    from scripts.tasks import preprocess

    _stub_overrides(
        monkeypatch,
        {"target_res": [1024], "multires_per_image": True},
    )
    monkeypatch.delenv("TARGET_RES", raising=False)
    monkeypatch.delenv("MULTIRES_PER_IMAGE", raising=False)

    with pytest.raises(ValueError, match="at least two target_res tiers"):
        preprocess.cmd_preprocess_vae([])


def test_preprocess_vae_strips_resize_lowres_arguments(monkeypatch):
    from scripts.tasks import preprocess

    calls: list[list[str]] = []
    _stub_overrides(
        monkeypatch,
        {"target_res": [1024], "multires_per_image": False},
    )
    monkeypatch.delenv("TARGET_RES", raising=False)
    monkeypatch.delenv("MULTIRES_PER_IMAGE", raising=False)
    monkeypatch.setattr(preprocess, "run", lambda command: calls.append(command))

    preprocess.cmd_preprocess_vae(
        ["--min_pixels", "250000", "--no_drop_lowres", "--overwrite"]
    )

    assert len(calls) == 1
    command = calls[0]
    assert "--min_pixels" not in command
    assert "--no_drop_lowres" not in command
    assert "--drop_lowres" not in command
    assert "--overwrite" in command


def test_caption_correction_config_parses_trigger_cli_args():
    from scripts.tasks.preprocess import _caption_correction_config

    config, cleaned = _caption_correction_config(
        [
            "--caption_trigger_word",
            "@foo",
            "--caption_trigger_at_front",
            "--other",
        ]
    )

    assert config["trigger_word"] == "@foo"
    assert config["trigger_at_front"] is True
    assert cleaned == ["--other"]


def test_preprocess_run_alias_is_stripped_and_explicit_value_wins(monkeypatch):
    from scripts.tasks.preprocess import _consume_preprocess_run

    monkeypatch.setenv("PREPROCESS_RUN", "/env/manifest.json")
    selected, cleaned = _consume_preprocess_run(
        ["--preprocess-run=/cli/manifest.json", "--overwrite"]
    )

    assert selected == "/cli/manifest.json"
    assert cleaned == ["--overwrite"]


def test_full_preprocess_failure_keeps_manifest_incomplete(tmp_path, monkeypatch):
    import json
    import pytest

    from library.preprocess.runs import resolve_preprocess_run
    from scripts.tasks import preprocess

    source = tmp_path / "source"
    source.mkdir()
    run = resolve_preprocess_run(
        source, {"target_res": [1024]}, post_image_dataset=tmp_path / "post"
    )
    monkeypatch.setattr(
        preprocess,
        "_resolve_stage_run",
        lambda extra, create=False: (run, []),
    )
    monkeypatch.setattr(preprocess, "_repa_pe_encoder", lambda: None)

    def fail_resize(extra):
        raise RuntimeError("resize failed")

    monkeypatch.setattr(preprocess, "cmd_preprocess_resize", fail_resize)

    with pytest.raises(RuntimeError, match="resize failed"):
        preprocess.cmd_preprocess([])

    manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["complete"] is False
    assert "resize failed" in manifest["error"]
