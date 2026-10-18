from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolated_task_environment(monkeypatch, tmp_path):
    """No test may launch a worker or create a run in the checkout."""
    import library.runtime.profiles as profiles
    from scripts.tasks import _common, _legacy_preprocess, preprocess

    monkeypatch.setattr(profiles, "is_v100_runtime", lambda: False)
    monkeypatch.setattr(_common, "_PATH_OVERRIDES_CACHE", {})
    monkeypatch.setattr(_legacy_preprocess, "_ACTIVE_PREPROCESS_RUN", None)
    for key in (
        "PREPROCESS_RUN",
        "ANIMA_PREPROCESS_RUN",
        "ANIMA_DAEMON_JOB_DIR",
        "TARGET_RES",
        "MULTIRES_PER_IMAGE",
        "DROP_LOWRES_IMAGES",
        "MIN_PIXELS",
        "PREPROCESS_STAGES_JSON",
        "PREPROCESS_PATH_PATTERN",
        "CAPTION_AUTOTAG",
        "CAPTION_POSITION_CLAUSES",
        "CAPTION_AUTOTAG_MODE",
        "CAPTION_AUTOTAG_MIN_CONFIDENCE",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(preprocess, "_ensure_danbooru_tags", lambda: None)
    monkeypatch.setattr(_legacy_preprocess, "_curation_decisions_args", lambda: [])
    monkeypatch.setattr(
        preprocess,
        "_curation_decisions_path",
        lambda: tmp_path / "absent-decisions.json",
    )
    monkeypatch.setenv("ANIMA_HOME", str(tmp_path))

    def forbidden(*args, **kwargs):
        raise AssertionError("A task test must stub its worker launch")

    for module in (_common, _legacy_preprocess, preprocess):
        monkeypatch.setattr(module, "run", forbidden)
    original = _legacy_preprocess.resolve_preprocess_run

    def guarded(source, config, **kwargs):
        from pathlib import Path

        if kwargs.get("create", True):
            post = Path(
                kwargs.get("post_image_dataset", "post_image_dataset")
            ).resolve()
            assert post.is_relative_to(tmp_path), (
                "Test attempted to create a real preprocessing run"
            )
        return original(source, config, **kwargs)

    monkeypatch.setattr(_legacy_preprocess, "resolve_preprocess_run", guarded)


def _entry(cmd: list[str]) -> str:
    """The child's entry: the module of a ``-m`` invocation (the ``anime_tools``
    caption stages) or the script path (trainer-side cache scripts)."""
    return cmd[2] if cmd[1] == "-m" else cmd[1]


def _stages_env(monkeypatch, **forms) -> None:
    """The GUI's stage forms (``PREPROCESS_STAGES_JSON``) for a run."""
    import json

    monkeypatch.setenv("PREPROCESS_STAGES_JSON", json.dumps(forms))


def _patch_run(monkeypatch, fn) -> None:
    """Stub both child launches: ``preprocess.run`` (the trainer-side cache
    scripts) and ``_common.run`` (what ``execute_stage`` uses for a curation
    stage from a shell). Also pin the shell path — a suite running as a daemon
    job would otherwise run the stages in-process."""
    from scripts.tasks import _common, preprocess

    monkeypatch.setattr(preprocess, "run", fn)
    monkeypatch.setattr(_common, "run", fn)
    monkeypatch.delenv("ANIMA_DAEMON_JOB_DIR", raising=False)


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

    _patch_run(monkeypatch, lambda cmd: calls.append(cmd))
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

    assert caption_cmd[:3] == [
        preprocess.PY,
        "-m",
        "anime_tools.stages.cli.correct_captions",
    ]
    # ``to_argv`` spells only what differs from the request default, and the
    # trainer's master tree *is* the package's own default, so ``--src`` is
    # elided rather than repeated. ``--dst`` differs and stays.
    from anime_tools import workspace as WS

    assert "--src" not in caption_cmd and WS.SOURCE_ROOT == "image_dataset"
    assert caption_cmd[caption_cmd.index("--dst") + 1] == "post_image_dataset/resized"
    assert caption_cmd[caption_cmd.index("--path_pattern") + 1] == "group/*"
    # The correction is what TE is about to encode, so it always applies.
    assert "--apply" in caption_cmd
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
    # The retired low-res flag is popped, never forwarded to the TE script.
    assert "--min_pixels" not in te_cmd and "--min_pixels" not in caption_cmd


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

    _patch_run(monkeypatch, lambda cmd: calls.append(cmd))
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
    assert _entry(caption_cmd) == "anime_tools.stages.cli.correct_captions"
    assert caption_cmd[caption_cmd.index("--caption_trigger_word") + 1] == (
        "@dataset-trigger"
    )
    assert te_cmd[te_cmd.index("--dir") + 1] == "post_image_dataset/resized"


def _stub_overrides(monkeypatch, overrides: dict) -> None:
    """Pin the merged-config read both builders fall back to when env is unset."""
    from scripts.tasks import _common

    monkeypatch.setattr(_common, "_path_overrides", lambda: dict(overrides))


def test_retired_lowres_args_are_popped(capsys):
    from scripts.tasks.preprocess import _pop_retired_lowres_args

    assert _pop_retired_lowres_args(
        ["--overwrite", "--min_pixels", "0", "--no_drop_lowres", "--drop_lowres"]
    ) == ["--overwrite"]
    assert "handled by resize" in capsys.readouterr().out
    assert _pop_retired_lowres_args(["--overwrite"]) == ["--overwrite"]
    assert capsys.readouterr().out == ""


def test_target_res_args_env_wins_over_config(monkeypatch):
    from scripts.tasks.preprocess import _target_res_args

    _stub_overrides(monkeypatch, {"target_res": [1024]})
    monkeypatch.setenv("TARGET_RES", "1024 896")

    assert _target_res_args([]) == ["--target_res", "1024", "896"]
    # An explicit CLI --target_res still wins over both env and config.
    assert _target_res_args(["--target_res", "768"]) == []


def test_preprocess_overrides_keep_method_specific_training_values_out(monkeypatch):
    from scripts.tasks import _common

    monkeypatch.setattr(_common, "_PATH_OVERRIDES_CACHE", None)
    monkeypatch.setenv("METHOD", "tlora")
    monkeypatch.setenv("PRESET", "low_vram")

    calls = []

    def fake_load(*, preset, method, methods_subdir):
        calls.append((preset, method, methods_subdir))
        if method is None:
            return {"target_res": [1024], "min_pixels": 250000}
        return {
            "target_res": [896, 1024],
            "min_pixels": 1,
            "network_dim": 64,
        }

    monkeypatch.setattr("library.config.io.load_path_overrides", fake_load)
    values = _common._path_overrides()

    assert values["target_res"] == [1024]
    assert values["min_pixels"] == 250000
    assert values["network_dim"] == 64
    assert calls == [
        ("low_vram", "tlora", "methods"),
        ("default", None, "methods"),
    ]


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
    _patch_run(monkeypatch, lambda command: calls.append(command))
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
    _patch_run(monkeypatch, lambda command: calls.append(command))
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
    _patch_run(monkeypatch, lambda command: calls.append(command))

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


def test_caption_position_clauses_is_not_a_correction_flag():
    """It gates its own stage — it must never reach ``correct_captions.py``.

    ``position_clauses`` rides in the caption-correction config dict (same
    family of caption-master rewrites, one GUI box), but enabling it alone must
    not turn the correction pass on or add an argv flag it doesn't know.
    """
    from scripts.tasks.preprocess import (
        _caption_correction_fields,
        _caption_correction_config,
        _caption_correction_enabled,
    )

    config, cleaned = _caption_correction_config(
        ["--caption_position_clauses", "--other"]
    )

    assert config["position_clauses"] is True
    assert cleaned == ["--other"]
    assert _caption_correction_enabled(config) is False
    assert _caption_correction_fields(config) == {}

    off, _ = _caption_correction_config(["--no_caption_position_clauses"])
    assert off["position_clauses"] is False


def test_preprocess_chains_position_clauses_before_the_caption_step(monkeypatch):
    """The stage rewrites the derived caption, so it must land before the mirror.

    Ordering is the whole contract: the caption step re-corrects that same file
    and writes the variant sidecars around it, so a position pass that ran after
    it would be encoded only on the *next* preprocess. It also has to run inline
    (plain ``run``) — this process is itself a daemon job on a serial queue.
    """
    from scripts.tasks import preprocess

    order: list[str] = []

    monkeypatch.setattr(preprocess, "_path", lambda key, default: default)
    monkeypatch.setattr(preprocess, "_repa_pe_encoder", lambda: None)
    monkeypatch.setattr(
        preprocess, "cmd_preprocess_resize", lambda *_a, **_k: order.append("resize")
    )
    monkeypatch.setattr(
        preprocess, "cmd_preprocess_vae", lambda *_a, **_k: order.append("vae")
    )
    monkeypatch.setattr(
        preprocess, "cmd_preprocess_te", lambda *_a, **_k: order.append("te")
    )
    monkeypatch.setattr(preprocess, "cmd_caption_index", lambda *_a, **_k: None)
    monkeypatch.setattr(preprocess.os.path, "exists", lambda _p: True)

    calls: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        order.append("position")

    _patch_run(monkeypatch, fake_run)
    monkeypatch.setenv("CAPTION_POSITION_CLAUSES", "1")

    monkeypatch.setattr(
        preprocess,
        "_resolve_stage_run",
        lambda extra, create=False: (None, list(extra)),
    )
    preprocess.cmd_preprocess([])

    assert order == ["resize", "vae", "position", "te"]
    (cmd,) = calls
    assert cmd[:3] == [preprocess.PY, "-m", "anime_tools.stages.cli.position_captions"]
    assert cmd[-1] == "--apply"


def test_preprocess_skips_position_clauses_when_unset(monkeypatch):
    from scripts.tasks import preprocess

    monkeypatch.setattr(preprocess, "_path", lambda key, default: default)
    monkeypatch.setattr(preprocess, "_repa_pe_encoder", lambda: None)
    for name in ("cmd_preprocess_resize", "cmd_preprocess_vae", "cmd_preprocess_te"):
        monkeypatch.setattr(preprocess, name, lambda *_a, **_k: None)
    monkeypatch.setattr(preprocess, "cmd_caption_index", lambda *_a, **_k: None)
    monkeypatch.setattr(preprocess.os.path, "exists", lambda _p: True)

    calls: list[list[str]] = []
    _patch_run(monkeypatch, lambda cmd, **_k: calls.append(cmd))
    monkeypatch.delenv("CAPTION_POSITION_CLAUSES", raising=False)
    monkeypatch.delenv("CAPTION_AUTOTAG", raising=False)

    monkeypatch.setattr(
        preprocess,
        "_resolve_stage_run",
        lambda extra, create=False: (None, list(extra)),
    )
    preprocess.cmd_preprocess([])

    assert calls == []


def test_preprocess_captions_runs_the_master_stages_when_configured(monkeypatch):
    """`preprocess-captions` mirrors the master, so it owns the stages too.

    Run on its own (not through the full chain) it must still honour
    ``caption_position_clauses`` / ``caption_autotag`` — otherwise the mirror,
    and every TE cache encoded from it, silently predates the rewrite.
    """
    from scripts.tasks import preprocess

    monkeypatch.setattr(preprocess, "_path", lambda key, default: default)
    monkeypatch.setattr(preprocess, "_ensure_danbooru_tags", lambda: None)
    monkeypatch.setattr(preprocess, "_variant_settings", lambda: ("4", "0.1", "0.0"))

    calls: list[list[str]] = []
    _patch_run(monkeypatch, lambda cmd, **_k: calls.append(cmd))
    monkeypatch.setenv("CAPTION_AUTOTAG", "1")
    monkeypatch.setenv("CAPTION_POSITION_CLAUSES", "1")

    preprocess.cmd_preprocess_captions([])

    scripts = [_entry(cmd) for cmd in calls]
    assert scripts == [
        "anime_tools.stages.cli.autotag_captions",
        "anime_tools.stages.cli.position_captions",
        "anime_tools.stages.cli.correct_captions",
    ]
    # In-pipeline stages write for real — a dry run here would leave the mirror
    # encoding the un-rewritten caption.
    assert calls[0][-1] == "--apply"
    assert calls[1][-1] == "--apply"


def test_master_stages_inherit_an_explicit_path_pattern(monkeypatch):
    """A scoped preprocess must not rewrite captions dataset-wide.

    The caption-MASTER stages are driven from the caption-config dict alone —
    the caller's ``extra`` never reaches them — so the resolved subset scope
    has to ride in that dict. Without it, ``make preprocess-captions
    ARGS="--path_pattern artistA/*"`` would cache the requested slice while
    autotag ``merge``/``overwrite`` rewrote every caption in the master.
    """
    from scripts.tasks import preprocess

    monkeypatch.setattr(preprocess, "_path", lambda key, default: default)
    monkeypatch.setattr(preprocess, "_ensure_danbooru_tags", lambda: None)
    monkeypatch.setattr(preprocess, "_variant_settings", lambda: ("0", "0.0", "0.0"))

    calls: list[list[str]] = []
    _patch_run(monkeypatch, lambda cmd, **_k: calls.append(cmd))
    monkeypatch.setenv("CAPTION_AUTOTAG", "1")
    monkeypatch.setenv("CAPTION_AUTOTAG_MODE", "merge")
    monkeypatch.setenv("CAPTION_POSITION_CLAUSES", "1")
    monkeypatch.delenv("PREPROCESS_PATH_PATTERN", raising=False)

    preprocess.cmd_preprocess_captions(["--path_pattern", "artistA/*"])

    assert [_entry(cmd) for cmd in calls] == [
        "anime_tools.stages.cli.autotag_captions",
        "anime_tools.stages.cli.position_captions",
        # The mirror rides along whenever position clauses are on — they land in
        # `resized/`, so every other caption has to be mirrored there too.
        "anime_tools.stages.cli.correct_captions",
    ]
    for cmd in calls:
        # Present, and emitted exactly once — the argv builder resolves the
        # scope and drops it from the tail rather than splatting it twice.
        assert cmd.count("--path_pattern") == 1
        assert cmd[cmd.index("--path_pattern") + 1] == "artistA/*"


def test_standalone_caption_target_does_not_duplicate_the_path_pattern():
    """`make caption-position ARGS="--path_pattern x"` passes it through once."""
    from scripts.tasks import preprocess

    argv = preprocess._caption_position_request(
        ["--path_pattern", "artistA/*", "--apply"]
    ).to_argv()
    assert argv.count("--path_pattern") == 1
    assert argv[argv.index("--path_pattern") + 1] == "artistA/*"
    assert argv[-1] == "--apply"


def test_preprocess_captions_runs_the_stages_even_with_correction_off(monkeypatch):
    """The early "correction disabled" return must not swallow the stages.

    With position clauses on, the mirror has to run as well even though nothing
    is being corrected or shuffled: the clause rewrite lands in ``resized/`` and
    the TE step therefore reads that tree, so every image the rewrite did *not*
    touch still needs its master caption mirrored there.
    """
    from scripts.tasks import preprocess

    monkeypatch.setattr(preprocess, "_path", lambda key, default: default)
    monkeypatch.setattr(preprocess, "_ensure_danbooru_tags", lambda: None)
    monkeypatch.setattr(preprocess, "_variant_settings", lambda: ("0", "0.0", "0.0"))

    calls: list[list[str]] = []
    _patch_run(monkeypatch, lambda cmd, **_k: calls.append(cmd))
    monkeypatch.delenv("CAPTION_AUTOTAG", raising=False)
    monkeypatch.setenv("CAPTION_POSITION_CLAUSES", "1")

    preprocess.cmd_preprocess_captions([])

    assert [_entry(cmd) for cmd in calls] == [
        "anime_tools.stages.cli.position_captions",
        "anime_tools.stages.cli.correct_captions",
    ]
    # Nothing to correct and no variants — the mirror runs in passthrough.
    assert "--no_correct" in calls[1]


def test_preprocess_te_reads_resized_when_position_clauses_are_on(monkeypatch):
    """Clauses live in ``resized/``, so TE must encode that tree, not the master.

    No correction + no variants would normally encode the caption master
    directly with a match filter — which would silently train the pre-clause
    caption, since the position pass never writes the master.
    """
    from scripts.tasks import preprocess

    monkeypatch.setattr(preprocess, "_path", lambda key, default: default)
    monkeypatch.setattr(preprocess, "_ensure_danbooru_tags", lambda: None)
    monkeypatch.setattr(preprocess, "_variant_settings", lambda: ("0", "0.0", "0.0"))

    calls: list[list[str]] = []
    _patch_run(monkeypatch, lambda cmd, **_k: calls.append(cmd))
    monkeypatch.delenv("CAPTION_AUTOTAG", raising=False)
    monkeypatch.setenv("CAPTION_POSITION_CLAUSES", "1")

    preprocess.cmd_preprocess_te([])

    assert [_entry(cmd) for cmd in calls] == [
        "anime_tools.stages.cli.position_captions",
        "anime_tools.stages.cli.correct_captions",
        "scripts/preprocess/cache_text_embeddings.py",
    ]
    te_cmd = calls[-1]
    assert te_cmd[te_cmd.index("--dir") + 1] == "post_image_dataset/resized"
    assert "--match_images_from" not in te_cmd


def test_preprocess_chain_runs_each_master_stage_once(monkeypatch):
    """`preprocess` runs the stages early; the caption/TE steps must not repeat.

    Both passes are idempotent on a rewritten caption, but each re-pays a SAM3 /
    tagger load over the whole tree. The shared caption-config dict is what
    carries "already ran" down the chain.
    """
    from scripts.tasks import preprocess

    monkeypatch.setattr(preprocess, "_path", lambda key, default: default)
    monkeypatch.setattr(preprocess, "_repa_pe_encoder", lambda: None)
    monkeypatch.setattr(preprocess, "_ensure_danbooru_tags", lambda: None)
    monkeypatch.setattr(preprocess, "_variant_settings", lambda: ("4", "0.1", "0.0"))
    monkeypatch.setattr(preprocess, "cmd_preprocess_resize", lambda *_a, **_k: None)
    monkeypatch.setattr(preprocess, "cmd_preprocess_vae", lambda *_a, **_k: None)
    monkeypatch.setattr(preprocess, "cmd_caption_index", lambda *_a, **_k: None)
    monkeypatch.setattr(preprocess.os.path, "exists", lambda _p: True)

    calls: list[list[str]] = []
    _patch_run(monkeypatch, lambda cmd, **_k: calls.append(cmd))
    monkeypatch.setenv("CAPTION_AUTOTAG", "1")
    monkeypatch.setenv("CAPTION_POSITION_CLAUSES", "1")

    monkeypatch.setattr(
        preprocess,
        "_resolve_stage_run",
        lambda extra, create=False: (None, list(extra)),
    )
    preprocess.cmd_preprocess([])

    scripts = [_entry(cmd) for cmd in calls]
    assert scripts.count("anime_tools.stages.cli.autotag_captions") == 1
    assert scripts.count("anime_tools.stages.cli.position_captions") == 1
    # Order is unchanged: the caption rewrites, then the mirror, then the encode.
    assert scripts == [
        "anime_tools.stages.cli.autotag_captions",
        "anime_tools.stages.cli.position_captions",
        "anime_tools.stages.cli.correct_captions",
        "scripts/preprocess/cache_text_embeddings.py",
    ]


def test_caption_autotag_is_not_a_correction_flag():
    """Like ``position_clauses``: own stage, must never reach correct_captions.py."""
    from scripts.tasks.preprocess import (
        _caption_correction_fields,
        _caption_correction_config,
        _caption_correction_enabled,
    )

    config, cleaned = _caption_correction_config(
        ["--caption_autotag", "--caption_autotag_mode", "merge", "--other"]
    )

    assert config["autotag"] is True
    assert config["autotag_mode"] == "merge"
    assert cleaned == ["--other"]
    assert _caption_correction_enabled(config) is False
    assert _caption_correction_fields(config) == {}

    off, _ = _caption_correction_config(["--no_caption_autotag"])
    assert off["autotag"] is False
    # Mode defaults to the non-destructive one when nothing selects it.
    assert off["autotag_mode"] == "missing"


def test_caption_autotag_rejects_an_unknown_mode():
    """Fail at argv-parse time, not minutes into a GPU job."""
    import pytest

    from scripts.tasks.preprocess import _caption_correction_config

    with pytest.raises(SystemExit):
        _caption_correction_config(["--caption_autotag_mode", "clobber"])


def test_caption_autotag_args_always_apply():
    """Typed stage requests preserve the apply and threshold contract."""
    from scripts.tasks.preprocess import _autotag_request

    request = _autotag_request({"autotag_mode": "missing"})
    assert request.mode == "missing" and request.apply is True
    assert "--min_confidence" not in request.to_argv()
    request = _autotag_request(
        {"autotag_mode": "merge", "autotag_min_confidence": 0.35}
    )
    assert request.mode == "merge" and request.apply is True
    assert request.min_confidence == 0.35
    argv = request.to_argv()
    assert argv[argv.index("--min_confidence") + 1] == "0.35"


def test_preprocess_chains_autotag_first(monkeypatch):
    """Autotag *creates* the captions every later stage reads, so it goes first.

    Ordering is the contract: position clauses append to the master, correction
    and TE read it. An autotag that ran after any of them would leave this run
    encoding the un-tagged caption.
    """
    from scripts.tasks import preprocess

    order: list[str] = []

    monkeypatch.setattr(preprocess, "_path", lambda key, default: default)
    monkeypatch.setattr(preprocess, "_repa_pe_encoder", lambda: None)
    monkeypatch.setattr(
        preprocess, "cmd_preprocess_resize", lambda *_a, **_k: order.append("resize")
    )
    monkeypatch.setattr(
        preprocess, "cmd_preprocess_vae", lambda *_a, **_k: order.append("vae")
    )
    monkeypatch.setattr(
        preprocess, "cmd_preprocess_te", lambda *_a, **_k: order.append("te")
    )
    monkeypatch.setattr(preprocess, "cmd_caption_index", lambda *_a, **_k: None)
    monkeypatch.setattr(preprocess.os.path, "exists", lambda _p: True)

    calls: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        order.append("autotag" if "autotag_captions" in _entry(cmd) else "position")

    _patch_run(monkeypatch, fake_run)
    monkeypatch.setenv("CAPTION_AUTOTAG", "1")
    monkeypatch.setenv("CAPTION_AUTOTAG_MODE", "merge")
    monkeypatch.setenv("CAPTION_POSITION_CLAUSES", "1")

    monkeypatch.setattr(
        preprocess,
        "_resolve_stage_run",
        lambda extra, create=False: (None, list(extra)),
    )
    preprocess.cmd_preprocess([])

    assert order == ["resize", "autotag", "vae", "position", "te"]
    autotag_cmd = calls[0]
    assert autotag_cmd[:3] == [
        preprocess.PY,
        "-m",
        "anime_tools.stages.cli.autotag_captions",
    ]
    assert "--apply" in autotag_cmd
    assert autotag_cmd[autotag_cmd.index("--mode") + 1] == "merge"


def test_preprocess_autotag_blank_env_confidence_is_zero(monkeypatch):
    """The GUI writes ``""`` for an empty field — that must not raise."""
    from scripts.tasks.preprocess import _caption_correction_config

    monkeypatch.setenv("CAPTION_AUTOTAG", "1")
    monkeypatch.setenv("CAPTION_AUTOTAG_MIN_CONFIDENCE", "")

    config, _ = _caption_correction_config([])
    assert config["autotag_min_confidence"] == 0.0


def test_preprocess_run_alias_is_stripped_and_explicit_value_wins(monkeypatch):
    from scripts.tasks.preprocess import _consume_preprocess_run

    monkeypatch.setenv("PREPROCESS_RUN", "/env/manifest.json")
    selected, cleaned = _consume_preprocess_run(
        ["--preprocess-run=/cli/manifest.json", "--overwrite"]
    )

    assert selected == "/cli/manifest.json"
    assert cleaned == ["--overwrite"]


def test_execution_controls_do_not_change_preprocess_fingerprint(monkeypatch):
    from library.preprocess.runs import config_fingerprint
    from scripts.tasks import preprocess

    _stub_overrides(monkeypatch, {"target_res": [1024]})
    baseline = preprocess._preprocess_config(["--resize_crop_anchor", "center"])
    controlled = preprocess._preprocess_config(
        [
            "--overwrite",
            "--workers=8",
            "--batch_size",
            "4",
            "--device",
            "cuda",
            "--resize_crop_anchor",
            "center",
        ]
    )

    assert controlled["preprocess_options"] == ["--resize_crop_anchor", "center"]
    assert config_fingerprint(controlled) == config_fingerprint(baseline)


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

    def fail_resize(extra, **kwargs):
        raise RuntimeError("resize failed")

    monkeypatch.setattr(preprocess, "cmd_preprocess_resize", fail_resize)

    with pytest.raises(RuntimeError, match="resize failed"):
        preprocess.cmd_preprocess([])

    manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["complete"] is False
    assert "resize failed" in manifest["error"]


def _make_preprocess_run(tmp_path, source_name="source", config=None):
    from library.preprocess.runs import resolve_preprocess_run

    source = tmp_path / source_name
    source.mkdir(exist_ok=True)
    return resolve_preprocess_run(
        source,
        config or {"target_res": [1024]},
        post_image_dataset=tmp_path / "post",
    )


def _select_stage_source(monkeypatch, source) -> None:
    from scripts.tasks import preprocess

    monkeypatch.setattr(
        preprocess,
        "_path",
        lambda key, default: str(source) if key == "source_image_dir" else default,
    )


def test_stage_run_adopts_explicit_manifest_despite_env_drift(tmp_path, monkeypatch):
    from scripts.tasks import preprocess

    run = _make_preprocess_run(
        tmp_path,
        config={"target_res": [1024], "caption_shuffle_variants": "4"},
    )
    _select_stage_source(monkeypatch, run.source_dir)
    monkeypatch.setenv("PREPROCESS_RUN", str(run.manifest_path))
    # The standalone stage task runs with different ambient env than the full
    # preprocess that created the run (the WebUI forwards CAPTION_*/TARGET_RES
    # only to `preprocess`/`preprocess-te`). The explicit selection must win.
    monkeypatch.setenv("CAPTION_SHUFFLE_VARIANTS", "2")
    monkeypatch.setenv("TARGET_RES", "896")

    resolved, cleaned = preprocess._resolve_stage_run([])

    assert resolved.root == run.root
    assert cleaned == []


def test_stage_run_rejects_manifest_from_other_source(tmp_path, monkeypatch):
    import pytest

    from scripts.tasks import preprocess

    run = _make_preprocess_run(tmp_path, source_name="source-a")
    other = tmp_path / "source-b"
    other.mkdir()
    _select_stage_source(monkeypatch, other)
    monkeypatch.setenv("PREPROCESS_RUN", str(run.manifest_path))

    with pytest.raises(SystemExit, match="different source dataset"):
        preprocess._resolve_stage_run([])


def test_stage_run_missing_manifest_fails_with_clear_error(tmp_path, monkeypatch):
    import pytest

    from scripts.tasks import preprocess

    source = tmp_path / "source"
    source.mkdir()
    _select_stage_source(monkeypatch, source)
    monkeypatch.setenv("PREPROCESS_RUN", str(tmp_path / "gone" / "manifest.json"))

    with pytest.raises(SystemExit, match="not found"):
        preprocess._resolve_stage_run([])


def test_stage_run_adopts_failed_run_for_resume(tmp_path, monkeypatch):
    import pytest

    from library.preprocess.runs import PreprocessRunError, run_from_manifest
    from scripts.tasks import preprocess

    run = _make_preprocess_run(tmp_path)
    run.write_manifest(status="failed", error="RuntimeError: resize failed")
    _select_stage_source(monkeypatch, run.source_dir)
    monkeypatch.setenv("PREPROCESS_RUN", str(run.manifest_path))

    resolved, _ = preprocess._resolve_stage_run([])
    assert resolved.root == run.root

    # Strict consumers of finished runs still reject an incomplete manifest.
    with pytest.raises(PreprocessRunError):
        run_from_manifest(run.manifest_path)


def test_preprocess_pe_routes_centroid_into_selected_run(tmp_path, monkeypatch):
    from pathlib import Path

    from scripts.tasks import preprocess

    run = _make_preprocess_run(tmp_path)
    calls: list[list[str]] = []
    _patch_run(monkeypatch, lambda cmd: calls.append(cmd))
    monkeypatch.setattr(
        preprocess,
        "_resolve_stage_run",
        lambda extra, create=False: (run, list(extra)),
    )

    preprocess.cmd_preprocess_pe([])

    cmd = calls[0]
    out = Path(cmd[cmd.index("--centroid_out") + 1])
    assert out == run.lora_cache_dir / "anima_pe_centroid_pe.safetensors"


def test_preprocess_pe_without_run_keeps_legacy_centroid(monkeypatch):
    from scripts.tasks import preprocess

    calls: list[list[str]] = []
    _patch_run(monkeypatch, lambda cmd: calls.append(cmd))
    monkeypatch.setattr(
        preprocess,
        "_resolve_stage_run",
        lambda extra, create=False: (None, list(extra)),
    )

    preprocess.cmd_preprocess_pe([])

    assert "--centroid_out" not in calls[0]


def test_preprocess_pe_respects_explicit_centroid_out(tmp_path, monkeypatch):
    from scripts.tasks import preprocess

    run = _make_preprocess_run(tmp_path)
    calls: list[list[str]] = []
    _patch_run(monkeypatch, lambda cmd: calls.append(cmd))
    monkeypatch.setattr(
        preprocess,
        "_resolve_stage_run",
        lambda extra, create=False: (run, list(extra)),
    )

    preprocess.cmd_preprocess_pe(["--centroid_out", "custom/centroid.safetensors"])

    cmd = calls[0]
    assert cmd.count("--centroid_out") == 1
    assert cmd[cmd.index("--centroid_out") + 1] == "custom/centroid.safetensors"


def test_stage_wrapper_restores_legacy_hooks_and_scopes_reports(tmp_path, monkeypatch):
    from scripts.tasks import _common, _legacy_preprocess, preprocess

    run = _make_preprocess_run(tmp_path)
    previous_run = _legacy_preprocess.run
    previous_path = _legacy_preprocess._path
    previous_config = _legacy_preprocess._preprocess_config
    previous_overrides = _common._PATH_OVERRIDES_CACHE
    monkeypatch.setattr(
        preprocess, "_resolve_stage_run", lambda extra, create=False: (run, list(extra))
    )
    calls = []
    _patch_run(monkeypatch, lambda command: calls.append(command))
    monkeypatch.setattr(_common, "run", lambda command, **kwargs: calls.append(command))
    monkeypatch.setattr(preprocess, "_variant_settings", lambda: ("1", "0", "0"))
    preprocess.cmd_preprocess_captions([], caption_config={"correct_order": False})
    assert str(run.resized_dir) in calls[0]
    assert str(run.root / "reports" / "correct") in calls[0]
    assert _legacy_preprocess.run is previous_run
    assert _legacy_preprocess._path is previous_path
    assert _legacy_preprocess._preprocess_config is previous_config
    assert _common._PATH_OVERRIDES_CACHE is previous_overrides


def test_selected_run_caption_index_is_not_shared(tmp_path, monkeypatch):
    from scripts.tasks import preprocess

    run = _make_preprocess_run(tmp_path)
    monkeypatch.setattr(
        preprocess, "_resolve_stage_run", lambda extra, create=False: (run, list(extra))
    )
    calls = []
    _patch_run(monkeypatch, lambda command: calls.append(command))
    preprocess.cmd_caption_index([])
    assert calls[0][calls[0].index("--out") + 1] == str(run.caption_index_path)


def test_explicit_lowres_filter_is_preserved_for_resize_only(monkeypatch):
    from scripts.tasks import preprocess

    calls = []
    _patch_run(monkeypatch, lambda command: calls.append(command))
    preprocess.cmd_preprocess_resize(["--min_pixels", "250000", "--overwrite"])
    assert calls[0][1] == "scripts/preprocess/resize_images.py"
    assert calls[0][calls[0].index("--min_pixels") + 1] == "250000"
    preprocess.cmd_preprocess_vae(["--min_pixels", "250000", "--target_res", "1024"])
    assert "--min_pixels" not in calls[1] and "--target_res" not in calls[1]


def test_autotag_environment_values_reach_typed_request_and_cli_wins(monkeypatch):
    from scripts.tasks import preprocess

    monkeypatch.setenv("CAPTION_AUTOTAG_MODE", "merge")
    monkeypatch.setenv("CAPTION_AUTOTAG_MIN_CONFIDENCE", "0.35")
    config, _ = preprocess._caption_correction_config([])
    request = preprocess._autotag_request(config)
    assert request.mode == "merge" and request.min_confidence == 0.35

    config, _ = preprocess._caption_correction_config(
        [
            "--caption_autotag_mode",
            "missing",
            "--caption_autotag_min_confidence",
            "0.5",
        ]
    )
    request = preprocess._autotag_request(config)
    assert request.mode == "missing" and request.min_confidence == 0.5
