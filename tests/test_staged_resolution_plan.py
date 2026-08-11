from __future__ import annotations

from pathlib import Path

import pytest
import toml
from fastapi.testclient import TestClient
from PIL import Image

from library.training import staged_resolution_plan as plans


def _write_minimal_configs(root: Path) -> None:
    configs = root / "configs"
    (configs / "gui-methods").mkdir(parents=True)
    (configs / "custom" / "variants").mkdir(parents=True)
    (configs / "base.toml").write_text(
        """
pretrained_model_name_or_path = "models/dit.safetensors"
qwen3 = "models/qwen3.safetensors"
vae = "models/vae.safetensors"
output_dir = "output/ckpt"

[[datasets]]
batch_size = 1
validation_split_num = 0
[[datasets.subsets]]
image_dir = "post_image_dataset/resized"
cache_dir = "post_image_dataset/lora"
num_repeats = 1
recursive = true

[general]
caption_extension = ".txt"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (configs / "presets.toml").write_text("[default]\n", encoding="utf-8")
    (configs / "gui-methods" / "lora.toml").write_text(
        """
network_module = "networks.lora_anima"
network_dim = 16
network_alpha = 16
learning_rate = 0.00002
max_train_epochs = 4
output_name = "staged-test"
""".strip()
        + "\n",
        encoding="utf-8",
    )


def test_profile_round_trip_and_name_whitelist(tmp_path):
    plan = plans.default_plan()
    saved = plans.save_profile("three_tier", plan, tmp_path)

    assert plans.list_profiles(tmp_path) == ["three_tier"]
    assert plans.load_profile("three_tier", tmp_path) == saved
    with pytest.raises(ValueError, match="profile name"):
        plans.save_profile("../escape", plan, tmp_path)


def test_plan_validation_rejects_bucket_semantics_errors():
    plan = plans.default_plan()
    plan["stages"][2]["resolution"] = 768
    with pytest.raises(ValueError, match="unique and strictly increasing"):
        plans.validate_plan(plan)

    plan = plans.default_plan()
    plan["stages"][0]["ratio"] = 10
    with pytest.raises(ValueError, match="sum to 100"):
        plans.validate_plan(plan)

    plan = plans.default_plan()
    plan["max_train_steps"] = 3
    plan["stages"][0]["ratio"] = 1
    plan["stages"][1]["ratio"] = 1
    plan["stages"][2]["ratio"] = 98
    with pytest.raises(ValueError, match="optimizer step"):
        plans.validate_plan(plan)


def _write_complete_stage_cache(
    root: Path, profile: str, plan: dict, stems: tuple[str, ...]
):
    for stage, paths in zip(plan["stages"], plans.stage_paths(profile, plan, root)):
        paths["resized_dir"].mkdir(parents=True)
        paths["cache_dir"].mkdir(parents=True)
        for stem in stems:
            image_path = paths["resized_dir"] / f"{stem}.png"
            Image.new("RGB", (stage["resolution"], stage["resolution"])).save(
                image_path
            )
            latent = (
                paths["cache_dir"]
                / f"{stem}_{stage['resolution']:04d}x{stage['resolution']:04d}_anima.npz"
            )
            latent.write_bytes(b"")
            (paths["cache_dir"] / f"{stem}_anima_te.safetensors").write_bytes(b"")


def test_status_requires_a_complete_cache_for_every_stage(tmp_path):
    plan = plans.default_plan()
    source = tmp_path / "image_dataset"
    source.mkdir()
    for stem in ("a", "b"):
        Image.new("RGB", (64, 64)).save(source / f"{stem}.png")
        (source / f"{stem}.txt").write_text("caption", encoding="utf-8")

    _write_complete_stage_cache(tmp_path, "default", plan, ("a", "b"))
    plans.write_profile_manifest("default", plan, tmp_path)

    status = plans.profile_status("default", plan, tmp_path)
    assert status["source_images"] == 2
    assert status["captions"] == 2
    assert status["all_ready"] is True
    assert all(stage["ready"] for stage in status["stages"])

    last_cache = plans.stage_paths("default", plan, tmp_path)[-1]["cache_dir"]
    (last_cache / "a_anima_te.safetensors").unlink()
    assert plans.profile_status("default", plan, tmp_path)["all_ready"] is False


def test_status_rejects_equal_count_stale_stems_and_source_inventory(tmp_path):
    plan = plans.default_plan()
    source = tmp_path / "image_dataset"
    source.mkdir()
    for stem in ("a", "b"):
        Image.new("RGB", (64, 64)).save(source / f"{stem}.png")

    _write_complete_stage_cache(tmp_path, "default", plan, ("a", "b"))
    plans.write_profile_manifest("default", plan, tmp_path)

    first_cache = plans.stage_paths("default", plan, tmp_path)[0]["cache_dir"]
    (first_cache / "a_0512x0512_anima.npz").unlink()
    (first_cache / "stale_0512x0512_anima.npz").write_bytes(b"")
    assert plans.profile_status("default", plan, tmp_path)["all_ready"] is False

    (first_cache / "stale_0512x0512_anima.npz").unlink()
    (first_cache / "a_0512x0512_anima.npz").write_bytes(b"")
    (source / "a.png").unlink()
    Image.new("RGB", (64, 64)).save(source / "c.png")
    assert plans.profile_status("default", plan, tmp_path)["all_ready"] is False


def test_stale_manifest_reset_is_scoped_to_profile_tree(tmp_path):
    plan = plans.default_plan()
    source = tmp_path / "image_dataset"
    source.mkdir()
    Image.new("RGB", (64, 64)).save(source / "a.png")
    plans.write_profile_manifest("default", plan, tmp_path)

    owned = tmp_path / "post_image_dataset" / "staged" / "default" / "512" / "cache"
    owned.mkdir(parents=True)
    (owned / "old_anima_te.safetensors").write_bytes(b"")
    outside = tmp_path / "post_image_dataset" / "keep.txt"
    outside.write_text("keep", encoding="utf-8")
    Image.new("RGB", (64, 64), "red").save(source / "a.png")

    assert plans.reset_profile_cache_if_stale("default", plan, tmp_path) is True
    assert not owned.exists()
    assert outside.read_text(encoding="utf-8") == "keep"


def test_compile_runtime_builds_three_full_dataset_rows(tmp_path):
    _write_minimal_configs(tmp_path)
    plan = plans.default_plan()

    output = plans.compile_runtime_config("default", plan, tmp_path)
    runtime = toml.loads(output.read_text(encoding="utf-8"))

    assert runtime["stage_schedule_enabled"] is True
    assert runtime["max_train_steps"] == 6000
    assert "max_train_epochs" not in runtime
    assert [stage["name"] for stage in runtime["stage_schedule"]] == [
        "512px",
        "768px",
        "1024px",
    ]
    assert [row["batch_size"] for row in runtime["datasets"]] == [4, 2, 1]
    assert [row["subsets"][0]["image_dir"] for row in runtime["datasets"]] == [
        "post_image_dataset/staged/default/512/resized",
        "post_image_dataset/staged/default/768/resized",
        "post_image_dataset/staged/default/1024/resized",
    ]


def test_compile_runtime_uses_compatible_external_blueprint_and_strips_legacy_keys(
    tmp_path,
):
    _write_minimal_configs(tmp_path)
    blueprint = tmp_path / "configs" / "datasets" / "external.toml"
    blueprint.parent.mkdir()
    blueprint.write_text(
        """
[general]
caption_extension = ".txt"

[[datasets]]
batch_size = 9
validation_split = 0.125
[[datasets.subsets]]
image_dir = "external-source"
cache_dir = "external-cache"
num_repeats = 3
recursive = false
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "configs" / "gui-methods" / "lora.toml").write_text(
        f"""
network_module = "networks.lora_anima"
dataset_config = "{blueprint.as_posix()}"
max_train_epochs = 4
target_res = [512, 768]
staged_resolution = true
staged_resolution_ratios = "10,30,60"
staged_resolution_base_sides = "512,768,1024"
stage_schedule_enabled = true
[[stage_schedule]]
name = "legacy"
subset_index = 0
start_pct = 0.0
end_pct = 1.0
""".strip()
        + "\n",
        encoding="utf-8",
    )

    output = plans.compile_runtime_config("default", plans.default_plan(), tmp_path)
    runtime = toml.loads(output.read_text(encoding="utf-8"))

    for key in (
        "dataset_config",
        "max_train_epochs",
        "target_res",
        "staged_resolution",
        "staged_resolution_ratios",
        "staged_resolution_base_sides",
    ):
        assert key not in runtime
    assert len(runtime["stage_schedule"]) == 3
    assert all(row["validation_split"] == 0.125 for row in runtime["datasets"])
    assert all(row["subsets"][0]["recursive"] is False for row in runtime["datasets"])


def test_compile_runtime_rejects_conditioning_blueprint(tmp_path):
    _write_minimal_configs(tmp_path)
    blueprint = tmp_path / "configs" / "datasets" / "controlnet.toml"
    blueprint.parent.mkdir()
    blueprint.write_text(
        """
[[datasets]]
batch_size = 1
[[datasets.subsets]]
image_dir = "target"
cache_dir = "cache"
conditioning_data_dir = "conditioning"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "configs" / "gui-methods" / "lora.toml").write_text(
        f'dataset_config = "{blueprint.as_posix()}"\n', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="conditioning_data_dir"):
        plans.compile_runtime_config("default", plans.default_plan(), tmp_path)


def test_staged_task_argument_parser_rejects_browser_overrides():
    from scripts.tasks.staged_resolution import _profile_arg

    assert _profile_arg(["default"], allow_training_flags=False) == ("default", [])
    assert _profile_arg(
        ["default", "--progress_jsonl", "job/progress.jsonl"],
        allow_training_flags=True,
    ) == ("default", ["--progress_jsonl", "job/progress.jsonl"])
    with pytest.raises(SystemExit, match="unsupported"):
        _profile_arg(["default", "--network_dim", "128"], allow_training_flags=True)


def test_staged_resume_uses_pinned_config_without_recompiling_profile(
    monkeypatch, tmp_path
):
    from scripts.tasks import staged_resolution as task

    snapshot = tmp_path / "config.snapshot.toml"
    snapshot.write_text('output_name = "pinned"\n', encoding="utf-8")
    state = tmp_path / "rolling-state"
    progress = tmp_path / "progress.jsonl"
    samples = tmp_path / "sample"
    launched: list[tuple[str, ...]] = []

    monkeypatch.setattr(
        task,
        "load_profile",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("resume must not reload a mutable profile")
        ),
    )
    monkeypatch.setattr(
        task,
        "compile_runtime_config",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("resume must not recompile a mutable profile")
        ),
    )
    monkeypatch.setattr(task, "accelerate_launch", lambda *args: launched.append(args))

    task.cmd_staged_train(
        [
            "default",
            "--config_file",
            str(snapshot),
            "--resume",
            str(state),
            "--progress_jsonl",
            str(progress),
            "--sample_dir",
            str(samples),
        ]
    )

    assert launched == [
        (
            "--config_file",
            str(snapshot),
            "--resume",
            str(state),
            "--progress_jsonl",
            str(progress),
            "--sample_dir",
            str(samples),
        )
    ]


def test_staged_preprocess_uses_lowres_matching_and_caption_variant_flags(
    monkeypatch, tmp_path
):
    from scripts.tasks import staged_resolution as task

    plan = plans.default_plan()
    runtime_path = tmp_path / "runtime.toml"
    runtime_path.write_text(
        'vae = "vae"\nqwen3 = "qwen"\npretrained_model_name_or_path = "dit"\n',
        encoding="utf-8",
    )
    stage_rows = plans.stage_paths("default", plan, tmp_path)
    commands: list[list[str]] = []
    statuses = [
        {
            "source_exists": True,
            "source_images": 1,
            "source_image_dir": str(tmp_path / "source"),
            "all_ready": False,
        },
        {"all_ready": True},
    ]
    monkeypatch.setattr(task, "load_profile", lambda _name: plan)
    monkeypatch.setattr(
        task, "compile_runtime_config", lambda _name, _plan: runtime_path
    )
    monkeypatch.setattr(task, "profile_status", lambda _name, _plan: statuses.pop(0))
    monkeypatch.setattr(task, "stage_paths", lambda _name, _plan: stage_rows)
    monkeypatch.setattr(
        task, "reset_profile_cache_if_stale", lambda _name, _plan: False
    )
    monkeypatch.setattr(task, "remove_profile_orphans", lambda _name, _plan: None)
    monkeypatch.setattr(task, "write_profile_manifest", lambda _name, _plan: None)
    monkeypatch.setattr(task, "run", lambda command: commands.append(command))

    task.cmd_staged_preprocess(["default"])

    te_commands = [
        command
        for command in commands
        if any(str(arg).endswith("cache_text_embeddings.py") for arg in command)
    ]
    assert len(te_commands) == 3
    for command, paths in zip(te_commands, stage_rows):
        assert command[command.index("--min_pixels") + 1] == "0"
        assert command[command.index("--match_images_from") + 1] == str(
            paths["resized_dir"]
        )
        assert "--caption_shuffle_variants" in command
        assert "--caption_tag_dropout_rate" in command
        assert "--caption_tag_randomize_rate" in command


def test_profile_api_persists_only_validated_structured_data(monkeypatch, tmp_path):
    from webui.api import staged_resolution as api

    _write_minimal_configs(tmp_path)
    monkeypatch.setattr(plans, "anima_home", lambda: tmp_path)
    body = api.StagedResolutionPlan.model_validate(plans.default_plan())

    response = api.put_profile("api_profile", body)

    assert response.name == "api_profile"
    assert response.persisted is True
    assert response.plan.max_train_steps == 6000
    assert plans.profile_path("api_profile", tmp_path).is_file()
    assert plans.runtime_path("api_profile", tmp_path).is_file()


def test_default_profile_response_marks_synthetic_plan_unpersisted(
    monkeypatch, tmp_path
):
    from webui.api import staged_resolution as api

    monkeypatch.setattr(plans, "anima_home", lambda: tmp_path)

    response = api.get_profile("default")

    assert response.name == "default"
    assert response.persisted is False
    assert not plans.profile_path("default", tmp_path).exists()


def test_cross_origin_staged_action_is_rejected_before_handler(monkeypatch):
    import library.config.io as config_io
    from webui import server
    from webui.api import staged_resolution as api

    called = 0

    async def fake_start(_name: str, _command: str):
        nonlocal called
        called += 1
        return {"task_id": "test-task"}

    monkeypatch.setattr(config_io, "migrate_custom_configs", lambda: None)
    monkeypatch.setattr(api, "_start_profile_task", fake_start)
    client = TestClient(server.create_app())

    rejected = client.post(
        "/api/staged-resolution/profiles/default/preprocess",
        headers={"Origin": "https://evil.example"},
        json={"version": 1},
    )
    assert rejected.status_code == 403
    assert called == 0

    bodyless = client.post(
        "/api/staged-resolution/profiles/default/preprocess",
        headers={"Origin": "http://127.0.0.1:8000"},
    )
    assert bodyless.status_code == 422
    assert called == 0

    allowed = client.post(
        "/api/staged-resolution/profiles/default/preprocess",
        headers={"Origin": "http://127.0.0.1:8000"},
        json={"version": 1},
    )
    assert allowed.status_code == 200
    assert called == 1


def test_same_origin_mutation_allows_runtime_webui_port(monkeypatch):
    import library.config.io as config_io
    from webui import server
    from webui.api import staged_resolution as api

    called = 0

    async def fake_start(_name: str, _command: str):
        nonlocal called
        called += 1
        return {"task_id": "test-task"}

    monkeypatch.setattr(config_io, "migrate_custom_configs", lambda: None)
    monkeypatch.setattr(api, "_start_profile_task", fake_start)
    client = TestClient(server.create_app())
    runtime_origin = "http://127.0.0.1:7460"

    allowed = client.post(
        "/api/staged-resolution/profiles/default/preprocess",
        headers={"Host": "127.0.0.1:7460", "Origin": runtime_origin},
        json={"version": 1},
    )

    assert allowed.status_code == 200
    assert called == 1

    rejected = client.post(
        "/api/staged-resolution/profiles/default/preprocess",
        headers={"Host": "127.0.0.1:7460", "Origin": "http://127.0.0.1:7459"},
        json={"version": 1},
    )

    assert rejected.status_code == 403
    assert called == 1
