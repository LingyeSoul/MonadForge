"""Failure injection for local settings; never writes repository configs."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import toml
import yaml

from webui.services import config_service as config
from webui.services import config_storage as storage
from webui.services import preprocess_service as preprocess


@pytest.fixture
def settings(tmp_path, monkeypatch):
    sam = tmp_path / "sam.yaml"
    gui = tmp_path / "webui_settings.json"
    tiers = tmp_path / "preprocess.toml"
    sam.write_text("prompts: [person]\nthreshold: 0.4\ndilate: 3\ncustom_key: keep\n")
    gui.write_text('{"unrelated": "keep", "mit_dilate": 5}')
    tiers.write_text(
        "target_res = [1024]\nmultires_per_image = false\nmin_pixels = 42\n"
    )
    monkeypatch.setattr(preprocess, "SAM_YAML", sam)
    monkeypatch.setattr(preprocess, "SETTINGS_FILE", gui)
    monkeypatch.setattr(preprocess, "PREPROCESS_TOML", tiers)
    monkeypatch.setattr(config, "_SETTINGS_FILE", gui)
    monkeypatch.setattr(
        preprocess,
        "get_target_res",
        lambda: toml.loads(tiers.read_text())["target_res"],
    )
    monkeypatch.setattr(
        preprocess,
        "get_multires_per_image",
        lambda: toml.loads(tiers.read_text()).get("multires_per_image", False),
    )
    return sam, gui, tiers


def _fail_replace(monkeypatch, destination):
    replace = Path.replace

    def fail(self, target):
        if Path(target) == destination and self.suffix == ".tmp":
            raise OSError("injected replace failure")
        return replace(self, target)

    monkeypatch.setattr(Path, "replace", fail)


def test_atomic_write_preserves_symlink_and_permissions(tmp_path):
    target = tmp_path / "actual.toml"
    target.write_text("old")
    target.chmod(0o640)
    link = tmp_path / "config.toml"
    link.symlink_to(target)
    storage.atomic_write_text(link, "new")
    assert link.is_symlink() and target.read_text() == "new"
    assert target.stat().st_mode & 0o777 == 0o640
    assert sorted(p.name for p in tmp_path.iterdir()) == ["actual.toml", "config.toml"]


def test_stage_failure_modifies_no_destination(tmp_path, monkeypatch):
    first, second = tmp_path / "first", tmp_path / "second"
    first.write_text("old one")
    second.write_text("old two")
    stage = storage._stage

    def fail(path, content, suffix, mode):
        if path == second and suffix == ".tmp":
            raise OSError("staging failed")
        return stage(path, content, suffix, mode)

    monkeypatch.setattr(storage, "_stage", fail)
    with pytest.raises(OSError, match="staging failed"):
        storage.atomic_write_many({first: "new one", second: "new two"})
    assert first.read_text() == "old one" and second.read_text() == "old two"
    assert set(tmp_path.iterdir()) == {first, second}


def test_failed_transaction_removes_new_file_and_restores_originals(
    tmp_path, monkeypatch
):
    existing, new, failed = (tmp_path / name for name in ("existing", "new", "failed"))
    existing.write_text("original")
    failed.write_text("last original")
    _fail_replace(monkeypatch, failed)
    with pytest.raises(OSError, match="injected"):
        storage.atomic_write_many({existing: "changed", new: "new", failed: "changed"})
    assert existing.read_text() == "original" and failed.read_text() == "last original"
    assert set(tmp_path.iterdir()) == {existing, failed}


def test_failed_rollback_retains_recovery_backup(tmp_path, monkeypatch):
    first, second = tmp_path / "first", tmp_path / "second"
    first.write_text("original one")
    second.write_text("original two")
    replace = Path.replace

    def fail(self, target):
        if Path(target) == second or self.suffix == ".bak":
            raise OSError("device unavailable")
        return replace(self, target)

    monkeypatch.setattr(Path, "replace", fail)
    with pytest.raises(OSError, match="Recovery backups retained") as failure:
        storage.atomic_write_many({first: "new one", second: "new two"})
    backups = list(tmp_path.glob("*.bak"))
    assert any(p.read_text() == "original one" for p in backups)
    assert all(str(p) in str(failure.value) for p in backups)
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize(
    "save",
    [
        lambda path: config._save(path, {"network_dim": 32}),
        lambda path: config.write_sample_prompts(str(path), [{"prompt": "new prompt"}]),
    ],
)
def test_config_replace_failure_keeps_old_bytes(tmp_path, monkeypatch, save):
    # Both writers refuse a path outside ROOT now, so the fixture's tmp dir is
    # the project root for this test (same pattern as the system-paths test).
    monkeypatch.setattr(config, "ROOT", tmp_path)
    path = tmp_path / "config.txt"
    path.write_text("old file")
    _fail_replace(monkeypatch, path)
    with pytest.raises(OSError, match="injected"):
        save(path)
    assert path.read_text() == "old file"
    assert list(tmp_path.iterdir()) == [path]


def test_system_model_paths_replace_failure_keeps_original(tmp_path, monkeypatch):
    from webui.api import system

    monkeypatch.setattr(system, "ROOT", tmp_path)
    path = tmp_path / "configs/custom/model.toml"
    path.parent.mkdir(parents=True)
    path.write_text('vae = "old"\n')
    _fail_replace(monkeypatch, path)
    with pytest.raises(OSError, match="injected"):
        system.update_model_paths([system.ModelPathUpdate(key="vae", value="new")])
    assert path.read_text() == 'vae = "old"\n'


@pytest.mark.parametrize("bad", ["{broken", "[]", '{"wandb": []}'])
def test_wandb_refuses_to_overwrite_invalid_settings(settings, bad):
    _, gui, _ = settings
    gui.write_text(bad)
    with pytest.raises(ValueError):
        config.save_wandb_settings({"enabled": True})
    assert gui.read_text() == bad


def test_tagger_refuses_to_overwrite_invalid_shared_settings(settings, monkeypatch):
    from webui.services import wd_tagger_service as tagger

    gui = settings[1]
    monkeypatch.setattr(tagger, "SETTINGS_FILE", gui)
    gui.write_text("{broken")
    with pytest.raises(ValueError):
        tagger.save_settings(tagger.DEFAULT_MODEL, 0.35)
    assert gui.read_text() == "{broken"


def test_tagger_failed_write_preserves_other_settings(settings, monkeypatch):
    from webui.services import wd_tagger_service as tagger

    gui = settings[1]
    original = gui.read_bytes()
    monkeypatch.setattr(tagger, "SETTINGS_FILE", gui)
    _fail_replace(monkeypatch, gui)
    with pytest.raises(OSError, match="injected"):
        tagger.save_settings(tagger.DEFAULT_MODEL, 0.35)
    assert gui.read_bytes() == original


def test_preprocess_validation_precedes_every_write(settings):
    original = {p: p.read_bytes() for p in settings}
    with pytest.raises(ValueError, match="at least two"):
        preprocess.save_settings(
            {
                "sam": {"threshold": 0.9},
                "target_res": [1024],
                "multires_per_image": True,
            }
        )
    assert {p: p.read_bytes() for p in settings} == original


@pytest.mark.parametrize(
    "index,bad", [(0, "prompts: ["), (1, "{broken"), (2, "target_res = [broken")]
)
def test_preprocess_parse_error_preserves_all_files(settings, index, bad):
    settings[index].write_text(bad)
    original = {p: p.read_bytes() for p in settings}
    with pytest.raises((ValueError, yaml.YAMLError)):
        preprocess.save_settings(
            {"target_res": [512, 1024], "multires_per_image": True}
        )
    assert {p: p.read_bytes() for p in settings} == original


@pytest.mark.parametrize("failed_index", [1, 2])
def test_preprocess_commit_failure_rolls_back_all_files(
    settings, monkeypatch, failed_index
):
    original = {p: p.read_bytes() for p in settings}
    _fail_replace(monkeypatch, settings[failed_index])
    with pytest.raises(OSError, match="injected"):
        preprocess.save_settings(
            {
                "sam": {"threshold": 0.9},
                "mit_dilate": 7,
                "target_res": [512, 1024],
                "multires_per_image": True,
            }
        )
    assert {p: p.read_bytes() for p in settings} == original
    assert set(settings[0].parent.iterdir()) == set(settings)


def test_preprocess_success_preserves_unrelated_keys(settings):
    result = preprocess.save_settings(
        {
            "sam": {"threshold": 0.8},
            "target_res": [512, 1024],
            "multires_per_image": True,
        }
    )
    assert result["target_res"] == [512, 1024] and result["multires_per_image"]
    assert "custom_key: keep" in settings[0].read_text()
    assert json.loads(settings[1].read_text())["unrelated"] == "keep"
    assert toml.loads(settings[2].read_text())["min_pixels"] == 42


def test_wandb_and_preprocess_concurrent_updates_preserve_both(settings, monkeypatch):
    entered, release, wandb_started, wandb_done = (threading.Event() for _ in range(4))
    write_many = preprocess.atomic_write_many

    def blocked_write(contents):
        entered.set()
        assert release.wait(3)
        write_many(contents)

    def save_wandb():
        wandb_started.set()
        result = config.save_wandb_settings({"project": "new project"})
        wandb_done.set()
        return result

    monkeypatch.setattr(preprocess, "atomic_write_many", blocked_write)
    with ThreadPoolExecutor(max_workers=2) as pool:
        prep = pool.submit(preprocess.save_settings, {"mit_dilate": 9})
        assert entered.wait(3)
        wandb = pool.submit(save_wandb)
        assert wandb_started.wait(3)
        wandb_done.wait(0.1)
        release.set()
        prep.result(timeout=3)
        wandb.result(timeout=3)
    saved = json.loads(settings[1].read_text())
    assert saved["mit_dilate"] == 9 and saved["wandb"]["project"] == "new project"
    assert saved["unrelated"] == "keep"


@pytest.mark.parametrize(
    "extra",
    ["network_dim = [broken", 'path = "C:\\bad"\nvalue = [broken', ["wrong type"]],
)
def test_invalid_extra_args_does_not_change_variant(tmp_path, monkeypatch, extra):
    monkeypatch.setattr(config, "CUSTOM_VARIANTS_DIR", tmp_path)
    path = tmp_path / "lora.toml"
    path.write_text("network_dim = 16\n")
    with pytest.raises(ValueError, match="extra_args"):
        config.save_variant_config("lora", {"network_dim": 32, "extra_args": extra})
    assert path.read_text() == "network_dim = 16\n"


def test_extra_args_still_accepts_windows_backslash_compatibility(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(config, "CUSTOM_VARIANTS_DIR", tmp_path)
    config.save_variant_config("lora", {"extra_args": 'path = "C:\\models\\custom"'})
    assert (
        toml.loads((tmp_path / "lora.toml").read_text())["path"] == "C:/models/custom"
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("learning_rate", "bad"),
        ("learning_rate", float("nan")),
        ("learning_rate", float("inf")),
        ("learning_rate", True),
        ("network_dim", True),
        ("network_dim", 1.5),
        ("max_train_epochs", 0),
        ("max_train_epochs", "bad"),
        ("batch_size", None),
        ("batch_size", False),
        ("batch_size", -1),
        ("num_repeats", 0),
        ("num_repeats", 1.2),
        ("sample_ratio", "bad"),
        ("sample_ratio", True),
        ("sample_ratio", float("nan")),
    ],
)
def test_invalid_known_training_values_are_rejected_before_save(
    tmp_path, monkeypatch, key, value
):
    monkeypatch.setattr(config, "CUSTOM_VARIANTS_DIR", tmp_path)
    path = tmp_path / "lora.toml"
    path.write_text("network_dim = 16\n")
    with pytest.raises(ValueError, match=key):
        config.save_variant_config("lora", {key: value})
    assert path.read_text() == "network_dim = 16\n"


def test_extra_args_override_form_before_validation_and_virtual_conversion(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(config, "CUSTOM_VARIANTS_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIGS_DIR", tmp_path)
    config.save_variant_config(
        "lora",
        {
            "learning_rate": "invalid form",
            "batch_size": "invalid form",
            "sample_ratio": False,
            "extra_args": 'learning_rate = 0.001\nbatch_size = 2\nsample_ratio = 0.5\ncustom_future_option = "keep"',
        },
    )
    saved = toml.loads((tmp_path / "lora.toml").read_text())
    assert saved["learning_rate"] == 0.001
    assert saved["datasets"][0]["batch_size"] == 2
    assert saved["datasets"][0]["subsets"][0]["sample_ratio"] == 0.5
    assert saved["custom_future_option"] == "keep"
    assert "batch_size" not in saved and "sample_ratio" not in saved


def test_invalid_extra_args_override_valid_form_without_persisting(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(config, "CUSTOM_VARIANTS_DIR", tmp_path)
    path = tmp_path / "lora.toml"
    path.write_text("network_dim = 16\n")
    with pytest.raises(ValueError, match="batch_size"):
        config.save_variant_config(
            "lora", {"batch_size": 2, "extra_args": 'batch_size = "bad"'}
        )
    assert path.read_text() == "network_dim = 16\n"


@pytest.mark.parametrize(
    "data",
    [
        {"anima_num_blocks": None},
        {"anima_num_blocks": {}},
        {"learning_rate": "bad"},
        {"max_train_epochs": True},
    ],
)
def test_validate_api_returns_invalid_for_bad_types(data):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from webui.api.config import router

    app = FastAPI()
    app.include_router(router)
    response = TestClient(app).post("/validate", json={"data": data})
    assert response.status_code == 200
    assert response.json()["valid"] is False
    assert response.json()["errors"]


def test_method_save_api_rejects_invalid_virtual_field_without_writing(
    tmp_path, monkeypatch
):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from webui.api.config import router

    monkeypatch.setattr(config, "CUSTOM_VARIANTS_DIR", tmp_path)
    path = tmp_path / "lora.toml"
    path.write_text("network_dim = 16\n")
    app = FastAPI()
    app.include_router(router)
    response = TestClient(app).put("/method", json={"data": {"batch_size": 0}})
    assert response.status_code == 400
    assert path.read_text() == "network_dim = 16\n"
