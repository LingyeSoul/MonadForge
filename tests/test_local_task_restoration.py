"""Restored local CLI dispatch: AST or mocks only, never a model/worker run."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESTORED = {
    "scripts/tasks/inference.py": {"cmd_test_dcw", "cmd_test_dcw_v4", "cmd_test_spectrum_dcw", "cmd_test_dcw_v4_spectrum", "_latest_fusion_head"},
    "scripts/tasks/tagger.py": {"cmd_preprocess_tagger"},
    "scripts/tasks/daemon.py": {"cmd_daemon_status"},
    "scripts/tasks/utilities.py": {"cmd_distill_prep"},
    "scripts/experimental_tasks/training.py": {"cmd_spd", "cmd_chimera", "cmd_byg"},
    "scripts/experimental_tasks/inference.py": {"cmd_test_spd", "cmd_test_byg", "cmd_invert_directedit", "_resolve_inference_base_flag", "_resolve_ref_image_pool"},
}


@pytest.mark.parametrize("path,expected", RESTORED.items())
def test_local_task_bodies_and_dependencies_exist(path, expected):
    tree = ast.parse((ROOT / path).read_text())
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert expected <= functions.keys()
    for name in expected:
        assert len(functions[name].body) > 1, name


def test_registered_commands_resolve_in_assigned_modules_without_importing_them():
    aliases = {
        "inference": "scripts/tasks/inference.py", "tagger": "scripts/tasks/tagger.py",
        "daemon": "scripts/tasks/daemon.py", "utilities": "scripts/tasks/utilities.py",
        "exp_training": "scripts/experimental_tasks/training.py",
        "exp_inference": "scripts/experimental_tasks/inference.py",
    }
    definitions = {alias: {node.name for node in ast.parse((ROOT / path).read_text()).body
                           if isinstance(node, ast.FunctionDef)} for alias, path in aliases.items()}
    missing = []
    for node in ast.walk(ast.parse((ROOT / "tasks.py").read_text())):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            module = node.value.id
            if module in definitions and node.attr.startswith("cmd_") and node.attr not in definitions[module]:
                missing.append(f"{module}.{node.attr}")
    assert missing == []


@pytest.mark.parametrize("name,expected", [
    ("cmd_test_dcw", "--dcw"), ("cmd_test_dcw_v4", "--dcw_calibrator"),
    ("cmd_test_spectrum_dcw", "--spectrum"), ("cmd_test_dcw_v4_spectrum", "--dcw_calibrator"),
])
def test_dcw_restored_targets_build_commands(monkeypatch, name, expected):
    from scripts.tasks import inference
    calls = []
    monkeypatch.setattr(inference, "run", lambda command: calls.append(command))
    monkeypatch.setattr(inference, "_base_test_args", lambda **kwargs: ["python", "inference.py"])
    monkeypatch.setattr(inference, "INFERENCE_BASE", ["python", "inference.py"])
    monkeypatch.setattr(inference, "latest_lora", lambda: Path("fixture-lora.safetensors"))
    monkeypatch.setattr(inference, "_latest_fusion_head", lambda: "fixture-calibrator.safetensors")
    monkeypatch.setattr(inference, "_spectrum_flags", lambda **kwargs: ["--spectrum"])
    getattr(inference, name)(["--seed", "9"])
    assert len(calls) == 1 and expected in calls[0]
    assert calls[0][-2:] == ["--seed", "9"]


def test_explicit_dcw_calibrator_does_not_scan_live_weights(monkeypatch):
    from scripts.tasks import inference
    calls = []
    monkeypatch.setattr(inference, "run", lambda command: calls.append(command))
    monkeypatch.setattr(inference, "_base_test_args", lambda **kwargs: ["python", "inference.py"])

    def forbidden():
        raise AssertionError("An explicit calibrator must not search real checkpoints")

    monkeypatch.setattr(inference, "_latest_fusion_head", forbidden)
    inference.cmd_test_dcw_v4(["--dcw_calibrator=fixture.safetensors"])
    assert calls[0][-1] == "--dcw_calibrator=fixture.safetensors"


def test_dual_encoder_preprocess_uses_complete_local_tagger(monkeypatch):
    from scripts.tasks import tagger
    calls = []
    monkeypatch.setattr(tagger, "run", lambda command: calls.append(command))
    tagger.cmd_preprocess_tagger(["--out_dir", "fixture"])
    assert [command[4] for command in calls] == ["build_vocab", "build_features"]
    assert all(command[2] == "scripts.anima_tagger.cli" for command in calls)
    assert all(command[-2:] == ["--out_dir", "fixture"] for command in calls)


def test_local_training_and_distill_targets_only_construct_commands(monkeypatch):
    from scripts.experimental_tasks import training
    from scripts.tasks import utilities
    calls = []
    train_calls = []
    monkeypatch.setattr(training, "run", lambda command: calls.append(command))
    monkeypatch.setattr(utilities, "run", lambda command: calls.append(command))
    monkeypatch.setattr(training, "train", lambda method, extra: train_calls.append((method, extra)))
    monkeypatch.setattr(training, "_preset", lambda: "fixture")
    monkeypatch.setattr(training, "bespoke_preset_flags", lambda preset: ["--blocks_to_swap", "3"])
    training.cmd_spd(["--iterations", "1"])
    training.cmd_chimera(["--max_train_steps", "1"])
    training.cmd_byg(["--max_train_steps", "1"])
    utilities.cmd_distill_prep(["--skip_synth"])
    assert calls[0][1:] == ["scripts/distill_spd.py", "--blocks_to_swap", "3", "--iterations", "1"]
    assert calls[1][1:] == ["-m", "scripts.distill_mod.prep", "--skip_synth"]
    assert [method for method, _ in train_calls] == ["chimera", "byg"]


def test_spd_inference_reads_mock_metadata_and_preserves_cli_override(monkeypatch):
    from scripts.experimental_tasks import inference
    calls = []

    class Header:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def metadata(self):
            return {"ss_spd_stages": "[0.5, 1.0]", "ss_spd_transition_sigmas": "[0.6]"}

    monkeypatch.setitem(sys.modules, "safetensors", SimpleNamespace(safe_open=lambda *_: Header()))
    monkeypatch.setattr(inference, "latest_output", lambda *_: Path("fixture-spd.safetensors"))
    monkeypatch.setattr(inference, "run", lambda command: calls.append(command))
    monkeypatch.setattr(inference, "INFERENCE_BASE", ["python", "inference.py", "--sampler", "heun"])
    inference.cmd_test_spd(["--spd_stages", "0.25", "1.0"])
    assert calls[0][calls[0].index("--sampler") + 1] == "euler"
    assert calls[0].count("--spd_stages") == 1
    assert calls[0][-3:] == ["--spd_stages", "0.25", "1.0"]


def test_daemon_status_uses_mock_passive_client(monkeypatch, capsys):
    from scripts.tasks import daemon
    client = SimpleNamespace(base="http://fixture", health=lambda: {"pid": 42, "paused": True},
                             list_jobs=lambda: [{"id": "fixture", "state": "done", "private": "omitted"}])
    monkeypatch.setattr(daemon._client, "DaemonClient", lambda: client)
    daemon.cmd_daemon_status([])
    payload = json.loads(capsys.readouterr().out)
    assert payload["up"] and payload["pid"] == 42
    assert "private" not in payload["jobs"][0]


def test_byg_preserves_existing_explicit_unavailable_boundary():
    from scripts.experimental_tasks import inference
    with pytest.raises(SystemExit, match="not wired"):
        inference.cmd_test_byg([])
