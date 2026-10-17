from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from library.runtime import profiles


def write_config(tmp_path: Path, content: str) -> Path:
    directory = tmp_path / "configs"
    (directory / "custom").mkdir(parents=True)
    (directory / "custom" / "runtime.toml").write_text(content)
    return directory


def test_default_and_curation_share_current_interpreter_without_registration(tmp_path):
    for name in ("default", "curation"):
        spec = profiles.resolve_runtime(name, tmp_path / "configs")
        assert spec.name == name
        assert spec.python == Path(os.path.abspath(sys.executable))
        assert spec.env == {}


def test_curation_inherits_default_and_overrides_local_environment(tmp_path):
    configs = write_config(tmp_path, f"""
[runtimes.default]
python = {json.dumps(sys.executable)}
env = {{ HF_HUB_OFFLINE = "0", SHARED_VALUE = "yes" }}
[runtimes.curation]
env = {{ HF_HUB_OFFLINE = "1" }}
""")
    spec = profiles.resolve_runtime("curation", configs)
    assert spec.python == Path(sys.executable)
    assert spec.env == {"HF_HUB_OFFLINE": "1", "SHARED_VALUE": "yes"}


@pytest.mark.skipif(os.name == "nt", reason="POSIX virtualenv symlink")
def test_relative_interpreter_preserves_venv_symlink_and_ignores_cwd(tmp_path, monkeypatch):
    interpreter = tmp_path / "runtime" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)
    configs = write_config(tmp_path, '[runtimes.default]\npython="runtime/bin/python"\n')
    monkeypatch.chdir(interpreter.parent)
    assert profiles.resolve_runtime("default", configs).python == interpreter


def test_missing_or_unexecutable_registration_fails(tmp_path):
    configs = write_config(tmp_path, '[runtimes.curation]\npython="missing/bin/python"\n')
    with pytest.raises(profiles.RuntimeProfileError, match="missing or not executable"):
        profiles.resolve_runtime("curation", configs)


def test_v100_does_not_fall_back_to_normal_stack(tmp_path, monkeypatch):
    monkeypatch.setattr(profiles, "is_v100_runtime", lambda: False)
    with pytest.raises(profiles.RuntimeProfileError, match="not registered"):
        profiles.resolve_runtime("v100", tmp_path / "configs")


def test_v100_can_use_current_protected_stack_or_explicit_registration(tmp_path, monkeypatch):
    monkeypatch.setattr(profiles, "is_v100_runtime", lambda: True)
    assert profiles.resolve_runtime("v100", tmp_path / "configs").python == Path(sys.executable)
    configs = write_config(tmp_path, f'[runtimes.v100]\npython={json.dumps(sys.executable)}\n')
    monkeypatch.setattr(profiles, "is_v100_runtime", lambda: False)
    assert profiles.resolve_runtime("v100", configs).python == Path(sys.executable)


@pytest.mark.parametrize("version,expected", [
    ("2.10.0+cu129", True), ("2.10.1+cu129", True),
    ("2.10.0+cpu", False), ("2.14.1+cu132", False), (None, False),
])
def test_v100_detection_uses_version_metadata_only(monkeypatch, version, expected):
    monkeypatch.setattr(profiles, "torch_version", lambda _python=None: version)
    assert profiles.is_v100_runtime() is expected


@pytest.mark.parametrize("content", [
    'runtimes = "bad"', '[runtimes.default]\npython=42',
    '[runtimes.default]\nenv={ BAD_VALUE=42 }', '[runtimes.default]\npython=""',
    '[runtimes.default]\nenv={ "BAD=KEY"="x" }', '[broken',
])
def test_invalid_local_runtime_config_is_actionable(tmp_path, content):
    configs = write_config(tmp_path, content)
    with pytest.raises(profiles.RuntimeProfileError):
        profiles.resolve_runtime("default", configs)


def test_unknown_profile_is_not_an_interpreter_path(tmp_path):
    with pytest.raises(profiles.RuntimeProfileError, match="Unknown runtime profile"):
        profiles.resolve_runtime("../../bin/python", tmp_path / "configs")


def test_resolution_does_not_import_torch(tmp_path):
    root = Path(__file__).parents[1]
    code = (
        f"import sys; sys.path.insert(0, {str(root)!r}); "
        "from library.runtime.profiles import resolve_runtime; "
        f"resolve_runtime('default', __import__('pathlib').Path({str(tmp_path)!r})); "
        "assert 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-I", "-c", code], check=True, timeout=20)
