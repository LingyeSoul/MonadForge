"""Fixed-operation RPC transport; subprocesses are replaced with fakes."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.curation import bridge
from scripts.curation.context import CurationUnavailable


def test_registered_runtime_receives_json_data_not_shell_code(monkeypatch, tmp_path):
    import library.runtime.profiles as profiles
    runtime = SimpleNamespace(python=tmp_path / "curation-python", env={"CURATION_TEST": "1"})
    monkeypatch.setattr(profiles, "resolve_runtime", lambda *_: runtime)
    calls = []

    def execute(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(stdout=json.dumps({"value": {"flat_tags": ["example"]}}), stderr="", returncode=0)

    monkeypatch.setattr(bridge.subprocess, "run", execute)

    @bridge.in_curation_runtime
    def parse(text):
        raise AssertionError("The protected host must not import the curation implementation")

    malicious_literal = '$(touch /tmp/never) `echo never`'
    assert parse(malicious_literal) == {"flat_tags": ["example"]}
    argv, options = calls[0]
    assert argv == [str(runtime.python), "-m", "scripts.curation.bridge"]
    assert "shell" not in options
    assert json.loads(options["input"])["args"] == [malicious_literal]
    assert options["env"]["CURATION_TEST"] == "1"


def test_bridge_rejects_unknown_operation_before_process_start(monkeypatch, tmp_path):
    import library.runtime.profiles as profiles
    monkeypatch.setattr(profiles, "resolve_runtime", lambda *_: SimpleNamespace(python=tmp_path / "other", env={}))

    @bridge.in_curation_runtime
    def arbitrary_module():
        pass

    with pytest.raises(ValueError, match="Unknown"):
        arbitrary_module()


def test_runtime_import_error_remains_a_service_unavailable(monkeypatch, tmp_path):
    import library.runtime.profiles as profiles
    monkeypatch.setattr(profiles, "resolve_runtime", lambda *_: SimpleNamespace(python=tmp_path / "other", env={}))
    monkeypatch.setattr(bridge.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        stdout=json.dumps({"error": {"type": "CurationUnavailable", "message": "register curation runtime"}}), stderr="", returncode=1,
    ))

    @bridge.in_curation_runtime
    def get_schema():
        return {}

    with pytest.raises(CurationUnavailable, match="register"):
        get_schema()
