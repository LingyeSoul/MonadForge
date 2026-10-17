from __future__ import annotations

import sys
from pathlib import Path

from library.runtime import profiles
from scripts import update


def update_fixture(monkeypatch, tmp_path):
    root = tmp_path / "installed"
    incoming = tmp_path / "incoming"
    root.mkdir()
    incoming.mkdir()
    (incoming / "new_code.py").write_text("VALUE = 1\n")
    monkeypatch.setattr(update, "ROOT", root)
    monkeypatch.setattr(update, "MANIFEST_FILE", root / ".anima_release.json")
    monkeypatch.setattr(update, "BACKUP_ROOT", root / ".anima-update-backups")
    monkeypatch.setattr(update, "_restart_daemon_if_idle", lambda: None)
    return root, incoming


def apply(incoming):
    return update._apply(
        incoming, "test-version", {}, dry_run=False,
        yes_overwrite=True, keep_conflicts=False, no_sync=False,
    )


def test_v100_update_changes_code_without_running_standard_sync(monkeypatch, tmp_path):
    root, incoming = update_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(update, "_v100_sync_protected", lambda: True)

    def no_subprocess(*args, **kwargs):
        raise AssertionError("V100 update must not run ordinary uv sync")

    monkeypatch.setattr(update.subprocess, "run", no_subprocess)
    assert apply(incoming) == 0
    assert (root / "new_code.py").read_text() == "VALUE = 1\n"


def test_standard_update_keeps_selected_torch_stack(monkeypatch, tmp_path):
    _root, incoming = update_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(update, "_v100_sync_protected", lambda: False)
    command = ["uv", "sync", "--no-default-groups", "--group", "torch-213"]
    monkeypatch.setattr(update, "_uv_sync_command", lambda: (command, None))
    calls = []
    monkeypatch.setattr(update.subprocess, "run", lambda cmd, **kwargs: calls.append(cmd))
    assert apply(incoming) == 0
    assert calls == [command]


def test_invalid_runtime_stops_update_before_copying_files(monkeypatch, tmp_path):
    root, incoming = update_fixture(monkeypatch, tmp_path)

    def invalid():
        raise profiles.RuntimeProfileError("invalid local registration")

    monkeypatch.setattr(update, "_v100_sync_protected", invalid)
    assert apply(incoming) == 2
    assert not (root / "new_code.py").exists()
    assert not update.MANIFEST_FILE.exists()


def test_updater_checks_project_venv_when_launched_with_system_python(monkeypatch, tmp_path):
    python = tmp_path / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    python.parent.mkdir(parents=True)
    python.write_text("not executed")
    monkeypatch.setattr(update, "ROOT", tmp_path)
    monkeypatch.delenv("ANIMA_RUNTIME_PROFILE", raising=False)
    monkeypatch.setattr(
        profiles, "resolve_runtime", lambda *args, **kwargs:
        profiles.RuntimeSpec("default", Path(sys.executable)),
    )
    monkeypatch.setattr(profiles, "is_v100_runtime", lambda candidate: candidate == python)
    assert update._v100_sync_protected()


def test_updater_preserves_local_runtime_registration_and_workspace():
    assert update.REPO == "LingyeSoul/MonadForge"
    assert update._is_preserved("configs/custom/runtime.toml")
    assert update._is_preserved("workspace/master/image.txt")
