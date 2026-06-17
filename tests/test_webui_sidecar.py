"""Tests for the WebUI sidecar supervisor.

Avoids launching a real uvicorn server by monkeypatching ``proc.spawn_detached``
to return a short-lived fake subprocess (a ``python -c "..."`` that exits after
a moment). This exercises the monitor loop's spawn → wait → respawn → give-up
logic without any network or GPU.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch


from scripts.daemon import config, webui_sidecar

# A fake "WebUI" that prints a line and exits 0 after ~150ms. spawn_detached is
# patched to launch this so the monitor sees a real (short-lived) process.
_SHORT_LIVED = [
    sys.executable,
    "-c",
    "import time; time.sleep(0.15)",
]

# A fake that runs long enough to be "healthy" (>= _HEALTHY_UPTIME) then exits.
_LONG_LIVED = [
    sys.executable,
    "-c",
    "import time; time.sleep(35)",
]


def _make_sidecar(tmp_path: Path, monkeypatch) -> webui_sidecar.WebUISidecar:
    """A sidecar whose state dir + cmd live in tmp, launching a fake process."""
    monkeypatch.setattr(config, "STATE_DIR", tmp_path)
    sc = webui_sidecar.WebUISidecar(port=9999)
    # Pin the pidfile to the tmp state dir (already done via STATE_DIR, but the
    # instance captured it at __init__ — reset it).
    sc._pidfile = tmp_path / "webui.json"
    return sc


def _patch_spawn(cmd_template):
    """Replace spawn_detached so it launches cmd_template as a real Popen."""
    def _fake(cmd, *, cwd, stdout_path, env=None):
        return subprocess.Popen(
            cmd_template,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )

    return patch("scripts.daemon.proc.spawn_detached", side_effect=_fake)


def _patch_build_cmd():
    """Make the sidecar build the fake cmd instead of `pythonw -m webui`."""
    return patch.object(
        webui_sidecar.WebUISidecar,
        "_build_cmd",
        return_value=_SHORT_LIVED,
    )


def test_sidecar_start_writes_pidfile(tmp_path: Path, monkeypatch):
    sc = _make_sidecar(tmp_path, monkeypatch)
    with _patch_spawn(_SHORT_LIVED), _patch_build_cmd():
        sc.start()
        # Give the spawn a moment.
        time.sleep(0.4)
        assert sc._pidfile.is_file()
        rec = json.load(open(sc._pidfile, encoding="utf-8"))
        assert rec["port"] == 9999
        assert rec["url"] == "http://127.0.0.1:9999"
    sc.stop()


def test_sidecar_stop_clears_pidfile(tmp_path: Path, monkeypatch):
    sc = _make_sidecar(tmp_path, monkeypatch)
    with _patch_spawn(_LONG_LIVED), patch.object(
        webui_sidecar.WebUISidecar, "_build_cmd", return_value=_LONG_LIVED
    ):
        sc.start()
        time.sleep(0.3)
        assert sc._pidfile.is_file()
        sc.stop()
    assert not sc._pidfile.is_file()


def test_sidecar_gives_up_after_repeated_crashes(tmp_path: Path, monkeypatch):
    """A process that dies 5× fast (under _HEALTHY_UPTIME) → monitor stops retrying."""
    sc = _make_sidecar(tmp_path, monkeypatch)
    # Shrink the healthy-uptime threshold so our 0.15s fake counts as a crash,
    # and the backoff so the test doesn't wait long.
    monkeypatch.setattr(webui_sidecar, "_HEALTHY_UPTIME", 999.0)
    monkeypatch.setattr(webui_sidecar, "_MAX_BACKOFF", 0.2)

    with _patch_spawn(_SHORT_LIVED), _patch_build_cmd():
        sc.start()
        # Wait long enough for give-up: 5 crashes × ~0.15s + backoffs (<1s each).
        time.sleep(6.0)
        # The monitor thread should have exited (given up) — joined quickly.
        assert sc._monitor_thread is not None
        sc._monitor_thread.join(timeout=2.0)
        assert not sc._monitor_thread.is_alive(), "monitor should have given up"
        assert not sc._pidfile.is_file(), "pidfile cleared on give-up"


def test_is_enabled_default_true(monkeypatch):
    monkeypatch.delenv("ANIMA_DAEMON_HOST_WEBUI", raising=False)
    assert webui_sidecar.is_enabled() is True


def test_is_enabled_disabled_by_zero(monkeypatch):
    monkeypatch.setenv("ANIMA_DAEMON_HOST_WEBUI", "0")
    assert webui_sidecar.is_enabled() is False


def test_resolve_port_default_and_override(monkeypatch):
    monkeypatch.delenv("ANIMA_WEBUI_PORT", raising=False)
    assert webui_sidecar.resolve_port() == 8000
    monkeypatch.setenv("ANIMA_WEBUI_PORT", "9000")
    assert webui_sidecar.resolve_port() == 9000
