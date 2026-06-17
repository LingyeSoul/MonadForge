"""Supervised WebUI sidecar — the daemon owns the uvicorn server process.

The WebUI server is a *long-running* process that must not participate in the
daemon's serial job queue (a never-exiting process would block every training
job and trip the stall watchdog). Instead the daemon spawns it as a supervised
**sidecar**: a detached subprocess on its own monitor thread, respawned on
unexpected exit (with backoff + a give-up cap so a port conflict or import
error can't loop forever), and tree-killed when the daemon shuts down.

The sidecar is deliberately **not** a ``Job`` — it never enters
``JobManager._jobs``, so ``active_job`` / the serial queue / the stall watchdog
all keep reflecting only training. Status (pid/port/url) is mirrored to
``output/daemon/webui.json`` so the tray / ``daemon-status`` / a future health
endpoint can report whether the WebUI is up.

Lifecycle:
    sidecar = WebUISidecar(port=8000)
    sidecar.start()      # spawn + launch monitor thread (returns immediately)
    ... daemon runs ...
    sidecar.stop()       # signal monitor + tree-kill the child
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Optional

from . import config, proc

logger = logging.getLogger(__name__)

# Respawn backoff: doubles each crash, capped here, gives up after this many
# consecutive fast failures (a healthy server rarely exits; a server that dies
# 5× in a row within the backoff window is almost certainly misconfigured — a
# held port, a missing model, a bad import — and looping only floods the log).
_MAX_BACKOFF = 30.0
_GIVE_UP_AFTER = 5
# A child that runs longer than this before exiting counts as "was healthy" →
# resets the crash counter (a slow leak that crashes once an hour shouldn't
# eventually trip the give-up cap).
_HEALTHY_UPTIME = 30.0


class WebUISidecar:
    """Owns the WebUI subprocess + a monitor thread that respawns it."""

    def __init__(self, *, port: int = 8000, host: str = "127.0.0.1") -> None:
        self._port = port
        self._host = host
        self._popen = None  # type: ignore[assignment]
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_requested = threading.Event()
        self._lock = threading.Lock()  # guards _popen across spawn/kill
        self._pidfile = config.STATE_DIR / "webui.json"

    # ── spawn ──────────────────────────────────────────────────────────

    def _build_cmd(self) -> list[str]:
        from .client import venv_python

        # ``-m webui`` launches uvicorn; the webui honors ANIMA_DAEMON_HOST_WEBUI
        # to skip its own ensure_daemon() (we're already its parent).
        py = venv_python(windowless=True)
        return [py, "-m", "webui", "--host", self._host, "--port", str(self._port)]

    def _env(self) -> dict:
        env = os.environ.copy()
        # Tell the child it was launched by the daemon so it doesn't try to
        # re-spawn one (which would race / no-op).
        env["ANIMA_DAEMON_HOST_WEBUI"] = "1"
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        return env

    def _spawn(self) -> None:
        """Launch (or re-launch) the WebUI subprocess."""
        stdout_path = config.STATE_DIR / "webui.log"
        cmd = self._build_cmd()
        logger.info("starting WebUI sidecar: %s", " ".join(cmd))
        popen = proc.spawn_detached(
            cmd,
            cwd=config.ROOT,
            stdout_path=stdout_path,
            env=self._env(),
        )
        with self._lock:
            self._popen = popen
        self._write_pidfile(popen.pid)

    # ── monitor ────────────────────────────────────────────────────────

    def _monitor(self) -> None:
        """Watch the child; respawn on unexpected exit, stop when asked."""
        crashes = 0
        backoff = 1.0
        while not self._stop_requested.is_set():
            with self._lock:
                popen = self._popen
            if popen is None:
                # stop() cleared it — we're tearing down.
                break
            started = time.time()
            # Block until the child exits. poll() returns None while running.
            rc = popen.wait()
            if self._stop_requested.is_set():
                logger.info("WebUI sidecar stopped (rc=%s)", rc)
                break
            uptime = time.time() - started
            # A run that lasted a while was healthy — reset the crash budget.
            if uptime >= _HEALTHY_UPTIME:
                crashes = 0
                backoff = 1.0
            crashes += 1
            logger.warning(
                "WebUI sidecar exited (rc=%s, uptime=%.1fs); crash #%d",
                rc,
                uptime,
                crashes,
            )
            if crashes > _GIVE_UP_AFTER:
                logger.error(
                    "WebUI sidecar gave up after %d consecutive crashes — "
                    "see output/daemon/webui.log. Not retrying.",
                    crashes,
                )
                self._clear_pidfile()
                break
            # Backoff before respawn (capped); wake early if stop() is called.
            if self._stop_requested.wait(backoff):
                break
            backoff = min(backoff * 2, _MAX_BACKOFF)
            try:
                self._spawn()
            except Exception:  # noqa: BLE001 — a spawn failure shouldn't kill the daemon
                logger.exception("WebUI sidecar respawn failed")

    # ── public lifecycle ───────────────────────────────────────────────

    def start(self) -> None:
        """Spawn the WebUI + start the monitor thread (non-blocking)."""
        self._stop_requested.clear()
        try:
            self._spawn()
        except Exception:  # noqa: BLE001 — boot must not crash the daemon
            logger.exception("WebUI sidecar failed to start")
            return
        self._monitor_thread = threading.Thread(
            target=self._monitor, name="mf-webui-sidecar", daemon=True
        )
        self._monitor_thread.start()

    def stop(self) -> None:
        """Signal the monitor + tree-kill the child. Idempotent."""
        self._stop_requested.set()
        with self._lock:
            popen = self._popen
            self._popen = None
        if popen is not None and popen.poll() is None:
            try:
                proc.kill_tree(popen.pid)
            except Exception:  # noqa: BLE001 — best-effort during shutdown
                logger.warning("could not kill WebUI sidecar tree", exc_info=True)
        self._clear_pidfile()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=5.0)

    # ── pidfile ────────────────────────────────────────────────────────

    def _write_pidfile(self, pid: int) -> None:
        record = {
            "pid": pid,
            "port": self._port,
            "host": self._host,
            "url": f"http://{self._host}:{self._port}",
        }
        try:
            config.STATE_DIR.mkdir(parents=True, exist_ok=True)
            tmp = self._pidfile.with_name(self._pidfile.name + ".tmp")
            tmp.write_text(json.dumps(record, indent=2), encoding="utf-8")
            os.replace(tmp, self._pidfile)
        except OSError as exc:
            logger.warning("could not write webui pidfile (%s)", exc)

    def _clear_pidfile(self) -> None:
        try:
            self._pidfile.unlink()
        except OSError:
            pass


def is_enabled() -> bool:
    """Whether the daemon should host the WebUI (env-toggleable, default on)."""
    return os.environ.get("ANIMA_DAEMON_HOST_WEBUI", "1") not in ("0", "", "false")


def resolve_port() -> int:
    """WebUI port from env (default 8000)."""
    return int(os.environ.get("ANIMA_WEBUI_PORT", "8000"))
