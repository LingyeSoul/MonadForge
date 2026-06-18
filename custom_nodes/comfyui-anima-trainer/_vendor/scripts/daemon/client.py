"""HTTP client for the daemon — used by the CLI commands and the ComfyUI node.

Pure stdlib (``urllib``) so it imports cleanly from inside ComfyUI without
dragging in ``library.*`` / torch. ``ensure_daemon`` auto-starts a console-
detached daemon and waits for ``/health`` — the "spawn it if it isn't up" path
both the ComfyUI node and ``make daemon`` rely on.
"""

from __future__ import annotations

import json
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterator, Optional

from . import config, proc


def venv_python(*, windowless: bool = False) -> str:
    """Resolve the anima_lora venv interpreter.

    The daemon must run under anima's venv (it builds ``accelerate launch``
    commands with ``sys.executable``), *not* whatever interpreter the caller
    happens to be — notably ComfyUI's. Probe the usual venv layouts under the
    repo root and its parent, then fall back to ``sys.executable``.

    ``windowless=True`` (Windows only) prefers ``pythonw.exe``: it never
    allocates a console, so the long-lived daemon has *no* window to pop up or,
    crucially, to be closed — closing a console window sends CTRL_CLOSE_EVENT
    and kills the process, which is how the daemon was dying and stranding its
    pidfile. (The uv venv ``python.exe`` is a trampoline that re-launches the
    real interpreter, so ``CREATE_NO_WINDOW`` on it doesn't reliably suppress
    the child's console — ``pythonw`` sidesteps that entirely.)
    """
    if sys.platform == "win32":
        exe = "pythonw.exe" if windowless else "python.exe"
        for base in (config.ROOT, config.ROOT.parent):
            cand = base / ".venv" / "Scripts" / exe
            if cand.exists():
                return str(cand)
    else:
        for base in (config.ROOT, config.ROOT.parent):
            cand = base / ".venv" / "bin" / "python"
            if cand.exists():
                return str(cand)
    return sys.executable


def _resolve_port() -> int:
    info = proc.read_pidfile(config.discover_pidfile())
    if info and info.get("port"):
        return int(info["port"])
    return config.DEFAULT_PORT


def _norm_root(path: str | Path) -> str:
    return str(Path(path).resolve()).casefold()


def daemon_matches_root(health: Optional[dict], expected_root: str | Path) -> bool:
    """True iff a daemon health response belongs to ``expected_root``.

    New daemons report ``root`` directly in ``/health`` and pidfiles. For
    legacy same-checkout daemons, a local in-repo pidfile is accepted even when
    the health payload lacks ``root``. A rootless daemon discovered only through
    the per-user global pidfile is treated as unknown, because that is exactly
    how a GUI can accidentally attach to another checkout's daemon.
    """
    if not health:
        return False
    expected = _norm_root(expected_root)
    root = health.get("root")
    if root:
        return _norm_root(root) == expected

    pidfile = config.discover_pidfile()
    info = proc.read_pidfile(pidfile) or {}
    root = info.get("root")
    if root:
        return _norm_root(root) == expected

    try:
        return pidfile.resolve() == config.PIDFILE.resolve()
    except OSError:
        return False


def _root_mismatch_message(health: dict, expected_root: str | Path) -> str:
    actual = health.get("root") or "unknown checkout"
    return (
        "training daemon belongs to a different anima_lora checkout "
        f"({actual}); expected {Path(expected_root).resolve()}"
    )


def _has_live_jobs(client: "DaemonClient") -> bool:
    try:
        return any(
            (job.get("state") or "") in {"queued", "running"}
            for job in client.list_jobs()
        )
    except Exception:
        return True


class DaemonClient:
    def __init__(self, port: Optional[int] = None) -> None:
        self.port = port or _resolve_port()

    @property
    def base(self) -> str:
        return f"http://{config.HOST}:{self.port}"

    def _request(
        self,
        method: str,
        path: str,
        body: Optional[dict] = None,
        *,
        timeout: float = 30.0,
    ):
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(
            self.base + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"} if data else {},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
        return json.loads(raw) if raw else None

    def health(self, *, timeout: float = 3.0) -> Optional[dict]:
        # Fast-fail when nothing is listening: on Windows a connect to a closed
        # port isn't refused for ~2s (SYN retransmit), so a bare urlopen stalls
        # every "is the daemon up?" probe (the GUI makes many, on the UI thread).
        # A short raw connect bounds the daemon-down answer at 0.25s.
        try:
            with socket.create_connection((config.HOST, self.port), timeout=0.25):
                pass
        except OSError:
            return None
        try:
            return self._request("GET", "/health", timeout=timeout)
        except (urllib.error.URLError, OSError, ValueError):
            return None

    def submit(
        self,
        *,
        method: str,
        preset: str = "default",
        methods_subdir: Optional[str] = None,
        config_snapshot: Optional[dict] = None,
        config_file: Optional[str] = None,
        overrides: Optional[dict] = None,
        extra: Optional[list[str]] = None,
        start: Optional[bool] = None,
    ) -> dict:
        return self._request(
            "POST",
            "/jobs",
            {
                "method": method,
                "preset": preset,
                "methods_subdir": methods_subdir,
                "config_snapshot": config_snapshot or None,
                "config_file": config_file,
                "overrides": overrides or {},
                "extra": extra or [],
                "start": start,
            },
        )

    def submit_command(
        self,
        *,
        label: str,
        argv: list[str],
        extra_env: Optional[dict] = None,
        chain_train: Optional[dict] = None,
        config_snapshot: Optional[dict] = None,
        config_file: Optional[str] = None,
        start: Optional[bool] = None,
    ) -> dict:
        return self._request(
            "POST",
            "/jobs",
            {
                "kind": "command",
                "label": label,
                "argv": list(argv),
                "extra_env": extra_env or {},
                "chain_train": chain_train or None,
                "config_snapshot": config_snapshot or None,
                "config_file": config_file,
                "start": start,
            },
        )

    def start_queue(self) -> Optional[dict]:
        """Resume a paused queue — the worker launches queued jobs in order."""
        return self._request("POST", "/queue/start")

    def pause_queue(self) -> Optional[dict]:
        """Hold the queue — queued jobs wait until ``start_queue``."""
        return self._request("POST", "/queue/pause")

    def list_jobs(self) -> list:
        return self._request("GET", "/jobs") or []

    def get(self, job_id: str) -> dict:
        return self._request("GET", f"/jobs/{job_id}")

    def stop(self, job_id: Optional[str] = None) -> dict:
        # No job_id → daemon's "stop the running job" semantics. We resolve the
        # active job here so the URL stays RESTful.
        if job_id is None:
            health = self.health() or {}
            job_id = health.get("active_job")
            if not job_id:
                return {"error": "no active job"}
        return self._request("POST", f"/jobs/{job_id}/stop")

    def shutdown(self, *, kill_jobs: bool = True) -> Optional[dict]:
        try:
            return self._request("POST", "/shutdown", {"kill_jobs": kill_jobs})
        except (urllib.error.URLError, OSError, ValueError):
            return None

    def stream(self, path: str) -> Iterator[str]:
        """Yield ``data:`` payloads from an SSE endpoint until the socket drops."""
        req = urllib.request.Request(self.base + path, method="GET")
        with urllib.request.urlopen(req, timeout=None) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").rstrip("\n")
                if line.startswith("data: "):
                    yield line[len("data: ") :]

    def stream_events(self) -> Iterator[str]:
        return self.stream("/events")

    def stream_logs(self, job_id: str) -> Iterator[str]:
        return self.stream(f"/jobs/{job_id}/logs")


def ensure_daemon(
    *,
    timeout: float = 60.0,
    port: Optional[int] = None,
    expected_root: Optional[str | Path] = None,
) -> DaemonClient:
    """Return a client to a live daemon, starting one if needed.

    Idempotent: if ``/health`` answers we just return a client. Otherwise spawn
    ``python -m scripts.daemon`` detached (stdout → ``daemon.log``) and poll
    ``/health`` until it answers or ``timeout`` elapses.

    The daemon may bind a *different* port than requested if the preferred one
    is taken by a stranger (see ``server.serve_with_fallback``); it records the
    actual port in the pidfile, so we re-resolve from there each tick and follow
    it rather than polling a port nothing is listening on.

    The poll cadence ramps: a freshly-spawned daemon is usually answering in
    well under a second (the package imports in ~tens of ms; the dominant cost
    is the OS process spawn), so a flat 0.5s interval would idle past a daemon
    that's already up. We poll fast at first (0.1s) and back off toward 0.5s, so
    the common case returns as soon as the daemon binds without busy-spinning on
    a genuinely slow start.
    """
    requested = port or _resolve_port()
    client = DaemonClient(requested)
    health = client.health()
    if health is not None and expected_root is None:
        return client
    if health is not None and expected_root is not None:
        if daemon_matches_root(health, expected_root):
            return client
        if health.get("active_job") or _has_live_jobs(client):
            raise RuntimeError(
                f"{_root_mismatch_message(health, expected_root)}; "
                "it still has queued or running jobs"
            )
        client.shutdown(kill_jobs=False)
        deadline = time.time() + min(timeout, 5.0)
        while time.time() < deadline and client.health() is not None:
            time.sleep(0.2)

    config.ensure_state_dirs()
    proc.spawn_detached(
        # pythonw.exe → no console window whose close button would kill the
        # daemon and strand the pidfile. Logs still go to daemon.log below.
        [venv_python(windowless=True), "-m", "scripts.daemon", str(requested)],
        cwd=config.ROOT,
        stdout_path=config.DAEMON_LOG,
    )
    deadline = time.time() + timeout
    interval = 0.1
    while time.time() < deadline:
        resolved = _resolve_port()  # follow a fallback-to-ephemeral daemon
        if resolved != client.port:
            client = DaemonClient(resolved)
        health = client.health()
        if health is not None and expected_root is None:
            return client
        if health is not None and expected_root is not None:
            if daemon_matches_root(health, expected_root):
                return client
            if health.get("active_job") or _has_live_jobs(client):
                raise RuntimeError(
                    f"{_root_mismatch_message(health, expected_root)}; "
                    "it still has queued or running jobs"
                )
        time.sleep(interval)
        interval = min(interval * 1.5, 0.5)  # ramp 0.1 → 0.5s
    raise RuntimeError(
        f"daemon did not come up within {timeout:.0f}s; see {config.DAEMON_LOG}"
    )


def is_running() -> bool:
    return DaemonClient().health() is not None
