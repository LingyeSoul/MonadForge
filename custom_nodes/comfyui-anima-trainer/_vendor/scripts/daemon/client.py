"""HTTP client for the daemon — used by the CLI commands and the ComfyUI node.

Pure stdlib (``urllib``) so it imports cleanly from inside ComfyUI without
dragging in ``library.*`` / torch. ``ensure_daemon`` auto-starts a console-
detached daemon and waits for ``/health`` — the "spawn it if it isn't up" path
both the ComfyUI node and ``make daemon`` rely on.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
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


class DaemonClient:
    def __init__(self, port: Optional[int] = None) -> None:
        self.port = port or _resolve_port()

    @property
    def base(self) -> str:
        return f"http://{config.HOST}:{self.port}"

    # ----- request plumbing -----

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

    # ----- typed endpoints -----

    def health(self, *, timeout: float = 3.0) -> Optional[dict]:
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
        overrides: Optional[dict] = None,
        extra: Optional[list[str]] = None,
    ) -> dict:
        return self._request(
            "POST",
            "/jobs",
            {
                "method": method,
                "preset": preset,
                "methods_subdir": methods_subdir,
                "overrides": overrides or {},
                "extra": extra or [],
            },
        )

    def submit_command(
        self,
        *,
        label: str,
        argv: list[str],
        extra_env: Optional[dict] = None,
        chain_train: Optional[dict] = None,
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
            },
        )

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

    # ----- SSE streams -----

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


def ensure_daemon(*, timeout: float = 60.0, port: Optional[int] = None) -> DaemonClient:
    """Return a client to a live daemon, starting one if needed.

    Idempotent: if ``/health`` answers we just return a client. Otherwise spawn
    ``python -m scripts.daemon`` detached (stdout → ``daemon.log``) and poll
    ``/health`` until it answers or ``timeout`` elapses.

    The daemon may bind a *different* port than requested if the preferred one
    is taken by a stranger (see ``server.serve_with_fallback``); it records the
    actual port in the pidfile, so we re-resolve from there each tick and follow
    it rather than polling a port nothing is listening on.
    """
    requested = port or _resolve_port()
    client = DaemonClient(requested)
    if client.health() is not None:
        return client

    config.ensure_state_dirs()
    proc.spawn_detached(
        # pythonw.exe → no console at all: nothing to clutter the screen, and
        # (the real fix) no window whose close button kills the daemon and
        # strands the pidfile, which made every later `make gui` spawn a fresh
        # one. Logs still go to daemon.log via the stdout redirect below.
        [venv_python(windowless=True), "-m", "scripts.daemon", str(requested)],
        cwd=config.ROOT,
        stdout_path=config.DAEMON_LOG,
    )
    deadline = time.time() + timeout
    while time.time() < deadline:
        resolved = _resolve_port()  # follow a fallback-to-ephemeral daemon
        if resolved != client.port:
            client = DaemonClient(resolved)
        if client.health() is not None:
            return client
        time.sleep(0.5)
    raise RuntimeError(
        f"daemon did not come up within {timeout:.0f}s; see {config.DAEMON_LOG}"
    )


def is_running() -> bool:
    return DaemonClient().health() is not None
