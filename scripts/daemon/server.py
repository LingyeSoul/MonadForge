"""Stdlib HTTP surface for the daemon — zero new deps, localhost only.

A hand-written ``(method, path)`` dispatch on a ``BaseHTTPRequestHandler``;
request bodies are plain ``json.loads``'d dicts (no Pydantic — the only callers
are trusted localhost clients: the ComfyUI node, an attached terminal, the MCP
server). Served by ``ThreadingHTTPServer`` so a parked SSE stream just holds one
blocked thread.

Endpoints
    POST /jobs              {method, preset, methods_subdir, overrides, extra} → {job_id}
                            or {kind:"command", label, argv, extra_env,
                                 chain_train?}                                 → {job_id}
    GET  /jobs              → [job, …]
    GET  /jobs/{id}         → job (+ latest progress event, stale_for)
    POST /jobs/{id}/stop    → {job}
    GET  /jobs/{id}/logs    → SSE: tail of the job's stdout.log
    GET  /events            → SSE: daemon-level lifecycle events
    GET  /health            → {ok, pid, port, active_job}
    POST /shutdown          {kill_jobs} → {ok}
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import config, tail
from .manager import JobManager

logger = logging.getLogger("anima.daemon")

_JOB_RE = re.compile(r"^/jobs/(?P<id>[^/]+)$")
_JOB_STOP_RE = re.compile(r"^/jobs/(?P<id>[^/]+)/stop$")
_JOB_LOGS_RE = re.compile(r"^/jobs/(?P<id>[^/]+)/logs$")


class _Handler(BaseHTTPRequestHandler):
    server_version = "AnimaDaemon/1.0"
    protocol_version = "HTTP/1.1"

    @property
    def manager(self) -> JobManager:
        return self.server.manager  # type: ignore[attr-defined]

    # ----- low-level write helpers -----

    def _send_json(self, obj, status: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw or b"{}")
        except ValueError:
            return {}
        return data if isinstance(data, dict) else {}

    def _open_sse(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

    def _sse(self, obj) -> bool:
        """Write one SSE event and flush. Returns False on a dropped client —
        ``wfile`` buffers, so without the flush the client sees nothing until
        the buffer fills."""
        try:
            payload = obj if isinstance(obj, str) else json.dumps(obj)
            self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError, OSError):
            return False

    def log_message(self, fmt, *args) -> None:  # quieter than default stderr spam
        logger.debug("http: " + fmt, *args)

    # ----- routing -----

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/health":
            self._handle_health()
        elif path == "/jobs":
            self._handle_list()
        elif path == "/events":
            self._handle_events()
        elif m := _JOB_LOGS_RE.match(path):
            self._handle_logs(m.group("id"))
        elif m := _JOB_RE.match(path):
            self._handle_get(m.group("id"))
        else:
            self._send_json({"error": "not found", "path": path}, 404)

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/jobs":
            self._handle_submit()
        elif path == "/shutdown":
            self._handle_shutdown()
        elif m := _JOB_STOP_RE.match(path):
            self._handle_stop(m.group("id"))
        else:
            self._send_json({"error": "not found", "path": path}, 404)

    # ----- handlers -----

    def _handle_health(self) -> None:
        active = self.manager.active_job()
        self._send_json(
            {
                "ok": True,
                "pid": os.getpid(),
                "port": self.server.server_address[1],
                "active_job": active.id if active else None,
            }
        )

    def _handle_submit(self) -> None:
        body = self._read_json()
        if (body.get("kind") or "train") == "command":
            argv = body.get("argv")
            if not isinstance(argv, list) or not argv:
                self._send_json({"error": "missing 'argv' for command job"}, 400)
                return
            job = self.manager.submit_command(
                label=body.get("label") or "command",
                argv=[str(a) for a in argv],
                extra_env=body.get("extra_env") or {},
                chain_train=body.get("chain_train") or None,
            )
            self._send_json({"job_id": job.id, "state": job.state}, 201)
            return
        method = body.get("method")
        if not method:
            self._send_json({"error": "missing 'method'"}, 400)
            return
        job = self.manager.submit(
            method=method,
            preset=body.get("preset") or "default",
            methods_subdir=body.get("methods_subdir"),
            overrides=body.get("overrides") or {},
            extra=body.get("extra") or [],
        )
        self._send_json({"job_id": job.id, "state": job.state}, 201)

    def _handle_list(self) -> None:
        self._send_json([j.public() for j in self.manager.list_jobs()])

    def _handle_get(self, job_id: str) -> None:
        job = self.manager.get(job_id)
        if job is None:
            self._send_json({"error": "no such job", "job_id": job_id}, 404)
            return
        out = job.public()
        out["latest"] = tail.last_event(job.progress_path)
        out["stale_for"] = self.manager.stale_for(job)
        self._send_json(out)

    def _handle_stop(self, job_id: str) -> None:
        job = self.manager.stop(job_id)
        if job is None:
            self._send_json({"error": "no such job", "job_id": job_id}, 404)
            return
        self._send_json({"job_id": job.id, "state": job.state})

    def _handle_shutdown(self) -> None:
        body = self._read_json()
        kill = bool(body.get("kill_jobs", True))
        self._send_json({"ok": True, "kill_jobs": kill})
        # Trigger shutdown after the response is flushed, off the handler thread
        # (server.shutdown() must not run in a request thread).
        threading.Thread(
            target=self.server.request_shutdown,  # type: ignore[attr-defined]
            args=(kill,),
            daemon=True,
        ).start()

    def _handle_events(self) -> None:
        q = self.manager.subscribe()
        self._open_sse()
        try:
            if not self._sse({"ev": "hello", "ts": time.time()}):
                return
            while True:
                try:
                    event = q.get(timeout=15)
                except Exception:
                    # idle keepalive comment so proxies/clients don't time out
                    try:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        return
                    continue
                if not self._sse(event):
                    return
        finally:
            self.manager.unsubscribe(q)

    def _handle_logs(self, job_id: str) -> None:
        job = self.manager.get(job_id)
        if job is None:
            self._send_json({"error": "no such job", "job_id": job_id}, 404)
            return
        self._open_sse()
        path = Path(job.stdout_path) if job.stdout_path else None
        if path is None:
            self._sse({"error": "no stdout for job"})
            return
        for line in tail.follow(path, from_start=True):
            if line:
                if not self._sse(line.rstrip("\n")):
                    return
            else:
                # heartbeat tick: stop once the job is terminal and drained.
                cur = self.manager.get(job_id)
                if cur is not None and cur.state not in ("queued", "running"):
                    # one more pass to flush any final lines already on disk
                    self._sse({"ev": "eof", "state": cur.state})
                    return


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, manager: JobManager):
        super().__init__(addr, _Handler)
        self.manager = manager

    def request_shutdown(self, kill_jobs: bool) -> None:
        self.manager.shutdown(kill_jobs=kill_jobs)
        self.shutdown()  # unblocks serve_forever()


def serve(manager: JobManager, *, port: int) -> _Server:
    """Bind 127.0.0.1:port and return the server (call ``serve_forever``)."""
    return _Server((config.HOST, port), manager)


def serve_with_fallback(manager: JobManager, *, port: int) -> _Server:
    """Bind ``port``; if it's already taken, fall back to an OS-chosen free one.

    The catch: don't blindly grab a new port on every collision, or a startup
    race (GUI auto-start + ``make daemon`` firing together) would spin up a
    *second* daemon that overwrites the pidfile — breaking the single-daemon
    invariant. So on ``EADDRINUSE`` we first probe the port: if an anima daemon
    already answers ``/health`` there (a sibling that won the race), we re-raise
    so the caller exits and defers to it. Only when a *stranger* holds the port
    do we move to an ephemeral one (the actual port is recorded in the pidfile,
    and ``ensure_daemon`` re-resolves it from there)."""
    try:
        return _Server((config.HOST, port), manager)
    except OSError:
        from .client import DaemonClient

        # A sibling may have bound the socket microseconds ago but not yet
        # reached serve_forever; probe a few times (short timeout) to be sure.
        for _ in range(3):
            if DaemonClient(port).health(timeout=0.5) is not None:
                raise  # an anima daemon owns it → let the caller stand down
            time.sleep(0.3)
        logger.warning(
            "127.0.0.1:%s held by a non-anima process; using an ephemeral port",
            port,
        )
        return _Server((config.HOST, 0), manager)
