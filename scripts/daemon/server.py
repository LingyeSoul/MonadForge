"""Stdlib HTTP surface for the daemon — zero new deps, localhost only.

A hand-written ``(method, path)`` dispatch on a ``BaseHTTPRequestHandler``;
request bodies are plain ``json.loads``'d dicts (no Pydantic — the only callers
are trusted localhost clients: the ComfyUI node, an attached terminal, the MCP
server). Served by ``ThreadingHTTPServer`` so a parked SSE stream just holds one
blocked thread.

Endpoints
    GET  /                  → README.md (self-description for agentic callers)
    GET  /tools             → [tool, …]  machine-readable manifest (JSON-Schema)
    POST /jobs              {method, preset, methods_subdir, overrides, extra} → {job_id}
                            or {kind:"command", label, argv, extra_env,
                                 chain_train?}                                 → {job_id}
    GET  /jobs              → [job, …] or {jobs, total, offset, limit} with query filters
    GET  /jobs/{id}         → job (+ latest progress event, stale_for)
    GET  /jobs/{id}/progress → filtered progress.jsonl events
                            ?events=step,val&since_step=N&every_nth=N&last_n=N
    POST /jobs/{id}/stop    → {job}; training stops cooperatively before timeout
    POST /queue/start       → {ok, paused:false}  (resume a paused queue)
    POST /queue/pause       → {ok, paused:true}   (hold queued jobs)
    GET  /jobs/{id}/logs    → SSE: tail of the job's stdout.log
    GET  /events            → SSE: daemon-level lifecycle events
    GET  /health            → {ok, pid, port, root, active_job, paused, worker_alive, worker_idle_for}
    DELETE /jobs/{id}       → delete terminal job metadata only
    POST /shutdown          {mode: detach|cooperative-stop|force} → {ok}
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import config, tail
from .manager import JobManager
from .jobs import TERMINAL_STATES

logger = logging.getLogger("anima.daemon")

_JOB_RE = re.compile(r"^/jobs/(?P<id>[^/]+)$")
_JOB_STOP_RE = re.compile(r"^/jobs/(?P<id>[^/]+)/stop$")
_JOB_RESUME_RE = re.compile(r"^/jobs/(?P<id>[^/]+)/resume$")
_JOB_DELETE_RE = re.compile(r"^/jobs/(?P<id>[^/]+)$")
_JOB_LOGS_RE = re.compile(r"^/jobs/(?P<id>[^/]+)/logs$")
_JOB_PROGRESS_RE = re.compile(r"^/jobs/(?P<id>[^/]+)/progress$")
_JOB_GROUP_RE = re.compile(r"^/job-groups/(?P<id>[^/]+)$")

_README = Path(__file__).resolve().parent / "README.md"

# Machine-readable self-description served at GET /tools — one entry per
# operation, JSON-Schema ``input_schema`` so a thin MCP bridge (or any LLM tool
# loop) can register these directly. Each tool names the underlying HTTP
# ``method`` + ``path`` so a caller can hit the endpoint itself. Kept in sync
# with the handlers below by hand; it is small and rarely changes. The prose
# walkthrough is GET / (README.md).
TOOLS = [
    {
        "name": "submit_training",
        "description": (
            "Enqueue a train.py run built from method + preset + overrides. "
            "Runs when it reaches the front of the serial queue. Returns {job_id, state}."
        ),
        "method": "POST",
        "path": "/jobs",
        "input_schema": {
            "type": "object",
            "required": ["method"],
            "properties": {
                "method": {
                    "type": "string",
                    "description": "Method/adapter config name (e.g. 'lora', 'chimera', 'easycontrol').",
                },
                "preset": {
                    "type": "string",
                    "default": "default",
                    "description": "Hardware preset: default | fast_16gb | low_vram | half.",
                },
                "methods_subdir": {
                    "type": "string",
                    "description": "Config subdir, e.g. 'gui-methods' for the clean per-variant tree. Omit for 'methods'.",
                },
                "overrides": {
                    "type": "object",
                    "description": '{key: value} → --key value CLI overrides, e.g. {"network_dim": 32, "max_train_epochs": 64}.',
                },
                "extra": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Extra CLI args appended verbatim.",
                },
                "config_snapshot": {
                    "type": "object",
                    "description": "A fully-merged config dict to pin instead of re-resolving the base→preset→method→overrides chain at launch.",
                },
                "config_file": {
                    "type": "string",
                    "description": "Path to a config snapshot file (alternative to config_snapshot).",
                },
                "start": {
                    "type": "boolean",
                    "description": "true → run now (resume queue); false → add to queue (held for Start Queue only if the queue is idle; auto-advances behind a job that's already running); omit → leave the gate as-is.",
                },
            },
        },
    },
    {
        "name": "submit_command",
        "description": (
            "Enqueue a plain `python <argv>` task (preprocess, mask, a distill loop). "
            "Optionally carries a chain_train spec the daemon auto-enqueues on success "
            "(how 'preprocess → train' survives the caller closing). Returns {job_id, state}."
        ),
        "method": "POST",
        "path": "/jobs",
        "input_schema": {
            "type": "object",
            "required": ["label", "argv"],
            "properties": {
                "kind": {
                    "type": "string",
                    "const": "command",
                    "description": "Must be 'command'.",
                },
                "label": {
                    "type": "string",
                    "description": "Display label (doubles as the job's 'method' field).",
                },
                "argv": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Argv run under the anima venv, e.g. ['tasks.py', 'preprocess-config', …].",
                },
                "extra_env": {
                    "type": "object",
                    "description": "Extra environment variables for the subprocess.",
                },
                "chain_train": {
                    "type": "object",
                    "description": "Training spec {method, preset, methods_subdir, overrides} auto-enqueued when this command finishes successfully.",
                },
                "config_snapshot": {
                    "type": "object",
                    "description": "Config dict to pin (forwarded to the chained train job).",
                },
                "config_file": {
                    "type": "string",
                    "description": "Config snapshot file path (alternative to config_snapshot).",
                },
                "start": {
                    "type": "boolean",
                    "description": "true → run now; false → add to queue (held only if the queue is idle; auto-advances behind a running job); omit → leave gate as-is.",
                },
            },
        },
    },
    {
        "name": "list_jobs",
        "description": "List historical and active jobs. Supports state/status, resumable, before/after time filters and offset/limit pagination.",
        "method": "GET",
        "path": "/jobs",
        "input_schema": {"type": "object", "properties": {
            "state": {"type": "string"}, "resumable": {"type": "boolean"},
            "offset": {"type": "integer"}, "limit": {"type": "integer"}
        }},
    },
    {
        "name": "get_job",
        "description": (
            "Fetch one job record plus live fields: 'latest' (last progress.jsonl event, "
            "train jobs only) and 'stale_for' (seconds since last progress tick). Poll this for completion."
        ),
        "method": "GET",
        "path": "/jobs/{id}",
        "input_schema": {
            "type": "object",
            "required": ["id"],
            "properties": {
                "id": {"type": "string", "description": "Job id from submit_*."}
            },
        },
    },
    {
        "name": "get_progress",
        "description": (
            "Query a train job's structured progress.jsonl — events are "
            "run_start | step (loss/lr/metrics) | val (cmmd) | ckpt | log "
            "(mirrored WARNING+ records) | run_end. Filters keep the payload "
            "small on long runs; returns {job_id, state, count, events}. "
            "This is the surface to debug/analyze a run from (loss curve, "
            "warnings, checkpoints) — stdout is only for raw crash output."
        ),
        "method": "GET",
        "path": "/jobs/{id}/progress",
        "input_schema": {
            "type": "object",
            "required": ["id"],
            "properties": {
                "id": {"type": "string", "description": "Job id from submit_*."},
                "events": {
                    "type": "string",
                    "description": "Comma-separated ev kinds to keep, e.g. 'step,val' or 'log,run_end'. Omit for all.",
                },
                "since_step": {
                    "type": "integer",
                    "description": "Keep events at/after this global_step (step-less events inherit the preceding step).",
                },
                "every_nth": {
                    "type": "integer",
                    "description": "Thin step events to every n-th (latest step always kept).",
                },
                "last_n": {
                    "type": "integer",
                    "default": 200,
                    "description": "Trailing cap on returned events.",
                },
            },
        },
    },
    {
        "name": "stop_job",
        "description": "Cancel a running or queued job. Training first saves cooperatively and is tree-killed only after the grace deadline. Returns {job_id, state}.",
        "method": "POST",
        "path": "/jobs/{id}/stop",
        "input_schema": {
            "type": "object",
            "required": ["id"],
            "properties": {"id": {"type": "string", "description": "Job id to stop."}},
        },
    },
    {
        "name": "resume_job",
        "description": "Create a new training job from the newest complete signature-matched state of a stopped/error/orphaned job.",
        "method": "POST",
        "path": "/jobs/{id}/resume",
        "input_schema": {"type": "object", "required": ["id"], "properties": {"id": {"type": "string"}}},
    },
    {
        "name": "delete_job_history",
        "description": "Delete terminal job metadata/logs/progress/sample only. Model weights and training state directories are preserved.",
        "method": "DELETE",
        "path": "/jobs/{id}",
        "input_schema": {"type": "object", "required": ["id"], "properties": {"id": {"type": "string"}}},
    },
    {
        "name": "tail_logs",
        "description": (
            "SSE stream of a job's combined stdout+stderr from the start of the file; "
            "ends with a {ev:'eof', state} event once the job is terminal and drained. "
            "There is no blocking 'wait' call — tail this or poll get_job."
        ),
        "method": "GET",
        "path": "/jobs/{id}/logs",
        "input_schema": {
            "type": "object",
            "required": ["id"],
            "properties": {"id": {"type": "string", "description": "Job id to tail."}},
        },
    },
    {
        "name": "pause_queue",
        "description": "Hold the queue gate — queued jobs wait until start_queue. Submissions still accepted.",
        "method": "POST",
        "path": "/queue/pause",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "start_queue",
        "description": "Resume a paused queue — the worker launches queued jobs in order.",
        "method": "POST",
        "path": "/queue/start",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "health",
        "description": "Daemon liveness: {ok, pid, port, root, active_job, paused, worker_alive, worker_idle_for}. 'root' is the checkout it belongs to; worker_idle_for is seconds since the job worker last advanced.",
        "method": "GET",
        "path": "/health",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "shutdown",
        "description": "Stop the daemon. Destructive: kill_jobs=true (default) tree-kills the running job too.",
        "method": "POST",
        "path": "/shutdown",
        "input_schema": {
            "type": "object",
            "properties": {
                "kill_jobs": {
                    "type": "boolean",
                    "default": True,
                    "description": "Kill the running job on the way down.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["detach", "cooperative-stop", "force"],
                    "description": "Detach preserves live/queued jobs for reattach; cooperative-stop waits for a checkpoint; force tree-kills explicitly."
                },
            },
        },
    },
]


class _Handler(BaseHTTPRequestHandler):
    server_version = "AnimaDaemon/1.0"
    protocol_version = "HTTP/1.1"

    @property
    def manager(self) -> JobManager:
        return self.server.manager  # type: ignore[attr-defined]

    def _send_json(self, obj, status: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, text: str, *, content_type: str, status: int = 200) -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
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

    def do_GET(self) -> None:  # noqa: N802
        path, _, query = self.path.partition("?")
        if path in ("/", "/readme"):
            self._handle_readme()
        elif path == "/tools":
            self._send_json(TOOLS)
        elif path == "/health":
            self._handle_health()
        elif path == "/jobs":
            self._handle_list(query)
        elif path == "/job-groups":
            self._handle_group_list(query)
        elif m := _JOB_GROUP_RE.match(path):
            self._handle_group_get(m.group("id"))
        elif path == "/events":
            self._handle_events()
        elif m := _JOB_LOGS_RE.match(path):
            self._handle_logs(m.group("id"))
        elif m := _JOB_PROGRESS_RE.match(path):
            self._handle_progress(m.group("id"), query)
        elif m := _JOB_RE.match(path):
            self._handle_get(m.group("id"))
        else:
            self._send_json({"error": "not found", "path": path}, 404)

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/jobs":
            self._handle_submit()
        elif path == "/queue/start":
            self.manager.resume()
            self._send_json({"ok": True, "paused": False})
        elif path == "/queue/pause":
            self.manager.pause()
            self._send_json({"ok": True, "paused": True})
        elif path == "/shutdown":
            self._handle_shutdown()
        elif m := _JOB_RESUME_RE.match(path):
            self._handle_resume(m.group("id"))
        elif m := _JOB_STOP_RE.match(path):
            self._handle_stop(m.group("id"))
        else:
            self._send_json({"error": "not found", "path": path}, 404)

    def do_DELETE(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if m := _JOB_GROUP_RE.match(path):
            group = self.manager.job_group(m.group("id"))
            if group is None:
                self._send_json({"error": "no such job group", "root_job_id": m.group("id")}, 404)
            elif any(attempt.get("state") not in TERMINAL_STATES for attempt in group["attempts"]):
                self._send_json({"error": "active job groups cannot be deleted", "root_job_id": group["root_job_id"]}, 409)
            else:
                try:
                    deleted = self.manager.delete_group(m.group("id"))
                except OSError as exc:
                    self._send_json({"error": f"could not delete job group: {exc}", "root_job_id": group["root_job_id"]}, 500)
                else:
                    self._send_json({"ok": True, "root_job_id": group["root_job_id"], "job_ids": deleted or []})
            return
        if m := _JOB_DELETE_RE.match(path):
            job = self.manager.get(m.group("id"))
            if job is None:
                self._send_json({"error": "no such job", "job_id": m.group("id")}, 404)
            elif job.state not in TERMINAL_STATES:
                self._send_json({"error": "running jobs cannot be deleted", "job_id": job.id}, 409)
            else:
                try:
                    self.manager.delete(job.id)
                except OSError as exc:
                    self._send_json({"error": f"could not delete job: {exc}", "job_id": job.id}, 500)
                else:
                    self._send_json({"ok": True, "job_id": job.id})
            return
        self._send_json({"error": "not found", "path": path}, 404)

    def _handle_readme(self) -> None:
        try:
            text = _README.read_text(encoding="utf-8")
        except OSError:
            # README not shipped alongside (e.g. a trimmed vendor tree) — point
            # the caller at the machine-readable manifest instead.
            self._send_json({"error": "README.md not found", "tools": "/tools"}, 404)
            return
        self._send_text(text, content_type="text/markdown; charset=utf-8")

    def _handle_health(self) -> None:
        active = self.manager.active_job()
        self._send_json(
            {
                "ok": True,
                "pid": os.getpid(),
                "port": self.server.server_address[1],
                "root": str(config.ROOT),
                "active_job": active.id if active else None,
                "paused": self.manager.is_paused(),
                "worker_alive": self.manager.worker_alive(),
                "worker_idle_for": self.manager.worker_idle_for(),
            }
        )

    def _handle_submit(self) -> None:
        body = self._read_json()
        # start: True → run now (resume queue), False → enqueue paused, None → as-is.
        start = body.get("start")
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
                config_snapshot=body.get("config_snapshot") or None,
                config_file=body.get("config_file") or None,
                start=start,
            )
            self._send_json(
                {"job_id": job.id, "state": job.state, "sample_dir": job.sample_dir},
                201,
            )
            return
        method = body.get("method")
        if not method:
            self._send_json({"error": "missing 'method'"}, 400)
            return
        job = self.manager.submit(
            method=method,
            preset=body.get("preset") or "default",
            methods_subdir=body.get("methods_subdir"),
            config_snapshot=body.get("config_snapshot") or None,
            config_file=body.get("config_file") or None,
            overrides=body.get("overrides") or {},
            extra=body.get("extra") or [],
            start=start,
        )
        self._send_json(
            {"job_id": job.id, "state": job.state, "sample_dir": job.sample_dir},
            201,
        )

    def _handle_list(self, query: str = "") -> None:
        params = urllib.parse.parse_qs(query)
        state = (params.get("state") or params.get("status") or [None])[0]
        raw_resume = (params.get("resumable") or [None])[0]
        resumable = None if raw_resume is None else raw_resume.lower() in {"1", "true", "yes"}
        def integer(name, default):
            try: return int((params.get(name) or [default])[0])
            except (TypeError, ValueError): return default
        def number(name):
            try: return float((params.get(name) or [None])[0])
            except (TypeError, ValueError): return None
        order = (params.get("order") or ["asc"])[0].lower()
        jobs, total = self.manager.list_jobs_filtered(state=state, resumable=resumable,
                                                       before=number("before") if number("before") is not None else number("to"),
                                                       after=number("after") if number("after") is not None else number("from"),
                                                       offset=integer("offset", 0), limit=integer("limit", 100),
                                                       newest_first=order in {"desc", "newest"})
        if not query:
            self._send_json([j.public() for j in jobs])
        else:
            self._send_json({"jobs": [j.public() for j in jobs], "total": total,
                             "offset": integer("offset", 0), "limit": integer("limit", 100)})

    def _handle_group_list(self, query: str = "") -> None:
        params = urllib.parse.parse_qs(query)
        state = (params.get("state") or params.get("status") or [None])[0]
        raw_resume = (params.get("resumable") or [None])[0]
        resumable = None if raw_resume is None else raw_resume.lower() in {"1", "true", "yes"}

        def integer(name, default):
            try:
                return int((params.get(name) or [default])[0])
            except (TypeError, ValueError):
                return default

        def number(name):
            try:
                return float((params.get(name) or [None])[0])
            except (TypeError, ValueError):
                return None

        order = (params.get("order") or ["asc"])[0].lower()
        before = number("before")
        after = number("after")
        groups, total = self.manager.list_job_groups_filtered(
            state=state,
            resumable=resumable,
            before=before if before is not None else number("to"),
            after=after if after is not None else number("from"),
            offset=integer("offset", 0),
            limit=integer("limit", 100),
            newest_first=order in {"desc", "newest"},
        )
        self._send_json(
            {
                "groups": groups,
                "total": total,
                "offset": integer("offset", 0),
                "limit": integer("limit", 100),
            }
        )

    def _handle_group_get(self, job_id: str) -> None:
        group = self.manager.job_group(job_id)
        if group is None:
            self._send_json({"error": "no such job group", "root_job_id": job_id}, 404)
            return
        self._send_json(group)

    def _handle_get(self, job_id: str) -> None:
        job = self.manager.get(job_id)
        if job is None:
            self._send_json({"error": "no such job", "job_id": job_id}, 404)
            return
        out = job.public()
        out["latest"] = tail.last_event(job.progress_path)
        out["stale_for"] = self.manager.stale_for(job)
        self._send_json(out)

    def _handle_progress(self, job_id: str, query: str) -> None:
        job = self.manager.get(job_id)
        if job is None:
            self._send_json({"error": "no such job", "job_id": job_id}, 404)
            return
        params = urllib.parse.parse_qs(query)

        def _int(name: str, default=None):
            raw = (params.get(name) or [None])[0]
            try:
                return int(raw) if raw is not None else default
            except ValueError:
                return default

        raw_events = (params.get("events") or [None])[0]
        kinds = (
            [s.strip() for s in raw_events.split(",") if s.strip()]
            if raw_events
            else None
        )
        evs = tail.read_events(
            job.progress_path,
            events=kinds,
            since_step=_int("since_step"),
            every_nth=_int("every_nth"),
            last_n=_int("last_n", 200),
        )
        self._send_json(
            {
                "job_id": job.id,
                "state": job.state,
                "progress_path": job.progress_path,
                "count": len(evs),
                "events": evs,
            }
        )

    def _handle_stop(self, job_id: str) -> None:
        job = self.manager.stop(job_id)
        if job is None:
            self._send_json({"error": "no such job", "job_id": job_id}, 404)
            return
        self._send_json({"job_id": job.id, "state": job.state})

    def _handle_resume(self, job_id: str) -> None:
        try:
            job = self.manager.resume_job(job_id)
        except ValueError as exc:
            self._send_json({"error": str(exc), "job_id": job_id}, 409)
            return
        if job is None:
            self._send_json({"error": "job is not a resumable training terminal", "job_id": job_id}, 409)
            return
        self._send_json({"job_id": job.id, "source_job_id": job_id, "root_job_id": job.root_job_id or job.id,
                         "parent_job_id": job.parent_job_id, "attempt_index": job.attempt_index,
                         "state": job.state, "resume_state": job.recovery_state,
                         "resume_step": job.recovery_step}, 201)

    def _handle_shutdown(self) -> None:
        body = self._read_json()
        mode = body.get("mode")
        if mode is None:
            mode = "force" if bool(body.get("kill_jobs", True)) else "detach"
        if mode not in {"detach", "cooperative-stop", "force"}:
            self._send_json(
                {
                    "error": "shutdown mode must be detach, cooperative-stop, or force",
                    "mode": mode,
                },
                400,
            )
            return
        self._send_json({"ok": True, "mode": mode})
        # Trigger shutdown after the response is flushed, off the handler thread
        # (server.shutdown() must not run in a request thread).
        threading.Thread(
            target=self.server.request_shutdown,  # type: ignore[attr-defined]
            args=(mode,),
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
    # SO_REUSEADDR means "rebind a TIME_WAIT socket" on POSIX (safe, wanted for
    # quick restarts) but "double-bind a live in-use port" on Windows — which
    # would silently spin up a second daemon on a port a sibling/stranger
    # already holds, defeating serve_with_fallback's collision detection. So
    # enable it only off-Windows; on Windows a contested bind must fail loudly.
    allow_reuse_address = os.name != "nt"

    def __init__(self, addr, manager: JobManager):
        super().__init__(addr, _Handler)
        self.manager = manager

    def request_shutdown(self, mode: str | bool) -> None:
        # Older in-process callers passed ``kill_jobs`` positionally. Keep that
        # contract working while the HTTP/API surface uses explicit modes.
        if isinstance(mode, bool):
            mode = "force" if mode else "detach"
        self.manager.shutdown(mode=mode)
        if mode == "cooperative-stop":
            deadline = time.time() + max(5.0, config.STOP_GRACE_SECONDS + 15.0)
            while time.time() < deadline:
                active = self.manager.active_job()
                if active is None or active.state in TERMINAL_STATES:
                    break
                time.sleep(0.25)
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
