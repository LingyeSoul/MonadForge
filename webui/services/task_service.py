"""Task lifecycle manager — submits jobs to the local training daemon.

The WebUI no longer spawns ``python tasks.py …`` subprocesses directly. Every
task (train / preprocess / mask / distill) is submitted as a daemon *command*
job (``POST /jobs {kind:"command", argv:[…]}``); the daemon owns the
subprocess, the serial queue (one job at a time → no GPU contention), the GPU
guard, and on-disk persistence so a job survives a WebUI restart.

This service is now a thin *adapter*: it maps the daemon's job model onto the
in-memory ``Task``/WS surface the frontend already speaks, preserving the
``{"type": "log|metrics|sample|done|…"}`` WS message shape — the frontend is
unchanged. Progress metrics still come from the same two channels:

- stdout: read from the daemon-managed ``<job_dir>/stdout.log`` (tailed here)
- progress.jsonl: derived as before (train.py writes it; we tail it)

The daemon assigns the ``job_id`` (sortable timestamp); we adopt it as the
``task.id`` so the REST/WS surface (which the frontend addresses by task_id)
lines up 1:1 with the daemon's job surface.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from webui.services.daemon_client import DaemonError, daemon_client
from webui.services.training_log_parser import TrainingLogParser

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent


class TaskState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    id: str
    command: str
    args: list[str]
    state: TaskState = TaskState.PENDING
    pid: Optional[int] = None
    exit_code: Optional[int] = None
    lines: list[str] = field(default_factory=list)
    # The daemon owns the subprocess now; we keep no live process handle.
    # `job_id` == `id` (the daemon's sortable job id is adopted as task_id);
    # `stdout_path` is the daemon-managed <job_dir>/stdout.log we tail.
    job_id: Optional[str] = field(default=None, repr=False)
    stdout_path: Optional[str] = field(default=None, repr=False)
    stdout_offset: int = field(default=0, repr=False)
    started_at: Optional[str] = field(default=None)
    _subscribers: list[asyncio.Queue] = field(default_factory=list, repr=False)
    parser: TrainingLogParser = field(default_factory=TrainingLogParser, repr=False)
    wandb_run_url: Optional[str] = None
    progress_started_at: Optional[float] = field(default=None, repr=False)
    progress_last_step: int = field(default=0, repr=False)
    progress_last_ts: Optional[float] = field(default=None, repr=False)
    # Set when the JSONL progress watcher is started (training tasks only).
    # Both the JSONL watcher and the stdout tqdm parser feed
    # ``parser.metrics`` for training tasks — the JSONL carries the
    # structured scalars (``avr_loss`` / ``lr`` / ``epoch``) and the
    # stdout tqdm bar carries the rest (``speed`` / ``elapsed`` /
    # ``eta`` / ``total_steps``). Disabling either channel makes
    # half the dashboard go blank, so we keep both.
    is_training: bool = field(default=False, repr=False)

    def info(self) -> dict:
        return {
            "task_id": self.id,
            "command": self.command,
            "state": self.state.value,
            "pid": self.pid,
            "exit_code": self.exit_code,
            "output_lines": len(self.lines),
            "started_at": self.started_at,
            "wandb_run_url": self.wandb_run_url,
        }


class TaskService:
    """Submits tasks to the local training daemon and mirrors them to the UI.

    Each task becomes a daemon *command* job (``[python, tasks.py, <cmd>, …]``).
    The daemon enforces a serial queue + GPU guard and persists state to disk;
    this service tails each job's ``stdout.log`` + derived ``progress.jsonl``
    and republishes them over the existing WS surface so the frontend is
    unchanged from the old direct-subprocess design.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}
        self._python = sys.executable

    def list_tasks(self) -> list[dict]:
        return [t.info() for t in self._tasks.values()]

    async def get_queue_status(self) -> dict:
        """Snapshot of the daemon queue: paused flag + per-job queue position.

        One ``GET /jobs`` (FIFO by ``submitted_at``) + one ``GET /health``.
        ``queue_position`` semantics: queued jobs are ``1, 2, 3…`` in submission
        order — i.e. "how many jobs finish before this one starts." Running and
        terminal jobs carry no position (the frontend only renders ``#N`` for
        ``state === 'pending'`` tasks). On daemon-down we return
        ``daemon_up: False`` so the frontend can disable the pause button /
        hide positions without erroring.

        ``positions`` maps ``id -> position`` for queued jobs only (``id`` is the
        job's id field as returned by ``GET /jobs`` — the same value the store
        keys ``TaskInfo.task_id`` by). The store merges these onto its
        ``TaskInfo`` list by id.
        """
        try:
            jobs = await daemon_client.list_jobs()
            health = await daemon_client.health()
        except DaemonError:
            return {
                "daemon_up": False,
                "paused": False,
                "positions": {},
            }

        # Daemon returns the full table incl. terminal + running jobs; only
        # queued jobs need a position. FIFO by submitted_at.
        queued = [j for j in jobs if (j.get("state") or "") == "queued"]
        queued.sort(key=lambda j: j.get("submitted_at") or 0)

        positions: dict[str, int] = {}
        for idx, j in enumerate(queued, start=1):
            jid = j.get("id")
            if jid is None:
                continue
            positions[jid] = idx

        return {
            "daemon_up": True,
            "paused": bool(health.get("paused")),
            "positions": positions,
        }

    async def pause_queue(self) -> dict:
        """Hold the queue — queued jobs wait until ``resume_queue``.

        Raises :class:`DaemonError` if the daemon is unreachable; the API layer
        maps that to HTTP 502 so a daemon-down click surfaces as a real failure
        rather than a 500.
        """
        return await daemon_client.pause_queue()

    async def resume_queue(self) -> dict:
        """Release a paused queue — the worker launches queued jobs in order.

        Raises :class:`DaemonError` if the daemon is unreachable; the API layer
        maps that to HTTP 502.
        """
        return await daemon_client.start_queue()

    def get_task(self, task_id: str) -> Optional[Task]:
        return self._tasks.get(task_id)

    def get_task_info(self, task_id: str) -> Optional[dict]:
        t = self._tasks.get(task_id)
        return t.info() if t else None

    def get_task_metrics(self, task_id: str) -> Optional[dict]:
        t = self._tasks.get(task_id)
        if t is None:
            return None
        snapshot = t.parser.metrics.snapshot()
        snapshot["wandb_run_url"] = t.wandb_run_url
        return snapshot

    async def start_task(
        self,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> Task:
        """Submit ``python tasks.py <command> [args...]`` as a daemon command job."""
        # The daemon prepends its own venv interpreter (venv_python) to command
        # jobs, so argv leads with the script — NOT the python executable.
        # See scripts/daemon/manager.py::_build_cmd (job.kind == "command").
        argv = ["tasks.py", command, *(args or [])]
        # The daemon already sets PYTHONUNBUFFERED/PYTHONUTF8/PYTHONIOENCODING
        # (manager._build_cmd), so we only forward caller-provided env (e.g.
        # PRESET overrides).
        extra_env = dict(env) if env else None

        logger.info(
            "Submitting task to daemon: tasks.py %s %s (env override keys: %s)",
            command,
            " ".join(args or []),
            list((env or {}).keys()),
        )
        if env and "PRESET" in env:
            logger.info("Task PRESET env override: %r", env["PRESET"])

        # Temporary task id; replaced with the daemon's job_id once enqueued so
        # task_id == job_id (the frontend addresses jobs by task_id over WS).
        temp_id = uuid.uuid4().hex[:12]
        task = Task(id=temp_id, command=command, args=args or [])
        self._tasks[temp_id] = task

        try:
            resp = await daemon_client.submit_command(
                argv,
                label=command,
                extra_env=extra_env,
                start=True,
            )
        except DaemonError as exc:
            task.state = TaskState.FAILED
            task.lines.append(f"[error] Failed to submit to daemon: {exc}")
            logger.exception("Failed to submit task to daemon")
            return task

        job_id = resp.get("job_id") or temp_id
        # Re-key the task under its daemon job_id so WS/REST addressing lines up.
        task.id = job_id
        task.job_id = job_id
        self._tasks[job_id] = task
        if temp_id != job_id:
            del self._tasks[temp_id]

        task.state = TaskState.PENDING  # queued; _poll_daemon_job flips to RUNNING
        task.started_at = datetime.now(timezone.utc).isoformat()

        # One poller drives state transitions + stdout tailing + terminal
        # signaling. The progress-JSONL watcher runs alongside it.
        asyncio.create_task(self._poll_daemon_job(task))
        jsonl_path = self._derive_progress_jsonl_path(resp, command, args or [], env or {})
        if jsonl_path:
            # Mark as a training task. Both the JSONL watcher and stdout tqdm
            # parser feed the same metrics snapshot: JSONL supplies structured
            # scalars, stdout supplies tqdm-only fields like speed / ETA.
            task.is_training = True
            asyncio.create_task(self._watch_progress_jsonl(task, jsonl_path))

        return task

    async def cancel_task(self, task_id: str) -> bool:
        """Stop a running/queued task via the daemon (tree-kill on its process)."""
        task = self._tasks.get(task_id)
        if not task or task.job_id is None:
            return False
        # Allow cancel from PENDING (queued) or RUNNING.
        if task.state not in (TaskState.RUNNING, TaskState.PENDING):
            return False
        try:
            await daemon_client.stop(task.job_id)
        except DaemonError:
            logger.exception("Failed to stop job %s", task.job_id)
            return False
        task.state = TaskState.CANCELLED
        task.lines.append(f"[cancelled] Task {task_id} terminated by user")
        await self._notify_subscribers(task, {"type": "cancelled", "task_id": task_id})
        return True

    def subscribe(self, task_id: str) -> asyncio.Queue:
        """Subscribe to log lines for a task. Returns an asyncio.Queue."""
        queue: asyncio.Queue = asyncio.Queue()
        task = self._tasks.get(task_id)
        if task:
            task._subscribers.append(queue)
            # Replay existing lines
            for line in task.lines:
                queue.put_nowait({"type": "log", "line": line})
            if task.state in (TaskState.SUCCESS, TaskState.FAILED, TaskState.CANCELLED):
                queue.put_nowait(
                    {
                        "type": "done",
                        "exit_code": task.exit_code,
                        "state": task.state.value,
                    }
                )
        else:
            queue.put_nowait({"type": "error", "message": f"Task {task_id} not found"})
        return queue

    def unsubscribe(self, task_id: str, queue: asyncio.Queue) -> None:
        task = self._tasks.get(task_id)
        if task and queue in task._subscribers:
            task._subscribers.remove(queue)

    async def _poll_daemon_job(self, task: Task) -> None:
        """Drive a submitted daemon job: poll state, tail stdout.log, finalize.

        Replaces the old ``_read_output`` (which read a live pipe). The daemon
        writes the subprocess's combined stdout+stderr to
        ``<job_dir>/stdout.log``; we tail it by offset and feed bytes through
        the same ``\\n``/``\\r`` splitter → ``_emit_line`` so tqdm bars still
        render as ``replace`` updates. State transitions come from
        ``GET /jobs/{id}`` (queued/running/done/error/stopped); on a terminal
        state we map to ``TaskState`` and emit the ``done`` message.
        """
        assert task.job_id is not None
        poll_interval = 0.5
        try:
            while task.state in (TaskState.PENDING, TaskState.RUNNING):
                try:
                    info = await daemon_client.get_job(task.job_id)
                except DaemonError:
                    logger.debug("daemon poll failed for %s; will retry", task.job_id)
                    await asyncio.sleep(poll_interval)
                    continue

                # Resolve / learn the stdout path lazily (daemon sets it once
                # the job is launched). Persisting it lets us tail even if a
                # later poll transiently fails.
                stdout_path = info.get("stdout_path")
                if stdout_path:
                    task.stdout_path = stdout_path

                # Map daemon state → TaskState. queued → PENDING, running →
                # RUNNING; terminal states break the loop after a final drain.
                dstate = info.get("state")
                if dstate == "running" and task.state == TaskState.PENDING:
                    task.state = TaskState.RUNNING
                    task.pid = info.get("pid")
                elif dstate in ("done", "error", "stopped"):
                    # Drain any final stdout before signaling done.
                    await self._drain_stdout(task)
                    await self._finalize_from_daemon(task, info)
                    return

                # Tail new stdout bytes each tick (no-op until path is known).
                await self._drain_stdout(task)
                await asyncio.sleep(poll_interval)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Error polling daemon job %s", task.job_id)
            task.state = TaskState.FAILED
            await self._notify_subscribers(
                task, {"type": "done", "exit_code": -1, "state": "failed"}
            )

    async def _finalize_from_daemon(self, task: Task, info: dict) -> None:
        """Map a terminal daemon job record onto the Task + emit ``done``."""
        dstate = info.get("state")
        if task.state == TaskState.CANCELLED:
            pass  # cancel_task already set it
        elif dstate == "done":
            task.state = TaskState.SUCCESS
            task.exit_code = 0
        elif dstate == "stopped":
            task.state = TaskState.CANCELLED
            task.exit_code = info.get("rc")
        else:  # error
            task.state = TaskState.FAILED
            task.exit_code = info.get("rc") or -1
            err = info.get("error") or info.get("status_detail")
            if err:
                task.lines.append(f"[error] {err}")
                await self._notify_subscribers(
                    task, {"type": "log", "line": f"[error] {err}"}
                )

        ckpt = info.get("ckpt_path")
        msg = {
            "type": "done",
            "exit_code": task.exit_code,
            "state": task.state.value,
        }
        if ckpt:
            msg["ckpt_path"] = ckpt
        await self._notify_subscribers(task, msg)

    async def _drain_stdout(self, task: Task) -> None:
        """Read new complete lines from the daemon-managed stdout.log.

        Tracks ``task.stdout_offset`` across calls. A trailing partial line
        (no ``\\n``) is left for the next poll — unless it's a tqdm ``\\r``
        update, which we emit immediately with ``replace=True``. Mirrors the
        old live-pipe splitter's semantics.
        """
        if not task.stdout_path:
            return
        try:
            data: Optional[tuple[bytes, int]] = await asyncio.to_thread(
                self._read_stdout_bytes, task.stdout_path, task.stdout_offset
            )
        except Exception:
            return  # file may not exist yet (job launching) — skip this tick
        if data is None:
            return
        new_bytes, new_offset = data
        if not new_bytes:
            return
        task.stdout_offset = new_offset
        await self._emit_stdout_bytes(task, new_bytes)

    @staticmethod
    def _read_stdout_bytes(path: str, offset: int) -> Optional[tuple[bytes, int]]:
        """Read complete stdout lines from *offset* (blocking I/O).

        Returns ``(b"", offset)`` when there's no complete line yet. A trailing
        fragment without ``\\n`` is held back (offset unchanged) so the next
        poll re-reads it once the ``\\n`` (or a ``\\r`` overwrite) lands.
        """
        try:
            size = os.path.getsize(path)
        except OSError:
            return None
        if size <= offset:
            return b"", offset
        try:
            with open(path, "rb") as fh:
                fh.seek(offset)
                raw = fh.read()
                end = fh.tell()
        except OSError:
            return None
        if not raw:
            return b"", offset
        # Keep back a trailing partial line (no \n) for the next poll.
        last_nl = raw.rfind(b"\n")
        if last_nl == -1:
            # No complete line yet. If it ends with \r it's a tqdm update we
            # should emit now (replace); otherwise wait for more.
            if raw.endswith(b"\r"):
                consumed = end
                return raw, consumed
            return b"", offset
        consumed = offset + last_nl + 1
        return raw[: last_nl + 1], consumed

    async def _emit_stdout_bytes(self, task: Task, data: bytes) -> None:
        """Split stdout bytes on \\n / \\r and emit each line (tqdm-aware)."""
        _LF = ord("\n")
        _CR = ord("\r")
        buf = bytearray()
        i = 0
        end = len(data)
        while i < end:
            b = data[i]
            if b == _CR:
                line = bytes(buf).decode("utf-8", errors="replace")
                buf.clear()
                if i + 1 < end and data[i + 1] == _LF:
                    i += 2
                    await self._emit_line(task, line, replace=False)
                    continue
                # bare \r → tqdm progress update
                i += 1
                await self._emit_line(task, line, replace=True)
            elif b == _LF:
                line = bytes(buf).decode("utf-8", errors="replace")
                buf.clear()
                i += 1
                await self._emit_line(task, line, replace=False)
            else:
                buf.append(b)
                i += 1
        # A trailing partial line without a terminator is held back by the
        # reader (offset not advanced past it), so nothing to flush here.

    # ── Progress JSONL tailing ────────────────────────────────────────

    @staticmethod
    def _arg_value(args: list[str], flag: str) -> Optional[str]:
        """Extract ``--flag value`` or ``--flag=value`` from an arg list."""
        prefix = flag + "="
        for i, a in enumerate(args):
            if a.startswith(prefix):
                return a[len(prefix) :]
            if a == flag and i + 1 < len(args):
                return args[i + 1]
        return None

    def _derive_progress_jsonl_path(
        self,
        job_resp: dict | str | None,
        command: str,
        args: list[str],
        env: dict[str, str] | None = None,
    ) -> Optional[str]:
        """Return the structured progress stream for this daemon job.

        WebUI command jobs run through ``tasks.py`` for compatibility, but the
        daemon assigns every job a private ``output/daemon/jobs/<job_id>/progress.jsonl``.
        Training command jobs are launched with ``--progress_jsonl`` pointing at
        that private file, so the dashboard must tail the same per-job path. Never
        fall back to ``output/logs/<output_name>.progress.jsonl`` here: that file is
        shared across runs and replays stale metrics at task start.
        """
        explicit = self._arg_value(args, "--progress_jsonl")
        if explicit is not None:
            explicit = explicit.strip()
            if explicit.lower() in ("", "none", "off"):
                return None
            if not os.path.isabs(explicit):
                return str((ROOT / explicit).resolve())
            return explicit

        if not self._command_runs_training(command):
            return None

        if isinstance(job_resp, dict):
            progress_path = job_resp.get("progress_path")
            if progress_path:
                return str(progress_path)
            job_id = job_resp.get("job_id")
        else:
            job_id = job_resp
        if not job_id:
            return None
        return str(ROOT / "output" / "daemon" / "jobs" / str(job_id) / "progress.jsonl")

    @staticmethod
    def _command_runs_training(command: str) -> bool:
        """Mirror of the daemon's ``_command_runs_train`` argv classification.

        Delegates to :func:`scripts.tasks._common.command_runs_training` so this
        and the daemon's ``--progress_jsonl`` injection decision can't drift —
        they must agree on which ``tasks.py`` commands route through ``train.py``
        (and thus honor ``--progress_jsonl``)."""
        from scripts.tasks._common import command_runs_training

        return command_runs_training(command)

    def _config_path_overrides(
        self,
        command: str,
        args: list[str],
        env: dict[str, str],
    ) -> dict:
        """Load merged top-level path scalars for command-style training jobs."""
        method: Optional[str] = None
        methods_subdir = "methods"
        preset = env.get("PRESET") or "default"

        if command == "lora-gui":
            variant = env.get("GUI_PRESETS")
            if not variant and args and not args[0].startswith("-"):
                variant = args[0]
            method = variant or "lora"
            methods_subdir = "gui-methods"
        else:
            method_by_command = {
                "lora": "lora",
                "easycontrol": "easycontrol",
            }
            method = method_by_command.get(command)

        if method is None:
            return {}

        try:
            from library.config.io import load_path_overrides

            return load_path_overrides(
                preset=preset,
                method=method,
                methods_subdir=methods_subdir,
            )
        except Exception as exc:  # noqa: BLE001 — fall back to safe defaults
            logger.debug(
                "Could not derive progress JSONL path from config for %s %s: %s",
                command,
                args,
                exc,
            )
            return {}

    async def _watch_progress_jsonl(self, task: Task, jsonl_path: str) -> None:
        """Tail the structured progress JSONL and push metrics to subscribers.

        The training subprocess writes line-buffered JSONL events
        (``{"ev":"step", ...}``) that carry structured scalars (loss, lr, …)
        at the ``log_every_n_steps`` cadence. This coroutine polls the file
        for new data and emits ``{"type":"metrics", "data":{...}}`` messages
        to WebSocket subscribers — the same message format the stdout parser
        produces, so the frontend code is unchanged.

        Each ``step`` event is forwarded as a single metrics message. The
        earlier debounce (a 0.3 s coalescing window) stacked every emit on
        the same window — and a subsequent event arriving just as the
        timer was about to fire reset it, so the dashboard never saw a
        metrics message until the ``finally`` flush at training end. The
        trainer only emits one ``step`` event per ``log_every_n_steps``
        (every 2 by default), so WS load is bounded; the loss / step
        history is already deduped by ``global_step`` so direct emits
        can't double-append.

        The ``run_start`` event carries ``total_steps`` / ``total_epochs`` —
        captured here so the dashboard shows ``step / total`` from step 1
        onward, instead of waiting for the first tqdm bar to leak through
        the daemon's stdout file (which can lag training by the full
        ``TQDM_MININTERVAL`` window).
        """
        # Wait for the task to be picked up by the daemon (state flips
        # PENDING → RUNNING inside ``_poll_daemon_job``). Under the old
        # direct-subprocess design ``start_task`` set RUNNING itself, so
        # this watcher entered its tail loop immediately. The daemon
        # introduces a queueing delay (the job may sit in the daemon's
        # serial queue behind another run); the watcher must wait for
        # the daemon to launch the subprocess before tailing — otherwise
        # the ``while state == RUNNING`` below exits on the first poll
        # and the dashboard never sees any step events.
        for _ in range(120):
            if task.state == TaskState.RUNNING:
                break
            if task.state in (TaskState.SUCCESS, TaskState.FAILED, TaskState.CANCELLED):
                return
            await asyncio.sleep(0.5)
        else:
            if task.state != TaskState.RUNNING:
                logger.debug(
                    "task %s never reached RUNNING; JSONL watcher exiting",
                    task.id,
                )
                return

        # Wait for the file to appear (training subprocess creates it on
        # first ``run_start``).  Give up after 5 minutes — the first
        # torch.compile / model-load trace on a long-running job can
        # easily take that long before the trainer writes its first
        # event, and a too-short timeout silently strands the
        # dashboard with no progress info for the rest of the run.
        for _ in range(600):
            if task.state in (TaskState.SUCCESS, TaskState.FAILED, TaskState.CANCELLED):
                return
            if os.path.isfile(jsonl_path):
                break
            await asyncio.sleep(0.5)
        else:
            logger.debug(
                "progress JSONL not found after 5 min, skipping watcher: %s",
                jsonl_path,
            )
            return

        logger.info("Tailing progress JSONL: %s", jsonl_path)
        offset = 0
        try:
            while task.state == TaskState.RUNNING:
                try:
                    size = os.path.getsize(jsonl_path)
                except OSError:
                    await asyncio.sleep(0.5)
                    continue
                if size <= offset:
                    await asyncio.sleep(0.3)
                    continue
                # Read new bytes in a thread so we don't block the event loop.
                result: Optional[tuple[list[str], int]] = await asyncio.to_thread(
                    self._read_jsonl_bytes, jsonl_path, offset
                )
                if result is not None:
                    new_lines, new_offset = result
                    offset = new_offset
                    for raw_line in new_lines:
                        try:
                            ev = json.loads(raw_line)
                        except (json.JSONDecodeError, ValueError):
                            continue
                        ev_type = ev.get("ev")
                        if ev_type == "run_start":
                            # Capture total_steps / total_epochs from the
                            # trainer's opening event so the dashboard can
                            # render "step / total" from the first step —
                            # don't wait for the first tqdm bar.
                            metrics = task.parser.metrics
                            ts = ev.get("ts")
                            task.progress_started_at = (
                                float(ts) if ts is not None else 0.0
                            )
                            task.progress_last_step = 0
                            task.progress_last_ts = task.progress_started_at
                            if "total_steps" in ev:
                                metrics.total_steps = int(ev["total_steps"])
                            if "total_epochs" in ev:
                                metrics.total_epochs = int(ev["total_epochs"])
                            # Wake up any late-joining subscriber waiting
                            # for total_steps — emit a snapshot now so the
                            # UI sees the denominator as soon as the trainer
                            # announces it.
                            await self._notify_subscribers(
                                task,
                                {"type": "metrics", "data": metrics.snapshot()},
                            )
                        elif ev_type == "step":
                            metrics = task.parser.metrics
                            if "global_step" in ev:
                                metrics.step = ev["global_step"]
                            if "epoch" in ev:
                                metrics.epoch = ev["epoch"]
                            # The trainer logs scalars under accelerate's
                            # ``group/prefix`` keys (``loss/average``,
                            # ``loss/current``, ``lr/unet``) — pull the
                            # values the dashboard renders (``avr_loss``,
                            # ``lr``) from those, with the legacy flat
                            # names as a fallback for any future writer
                            # that emits the older shape.
                            #
                            # IMPORTANT: parse lr BEFORE appending to
                            # histories so lr_history records the *current*
                            # step's lr, not the previous step's value.
                            lr = ev.get("lr/unet")
                            if lr is None:
                                lr = ev.get("lr")
                            if lr is not None:
                                metrics.lr = float(lr)
                            loss = ev.get("loss/average")
                            if loss is None:
                                loss = ev.get("avr_loss")
                            if loss is not None:
                                metrics.avr_loss = float(loss)
                                s = metrics.step
                                if (
                                    not metrics.step_history
                                    or s != metrics.step_history[-1]
                                ):
                                    metrics.loss_history.append(metrics.avr_loss)
                                    metrics.step_history.append(s)
                                    metrics.lr_history.append(metrics.lr)
                            self._update_jsonl_timing_metrics(task, ev)
                            # Emit a fresh snapshot per step event. The
                            # earlier debounce was meant to coalesce a
                            # burst of step events into one WS message,
                            # but in practice it stacked every emit on
                            # the same 0.3 s window — and a subsequent
                            # event arrived just as the timer was about
                            # to fire, so the dashboard never saw a
                            # metrics message until the ``finally``
                            # flush. Emit directly per event instead:
                            # the loss / step history are already
                            # deduped by ``global_step`` so duplicate
                            # appends are impossible, and the trainer
                            # only emits one ``step`` per
                            # ``log_every_n_steps`` (every 2 by
                            # default) so the WS load is bounded.
                            await self._notify_subscribers(
                                task,
                                {"type": "metrics", "data": metrics.snapshot()},
                            )
                        elif ev_type == "sample":
                            # Training emitted a preview image; relay as a
                            # dedicated ``sample`` message so the dashboard
                            # can append it to its gallery in arrival order.
                            sample_path = ev.get("path")
                            if not sample_path:
                                continue
                            await self._notify_subscribers(
                                task,
                                {
                                    "type": "sample",
                                    "path": sample_path,
                                    "step": ev.get("global_step"),
                                    "epoch": ev.get("epoch"),
                                    "prompt": ev.get("prompt"),
                                    "ts": ev.get("ts"),
                                },
                            )
                        elif ev_type == "run_end":
                            # Trainer announced terminal state — keep its
                            # ``final_step`` so the dashboard ends on the
                            # exact step training reached.
                            metrics = task.parser.metrics
                            if "final_step" in ev:
                                metrics.step = int(ev["final_step"])
                            # Flush any pending debounced metrics so the
                            # last step the trainer logged is on screen
                            # before the ``done`` message lands — and
                            # schedule a fresh snapshot so the final_step
                            # value (which may differ from the last logged
                            # step) is also visible to subscribers.
                            await self._flush_metrics_emit(task)
                            await self._notify_subscribers(
                                task,
                                {"type": "metrics", "data": metrics.snapshot()},
                            )
                else:
                    # File may have been rotated; reset.
                    offset = 0
                await asyncio.sleep(0.3)
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Progress JSONL watcher failed for task %s", task.id)
        finally:
            # Flush any pending debounced metrics so a fast-finishing run
            # still surfaces its last step to subscribers (the loop exits
            # when ``task.state`` flips terminal, which can race with the
            # debounce window — e.g. a step event lands 50 ms before
            # ``done`` and the scheduled send never fires).
            await self._flush_metrics_emit(task)

    def _update_jsonl_timing_metrics(self, task: Task, ev: dict) -> None:
        """Derive speed / elapsed / ETA from structured JSONL step timing."""
        metrics = task.parser.metrics
        try:
            step = int(ev.get("global_step") or metrics.step or 0)
        except (TypeError, ValueError):
            return
        ts_raw = ev.get("ts")
        if ts_raw is None:
            return
        try:
            ts = float(ts_raw)
        except (TypeError, ValueError):
            return

        if task.progress_started_at is None:
            task.progress_started_at = 0.0
        elapsed = max(0.0, ts - task.progress_started_at)
        metrics.elapsed = self._format_duration(elapsed)

        last_ts = task.progress_last_ts
        last_step = task.progress_last_step
        delta_steps = step - last_step
        delta_ts = ts - last_ts if last_ts is not None else 0.0
        if delta_steps > 0 and delta_ts > 0:
            steps_per_sec = delta_steps / delta_ts
            if steps_per_sec >= 1.0:
                metrics.speed = f"{steps_per_sec:.2f} it/s"
            else:
                metrics.speed = f"{(1.0 / steps_per_sec):.2f} s/it"
            if metrics.total_steps > step:
                remaining = metrics.total_steps - step
                metrics.eta = self._format_duration(remaining / steps_per_sec)

        if step >= task.progress_last_step:
            task.progress_last_step = step
            task.progress_last_ts = ts

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds = max(0, int(round(seconds)))
        hours, rem = divmod(seconds, 3600)
        minutes, secs = divmod(rem, 60)
        if hours:
            return f"{hours:d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    async def _flush_metrics_emit(self, task: Task) -> None:
        """No-op now that the watcher emits per step event directly.

        Kept for callers (the ``run_end`` handler and the watcher's
        ``finally`` clause) that previously relied on the debounced
        flush to surface the last step. Direct emits make it redundant
        but harmless — both call sites still want a deterministic
        "send whatever is pending" hook in case future code adds
        async-side bookkeeping that must fire on terminal state.
        """
        return None

    @staticmethod
    def _read_jsonl_bytes(path: str, offset: int) -> Optional[tuple[list[str], int]]:
        """Read complete JSONL lines starting at *offset* (blocking I/O).

        Returns ``(None, -)`` if the file is shorter than *offset* (rotation),
        otherwise ``(lines, new_offset)`` where *lines* may be empty.
        """
        try:
            with open(path, "r", encoding="utf-8") as fh:
                fh.seek(offset)
                raw = fh.read()
                end = fh.tell()
        except FileNotFoundError:
            return None
        if not raw:
            return [], offset
        # The last line may be incomplete (no trailing \n yet); don't
        # consume it — the next poll will pick it up.
        lines = raw.split("\n")
        if not raw.endswith("\n"):
            incomplete = lines.pop()
            # Adjust offset back by the length of the incomplete tail.
            end -= len(incomplete.encode("utf-8")) + 1  # +1 for the \n split
        else:
            # Strip trailing empty string from split on final \n.
            if lines and lines[-1] == "":
                lines.pop()
        return [ln for ln in lines if ln.strip()], end

    async def _emit_line(self, task: Task, line: str, *, replace: bool) -> None:
        """Append (or replace) a line and notify subscribers.

        When *replace* is ``True`` the line is a tqdm progress update that
        should overwrite the previous line in the frontend.
        """
        if replace and task.lines:
            task.lines[-1] = line
        else:
            task.lines.append(line)

        msg: dict = {"type": "log", "line": line}
        if replace:
            msg["replace"] = True
        await self._notify_subscribers(task, msg)

        # Parse training metrics from the line. For training tasks both
        # the stdout tqdm redraws and the JSONL ``step`` events feed
        # ``task.parser.metrics``: the stdout path is the *only* source
        # for tqdm-derived scalars (``speed``, ``elapsed``, ``eta``,
        # ``total_steps``), the JSONL path carries the structured
        # scalars (``avr_loss``, ``lr``, ``epoch``). They update the
        # same parser object so the debounced JSONL emit and the
        # stdout emit both carry the merged snapshot — no data race,
        # just two channels covering two halves of the dashboard.
        if task.parser.feed(line):
            snapshot = task.parser.metrics.snapshot()
            print(
                f"[metrics] step={snapshot.get('step')}/{snapshot.get('total_steps')} "
                f"loss={snapshot.get('avr_loss')} lr={snapshot.get('lr')} speed={snapshot.get('speed')}",
                flush=True,
            )
            await self._notify_subscribers(
                task,
                {"type": "metrics", "data": snapshot},
            )

        # Capture wandb run URL from stdout (wandb prints "Run page: https://...")
        if task.wandb_run_url is None:
            _wb_match = re.search(r"https://wandb\.ai/\S+", line)
            if _wb_match:
                task.wandb_run_url = _wb_match.group(0)
                await self._notify_subscribers(
                    task,
                    {"type": "wandb_url", "url": task.wandb_run_url},
                )

    async def _notify_subscribers(self, task: Task, msg: dict) -> None:
        dead: list[asyncio.Queue] = []
        for q in task._subscribers:
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            task._subscribers.remove(q)


# Singleton
task_service = TaskService()
