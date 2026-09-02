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
from webui.services.task_catalog import task_category
from webui.services.training_log_parser import TrainingLogParser

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent

# Cap on the in-memory output buffer per task. The full stdout stays on
# disk (the daemon's stdout.log); this buffer only feeds the UI, so once
# the cap trips, oldest lines are dropped FIFO. Keeps WS replay, the
# /output REST payload and frontend memory bounded on multi-day runs.
MAX_LINES = 5000
TRIM_LINES_TO = 4000

# Minimum wall-clock interval between metrics snapshot broadcasts per
# task. Both the stdout tqdm parser and the JSONL watcher emit cumulative
# snapshots, so dropping intermediates loses nothing — the next snapshot
# that does go out carries the full state.
METRICS_MIN_INTERVAL = 0.4


class TaskState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    STOPPING = "stopping"
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
    # Total lines ever seen (incl. ones dropped by the MAX_LINES cap) so
    # the /output endpoint can flag the buffer as truncated.
    lines_total: int = field(default=0, repr=False)
    # The daemon owns the subprocess now; we keep no live process handle.
    # `job_id` == `id` (the daemon's sortable job id is adopted as task_id);
    # `stdout_path` is the daemon-managed <job_dir>/stdout.log we tail.
    job_id: Optional[str] = field(default=None, repr=False)
    # Per-job preview-image dir returned by the daemon on submit. Read by the
    # preview API to locate the gallery — kept here (not re-parsed from args)
    # so it survives a WebUI restart via ``daemon_client.get_job``.
    sample_dir: Optional[str] = field(default=None, repr=False)
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
    recovery_step: Optional[int] = None
    resume_state: Optional[str] = field(default=None, repr=False)
    terminal_reason: Optional[str] = field(default=None, repr=False)
    resumable: bool = False
    legacy: bool = False
    target_steps: Optional[int] = None
    target_epochs: Optional[int] = None
    # Full physical daemon records for this logical task, oldest first.
    attempts: list[dict] = field(default_factory=list, repr=False)
    monitored_job_id: Optional[str] = field(default=None, repr=False)
    # Attempt-id signature the on-disk output was last rebuilt from. The
    # periodic queue-status reconcile calls _apply_daemon_state for every
    # task; without this guard each pass re-read every terminal task's
    # stdout files from disk (synchronously, on the event loop).
    loaded_attempt_ids: list[str] = field(default_factory=list, repr=False)
    # monotonic() timestamp of the last metrics snapshot broadcast.
    last_metrics_ts: float = field(default=0.0, repr=False)

    def info(self) -> dict:
        attempt_infos = [
            {
                "job_id": str(attempt.get("id") or attempt.get("job_id") or ""),
                "attempt_index": int(attempt.get("attempt_index") or 0),
                "state": attempt.get("state"),
                "started_at": TaskService._format_daemon_timestamp(
                    attempt.get("started_at") or attempt.get("submitted_at")
                ),
                "ended_at": TaskService._format_daemon_timestamp(attempt.get("ended_at")),
                "recovery_step": attempt.get("recovery_step"),
                "exit_code": attempt.get("rc"),
                "terminal_reason": attempt.get("terminal_reason")
                or attempt.get("status_detail")
                or attempt.get("error"),
            }
            for attempt in self.attempts
        ]
        return {
            "task_id": self.id,
            "current_job_id": self.job_id,
            "attempt_count": max(1, len(attempt_infos)),
            "attempts": attempt_infos,
            "command": self.command,
            "state": self.state.value,
            "pid": self.pid,
            "exit_code": self.exit_code,
            "output_lines": len(self.lines),
            "started_at": self.started_at,
            "wandb_run_url": self.wandb_run_url,
            "category": task_category(self.command),
            "recovery_step": self.recovery_step,
            "resume_state": self.resume_state,
            "terminal_reason": self.terminal_reason,
            "resumable": self.resumable,
            "legacy": self.legacy,
            "target_steps": self.target_steps,
            "target_epochs": self.target_epochs,
            "last_progress": {
                "step": self.parser.metrics.step,
                "total_steps": self.parser.metrics.total_steps,
                "epoch": self.parser.metrics.epoch,
            },
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
        self._pollers: dict[str, asyncio.Task] = {}
        self._progress_watchers: dict[str, asyncio.Task] = {}
        self._python = sys.executable

    async def reconcile_daemon_jobs(self) -> None:
        """Restore active daemon jobs after a WebUI process restart."""
        try:
            page = await self._list_job_groups_page(limit=500)
        except DaemonError as exc:
            logger.warning("Could not reconcile daemon jobs at WebUI startup: %s", exc)
            return

        self._reconcile_job_list(list(page.get("groups") or []))

    @staticmethod
    def _physical_job_group(info: dict) -> dict:
        job_id = str(info.get("id") or "")
        group = dict(info)
        group.update(
            {
                "id": str(info.get("root_job_id") or job_id),
                "root_job_id": str(info.get("root_job_id") or job_id),
                "current_job_id": job_id,
                "attempt_count": 1,
                "attempts": [dict(info)],
            }
        )
        return group

    async def _list_job_groups_page(self, **kwargs) -> dict:
        """Use lineage-aware daemon APIs, with one-attempt compatibility."""
        try:
            return await daemon_client.list_job_groups_page(**kwargs)
        except DaemonError:
            page = await daemon_client.list_jobs_page(**kwargs)
            jobs = list(page.get("jobs") or [])
            states = {
                part.strip()
                for part in str(kwargs.get("state") or "").split(",")
                if part.strip()
            }
            if states:
                jobs = [job for job in jobs if job.get("state") in states]
            return {
                "groups": [self._physical_job_group(job) for job in jobs],
                "total": len(jobs) if states else int(page.get("total", len(jobs)) or 0),
                "offset": int(page.get("offset", kwargs.get("offset", 0)) or 0),
                "limit": int(page.get("limit", kwargs.get("limit", len(jobs))) or 0),
            }

    def _reconcile_job_list(self, jobs: list[dict]) -> None:
        """Merge active daemon jobs into the WebUI's in-memory task table."""

        for raw_info in jobs:
            info = raw_info if raw_info.get("attempts") else self._physical_job_group(raw_info)
            root_id = info.get("root_job_id") or info.get("id")
            if not root_id:
                continue
            task = self._tasks.get(str(root_id))
            if task is None:
                task = self._task_from_daemon(info)
                self._tasks[task.id] = task
            else:
                self._apply_daemon_state(task, info)
            self._ensure_monitors(task, info.get("progress_path"))

    def _task_from_daemon(self, info: dict) -> Task:
        group = info if info.get("attempts") else self._physical_job_group(info)
        attempts = list(group.get("attempts") or [])
        current_job_id = str(group.get("current_job_id") or attempts[-1].get("id") or group["id"])
        current = next(
            (attempt for attempt in attempts if str(attempt.get("id")) == current_job_id),
            attempts[-1] if attempts else group,
        )
        root_id = str(group.get("root_job_id") or group.get("id") or current_job_id)
        argv = [str(value) for value in (current.get("argv") or [])]
        command = str(current.get("method") or "unknown")
        args: list[str] = []
        if len(argv) >= 2 and Path(argv[0]).name.lower() == "tasks.py":
            command = argv[1]
            args = argv[2:]

        task = Task(
            id=root_id,
            command=command,
            args=args,
            state=self._map_daemon_state(current.get("state"), stop_requested=bool(current.get("stop_requested"))),
            pid=current.get("pid"),
            exit_code=current.get("rc"),
            job_id=current_job_id,
            sample_dir=current.get("sample_dir"),
            stdout_path=current.get("stdout_path"),
            started_at=self._format_daemon_timestamp(
                group.get("started_at") or group.get("submitted_at")
            ),
            attempts=attempts,
        )
        task.is_training = current.get("kind") == "train" or self._command_runs_training(command)
        task.recovery_step = current.get("recovery_step")
        task.resume_state = current.get("recovery_state")
        task.terminal_reason = current.get("terminal_reason") or current.get("status_detail") or current.get("error")
        # A terminal status alone does not imply that a complete, signature-
        # matched state exists. Keep Continue hidden until the daemon has
        # discovered and persisted an actual recovery directory.
        task.resumable = bool(current.get("recovery_state"))
        task.legacy = bool(current.get("legacy"))
        task.target_steps = current.get("target_steps")
        task.target_epochs = current.get("target_epochs")
        self._load_historical_output(task, group)
        return task

    @staticmethod
    def _format_daemon_timestamp(value) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        try:
            return datetime.fromtimestamp(float(value), timezone.utc).isoformat()
        except (TypeError, ValueError, OSError):
            return None

    @staticmethod
    def _map_daemon_state(state: Optional[str], *, stop_requested: bool = False) -> TaskState:
        return {
            "queued": TaskState.PENDING,
            "running": TaskState.STOPPING if stop_requested else TaskState.RUNNING,
            "done": TaskState.SUCCESS,
            "stopped": TaskState.CANCELLED,
            "error": TaskState.FAILED,
        }.get(state or "", TaskState.PENDING)

    def _apply_daemon_state(self, task: Task, info: dict) -> None:
        group = info if info.get("attempts") else self._physical_job_group(info)
        attempts = list(group.get("attempts") or [])
        current_job_id = str(
            group.get("current_job_id")
            or (attempts[-1].get("id") if attempts else task.job_id or task.id)
        )
        current = next(
            (attempt for attempt in attempts if str(attempt.get("id")) == current_job_id),
            attempts[-1] if attempts else group,
        )
        attempt_ids_before = [str(attempt.get("id")) for attempt in task.attempts]
        attempt_ids_after = [str(attempt.get("id")) for attempt in attempts]
        current_changed = task.job_id != current_job_id

        task.state = self._map_daemon_state(
            current.get("state"), stop_requested=bool(current.get("stop_requested"))
        )
        task.pid = current.get("pid")
        task.exit_code = current.get("rc")
        task.job_id = current_job_id
        task.attempts = attempts
        task.sample_dir = current.get("sample_dir") or task.sample_dir
        task.stdout_path = current.get("stdout_path") or task.stdout_path
        task.recovery_step = current.get("recovery_step")
        task.resume_state = current.get("recovery_state")
        task.terminal_reason = current.get("terminal_reason") or current.get("status_detail") or current.get("error")
        task.resumable = bool(current.get("recovery_state"))
        task.legacy = bool(current.get("legacy"))
        task.target_steps = current.get("target_steps")
        task.target_epochs = current.get("target_epochs")
        if current_changed or attempt_ids_before != attempt_ids_after:
            task.monitored_job_id = None
            self._load_historical_output(task, group)
        elif task.loaded_attempt_ids != attempt_ids_after:
            # First _apply_daemon_state for a task that was created without
            # its on-disk output (e.g. a stale in-memory row). Guarded by the
            # loaded-attempt signature so the periodic queue-status reconcile
            # — which lands here for every terminal task — stops re-reading
            # every attempt's stdout files from disk on each pass.
            self._load_historical_output(task, group)

    @staticmethod
    def _load_historical_output(task: Task, info: dict) -> None:
        """Rebuild one logical task from all physical attempt artifacts."""
        attempts = list(info.get("attempts") or [info])
        task.lines = []
        task.lines_total = 0
        task.parser = TrainingLogParser()
        task.stdout_offset = 0
        task.progress_last_step = 0
        task.progress_last_ts = None
        points: dict[int, tuple[float, float, int | None]] = {}
        final_steps: list[int] = []

        for position, attempt in enumerate(attempts, start=1):
            attempt_id = str(attempt.get("id") or attempt.get("job_id") or position)
            if len(attempts) > 1:
                recovery = attempt.get("recovery_step")
                detail = f" from step {recovery}" if recovery is not None else ""
                task.lines.append(
                    f"[attempt {position}/{len(attempts)} · {attempt_id}{detail}]"
                )

            stdout_path = attempt.get("stdout_path")
            if stdout_path:
                try:
                    raw = Path(stdout_path).read_text(
                        encoding="utf-8", errors="replace"
                    )
                    lines = [line for line in re.split(r"\r?\n", raw) if line]
                    task.lines.extend(lines)
                    for line in lines:
                        task.parser.feed(line)
                    if attempt_id == str(task.job_id):
                        task.stdout_offset = len(raw.encode("utf-8"))
                except OSError:
                    pass

            progress_path = attempt.get("progress_path")
            if not progress_path or not task.is_training:
                continue
            try:
                raw_events = Path(progress_path).read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
            except OSError:
                continue
            for raw_event in raw_events:
                try:
                    ev = json.loads(raw_event)
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue
                kind = ev.get("ev")
                metrics = task.parser.metrics
                if kind == "run_start":
                    if ev.get("total_steps") is not None:
                        metrics.total_steps = int(ev["total_steps"])
                    if ev.get("total_epochs") is not None:
                        metrics.total_epochs = int(ev["total_epochs"])
                    metrics.sampling_enabled = (
                        metrics.sampling_enabled
                        or bool(ev.get("sampling_enabled", False))
                    )
                elif kind == "step" and ev.get("global_step") is not None:
                    step = int(ev["global_step"])
                    loss = ev.get("loss/average", ev.get("avr_loss"))
                    lr = ev.get("lr/d*lr/unet", ev.get("lr/unet", ev.get("lr")))
                    epoch = ev.get("epoch")
                    if loss is not None:
                        previous_lr = points.get(step, (0.0, metrics.lr, None))[1]
                        points[step] = (
                            float(loss),
                            float(lr) if lr is not None else previous_lr,
                            int(epoch) if epoch is not None else None,
                        )
                    task.progress_last_step = max(task.progress_last_step, step)
                elif kind == "run_end" and ev.get("final_step") is not None:
                    final_steps.append(int(ev["final_step"]))

        metrics = task.parser.metrics
        # Structured JSONL is authoritative for chart points; stdout is retained
        # for speed/ETA and event parsing only.
        metrics.step_history = []
        metrics.loss_history = []
        metrics.lr_history = []
        for step in sorted(points):
            loss, lr, epoch = points[step]
            metrics.upsert_step(step, loss, lr)
            metrics.avr_loss = loss
            metrics.lr = lr
            if epoch is not None:
                metrics.epoch = epoch
        metrics.step = max(
            [task.progress_last_step, *final_steps, *(points.keys() or [0])]
        )
        task.lines_total = len(task.lines)
        TaskService._cap_lines(task)
        task.loaded_attempt_ids = [
            str(a.get("id") or a.get("job_id") or pos)
            for pos, a in enumerate(attempts, start=1)
        ]

    @staticmethod
    def _cap_lines(task: Task) -> None:
        """Trim the UI output buffer FIFO once it outgrows MAX_LINES.

        Batched (trim to TRIM_LINES_TO only past MAX_LINES) so a chatty
        tqdm stream doesn't pay an O(n) splice on every line.
        """
        if len(task.lines) > MAX_LINES:
            del task.lines[: len(task.lines) - TRIM_LINES_TO]

    def _ensure_monitors(self, task: Task, progress_path: Optional[str] = None) -> None:
        """Start the per-job pollers once, including for restored jobs."""
        if task.job_id is None or task.state not in (
            TaskState.PENDING,
            TaskState.RUNNING,
            TaskState.STOPPING,
        ):
            return

        if task.monitored_job_id != task.job_id:
            old_poller = self._pollers.get(task.id)
            old_watcher = self._progress_watchers.get(task.id)
            if old_poller is not None and not old_poller.done():
                old_poller.cancel()
            if old_watcher is not None and not old_watcher.done():
                old_watcher.cancel()
            task.monitored_job_id = task.job_id

        poller = self._pollers.get(task.id)
        if poller is None or poller.done():
            poller = asyncio.create_task(self._poll_daemon_job(task))
            self._pollers[task.id] = poller
            poller.add_done_callback(
                lambda done, task_id=task.id: self._forget_monitor(
                    self._pollers, task_id, done
                )
            )

        if task.is_training and progress_path:
            watcher = self._progress_watchers.get(task.id)
            if watcher is None or watcher.done():
                watcher = asyncio.create_task(
                    self._watch_progress_jsonl(task, str(progress_path))
                )
                self._progress_watchers[task.id] = watcher
                watcher.add_done_callback(
                    lambda done, task_id=task.id: self._forget_monitor(
                        self._progress_watchers, task_id, done
                    )
                )

    @staticmethod
    def _forget_monitor(
        registry: dict[str, asyncio.Task], task_id: str, done: asyncio.Task
    ) -> None:
        if registry.get(task_id) is done:
            registry.pop(task_id, None)

    async def close(self) -> None:
        """Stop WebUI-owned watchers without touching daemon jobs."""
        monitors = [*self._pollers.values(), *self._progress_watchers.values()]
        for monitor in monitors:
            monitor.cancel()
        if monitors:
            await asyncio.gather(*monitors, return_exceptions=True)
        self._pollers.clear()
        self._progress_watchers.clear()

    def list_tasks(
        self, *, state: Optional[str] = None, limit: int = 100, offset: int = 0
    ) -> list[dict]:
        items = sorted(
            (t.info() for t in self._tasks.values()),
            key=lambda item: item.get("started_at") or "",
            reverse=True,
        )
        if state:
            items = [item for item in items if item.get("state") == state]
        start = max(0, int(offset))
        return items[start : start + max(0, int(limit))]

    async def list_tasks_page(
        self, *, state: Optional[str] = None, limit: int = 100, offset: int = 0
    ) -> tuple[list[dict], int]:
        """Read one exact page from the daemon's durable job table.

        The in-memory task map remains the WebSocket authority, but it must not
        be the source of pagination: after a WebUI restart it only contains the
        pages that have been requested so far.  The daemon therefore supplies
        both the filtered page and its total; reconciliation hydrates the page
        into the existing Task objects so historical logs still work.
        """
        daemon_state = {
            "success": "done",
            "failed": "error",
            "cancelled": "stopped",
            "active": "queued,running",
        }.get(state or "")
        page = await self._list_job_groups_page(
            state=daemon_state,
            offset=max(0, int(offset)),
            limit=max(1, min(int(limit), 500)),
            newest_first=True,
        )
        jobs = list(page.get("groups") or [])
        total = int(page.get("total", len(jobs)) or 0)
        self._reconcile_job_list(jobs)
        items_by_id = {task.id: task.info() for task in self._tasks.values()}
        items = [
            items_by_id[str(job["id"])]
            for job in jobs
            if str(job.get("id")) in items_by_id
        ]
        items.sort(key=lambda item: item.get("started_at") or "", reverse=True)
        if state == "stopping":
            items = [item for item in items if item.get("state") == "stopping"]
        return items, total

    async def resume_task(self, task_id: str) -> Optional[Task]:
        source = self._tasks.get(task_id)
        if source is None:
            try:
                info = await daemon_client.get_job_group(task_id)
            except DaemonError:
                return None
            source = self._task_from_daemon(info)
            self._tasks[source.id] = source
        try:
            response = await daemon_client.resume(source.job_id or task_id)
            info = await daemon_client.get_job_group(
                str(response.get("root_job_id") or source.id)
            )
        except DaemonError:
            raise
        self._apply_daemon_state(source, info)
        self._ensure_monitors(source, info.get("progress_path"))
        return source

    async def delete_task(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        if task is None:
            try:
                info = await daemon_client.get_job_group(task_id)
            except DaemonError:
                return False
            state = self._map_daemon_state(
                info.get("state"), stop_requested=bool(info.get("stop_requested"))
            )
            if state not in (TaskState.SUCCESS, TaskState.FAILED, TaskState.CANCELLED):
                return False
        elif task.state not in (TaskState.SUCCESS, TaskState.FAILED, TaskState.CANCELLED):
            return False
        await daemon_client.delete_job_group(task_id)
        self._tasks.pop(task_id, None)
        return True

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
            group_page = await self._list_job_groups_page(limit=500)
            health = await daemon_client.health()
        except DaemonError:
            return {
                "daemon_up": False,
                "paused": False,
                "positions": {},
            }

        # Queue status is already polled by the task page. Reuse that daemon
        # snapshot to heal a startup race (WebUI came up before the daemon) or
        # an unexpectedly exited monitor without adding another HTTP request.
        self._reconcile_job_list(list(group_page.get("groups") or []))

        # Daemon returns the full table incl. terminal + running jobs; only
        # queued jobs need a position. FIFO by submitted_at.
        queued = [j for j in jobs if (j.get("state") or "") == "queued"]
        queued.sort(key=lambda j: j.get("submitted_at") or 0)

        positions: dict[str, int] = {}
        for idx, j in enumerate(queued, start=1):
            jid = j.get("id")
            if jid is None:
                continue
            positions[str(j.get("root_job_id") or jid)] = idx

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

    async def shutdown_daemon(self, *, kill_jobs: bool = True, mode: str | None = None) -> dict:
        """Fully stop the training daemon — a complete exit (process + worker).

        Posts to the daemon's ``/shutdown``; by default ``kill_jobs=True`` also
        tree-kills the running job so nothing is left behind. When the daemon
        hosts the WebUI as a sidecar (the default), the WebUI process is taken
        down too, so the response may arrive after the connection drops —
        callers should not treat a connection-reset as failure.

        Raises :class:`DaemonError` if the daemon is unreachable.
        """
        return await daemon_client.shutdown(kill_jobs=kill_jobs, mode=mode)

    def get_task(self, task_id: str) -> Optional[Task]:
        return self._get_or_load_task(task_id)

    def get_task_info(self, task_id: str) -> Optional[dict]:
        t = self._get_or_load_task(task_id)
        return t.info() if t else None

    def get_task_metrics(self, task_id: str) -> Optional[dict]:
        t = self._get_or_load_task(task_id)
        if t is None:
            return None
        snapshot = t.parser.metrics.snapshot()
        snapshot["wandb_run_url"] = t.wandb_run_url
        return snapshot

    def _get_or_load_task(self, task_id: str) -> Optional[Task]:
        """Hydrate a task on demand for direct historical REST/WS access."""
        task = self._tasks.get(task_id)
        if task is not None:
            return task
        try:
            info = daemon_client.get_job_group_sync(task_id)
        except DaemonError:
            try:
                info = self._physical_job_group(daemon_client.get_job_sync(task_id))
            except DaemonError:
                return None
        task = self._task_from_daemon(info)
        self._tasks[task.id] = task
        return task

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
        config_snapshot = self._training_config_snapshot(command, argv, extra_env or {})

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
        # ``sampling_enabled`` is NOT derived here: the GUI training path
        # sources its ``--sample_*`` flags from the method TOML, merged inside
        # ``train.py``, so the raw ``args`` never carry them. The trainer is
        # the single source of truth — it emits ``sampling_enabled`` on the
        # ``run_start`` progress event, which ``_watch_progress_jsonl`` reads
        # back into ``metrics.sampling_enabled`` once the run actually starts.
        self._tasks[temp_id] = task

        try:
            resp = await daemon_client.submit_command(
                argv,
                label=command,
                extra_env=extra_env,
                config_snapshot=config_snapshot,
                start=True,
            )
        except DaemonError as exc:
            self._tasks.pop(temp_id, None)
            logger.warning("Failed to submit task to daemon: %s", exc)
            raise

        job_id = resp.get("job_id") or temp_id
        # Re-key the task under its daemon job_id so WS/REST addressing lines up.
        task.id = job_id
        task.job_id = job_id
        task.sample_dir = resp.get("sample_dir")
        self._tasks[job_id] = task
        if temp_id != job_id:
            del self._tasks[temp_id]

        task.state = TaskState.PENDING  # queued; _poll_daemon_job flips to RUNNING
        task.started_at = datetime.now(timezone.utc).isoformat()

        jsonl_path = self._derive_progress_jsonl_path(
            resp, command, args or [], env or {}
        )
        if jsonl_path:
            # Mark as a training task. Both the JSONL watcher and stdout tqdm
            # parser feed the same metrics snapshot: JSONL supplies structured
            # scalars, stdout supplies tqdm-only fields like speed / ETA.
            task.is_training = True
        self._ensure_monitors(task, jsonl_path)

        return task

    @staticmethod
    def _training_config_snapshot(
        command: str, argv: list[str], env: dict[str, str]
    ) -> Optional[dict]:
        """Capture the GUI merged config at submission time.

        The daemon stores this under the job directory and uses it for an
        explicit resume. Staged-resolution training gets the generated runtime
        config plus its profile manifest pinned at submission time; other
        command jobs do not need a synthetic snapshot.
        """
        if command == "staged-train":
            profile = argv[2] if len(argv) >= 3 and not argv[2].startswith("-") else "default"
            try:
                import toml
                from library.training.staged_resolution_plan import (
                    compile_runtime_config,
                    load_profile,
                    manifest_path,
                )

                plan = load_profile(str(profile), ROOT)
                runtime_path = compile_runtime_config(str(profile), plan, ROOT)
                snapshot = toml.load(runtime_path)
                snapshot["dataset_manifest"] = str(manifest_path(str(profile), ROOT))
                return snapshot
            except (FileNotFoundError, ValueError, OSError):
                logger.warning("Could not capture staged training snapshot for %s", profile)
                return None
        if command != "lora-gui":
            return None
        variant = env.get("GUI_PRESETS")
        if not variant and len(argv) >= 3 and not argv[2].startswith("-"):
            variant = argv[2]
        variant = variant or "lora"
        try:
            from webui.services.config_service import merged_gui_variant_preset

            merged, origin = merged_gui_variant_preset(
                str(variant), env.get("PRESET") or "default"
            )
        except (FileNotFoundError, ValueError, OSError):
            logger.warning("Could not capture GUI config snapshot for %s", variant)
            return None
        # ``base.toml`` carries argparse's default max_train_steps for legacy
        # configs, while GUI variants normally select an epoch-only budget.
        # Persisting that inherited default in a standalone snapshot would make
        # read_config_from_file treat it as an explicit step budget on resume.
        if (
            origin.get("max_train_steps") == "base"
            and origin.get("max_train_epochs") in {"method", "preset"}
        ):
            merged.pop("max_train_steps", None)
        for index, token in enumerate(argv):
            if token == "--preprocess_run" and index + 1 < len(argv):
                merged["preprocess_run"] = argv[index + 1]
                break
        return merged

    async def cancel_task(self, task_id: str) -> bool:
        """Request a stop and mirror the daemon until it reaches a terminal state.

        A training stop can spend time saving a complete checkpoint, so the
        local ``stopping`` state must not terminate the daemon poller early.
        """
        task = self._tasks.get(task_id)
        if not task or task.job_id is None:
            return False
        if task.state == TaskState.STOPPING:
            return True
        # Allow cancel from PENDING (queued) or RUNNING.
        if task.state not in (TaskState.RUNNING, TaskState.PENDING):
            return False
        try:
            await daemon_client.stop(task.job_id)
        except DaemonError:
            logger.exception("Failed to stop job %s", task.job_id)
            # The poller may have observed a terminal daemon record while the
            # stop request was in flight. Do not report a spurious failure in
            # that race, and never overwrite the authoritative terminal state.
            return task.state in (
                TaskState.SUCCESS,
                TaskState.FAILED,
                TaskState.CANCELLED,
            )
        if task.state in (TaskState.SUCCESS, TaskState.FAILED, TaskState.CANCELLED):
            return True
        task.state = TaskState.STOPPING
        task.lines.append(f"[stopping] Stop requested for task {task_id}")
        task.lines_total += 1
        await self._notify_subscribers(task, {"type": "stopping", "task_id": task_id})
        return True

    def subscribe(self, task_id: str) -> asyncio.Queue:
        """Subscribe to log lines for a task. Returns an asyncio.Queue."""
        queue: asyncio.Queue = asyncio.Queue()
        task = self._get_or_load_task(task_id)
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
        while task.state in (
            TaskState.PENDING,
            TaskState.RUNNING,
            TaskState.STOPPING,
        ):
            try:
                info = await daemon_client.get_job(task.job_id)
            except asyncio.CancelledError:
                raise
            except DaemonError:
                logger.debug("daemon poll failed for %s; will retry", task.job_id)
                await asyncio.sleep(poll_interval)
                continue
            except Exception:
                logger.exception("Unexpected daemon poll error for %s", task.job_id)
                await asyncio.sleep(poll_interval)
                continue

            stdout_path = info.get("stdout_path")
            if stdout_path:
                task.stdout_path = stdout_path

            dstate = info.get("state")
            if dstate == "running":
                if task.state != TaskState.STOPPING:
                    task.state = TaskState.RUNNING
                task.pid = info.get("pid")
            elif dstate == "queued":
                if task.state != TaskState.STOPPING:
                    task.state = TaskState.PENDING
            elif dstate in ("done", "error", "stopped"):
                await self._safe_drain_stdout(task)
                await self._finalize_from_daemon(task, info)
                return

            await self._safe_drain_stdout(task)
            await asyncio.sleep(poll_interval)

    async def _safe_drain_stdout(self, task: Task) -> None:
        """Keep telemetry failures from changing daemon-authoritative state."""
        try:
            await self._drain_stdout(task)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Task telemetry failed for daemon job %s", task.job_id)

    async def _finalize_from_daemon(self, task: Task, info: dict) -> None:
        """Map a terminal daemon job record onto the Task + emit ``done``."""
        dstate = info.get("state")
        task.recovery_step = info.get("recovery_step")
        task.resume_state = info.get("recovery_state")
        task.terminal_reason = info.get("terminal_reason") or info.get("status_detail") or info.get("error")
        task.resumable = bool(info.get("recovery_state"))
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
                task.lines_total += 1
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
            # The WebUI lists custom overlays as ``custom/<stem>`` and feeds
            # that identifier back here; strip the prefix (and canonicalize
            # casing on case-insensitive FSes) so load_path_overrides gets the
            # bare stem it expects. Mirrors scripts/tasks/training.py.
            method = (variant or "lora").split("/", 1)[-1]
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
        while task.state == TaskState.PENDING:
            await asyncio.sleep(0.5)
        if task.state != TaskState.RUNNING:
            return

        # Wait for the file to appear (training subprocess creates it on
        # first ``run_start``). Keep waiting for as long as the daemon says the
        # job is running; model load/compile time has no reliable upper bound.
        while task.state == TaskState.RUNNING and not os.path.isfile(jsonl_path):
            await asyncio.sleep(0.5)
        if task.state != TaskState.RUNNING:
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
                            task.progress_last_step = int(
                                task.recovery_step or metrics.step or 0
                            )
                            task.progress_last_ts = task.progress_started_at
                            if "total_steps" in ev:
                                metrics.total_steps = int(ev["total_steps"])
                            if "total_epochs" in ev:
                                metrics.total_epochs = int(ev["total_epochs"])
                            # ``sampling_enabled`` is the trainer's authoritative
                            # decision (sample_prompts set AND a trigger) — read
                            # it here instead of re-deriving from CLI argv, which
                            # carries no ``--sample_*`` flags on the GUI path.
                            if "sampling_enabled" in ev:
                                metrics.sampling_enabled = bool(ev["sampling_enabled"])
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
                            # IMPORTANT: parse loss AND lr BEFORE appending
                            # to the histories. Parsing lr after the append
                            # (the old shape) recorded the *previous* step's
                            # lr under the current step — and, before the
                            # first lr is seen, the default ``0.0``. That
                            # made the dashboard's LR curve visibly dip to
                            # zero / kink at the first logged step and at
                            # every preview-sampling boundary (the sample
                            # event arrives just before its matching step
                            # event, so the stale-lr entry lined up exactly
                            # with the sample on the chart). The loss curve
                            # showed the same kink because
                            # ``lr_history``/``step_history``/``loss_history``
                            # are index-aligned and the dashboard zips them
                            # by position. Mirror the already-correct order
                            # in ``training_log_parser._parse_tqdm``.
                            loss = ev.get("loss/average")
                            if loss is None:
                                loss = ev.get("avr_loss")
                            # Prefer the effective lr (``d*lr`` for
                            # Prodigy/D-Adaptation). New trainer builds write
                            # it directly under ``lr/unet``; legacy progress
                            # JSONL from before that change carried only the
                            # base lr there and the effective value under
                            # ``lr/d*lr/unet`` — fall back to that so old runs
                            # still show the real (rising) lr instead of a flat 1.0.
                            lr = ev.get("lr/d*lr/unet")
                            if lr is None:
                                lr = ev.get("lr/unet")
                            if lr is None:
                                lr = ev.get("lr")
                            if loss is not None:
                                metrics.avr_loss = float(loss)
                            if lr is not None:
                                metrics.lr = float(lr)
                            # Only grow the aligned histories when this
                            # step carries a loss value (a ``step`` event
                            # without ``loss/average`` — e.g. the trainer's
                            # ``loss/epoch_average`` flush — is not a
                            # plottable point). Dedupe by ``global_step``
                            # so a replayed event can't double-append.
                            if loss is not None:
                                s = metrics.step
                                metrics.upsert_step(
                                    int(s), metrics.avr_loss, metrics.lr
                                )
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
                            # ``_metrics_due`` rate-limits the full
                            # snapshot copies on top of that.
                            if self._metrics_due(task):
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
                                    "attempt_id": task.job_id,
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

    @staticmethod
    def _metrics_due(task: Task) -> bool:
        """True when a metrics snapshot may be broadcast for *task* now.

        Rate-limits the per-step/per-tqdm snapshot broadcasts (each carries
        the full loss/step/lr histories) to one per METRICS_MIN_INTERVAL.
        Snapshots are cumulative, so a dropped intermediate is fully covered
        by the next one that goes out.
        """
        now = time.monotonic()
        if now - task.last_metrics_ts < METRICS_MIN_INTERVAL:
            return False
        task.last_metrics_ts = now
        return True

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
            task.lines_total += 1
        self._cap_lines(task)

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
        # Broadcasts are rate-limited by ``_metrics_due``: snapshots are
        # cumulative, so dropping intermediates loses nothing.
        if task.parser.feed(line) and self._metrics_due(task):
            await self._notify_subscribers(
                task,
                {"type": "metrics", "data": task.parser.metrics.snapshot()},
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
