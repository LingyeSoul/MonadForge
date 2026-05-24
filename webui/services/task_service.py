"""Async subprocess lifecycle manager for tasks.py commands."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import re
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
    process: Optional[asyncio.subprocess.Process] = field(default=None, repr=False)
    started_at: Optional[str] = field(default=None)
    _subscribers: list[asyncio.Queue] = field(default_factory=list, repr=False)
    parser: TrainingLogParser = field(default_factory=TrainingLogParser, repr=False)
    wandb_run_url: Optional[str] = None

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
    """Manages subprocess lifecycle for tasks.py commands."""

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}
        self._python = sys.executable

    def list_tasks(self) -> list[dict]:
        return [t.info() for t in self._tasks.values()]

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
        """Launch ``python tasks.py <command> [args...]`` as a subprocess."""
        task_id = uuid.uuid4().hex[:12]
        task = Task(id=task_id, command=command, args=args or [])
        self._tasks[task_id] = task

        cmd = [self._python, "tasks.py", command, *(args or [])]
        merged_env = {**os.environ, **(env or {})}
        # Force unbuffered stdout/stderr so readline() sees output immediately.
        # Without this, Python detects the pipe and uses full buffering (4-8 KB),
        # causing the first N kilobytes of output to be silently lossy.
        merged_env["PYTHONUNBUFFERED"] = "1"
        # Force child Python processes to use UTF-8 for stdout/stderr.
        # Without this, Windows uses the system locale encoding (e.g. GBK/CP936
        # on Chinese Windows), causing non-ASCII characters to be garbled when
        # decoded as UTF-8 on the receiving end.
        merged_env["PYTHONIOENCODING"] = "utf-8"

        logger.info("Starting task %s: %s (env override keys: %s)", task_id, " ".join(cmd), list((env or {}).keys()))
        if env and "PRESET" in env:
            logger.info("Task %s PRESET env override: %r", task_id, env["PRESET"])

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(ROOT),
                env=merged_env,
            )
            task.process = process
            task.pid = process.pid
            task.state = TaskState.RUNNING
            task.started_at = datetime.now(timezone.utc).isoformat()

            # Fire-and-forget reader — updates task state when done
            asyncio.create_task(self._read_output(task))

            # Tail the structured progress JSONL for richer metrics.
            # The training subprocess auto-derives the path from
            # output_dir/output_name when --progress_jsonl is not passed.
            jsonl_path = self._derive_progress_jsonl_path(args or [])
            if jsonl_path:
                asyncio.create_task(self._watch_progress_jsonl(task, jsonl_path))
        except Exception as exc:
            task.state = TaskState.FAILED
            task.lines.append(f"[error] Failed to start: {exc}")
            logger.exception("Failed to start task %s", task_id)

        return task

    async def cancel_task(self, task_id: str) -> bool:
        """Terminate a running task (tree-kill on Windows)."""
        task = self._tasks.get(task_id)
        if not task or task.state != TaskState.RUNNING:
            return False
        if task.process is None:
            return False

        try:
            if sys.platform == "win32":
                # /T kills the process tree, /F forces
                os.system(f"taskkill /PID {task.process.pid} /T /F")
            else:
                task.process.terminate()
            task.state = TaskState.CANCELLED
            task.lines.append(f"[cancelled] Task {task_id} terminated by user")
            await self._notify_subscribers(
                task, {"type": "cancelled", "task_id": task_id}
            )
            return True
        except Exception:
            logger.exception("Failed to cancel task %s", task_id)
            return False

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

    async def _read_output(self, task: Task) -> None:
        """Read subprocess stdout and dispatch to subscribers.

        Splits on both ``\\n`` and ``\\r`` so tqdm progress-bar updates
        (which use bare ``\\r`` to overwrite the previous line) arrive as
        individual messages instead of being concatenated into one giant line.

        Lines ending with ``\\r`` (progress updates) are sent with
        ``replace: true`` so the frontend can overwrite the previous line
        instead of appending.  They also replace the last entry in
        ``task.lines`` so that late-joining subscribers (REST replay) see
        only the final progress-bar state.
        """
        assert task.process is not None
        assert task.process.stdout is not None

        _LF = ord("\n")
        _CR = ord("\r")
        buf = bytearray()
        pending_cr = False  # \r at end of previous chunk; check next chunk for \n
        try:
            while True:
                chunk = await task.process.stdout.read(4096)
                if not chunk:
                    break
                i = 0
                end = len(chunk)
                # Handle \r from previous chunk boundary
                if pending_cr:
                    pending_cr = False
                    line = bytes(buf).decode("utf-8", errors="replace")
                    buf.clear()
                    if chunk[0] == _LF:
                        # \r\n across chunk boundary → regular line
                        i = 1
                        await self._emit_line(task, line, replace=False)
                    else:
                        # bare \r → progress update
                        await self._emit_line(task, line, replace=True)
                while i < end:
                    b = chunk[i]
                    if b == _CR:
                        line = bytes(buf).decode("utf-8", errors="replace")
                        buf.clear()
                        if i + 1 < end:
                            # \r\n within same chunk → regular line ending
                            if chunk[i + 1] == _LF:
                                i += 2
                                await self._emit_line(task, line, replace=False)
                                continue
                            # bare \r → tqdm progress update
                            i += 1
                            await self._emit_line(task, line, replace=True)
                        else:
                            # \r at chunk boundary — defer until next chunk
                            pending_cr = True
                            i += 1
                    elif b == _LF:
                        line = bytes(buf).decode("utf-8", errors="replace")
                        buf.clear()
                        i += 1
                        await self._emit_line(task, line, replace=False)
                    else:
                        buf.append(b)
                        i += 1

            # Flush remaining bytes (partial line without terminator)
            if buf or pending_cr:
                line = bytes(buf).decode("utf-8", errors="replace")
                await self._emit_line(
                    task, line, replace=pending_cr
                )

            task.exit_code = await task.process.wait()
            if task.state == TaskState.CANCELLED:
                pass  # already set
            elif task.exit_code == 0:
                task.state = TaskState.SUCCESS
            else:
                task.state = TaskState.FAILED

            msg = {
                "type": "done",
                "exit_code": task.exit_code,
                "state": task.state.value,
            }
            await self._notify_subscribers(task, msg)
        except Exception:
            logger.exception("Error reading output for task %s", task.id)
            task.state = TaskState.FAILED
            await self._notify_subscribers(
                task, {"type": "done", "exit_code": -1, "state": "failed"}
            )

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

    def _derive_progress_jsonl_path(self, args: list[str]) -> Optional[str]:
        """Mirror ``ProgressSink.resolve_path`` logic on the arg list.

        Returns the expected JSONL path so the watcher can tail it, or
        ``None`` when the path cannot be determined (e.g. no output_dir).
        """
        output_dir = self._arg_value(args, "--output_dir") or "output/ckpt"
        output_name = self._arg_value(args, "--output_name") or "anima_lora"
        parent = os.path.dirname(os.path.normpath(output_dir))
        logs_dir = os.path.join(parent or output_dir, "logs")
        return os.path.join(ROOT, logs_dir, f"{output_name}.progress.jsonl")

    async def _watch_progress_jsonl(
        self, task: Task, jsonl_path: str
    ) -> None:
        """Tail the structured progress JSONL and push metrics to subscribers.

        Runs alongside ``_read_output``.  The training subprocess writes
        line-buffered JSONL events (``{"ev":"step", ...}``) that carry
        structured scalars (loss, lr, …) at the ``log_every_n_steps``
        cadence.  This coroutine polls the file for new data and emits
        ``{"type":"metrics", "data":{...}}`` messages to WebSocket
        subscribers — the same message format the stdout parser produces,
        so the frontend code is unchanged.
        """
        # Wait for the file to appear (training subprocess creates it on
        # first run_start).  Give up after 60 s — if the file never
        # appears the stdout fallback still works.
        for _ in range(120):
            if task.state != TaskState.RUNNING:
                return
            if os.path.isfile(jsonl_path):
                break
            await asyncio.sleep(0.5)
        else:
            logger.debug(
                "progress JSONL not found after 60 s, skipping watcher: %s",
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
                result: Optional[tuple[list[str], int]] = (
                    await asyncio.to_thread(
                        self._read_jsonl_bytes, jsonl_path, offset
                    )
                )
                if result is not None:
                    new_lines, new_offset = result
                    offset = new_offset
                    for raw_line in new_lines:
                        try:
                            ev = json.loads(raw_line)
                        except (json.JSONDecodeError, ValueError):
                            continue
                        if ev.get("ev") != "step":
                            continue
                        metrics = task.parser.metrics
                        if "global_step" in ev:
                            metrics.step = ev["global_step"]
                        if "epoch" in ev:
                            metrics.epoch = ev["epoch"]
                        if "avr_loss" in ev:
                            loss = ev["avr_loss"]
                            metrics.avr_loss = loss
                            s = metrics.step
                            if (
                                not metrics.step_history
                                or s != metrics.step_history[-1]
                            ):
                                metrics.loss_history.append(loss)
                                metrics.step_history.append(s)
                                metrics.lr_history.append(metrics.lr)
                        if "lr" in ev:
                            metrics.lr = ev["lr"]
                        snapshot = metrics.snapshot()
                        await self._notify_subscribers(
                            task,
                            {"type": "metrics", "data": snapshot},
                        )
                else:
                    # File may have been rotated; reset.
                    offset = 0
                await asyncio.sleep(0.3)
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception(
                "Progress JSONL watcher failed for task %s", task.id
            )

    @staticmethod
    def _read_jsonl_bytes(
        path: str, offset: int
    ) -> Optional[tuple[list[str], int]]:
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
        return [l for l in lines if l.strip()], end

    async def _emit_line(
        self, task: Task, line: str, *, replace: bool
    ) -> None:
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

        # Parse training metrics from the line
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
