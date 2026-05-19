"""Async subprocess lifecycle manager for tasks.py commands."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

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

    def info(self) -> dict:
        return {
            "task_id": self.id,
            "command": self.command,
            "state": self.state.value,
            "pid": self.pid,
            "exit_code": self.exit_code,
            "output_lines": len(self.lines),
            "started_at": self.started_at,
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
        # causing the first N kilobytes of output to be silently lost.
        merged_env["PYTHONUNBUFFERED"] = "1"

        logger.info("Starting task %s: %s", task_id, " ".join(cmd))

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
        """Read subprocess stdout line-by-line and dispatch to subscribers."""
        assert task.process is not None
        assert task.process.stdout is not None
        try:
            while True:
                raw = await task.process.stdout.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").rstrip("\n\r")
                task.lines.append(line)
                await self._notify_subscribers(task, {"type": "log", "line": line})

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
