"""Pydantic schemas for task API."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class TaskState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    STOPPING = "stopping"


class TaskInfo(BaseModel):
    task_id: str
    current_job_id: Optional[str] = None
    attempt_count: int = 1
    attempts: list[dict] = Field(default_factory=list)
    command: str
    state: TaskState
    pid: Optional[int] = None
    exit_code: Optional[int] = None
    output_lines: int = 0
    recovery_step: Optional[int] = None
    resume_state: Optional[str] = None
    terminal_reason: Optional[str] = None
    resumable: bool = False
    started_at: Optional[str] = None
    category: str = "task"
