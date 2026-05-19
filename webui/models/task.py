"""Pydantic schemas for task API."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel


class TaskState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskInfo(BaseModel):
    id: str
    command: str
    state: TaskState
    pid: Optional[int] = None
    exit_code: Optional[int] = None
    lines: int = 0
