"""Preview-image API for the training dashboard.

Exposes the per-task ``output_dir/sample/`` directory so the dashboard can
both list the PNG previews already written and stream individual files for
the gallery + click-to-enlarge dialog.

Path safety: every served file must live under the task's registered
``output_dir/sample/`` root. Anything outside (or with ``..`` components, or
matching a non-image extension) is rejected with 404.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from webui.services.daemon_client import DaemonError, daemon_client as _daemon_client
from webui.services.task_service import ROOT, task_service

router = APIRouter()

_SAMPLE_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

# Whitelist: only [a-zA-Z0-9_./-] is allowed in the relative path. No spaces,
# no NULs, no parent-traversal dots-as-segments — combined with the
# ``.resolve() / relative_to(root)`` check below, this is defense-in-depth.
_SAFE_REL = re.compile(r"^[a-zA-Z0-9_./-]+$")


class SampleImage(BaseModel):
    path: str  # relative to the task's sample dir
    filename: str
    stem: str
    size: int
    mtime: str
    mtime_unix: float


class SampleListResponse(BaseModel):
    task_id: str
    sample_dir: str | None
    items: list[SampleImage]
    total: int


def _resolve_task_sample_dir(task_id: str) -> Path | None:
    """Resolve the on-disk sample directory for *task_id*.

    Resolution order (mirrors how ``--progress_jsonl`` is located):

    1. ``task.sample_dir`` — the daemon injects ``<job_dir>/sample`` and
       returns it on submit; this is the per-job isolation path that keeps a
       new task's gallery from replaying the previous task's previews.
    2. ``--sample_dir`` in ``task.args`` — covers a direct ``train.py`` run or
       a task from before per-job isolation.
    3. ``<output_dir>/sample`` — final fallback for non-daemon tasks.

    When the task isn't tracked in this WebUI session (e.g. after a restart —
    ``_tasks`` is session-only memory), fall back to the daemon's persisted job
    record, whose ``sample_dir`` survives restarts. Returns ``None`` when the
    directory doesn't exist yet (training hasn't reached its first sample
    event) or when the task isn't known to either the WebUI or the daemon.
    """
    task = task_service.get_task(task_id)
    sample_dir_raw: str | None = None
    if task is not None:
        sample_dir_raw = task.sample_dir or task_service._arg_value(
            task.args, "--sample_dir"
        )
        if not sample_dir_raw:
            output_dir = (
                task_service._arg_value(task.args, "--output_dir") or "output/ckpt"
            )
            sample_dir_raw = os.path.join(output_dir, "sample")
    else:
        # Session-only task list lost (WebUI restarted). The daemon persists
        # job records (including sample_dir) to job.json — recover from there.
        sample_dir_raw = _sample_dir_from_daemon_job(task_id)

    if not sample_dir_raw:
        return None
    sample_dir = ROOT / sample_dir_raw

    # Resolve then guard against a path that escapes ROOT via a malicious
    # ``--sample_dir`` / ``--output_dir`` arg (e.g. ``../../../etc``).
    try:
        resolved = sample_dir.resolve()
        resolved.relative_to(ROOT.resolve())
    except (ValueError, OSError):
        return None
    return resolved if resolved.is_dir() else None


def _sample_dir_from_daemon_job(job_id: str) -> str | None:
    """Recover ``sample_dir`` from the daemon's persisted job record.

    Used when the WebUI's in-memory task list no longer knows about *job_id*
    (after a restart). Tolerates the daemon being down — returns ``None``
    rather than propagating, so preview simply reports "no samples".
    """
    try:
        job = _daemon_client.get_job_sync(job_id)
    except (DaemonError, OSError):
        return None
    if not isinstance(job, dict):
        return None
    return job.get("sample_dir")


def _scan_sample_dir(sample_dir: Path) -> list[Path]:
    """Return PNG files in *sample_dir* (non-recursive), newest first."""
    try:
        files = [p for p in sample_dir.iterdir() if p.is_file()]
    except OSError:
        return []
    return sorted(
        (p for p in files if p.suffix.lower() in _SAMPLE_IMAGE_EXTS),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


@router.get("/runs/{task_id}/samples", response_model=SampleListResponse)
def list_task_samples(
    task_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(60, ge=1, le=500),
):
    """List previews already written for *task_id* (newest first)."""
    task = task_service.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    sample_dir = _resolve_task_sample_dir(task_id)
    if sample_dir is None:
        return SampleListResponse(task_id=task_id, sample_dir=None, items=[], total=0)

    all_files = _scan_sample_dir(sample_dir)
    total = len(all_files)
    start = (page - 1) * page_size
    page_files = all_files[start : start + page_size]

    items: list[SampleImage] = []
    for p in page_files:
        try:
            stat = p.stat()
        except OSError:
            continue
        items.append(
            SampleImage(
                path=p.name,
                filename=p.name,
                stem=p.stem,
                size=stat.st_size,
                mtime=datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat(),
                mtime_unix=stat.st_mtime,
            )
        )

    return SampleListResponse(
        task_id=task_id,
        sample_dir=str(sample_dir),
        items=items,
        total=total,
    )


@router.get("/runs/{task_id}/samples/file")
def get_sample_file(
    task_id: str,
    path: str = Query(..., description="Filename within the task's sample dir"),
):
    """Stream a preview PNG by filename.

    ``path`` must be a plain filename (no slashes, no ``..``); we still
    resolve it under the task's sample dir and verify it stays inside.
    """
    task = task_service.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    if not _SAFE_REL.match(path) or ".." in path.split("/"):
        raise HTTPException(status_code=400, detail="Invalid sample path")

    sample_dir = _resolve_task_sample_dir(task_id)
    if sample_dir is None:
        raise HTTPException(
            status_code=404,
            detail="No samples for this task (sampling disabled or not yet generated)",
        )

    candidate = (sample_dir / path).resolve()
    try:
        candidate.relative_to(sample_dir)
    except ValueError:
        raise HTTPException(status_code=403, detail="Path escapes sample dir")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail=f"Sample not found: {path}")
    if candidate.suffix.lower() not in _SAMPLE_IMAGE_EXTS:
        raise HTTPException(status_code=400, detail="Not a sample image")

    return FileResponse(str(candidate))
