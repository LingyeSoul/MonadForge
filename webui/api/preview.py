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

# Reject only the path-traversal vectors: NUL/control bytes, any path separator
# (the served path is meant to be a bare filename within the sample dir), and a
# ``..`` segment. We deliberately do NOT use a positive character whitelist:
# ``output_name`` is user-controlled and legitimately contains characters
# outside ``[a-zA-Z0-9_./-]`` (e.g. ``@``, spaces, CJK, parens), and a whitelist
# silently broke the gallery for such runs — the file showed up in the listing
# but ``<img>`` got a 400 and fell back to the filename text. Containment is
# still enforced by the ``resolve() / relative_to(sample_dir)`` check below;
# this check is defense-in-depth against the traversal itself.
_UNSAFE_PATH = re.compile(r"[\x00-\x1f\\/]|\.\.|^\.")


class SampleImage(BaseModel):
    attempt_id: str
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


def _resolve_task_sample_dir(task_id: str, attempt_id: str | None = None) -> Path | None:
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
        if attempt_id:
            if task.attempts:
                attempt = next(
                    (
                        item
                        for item in task.attempts
                        if str(item.get("id") or item.get("job_id")) == attempt_id
                    ),
                    None,
                )
                if attempt is None:
                    return None
                sample_dir_raw = attempt.get("sample_dir")
                if not sample_dir_raw:
                    return None
            else:
                current_attempt_id = str(task.job_id or task.id)
                if attempt_id != current_attempt_id:
                    return None
                sample_dir_raw = task.sample_dir
        else:
            sample_dir_raw = task.sample_dir
        sample_dir_raw = sample_dir_raw or task_service._arg_value(
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
        sample_dir_raw = _sample_dir_from_daemon_job(task_id, attempt_id)

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


def _sample_dir_from_daemon_job(job_id: str, attempt_id: str | None = None) -> str | None:
    """Recover ``sample_dir`` from the daemon's persisted job record.

    Used when the WebUI's in-memory task list no longer knows about *job_id*
    (after a restart). Tolerates the daemon being down — returns ``None``
    rather than propagating, so preview simply reports "no samples".
    """
    try:
        group = _daemon_client.get_job_group_sync(job_id)
    except (DaemonError, OSError):
        try:
            group = _daemon_client.get_job_sync(job_id)
        except (DaemonError, OSError):
            return None
    if not isinstance(group, dict):
        return None
    attempts = list(group.get("attempts") or [group])
    target_id = attempt_id or str(group.get("current_job_id") or job_id)
    attempt = next(
        (
            item
            for item in attempts
            if str(item.get("id") or item.get("job_id")) == target_id
        ),
        None,
    )
    if attempt is None and attempt_id is None and attempts:
        attempt = attempts[-1]
    return attempt.get("sample_dir") if attempt else None


def _resolve_task_sample_dirs(task_id: str) -> list[tuple[str, Path]]:
    task = task_service.get_task(task_id)
    if task is None:
        return []
    attempts = task.attempts or [
        {"id": task.job_id or task.id, "sample_dir": task.sample_dir}
    ]
    out: list[tuple[str, Path]] = []
    for attempt in attempts:
        attempt_id = str(attempt.get("id") or attempt.get("job_id") or "")
        if not attempt_id:
            continue
        sample_dir = _resolve_task_sample_dir(task_id, attempt_id)
        if sample_dir is not None:
            out.append((attempt_id, sample_dir))
    return out


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

    sample_dirs = _resolve_task_sample_dirs(task_id)
    if not sample_dirs:
        return SampleListResponse(task_id=task_id, sample_dir=None, items=[], total=0)

    all_files = [
        (attempt_id, sample_dir, path)
        for attempt_id, sample_dir in sample_dirs
        for path in _scan_sample_dir(sample_dir)
    ]
    all_files.sort(key=lambda item: item[2].stat().st_mtime, reverse=True)
    total = len(all_files)
    start = (page - 1) * page_size
    page_files = all_files[start : start + page_size]

    items: list[SampleImage] = []
    for attempt_id, _sample_dir, p in page_files:
        try:
            stat = p.stat()
        except OSError:
            continue
        items.append(
            SampleImage(
                attempt_id=attempt_id,
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
        sample_dir=str(sample_dirs[-1][1]),
        items=items,
        total=total,
    )


@router.get("/runs/{task_id}/samples/file")
def get_sample_file(
    task_id: str,
    path: str = Query(..., description="Filename within the task's sample dir"),
    attempt_id: str | None = None,
):
    """Stream a preview PNG by filename.

    ``path`` must be a plain filename (no slashes, no ``..``); we still
    resolve it under the task's sample dir and verify it stays inside.
    """
    task = task_service.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    if _UNSAFE_PATH.search(path):
        raise HTTPException(status_code=400, detail="Invalid sample path")

    sample_dir = _resolve_task_sample_dir(task_id, attempt_id)
    if sample_dir is None and attempt_id is None:
        # Backward compatibility for clients that only send a filename: pick
        # the newest matching artifact across the logical task.
        matches = []
        for _candidate_attempt, candidate_dir in _resolve_task_sample_dirs(task_id):
            candidate = (candidate_dir / path).resolve()
            if candidate.is_file():
                matches.append((candidate.stat().st_mtime, candidate_dir))
        if matches:
            sample_dir = max(matches, key=lambda item: item[0])[1]
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
