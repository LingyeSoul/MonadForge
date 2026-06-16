"""Preview-image API for the training dashboard.

Exposes the per-task ``output_dir/sample/`` directory so the dashboard can
both list the PNG previews already written and stream individual files for
the gallery + click-to-enlarge dialog.

Path safety: every served file must live under the task's registered
``output_dir/sample/`` root. Anything outside (or with ``..`` components, or
matching a non-image extension) is rejected with 404.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

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
    """Resolve the on-disk ``<output_dir>/sample`` directory for *task_id*.

    Returns ``None`` when the task isn't tracked, sampling was never enabled
    (``--sample_prompts`` not set on the command line), or the directory
    doesn't exist yet (training hasn't reached its first sample event).
    """
    task = task_service.get_task(task_id)
    if task is None:
        return None

    output_dir = task_service._arg_value(task.args, "--output_dir")
    if not output_dir:
        # Fall back to the default: ``output/ckpt`` (matches the
        # _derive_progress_jsonl_path default).
        output_dir = "output/ckpt"

    sample_dir = ROOT / output_dir / "sample"
    # Resolve then guard against a path that escapes ROOT via a malicious
    # ``--output_dir`` arg (e.g. ``../../../etc``).
    try:
        resolved = sample_dir.resolve()
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        return None
    return resolved if resolved.is_dir() else None


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
        return SampleListResponse(
            task_id=task_id, sample_dir=None, items=[], total=0
        )

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
            status_code=404, detail="No samples for this task (sampling disabled or not yet generated)"
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
