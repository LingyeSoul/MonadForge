"""Task management API endpoints."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field

from webui.services.daemon_client import DaemonError
from webui.services.task_catalog import COMMAND_CATALOG
from webui.services.task_service import task_service

router = APIRouter()


# Flags the WebUI caller must NOT set: they're either injected by the daemon
# (``--progress_jsonl`` / ``--sample_dir`` → per-job paths the daemon owns, and
# letting the caller override them breaks dashboard gallery isolation) or they
# point train.py at arbitrary filesystem locations (``--output_dir`` /
# ``--config_file`` / ``--dataset_config`` → arbitrary dir-create / file-write
# primitive, since train.py ``os.makedirs``/writes at these). The daemon runs
# localhost no-auth, but the WebUI is the public surface — filter at the edge.
_FORBIDDEN_ARG_FLAGS = frozenset(
    {
        "--sample_dir",
        "--output_dir",
        "--progress_jsonl",
        "--config_file",
        "--dataset_config",
    }
)


class TaskStartRequest(BaseModel):
    command: str = Field(min_length=1, max_length=64)
    args: list[str] = Field(default_factory=list, max_length=256)
    env: dict[str, str] = Field(default_factory=dict, max_length=64)


class TaskCommandListResponse(BaseModel):
    commands: dict[str, str]


_COMMAND_DESCRIPTIONS = COMMAND_CATALOG


@router.get("", response_model=list[dict])
async def list_tasks(
    response: Response,
    state: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
):
    """List one durable task-history page.

    Keep the response body backward-compatible as a list.  Pagination metadata
    is exposed in headers so older consumers do not need a schema migration.
    """
    page, total = await task_service.list_tasks_page(
        state=state,
        limit=min(max(limit, 1), 500),
        offset=max(offset, 0),
    )
    response.headers["X-Total-Count"] = str(total)
    response.headers["X-Page-Offset"] = str(max(offset, 0))
    response.headers["X-Page-Limit"] = str(min(max(limit, 1), 500))
    return page


@router.get("/commands", response_model=TaskCommandListResponse)
def list_commands():
    """Return available task commands with descriptions."""
    return TaskCommandListResponse(commands=_COMMAND_DESCRIPTIONS)


# ── Queue control ──────────────────────────────────────────────────────────
# These MUST be declared before /{task_id} routes, otherwise FastAPI matches
# "queue" as a task_id.


@router.get("/queue/status")
async def queue_status():
    """Queue snapshot: daemon_up + paused + {job_id: queue_position} map.

    Polled by the Tasks view (5s) alongside ``GET /api/tasks`` to render
    per-task queue positions and the pause/resume button state.
    """
    return await task_service.get_queue_status()


@router.post("/queue/pause")
async def queue_pause():
    """Hold the queue — queued jobs wait until ``/queue/resume``."""
    try:
        return await task_service.pause_queue()
    except DaemonError as exc:
        raise HTTPException(status_code=502, detail=f"daemon: {exc}") from exc


@router.post("/queue/resume")
async def queue_resume():
    """Release a paused queue — the worker launches queued jobs in order."""
    try:
        return await task_service.resume_queue()
    except DaemonError as exc:
        raise HTTPException(status_code=502, detail=f"daemon: {exc}") from exc


class DaemonShutdownRequest(BaseModel):
    kill_jobs: bool = True
    mode: Optional[str] = None


@router.post("/daemon/shutdown")
async def daemon_shutdown(body: DaemonShutdownRequest):
    """Fully stop the training daemon (complete exit).

    A connection-reset / 5xx is expected here when the daemon hosts the WebUI
    as a sidecar — its shutdown tree-kills this server too. The shutdown has
    still been triggered in that case.
    """
    if body.mode is not None and body.mode not in {"detach", "cooperative-stop", "force"}:
        raise HTTPException(
            status_code=400,
            detail="shutdown mode must be detach, cooperative-stop, or force",
        )
    try:
        return await task_service.shutdown_daemon(kill_jobs=body.kill_jobs, mode=body.mode)
    except DaemonError as exc:
        raise HTTPException(status_code=502, detail=f"daemon: {exc}") from exc


@router.get("/{task_id}")
def get_task(task_id: str):
    """Get task status."""
    info = task_service.get_task_info(task_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return info


@router.get("/{task_id}/output")
def get_task_output(task_id: str):
    """Get accumulated output lines for a task.

    ``lines`` is capped server-side (oldest dropped FIFO); ``total`` is the
    all-time line count so clients can show a truncation notice.
    """
    task = task_service.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return {
        "lines": task.lines,
        "state": task.state.value,
        "exit_code": task.exit_code,
        "total": task.lines_total,
        "truncated": task.lines_total > len(task.lines),
    }


@router.get("/{task_id}/metrics")
def get_task_metrics(task_id: str):
    """Get parsed training metrics for a task."""
    metrics = task_service.get_task_metrics(task_id)
    if metrics is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return metrics


@router.post("")
async def start_task(body: TaskStartRequest):
    """Start a new task."""
    if body.command not in _COMMAND_DESCRIPTIONS:
        raise HTTPException(status_code=400, detail=f"Unknown command: {body.command}")
    _reject_forbidden_args(body.args)
    try:
        task = await task_service.start_task(body.command, body.args, body.env or None)
    except DaemonError as exc:
        raise HTTPException(
            status_code=502, detail="Training daemon is unavailable"
        ) from exc
    return task.info()


def _reject_forbidden_args(args: list[str]) -> None:
    """Reject path/daemon-owned flags a WebUI caller must not set.

    Matches both ``--flag value`` and ``--flag=value`` forms; the daemon's own
    argv injection uses ``not in`` (bare-flag only), so filtering here closes
    the ``--flag=value`` bypass at the public edge.
    """
    for a in args:
        token = a.split("=", 1)[0]
        if token in _FORBIDDEN_ARG_FLAGS:
            raise HTTPException(
                status_code=400,
                detail=f"Argument {token} is reserved (set by the daemon/train.py, "
                "not the WebUI).",
            )


@router.delete("/{task_id}")
async def stop_task(task_id: str):
    """Request cancellation; completion is reported after daemon teardown."""
    ok = await task_service.cancel_task(task_id)
    if not ok:
        raise HTTPException(
            status_code=404, detail=f"Task {task_id} not found or not cancellable"
        )
    return {"status": "stopping", "task_id": task_id}


@router.post("/{task_id}/resume")
async def resume_task(task_id: str):
    try:
        task = await task_service.resume_task(task_id)
    except DaemonError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return task.info()


@router.delete("/{task_id}/history")
async def delete_history(task_id: str):
    try:
        ok = await task_service.delete_task(task_id)
    except DaemonError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not ok:
        raise HTTPException(status_code=409, detail="Only terminal tasks can be deleted")
    return {"status": "deleted", "task_id": task_id}
