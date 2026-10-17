"""MOSS forms, profiles and task-bound results on the existing daemon queue."""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Iterator, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from library.moss.requests import CacheRequest, TrainRequest
from webui.services import moss_service as svc, moss_workspace as workspace
from webui.services.daemon_client import DaemonError, daemon_client
from webui.services.task_service import task_service

router = APIRouter()


class JobResponse(BaseModel):
    task_id: str
    command: str


@contextmanager
def workspace_errors() -> Iterator[None]:
    try:
        yield
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DaemonError as exc:
        raise HTTPException(
            status_code=502, detail=f"MOSS daemon request failed: {exc}"
        ) from exc


@router.get("/schema", response_model=svc.FormSchema)
def schema() -> svc.FormSchema:
    return svc.read_schema()


@router.post("/resolve", response_model=svc.ResolvedPaths)
def resolve(body: svc.ValuesRequest) -> svc.ResolvedPaths:
    with workspace_errors():
        return svc.resolve_values(body.values)


@router.get("/profiles", response_model=list[str])
def profiles() -> list[str]:
    with workspace_errors():
        return workspace.list_profiles()


@router.get("/profiles/{name}", response_model=workspace.Profile)
def profile(name: str) -> workspace.Profile:
    with workspace_errors():
        return workspace.read_profile(name)


@router.put("/profiles/{name}", response_model=workspace.Profile)
def put_profile(name: str, body: workspace.Profile) -> workspace.Profile:
    with workspace_errors():
        return workspace.save_profile(name, body)


@router.post("/cache-status", response_model=workspace.CacheStatus)
def cache_status(body: svc.ValuesRequest) -> workspace.CacheStatus:
    with workspace_errors():
        req = svc.request_from_values(CacheRequest, body.values)
        return workspace.inspect_cache(req.out)


@router.post("/preflight", response_model=workspace.CacheStatus)
def preflight(body: svc.ValuesRequest) -> workspace.CacheStatus:
    with workspace_errors():
        req = svc.request_from_values(TrainRequest, body.values)
        svc.validate_paths(req)
        return workspace.inspect_cache(req.cache)


@router.post("/jobs/{phase}", response_model=JobResponse)
async def submit_job(
    phase: Literal["cache", "train", "generate", "tune"], body: svc.ValuesRequest
) -> JobResponse:
    with workspace_errors():
        command = f"moss-{phase}"
        req = svc.request_from_values(svc.REQUEST_TYPES[command], body.values)
        task = await task_service.start_task(
            command,
            req.to_argv(),
            context={"model_type": "moss", "phase": phase},
            runtime_profile=body.runtime_profile,
        )
        return JobResponse(task_id=task.id, command=task.command)


@router.post("/chain", response_model=JobResponse)
async def chain(body: svc.ChainRequest) -> JobResponse:
    with workspace_errors():
        cache = svc.request_from_values(CacheRequest, body.cache.values)
        train = svc.request_from_values(TrainRequest, body.train.values)
        args = [
            "--cache-args",
            json.dumps(cache.to_argv()),
            "--train-args",
            json.dumps(train.to_argv()),
        ]
        task = await task_service.start_task(
            "moss-cache-train",
            args,
            context={"model_type": "moss", "phase": "cache-train"},
            runtime_profile=body.runtime_profile,
        )
        return JobResponse(task_id=task.id, command=task.command)


async def task_request(task_id: str, command: str):
    job = await daemon_client.get_job(task_id)
    if job.get("kind") != "command" or job.get("method") != command:
        raise ValueError(f"Task {task_id} is not a {command} job")
    argv = job.get("argv") or []
    module = {
        "moss-generate": "scripts.moss.generate",
        "moss-tune": "scripts.moss.tune_swap",
        "moss-train": "scripts.moss.train",
    }[command]
    if argv[:2] not in (["tasks.py", command], ["-m", module]):
        raise ValueError(f"Unsupported persisted MOSS command for task {task_id}")
    try:
        return svc.REQUEST_TYPES[command].from_argv(argv[2:])
    except SystemExit as exc:
        raise ValueError(
            f"Invalid persisted MOSS arguments for task {task_id}"
        ) from exc


@router.get("/tasks/{task_id}/results", response_model=svc.ResultManifest)
async def task_results(task_id: str) -> svc.ResultManifest:
    with workspace_errors():
        req = await task_request(task_id, "moss-generate")
        return svc.read_results(req.out_dir)


@router.get("/tasks/{task_id}/samples/file")
async def task_file(task_id: str, path: str) -> FileResponse:
    with workspace_errors():
        req = await task_request(task_id, "moss-generate")
        manifest = svc.read_results(req.out_dir)
        if path not in {image.file for image in manifest.images}:
            raise ValueError("Image does not belong to this MOSS job")
        return FileResponse(
            svc.result_image_path(req.out_dir, path), media_type="image/png"
        )


@router.get("/tasks/{task_id}/samples/thumb")
async def task_thumb(task_id: str, path: str) -> Response:
    from io import BytesIO
    from PIL import Image

    with workspace_errors():
        req = await task_request(task_id, "moss-generate")
        manifest = svc.read_results(req.out_dir)
        if path not in {image.file for image in manifest.images}:
            raise ValueError("Image does not belong to this MOSS job")
        with Image.open(svc.result_image_path(req.out_dir, path)) as image:
            image.thumbnail((320, 320))
            buffer = BytesIO()
            image.convert("RGB").save(buffer, format="JPEG")
        return Response(buffer.getvalue(), media_type="image/jpeg")


@router.get("/tasks/{task_id}/report")
async def task_report(task_id: str) -> dict:
    with workspace_errors():
        job = await daemon_client.get_job(task_id)
        command = job.get("method")
        if command not in {"moss-tune", "moss-train"}:
            raise ValueError("Only MOSS tuning/training jobs have a report")
        req = await task_request(task_id, command)
        path = svc.output_path(req.output)
        if command == "moss-train":
            path = path.with_suffix(".json")
        report = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            raise ValueError("Invalid MOSS task report")
        return report
