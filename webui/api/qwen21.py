"""Qwen-specific typed forms and jobs on the existing daemon task surface."""

from __future__ import annotations

import json
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from library.qwen21.requests import CacheRequest, GenerateRequest, TrainRequest
from webui.services import qwen21_service as svc
from webui.services.daemon_client import DaemonError
from webui.services.task_service import task_service

# Workspace operations never initialize torch or a GPU context.
from contextlib import contextmanager
from typing import Iterator

from library.env import resolve_under_home
from webui.services import qwen21_workspace as workspace
from webui.services.daemon_client import daemon_client


router = APIRouter()


class JobResponse(BaseModel):
    task_id: str
    command: str


@router.get("/schema", response_model=svc.FormSchema)
def get_schema() -> svc.FormSchema:
    return svc.read_schema()


@router.post("/resolve", response_model=svc.ResolvedPaths)
def resolve_sources(body: svc.ValuesRequest) -> svc.ResolvedPaths:
    try:
        return svc.resolve_values(body.values)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/jobs/{phase}", response_model=JobResponse)
async def submit_job(
    phase: Literal["cache", "train", "generate"], body: svc.ValuesRequest
) -> JobResponse:
    request_type = {
        "cache": CacheRequest,
        "train": TrainRequest,
        "generate": GenerateRequest,
    }[phase]
    try:
        req = svc.request_from_values(request_type, body.values)
        task = await task_service.start_task(f"qwen21-{phase}", req.to_argv())
        return JobResponse(task_id=task.id, command=task.command)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DaemonError as exc:
        raise HTTPException(
            status_code=502, detail=f"Qwen job submission failed: {exc}"
        ) from exc


@router.post("/chain", response_model=JobResponse)
async def submit_chain(body: svc.ChainRequest) -> JobResponse:
    try:
        cache = svc.request_from_values(CacheRequest, body.cache.values)
        train = svc.request_from_values(TrainRequest, body.train.values)
        args = [
            "--cache-args",
            json.dumps(cache.to_argv()),
            "--train-args",
            json.dumps(train.to_argv()),
        ]
        task = await task_service.start_task("qwen21-cache-train", args)
        return JobResponse(task_id=task.id, command=task.command)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DaemonError as exc:
        raise HTTPException(
            status_code=502, detail=f"Qwen workflow submission failed: {exc}"
        ) from exc


@router.get("/results", response_model=svc.ResultManifest)
def get_results(directory: str) -> svc.ResultManifest:
    try:
        return svc.read_results(directory)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/result-image")
def get_result_image(directory: str, file: str) -> FileResponse:
    try:
        path = svc.result_image_path(directory, file)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return FileResponse(path, media_type="image/png")


@contextmanager
def workspace_errors() -> Iterator[None]:
    try:
        yield
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DaemonError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/profiles", response_model=list[str])
def profiles() -> list[str]:
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
def training_preflight(body: svc.ValuesRequest) -> workspace.CacheStatus:
    with workspace_errors():
        req = svc.request_from_values(TrainRequest, body.values)
        svc.validate_training_model(req)
        return workspace.inspect_cache(req.cache)


class GenerationJob(BaseModel):
    method: str
    kind: Literal["command"]
    argv: list[str]


async def task_result_directory(task_id: str) -> str:
    job = GenerationJob.model_validate(await daemon_client.get_job(task_id))
    if job.method != "qwen21-generate":
        raise ValueError(f"Task {task_id} is not a Qwen generation task")
    if job.argv[:2] == ["tasks.py", "qwen21-generate"]:
        args = job.argv[2:]
    elif job.argv[:2] == ["-m", "scripts.qwen21.generate"]:
        args = job.argv[2:]
    else:
        raise ValueError(
            f"Unsupported Qwen generation argv for task {task_id}: {job.argv}"
        )
    try:
        req = GenerateRequest.from_argv(args)
    except SystemExit as exc:
        raise ValueError(
            f"Invalid persisted Qwen arguments for task {task_id}"
        ) from exc
    if not resolve_under_home(req.out_dir).is_absolute() or not req.out_dir:
        raise ValueError(f"Task {task_id} has no generation output directory")
    return str(svc.output_path(req.out_dir))


@router.get("/tasks/{task_id}/results", response_model=svc.ResultManifest)
async def task_results(task_id: str) -> svc.ResultManifest:
    with workspace_errors():
        return svc.read_results(await task_result_directory(task_id))


@router.get("/tasks/{task_id}/samples/file")
async def task_result_file(task_id: str, path: str) -> FileResponse:
    with workspace_errors():
        directory = await task_result_directory(task_id)
        manifest = svc.read_results(directory)
        if path not in {image.file for image in manifest.images}:
            raise ValueError(f"Image {path!r} does not belong to Qwen task {task_id}")
        return FileResponse(
            svc.result_image_path(directory, path), media_type="image/png"
        )


@router.get("/tasks/{task_id}/samples/thumb")
async def task_result_thumb(task_id: str, path: str) -> Response:
    from io import BytesIO

    from PIL import Image

    with workspace_errors():
        directory = await task_result_directory(task_id)
        manifest = svc.read_results(directory)
        if path not in {image.file for image in manifest.images}:
            raise ValueError(f"Image {path!r} does not belong to Qwen task {task_id}")
        with Image.open(svc.result_image_path(directory, path)) as image:
            image.thumbnail((320, 320))
            buffer = BytesIO()
            image.save(buffer, format="WEBP")
        return Response(buffer.getvalue(), media_type="image/webp")
