"""Validated WebUI interface for staged-resolution profiles."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from library.training import staged_resolution_plan as plans
from webui.services.daemon_client import DaemonError
from webui.services.task_service import task_service

router = APIRouter()


class ResolutionStage(BaseModel):
    resolution: int
    ratio: float
    batch_size: int
    num_repeats: int = 1


class StagedResolutionPlan(BaseModel):
    version: int = 1
    method: str = Field(default="lora", min_length=1, max_length=71)
    variant: str = Field(default="lora", min_length=1, max_length=71)
    preset: str = Field(default="default", min_length=1, max_length=71)
    source_image_dir: str = Field(min_length=1, max_length=1024)
    max_train_steps: int = Field(ge=3)
    stages: list[ResolutionStage] = Field(min_length=3, max_length=3)


class ProfileResponse(BaseModel):
    name: str
    plan: StagedResolutionPlan
    persisted: bool


class ProfileListResponse(BaseModel):
    profiles: list[str]


class ProfileActionRequest(BaseModel):
    version: Literal[1] = 1


def _http_error(exc: Exception, status: int = 400) -> HTTPException:
    return HTTPException(status_code=status, detail=str(exc))


@router.get("/profiles", response_model=ProfileListResponse)
def list_profiles():
    return ProfileListResponse(profiles=plans.list_profiles())


@router.get("/profiles/{name}", response_model=ProfileResponse)
def get_profile(name: str):
    try:
        persisted = plans.profile_path(name).is_file()
        plan = plans.load_profile(
            name,
            default_if_missing=name == plans.DEFAULT_PROFILE,
        )
    except ValueError as exc:
        raise _http_error(exc) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail="staged-resolution profile not found"
        ) from exc
    return ProfileResponse(
        name=plans.normalize_profile_name(name),
        plan=plan,
        persisted=persisted,
    )


@router.put("/profiles/{name}", response_model=ProfileResponse)
def put_profile(name: str, body: StagedResolutionPlan):
    try:
        data = body.model_dump()
        normalized = plans.validate_plan(data)
        # Compile before persisting so an unknown variant/preset cannot leave a
        # saved profile that only fails much later in the daemon queue.
        plans.compile_runtime_config(name, normalized)
        saved = plans.save_profile(name, normalized)
    except ValueError as exc:
        raise _http_error(exc) from exc
    except (FileNotFoundError, KeyError) as exc:
        raise HTTPException(
            status_code=400, detail="selected variant or preset is unavailable"
        ) from exc
    return ProfileResponse(
        name=plans.normalize_profile_name(name),
        plan=saved,
        persisted=True,
    )


@router.get("/profiles/{name}/status")
def get_profile_status(name: str) -> dict[str, Any]:
    try:
        plan = plans.load_profile(
            name,
            default_if_missing=name == plans.DEFAULT_PROFILE,
        )
        return plans.profile_status(name, plan)
    except ValueError as exc:
        raise _http_error(exc) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail="staged-resolution profile not found"
        ) from exc


async def _start_profile_task(name: str, command: str) -> dict[str, Any]:
    try:
        plans.load_profile(name)
        task = await task_service.start_task(
            command, [plans.normalize_profile_name(name)]
        )
    except ValueError as exc:
        raise _http_error(exc) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail="staged-resolution profile not found"
        ) from exc
    except DaemonError as exc:
        raise HTTPException(
            status_code=502, detail="Training daemon is unavailable"
        ) from exc
    return task.info()


@router.post("/profiles/{name}/preprocess")
async def start_preprocess(name: str, _body: ProfileActionRequest):
    return await _start_profile_task(name, "staged-preprocess")


@router.post("/profiles/{name}/train")
async def start_training(name: str, _body: ProfileActionRequest):
    return await _start_profile_task(name, "staged-train")
