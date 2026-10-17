"""Native dataset curation API; work is executed by the shared daemon queue."""

from __future__ import annotations

import mimetypes
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field

from scripts.curation.context import CurationUnavailable
from webui.services import curation_service as svc
from webui.services.daemon_client import DaemonError


class CurationRoute(APIRoute):
    def get_route_handler(self):
        original = super().get_route_handler()

        async def handle(request: Request):
            try:
                return await original(request)
            except CurationUnavailable as exc:
                raise HTTPException(503, str(exc)) from exc
            except FileNotFoundError as exc:
                raise HTTPException(404, str(exc)) from exc
            except (ValueError, KeyError) as exc:
                raise HTTPException(400, str(exc)) from exc
            except DaemonError as exc:
                raise HTTPException(502, str(exc)) from exc
        return handle


router = APIRouter(route_class=CurationRoute)


class Payload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Caption(Payload):
    rel: str = Field(min_length=1, max_length=4096)
    kind: Literal["master", "revised"]
    text: str = Field(min_length=1, max_length=2_000_000)


class Text(Payload):
    text: str = Field(max_length=2_000_000)


class Clause(Payload):
    header: str = ""
    prefix: str = "On the "
    position: str = ""
    tags: list[str] = Field(default_factory=list, max_length=20000)


class Composition(Payload):
    flat_tags: list[str] = Field(default_factory=list, max_length=20000)
    clauses: list[Clause] = Field(default_factory=list, max_length=2000)


class Exclusion(Payload):
    rel: str = Field(min_length=1, max_length=4096)
    excluded: bool
    note: str = Field(default="", max_length=10000)


class Submission(Payload):
    stage: str = Field(min_length=1, max_length=64)
    values: dict[str, Any] = Field(default_factory=dict, max_length=256)
    apply: bool = False
    rel: str = Field(default="", max_length=4096)


@router.get("/schema")
def schema():
    return svc.get_schema()


@router.get("/roots")
def roots():
    return svc.get_roots()


@router.put("/roots")
def save_roots(body: dict[str, str]):
    return svc.save_roots(body)


@router.get("/settings")
def settings():
    return svc.get_settings()


@router.put("/settings")
def save_settings(body: dict[str, Any]):
    return svc.save_settings(body)


@router.get("/dataset")
def dataset(search: str = "", limit: int = Query(20000, ge=1, le=20000)):
    return svc.dataset(search, limit)


@router.get("/groups")
def groups():
    return svc.groups()


@router.get("/items")
def item(rel: str = Query(..., min_length=1)):
    return svc.item(rel)


@router.put("/caption")
def caption(body: Caption):
    return svc.caption(body.rel, body.kind, body.text)


@router.post("/parse")
def parse(body: Text):
    return svc.parse(body.text)


@router.post("/compose")
def compose(body: Composition):
    return svc.compose(body.flat_tags, [clause.model_dump() for clause in body.clauses])


@router.post("/exclude")
def exclude(body: Exclusion):
    return svc.exclude(body.rel, body.excluded, body.note)


@router.get("/files")
def files(path: str):
    resolved = svc.file_path(path)
    return FileResponse(resolved, media_type=mimetypes.guess_type(resolved.name)[0] or "application/octet-stream", headers={"cache-control": "no-cache"})


@router.get("/thumb")
def thumb(path: str, size: int = Query(256, ge=16, le=1024)):
    return Response(svc.thumbnail(path, size), media_type="image/webp", headers={"cache-control": "no-cache"})


@router.get("/jobs")
def jobs():
    return svc.list_jobs()


@router.post("/jobs")
async def submit(body: Submission):
    return await svc.submit(body.stage, body.values, body.apply, body.rel)


@router.get("/jobs/{job_id}/report")
def report(job_id: str):
    return svc.report(job_id)


@router.get("/jobs/{job_id}/proposals")
def proposals(job_id: str):
    return svc.proposals(job_id)


@router.get("/jobs/{job_id}/proposal")
def proposal(job_id: str, rel: str):
    return svc.proposal(job_id, rel)


@router.post("/jobs/{job_id}/{operation}")
async def replay(job_id: str, operation: Literal["apply", "undo"]):
    return await svc.replay(job_id, operation)
