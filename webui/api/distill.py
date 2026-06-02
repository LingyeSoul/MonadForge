"""API endpoints for SPD/Turbo distill config editing."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from webui.services import distill_service as svc

router = APIRouter()


# ── Response models ─────────────────────────────────────────────


class DistillMethodSummary(BaseModel):
    key: str
    label: str
    config_path: str
    task_command: str


class DistillField(BaseModel):
    key: str
    value: Any
    type: str
    comment: str = ""


class DistillSection(BaseModel):
    name: str
    fields: list[DistillField]


class DistillConfigResponse(BaseModel):
    method: str
    sections: list[DistillSection]


class DistillSaveRequest(BaseModel):
    updates: dict[str, dict[str, Any]]


# ── Endpoints ───────────────────────────────────────────────────


@router.get("/methods", response_model=list[DistillMethodSummary])
def list_methods():
    """List available distill methods."""
    return svc.list_distill_methods()


@router.get("/config", response_model=DistillConfigResponse)
def get_config(method: str = Query(...)):
    """Read a distill method's sectioned TOML config."""
    try:
        return svc.read_distill_config(method)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.put("/config")
def save_config(body: DistillSaveRequest, method: str = Query(...)):
    """Save field values to a distill method's TOML config."""
    try:
        svc.save_distill_config(method, body.updates)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"ok": True}


@router.get("/guide")
def get_guide(method: str = Query(...), lang: str = Query("cn")):
    """Return guide HTML for a distill method."""
    html = svc.load_guide(method, lang)
    return {"html": html}
