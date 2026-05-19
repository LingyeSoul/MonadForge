"""Merge API — adapter directory listing and bakeability scanning."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from webui.services import merge_service as svc

router = APIRouter()


class ScanRequest(BaseModel):
    file_path: str


@router.get("/dirs")
def list_dirs():
    """List directories containing .safetensors adapter files."""
    return {"dirs": svc.list_adapter_dirs()}


@router.get("/files")
def list_files(dir: str = Query(...)):
    """List .safetensors files in a directory."""
    return {"files": svc.list_files(dir)}


@router.post("/scan")
def scan_adapter(body: ScanRequest):
    """Scan a .safetensors file for bakeability."""
    result = svc.scan_adapter(body.file_path)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result
