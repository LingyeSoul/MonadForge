"""Models API — trained adapter weight listing, download, and deletion."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from webui.services import model_service as svc
from webui.services.paths import resolve_path

router = APIRouter()


@router.get("")
def list_models():
    """List every final adapter weight below the default output directory."""
    return {"models": svc.list_models()}


@router.get("/metadata")
def model_metadata(path: str = Query(...)):
    """Return the ss_* header metadata of one weight file."""
    result = svc.read_model_metadata(path)
    if result is None:
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    return result


@router.get("/download")
def download_model(path: str = Query(...)):
    """Stream a weight file as an attachment download."""
    p = resolve_path(path, expect_file=True)
    if p is None:
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    return FileResponse(str(p), filename=p.name)


@router.delete("")
def delete_model(path: str = Query(...)):
    """Delete one final adapter weight."""
    deleted, error = svc.delete_model(path)
    if not deleted:
        status = 404 if "not found" in error.lower() else 400
        raise HTTPException(status_code=status, detail=error)
    return {"status": "deleted", "path": path}
