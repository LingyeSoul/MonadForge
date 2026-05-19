"""Image / caption / version / mask API endpoints."""

from __future__ import annotations


from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from webui.services import image_service as svc

router = APIRouter()


# ── request/response models ─────────────────────────────────────


class CaptionUpdate(BaseModel):
    content: str


class ImagePageResponse(BaseModel):
    items: list[dict]
    total: int
    page: int
    pages: int


# ── directory endpoints ─────────────────────────────────────────


@router.get("/directories")
def list_directories():
    """Return available dataset directories."""
    return svc.list_directories()


# ── image listing ───────────────────────────────────────────────


@router.get("", response_model=ImagePageResponse)
def list_images(
    directory: str = Query("image_dataset"),
    search: str = Query("", description="Filename filter (case-insensitive)"),
    sort_desc: bool = Query(False),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
):
    """Paginated image listing with search and sort."""
    return svc.list_images(
        directory=directory,
        search=search,
        sort_desc=sort_desc,
        page=page,
        page_size=page_size,
    )


# ── image file serving ─────────────────────────────────────────


@router.get("/file/{path:path}")
def get_image_file(path: str, directory: str = Query("image_dataset")):
    """Serve an image file by its relative path."""
    img = svc.resolve_image_path(directory, path)
    if img is None:
        raise HTTPException(status_code=404, detail=f"Image not found: {path}")
    return FileResponse(str(img))


# ── mask overlay serving ───────────────────────────────────────


@router.get("/mask-file/{path:path}")
def get_mask_file(path: str, directory: str = Query("image_dataset")):
    """Serve a mask overlay file by its relative path."""
    img = svc.resolve_image_path(directory, path)
    if img is None:
        raise HTTPException(status_code=404, detail=f"Image not found: {path}")
    base = svc.resolve_directory(directory)
    mask = svc.resolve_mask_path(img, base)
    if mask is None:
        raise HTTPException(status_code=404, detail="No mask for this image")
    return FileResponse(str(mask))


# ── caption CRUD ───────────────────────────────────────────────


@router.get("/caption/{path:path}")
def get_caption(path: str, directory: str = Query("image_dataset")):
    """Read caption for an image."""
    try:
        return svc.get_caption(directory, path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.put("/caption/{path:path}")
def update_caption(
    path: str, body: CaptionUpdate, directory: str = Query("image_dataset")
):
    """Write caption + append previous version to history."""
    try:
        return svc.save_caption(directory, path, body.content)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ── version history ────────────────────────────────────────────


@router.get("/versions/{path:path}")
def get_versions(path: str, directory: str = Query("image_dataset")):
    """Return caption version history (newest first)."""
    try:
        return svc.get_versions(directory, path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ── mask info ──────────────────────────────────────────────────


@router.get("/mask-info/{path:path}")
def get_mask_info(path: str, directory: str = Query("image_dataset")):
    """Return mask metadata for an image."""
    try:
        return svc.get_mask_info(directory, path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
