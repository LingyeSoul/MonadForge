"""Image / caption / version / mask API endpoints."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from webui.services import image_service as svc
from webui.services.config_service import get_path_overrides

router = APIRouter()


def _default_directory() -> str:
    """Default directory name from the config chain."""
    return get_path_overrides()["source_image_dir"]


# ── request/response models ─────────────────────────────────────


class CaptionUpdate(BaseModel):
    content: str


class BatchCaptionRequest(BaseModel):
    directory: str
    paths: list[str]
    action: str
    tag: str | None = None
    find: str | None = None
    replace: str | None = None
    use_regex: bool = False


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
    directory: str | None = Query(None),
    search: str = Query("", description="Filename filter (case-insensitive)"),
    sort_desc: bool = Query(False),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
):
    """Paginated image listing with search and sort."""
    return svc.list_images(
        directory=directory or _default_directory(),
        search=search,
        sort_desc=sort_desc,
        page=page,
        page_size=page_size,
    )


# ── image file serving ─────────────────────────────────────────


@router.get("/file/{path:path}")
def get_image_file(path: str, directory: str | None = Query(None)):
    """Serve an image file by its relative path."""
    img = svc.resolve_image_path(directory or _default_directory(), path)
    if img is None:
        raise HTTPException(status_code=404, detail=f"Image not found: {path}")
    return FileResponse(str(img))


# ── mask overlay serving ───────────────────────────────────────


@router.get("/mask-file/{path:path}")
def get_mask_file(path: str, directory: str | None = Query(None)):
    """Serve a mask overlay file by its relative path."""
    directory = directory or _default_directory()
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
def get_caption(path: str, directory: str | None = Query(None)):
    """Read caption for an image."""
    try:
        return svc.get_caption(directory or _default_directory(), path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.put("/caption/{path:path}")
def update_caption(path: str, body: CaptionUpdate, directory: str | None = Query(None)):
    """Write caption + append previous version to history."""
    try:
        return svc.save_caption(directory or _default_directory(), path, body.content)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ── version history ────────────────────────────────────────────


@router.get("/versions/{path:path}")
def get_versions(path: str, directory: str | None = Query(None)):
    """Return caption version history (newest first)."""
    try:
        return svc.get_versions(directory or _default_directory(), path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/tag-index")
def get_tag_index(directory: str | None = Query(None)):
    """Return tag frequency table for all images in directory."""
    return svc.build_tag_index(directory or _default_directory())


@router.put("/batch-caption")
def batch_caption(body: BatchCaptionRequest):
    """Batch update captions for multiple images."""
    return svc.batch_update_captions(
        directory=body.directory,
        paths=body.paths,
        action=body.action,
        tag=body.tag,
        find=body.find,
        replace=body.replace,
        use_regex=body.use_regex,
    )


# ── dataset upload ─────────────────────────────────────────────


@router.post("/upload-archive", status_code=202)
async def upload_archive(
    file: UploadFile = File(...),
    target: str = Form(..., description="Extraction directory (relative to project root or absolute)"),
):
    """Upload a dataset archive (zip / tar.*) and start extracting into *target*.

    The upload is streamed to a temp file so large archives don't sit in
    memory. Extraction runs as a background task — the response carries the
    task id; poll ``GET /upload-archive/{task_id}`` until *status* becomes
    ``"done"`` or ``"error"``.
    """
    filename = file.filename or ""
    if not filename.lower().endswith(svc.ARCHIVE_SUFFIXES):
        raise HTTPException(
            status_code=400,
            detail="Unsupported archive format: expected .zip, .tar, .tar.gz, .tgz, .tar.bz2 or .tar.xz",
        )
    tmp = tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False)
    try:
        while chunk := await file.read(1 << 20):
            tmp.write(chunk)
        tmp.close()
        # Raises ValueError (→ 400) synchronously for invalid targets; once
        # the task starts, the worker owns the temp file's cleanup.
        # The original filename is passed for format detection: the temp
        # file's suffix is truncated (pack.tar.gz → .gz).
        return svc.start_extract_task(Path(tmp.name), target, archive_name=filename)
    except ValueError as e:
        Path(tmp.name).unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/upload-archive/{task_id}")
def get_upload_task(task_id: str):
    """Poll a background extraction task started by POST /upload-archive."""
    task = svc.get_extract_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Unknown extraction task: {task_id}")
    return task


# ── mask info ──────────────────────────────────────────────────


@router.get("/mask-info/{path:path}")
def get_mask_info(path: str, directory: str | None = Query(None)):
    """Return mask metadata for an image."""
    try:
        return svc.get_mask_info(directory or _default_directory(), path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
