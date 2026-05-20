"""File browser API — directory listing and path validation for model file picker."""

from __future__ import annotations

from fastapi import APIRouter, Query

from webui.services import file_service

router = APIRouter()


@router.get("/browse")
def browse(
    dir: str = Query("models", description="Directory path (relative to project root or absolute)"),
    ext: str = Query(".safetensors", description="File extension filter"),
):
    """List subdirectories and files in the given directory."""
    return file_service.browse_directory(dir, ext=ext)


@router.get("/validate")
def validate(
    path: str = Query(..., description="File path to validate (relative to project root or absolute)"),
):
    """Check whether a model file exists at the given path."""
    return file_service.validate_model_path(path)
