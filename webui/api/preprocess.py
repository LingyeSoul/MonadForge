"""Preprocessing settings and status API."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from webui.services import preprocess_service as svc

router = APIRouter()


# ── Request / response models ────────────────────────────────────


class SamSettings(BaseModel):
    prompts: list[str] = ["speech bubble", "text bubble"]
    threshold: float = 0.5
    dilate: int = 5


class PreprocessSettings(BaseModel):
    sam: SamSettings = SamSettings()
    run_sam_mask: bool = True
    run_mit_mask: bool = True
    caption_shuffle_variants: int = 4
    caption_tag_dropout_rate: float = 0.1
    mit_text_threshold: float = 0.8
    mit_dilate: int = 5


class CacheCounts(BaseModel):
    latents: int = 0
    te: int = 0
    pe: int = 0


class PreprocessStatus(BaseModel):
    resized: int = 0
    masks: int = 0
    cache: CacheCounts = CacheCounts()


class DatasetPaths(BaseModel):
    source_image_dir: str = ""
    resized_image_dir: str = ""
    lora_cache_dir: str = ""


class SavePathsRequest(BaseModel):
    source_image_dir: str | None = None
    resized_image_dir: str | None = None
    lora_cache_dir: str | None = None


# ── Endpoints ────────────────────────────────────────────────────


@router.get("/settings", response_model=PreprocessSettings)
def get_settings():
    """Read current preprocessing settings from config files."""
    return svc.get_settings()


@router.put("/settings", response_model=PreprocessSettings)
def put_settings(body: PreprocessSettings):
    """Save preprocessing settings to config files."""
    return svc.save_settings(body.model_dump())


@router.get("/status", response_model=PreprocessStatus)
def get_status(
    variant: str | None = Query(None),
    preset: str | None = Query(None),
):
    """Return current preprocess pipeline counts."""
    return svc.get_status(variant=variant, preset=preset)


@router.get("/paths", response_model=DatasetPaths)
def get_paths(
    variant: str | None = Query(None),
    preset: str | None = Query(None),
):
    """Return resolved dataset paths from the config chain."""
    return svc.get_paths(variant=variant, preset=preset)


@router.put("/paths", response_model=DatasetPaths)
def save_paths(
    body: SavePathsRequest,
    variant: str = Query(""),
):
    """Save path overrides to the variant TOML."""
    if not variant:
        raise HTTPException(status_code=400, detail="variant query parameter is required")
    return svc.save_path_overrides(variant, body.model_dump(exclude_none=True))
