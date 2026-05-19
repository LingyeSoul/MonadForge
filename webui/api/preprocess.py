"""Preprocessing settings and status API."""

from __future__ import annotations

from fastapi import APIRouter
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
def get_status():
    """Return current preprocess pipeline counts."""
    return svc.get_status()
