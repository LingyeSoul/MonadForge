"""Preprocessing settings and status API."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, field_validator, model_validator

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
    # Free-fit tier edges (allowed: 512 768 896 1024 1280 1536). This is the
    # value resize actually consumes — the old vestigial resize_resolution
    # scalar was dropped under free-fit. Saved to configs/custom/preprocess.toml
    # so WebUI edits don't dirty the git-tracked repo copy.
    target_res: list[int] = [1024]
    multires_per_image: bool = False

    @field_validator("target_res")
    @classmethod
    def validate_target_res(cls, edges: list[int]) -> list[int]:
        allowed = {512, 768, 896, 1024, 1280, 1536}
        normalized = sorted(set(edges))
        invalid = [edge for edge in normalized if edge not in allowed]
        if not normalized:
            raise ValueError("target_res must contain at least one tier")
        if invalid:
            raise ValueError(f"unsupported target_res tier(s): {invalid}")
        return normalized

    @model_validator(mode="after")
    def validate_multires_tiers(self):
        if self.multires_per_image and len(self.target_res) < 2:
            raise ValueError(
                "multires_per_image requires at least two target_res tiers"
            )
        return self


class CacheCounts(BaseModel):
    latents: int = 0
    te: int = 0
    pe: int = 0


class PreprocessStatus(BaseModel):
    resized: int = 0
    masks: int = 0
    cache: CacheCounts = CacheCounts()
    cond_resized: int = 0


class DatasetPaths(BaseModel):
    source_image_dir: str = ""
    resized_image_dir: str = ""
    lora_cache_dir: str = ""
    conditioning_data_dir: str = ""
    conditioning_resized_dir: str = ""


class AdapterCacheStats(BaseModel):
    latents: int = 0
    te: int = 0
    pe: int = 0


class AdapterStats(BaseModel):
    source_count: int = 0
    caption_count: int = 0
    cache: AdapterCacheStats = AdapterCacheStats()


class SavePathsRequest(BaseModel):
    source_image_dir: str | None = None
    resized_image_dir: str | None = None
    lora_cache_dir: str | None = None
    conditioning_data_dir: str | None = None
    conditioning_resized_dir: str | None = None


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
        raise HTTPException(
            status_code=400, detail="variant query parameter is required"
        )
    return svc.save_path_overrides(variant, body.model_dump(exclude_none=True))


@router.get("/adapter-stats", response_model=AdapterStats)
def get_adapter_stats(dir: str = Query(...)):
    """Return dataset statistics for an adapter's source directory."""
    return svc.adapter_stats(dir)
