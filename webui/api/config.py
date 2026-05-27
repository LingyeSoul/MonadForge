"""Config API endpoints."""

from __future__ import annotations


from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from webui.models.config import (
    ConfigLayerResponse,
    ConfigUpdateRequest,
    ConfigValidationResult,
    CreatePresetRequest,
    MergedConfigResponse,
    MethodsResponse,
    PresetsResponse,
    VariantMetaResponse,
    VariantsResponse,
)
from webui.services import config_service as svc

router = APIRouter()


@router.get("/methods", response_model=MethodsResponse)
def get_methods():
    return MethodsResponse(methods=svc.list_methods())


@router.get("/variants", response_model=VariantsResponse)
def get_variants(method: str = Query("lora")):
    variants = svc.list_gui_variants(method)
    labels = svc.variant_labels(method)
    return VariantsResponse(variants=variants, labels=labels)


@router.get("/presets", response_model=PresetsResponse)
def get_presets():
    return PresetsResponse(presets=svc.list_presets())


@router.post("/presets", response_model=PresetsResponse)
def create_preset(body: CreatePresetRequest):
    try:
        presets = svc.create_custom_preset(body.name, body.data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return PresetsResponse(presets=presets)


@router.delete("/presets/{name}", response_model=PresetsResponse)
def delete_preset(name: str):
    try:
        presets = svc.delete_custom_preset(name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return PresetsResponse(presets=presets)


@router.get("/variant-meta", response_model=VariantMetaResponse)
def get_variant_meta(variant: str = Query(...)):
    meta = svc.variant_metadata(variant)
    return VariantMetaResponse(**meta)


@router.get("/merged", response_model=MergedConfigResponse)
def get_merged_config(
    variant: str = Query("lora"),
    preset: str = Query("default"),
    lang: str = Query("cn"),
):
    result = svc.build_merged_config(variant, preset, lang=lang)
    return MergedConfigResponse(**result)


# ── Sample prompts file I/O ───────────────────────────────────


class SamplePromptEntry(BaseModel):
    prompt: str
    negative_prompt: str = ""
    controlnet_image: str = ""


class SamplePromptsResponse(BaseModel):
    path: str
    entries: list[SamplePromptEntry]


class SamplePromptsUpdate(BaseModel):
    path: str
    entries: list[SamplePromptEntry]


@router.get("/sample-prompts", response_model=SamplePromptsResponse)
def get_sample_prompts(path: str = Query("sample_prompts.txt")):
    entries = svc.read_sample_prompts(path)
    return SamplePromptsResponse(
        path=path,
        entries=[SamplePromptEntry(**e) for e in entries],
    )


@router.put("/sample-prompts")
def put_sample_prompts(body: SamplePromptsUpdate):
    svc.write_sample_prompts(body.path, [e.model_dump() for e in body.entries])
    return {"status": "ok"}


# ── Field help + method guide ───────────────────────────────────


class FieldHelpResponse(BaseModel):
    field_help: dict[str, str]
    guide_html: str = ""


class CreateVariantRequest(BaseModel):
    name: str
    seed_from: str | None = None


@router.post("/variants", response_model=VariantsResponse)
def create_variant(body: CreateVariantRequest):
    """Create a new custom variant, optionally seeded from an existing one."""
    try:
        variants = svc.create_variant(body.name, seed_from=body.seed_from)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return VariantsResponse(variants=variants, labels={})


@router.get("/field-help", response_model=FieldHelpResponse)
def get_field_help(
    variant: str = Query("lora"),
    lang: str = Query("cn"),
):
    """Return per-field help text + method guide HTML for the config editor."""
    try:
        result = svc.get_field_help_data(variant, lang=lang)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return FieldHelpResponse(**result)


# ── Prelaunch check & checkpoint management ───────────────────


class PrelaunchCheckResponse(BaseModel):
    cache_counts: dict[str, int]
    has_cache: bool
    checkpoint: dict | None = None
    requires_pe: bool = False


class WipeCheckpointRequest(BaseModel):
    output_dir: str
    output_name: str = "last"


@router.get("/prelaunch-check", response_model=PrelaunchCheckResponse)
def prelaunch_check(
    variant: str = Query(...),
    preset: str = Query("default"),
):
    """Check cache counts and checkpoint state before training launch."""
    try:
        result = svc.prelaunch_check(variant, preset)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return PrelaunchCheckResponse(**result)


@router.post("/wipe-checkpoint")
def wipe_checkpoint(body: WipeCheckpointRequest):
    """Delete a checkpoint state directory and its sidecar adapter file."""
    try:
        svc.wipe_checkpoint(body.output_dir, body.output_name)
    except OSError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "ok"}


# ── Layer read/write (catch-all) ───────────────────────────────


@router.get("/{layer}", response_model=ConfigLayerResponse)
def get_layer(
    layer: str,
    variant: str = Query("lora"),
    preset: str = Query("default"),
):
    try:
        data = svc.read_layer(layer, variant=variant, preset=preset)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return ConfigLayerResponse(layer=layer, data=data)


@router.put("/{layer}", response_model=ConfigLayerResponse)
def put_layer(
    layer: str,
    body: ConfigUpdateRequest,
    variant: str = Query("lora"),
    preset: str = Query("default"),
):
    if layer in ("base", "preset"):
        raise HTTPException(
            status_code=403,
            detail=f"Layer '{layer}' is read-only. Use custom overlays instead.",
        )
    try:
        if layer == "method":
            svc.save_variant_config(variant, body.data)
        else:
            svc.write_layer(layer, body.data, variant=variant, preset=preset)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    data = svc.read_layer(layer, variant=variant, preset=preset)
    return ConfigLayerResponse(layer=layer, data=data)


@router.post("/validate", response_model=ConfigValidationResult)
def validate_config(body: ConfigUpdateRequest):
    errors = svc.validate_config(body.data)
    return ConfigValidationResult(valid=len(errors) == 0, errors=errors)


@router.get("/schema/groups")
def get_groups():
    """Return field groupings for the form layout."""
    return svc.get_field_groups()


@router.get("/wandb-settings")
def get_wandb_settings():
    """Read wandb tracking settings from webui_settings.json."""
    return svc.get_wandb_settings()


@router.put("/wandb-settings")
def update_wandb_settings(body: dict):
    """Write wandb tracking settings to webui_settings.json."""
    return svc.save_wandb_settings(body)
