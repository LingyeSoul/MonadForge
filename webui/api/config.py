"""Config API endpoints."""

from __future__ import annotations


from fastapi import APIRouter, HTTPException, Query

from webui.models.config import (
    ConfigLayerResponse,
    ConfigUpdateRequest,
    ConfigValidationResult,
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


@router.get("/variant-meta", response_model=VariantMetaResponse)
def get_variant_meta(variant: str = Query(...)):
    meta = svc.variant_metadata(variant)
    return VariantMetaResponse(**meta)


@router.get("/merged", response_model=MergedConfigResponse)
def get_merged_config(
    variant: str = Query("lora"),
    preset: str = Query("default"),
):
    result = svc.build_merged_config(variant, preset)
    return MergedConfigResponse(**result)


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
