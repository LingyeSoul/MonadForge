"""Pydantic schemas for config API."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class FieldMeta(BaseModel):
    """Metadata for a single config field."""

    key: str
    value: Any
    origin: str  # "base" | "preset" | "method"
    field_type: str  # "bool" | "int" | "float" | "str" | "list" | "select"
    group: Optional[str] = None  # Architecture / Training / Performance / Paths
    is_basic: bool = False
    is_virtual: bool = False
    options: Optional[list[str]] = None  # for select fields (attn_mode, etc.)
    description: Optional[str] = None  # localized help text
    description_en: Optional[str] = None  # English tip (always English)


class ConfigLayerResponse(BaseModel):
    """A single config layer (base / preset / method)."""

    layer: str
    data: dict[str, Any]


class MergedConfigResponse(BaseModel):
    """Merged config with per-field metadata."""

    fields: list[FieldMeta]
    variant: str
    preset: str


class ConfigUpdateRequest(BaseModel):
    """Body for PUT /api/config/{layer}."""

    data: dict[str, Any]


class ConfigValidationResult(BaseModel):
    valid: bool
    errors: list[str] = []


class MethodsResponse(BaseModel):
    methods: list[str]


class VariantsResponse(BaseModel):
    variants: list[str]
    labels: dict[str, str] = {}


class PresetsResponse(BaseModel):
    presets: list[str]


class CreatePresetRequest(BaseModel):
    name: str
    data: dict[str, Any]


class VariantMetaResponse(BaseModel):
    variant: str
    family: Optional[str] = None
    label: Optional[str] = None
    description: Optional[str] = None
    experimental: bool = False
    order: int = 100
