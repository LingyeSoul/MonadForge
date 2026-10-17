"""Torch-free MOSS profiles and cache-header inspection."""

from __future__ import annotations

import dataclasses
import json
import os
import re
import tempfile
from pathlib import Path

import toml
from pydantic import BaseModel, ConfigDict, TypeAdapter

from library.env import resolve_under_home
from library.moss.requests import (
    CacheRequest,
    GenerateRequest,
    TrainRequest,
    TuneSwapRequest,
)
from library.moss.scan import LATENT_RE, LATENT_SUFFIX, TE_SUFFIX
from webui.services.qwen21_workspace import CacheStatus, read_header

Scalar = str | int | float | bool | None
MODEL_FIELDS = frozenset({"model_dir"})
SECTION_TYPES = {
    "cache": CacheRequest,
    "train": TrainRequest,
    "generate": GenerateRequest,
    "tune": TuneSwapRequest,
}


class Profile(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore", allow_inf_nan=False)
    models: dict[str, Scalar] = {"model_dir": ""}
    cache: dict[str, Scalar] = {}
    train: dict[str, Scalar] = {}
    generate: dict[str, Scalar] = {}
    tune: dict[str, Scalar] = {}


def normalize_profile(profile: Profile) -> Profile:
    models = {
        key: value for key, value in profile.models.items() if key in MODEL_FIELDS
    }
    sections = {}
    for name, cls in SECTION_TYPES.items():
        values = {
            key: value
            for key, value in getattr(profile, name).items()
            if key not in MODEL_FIELDS
        }
        req = TypeAdapter(cls).validate_json(
            json.dumps({**values, **models}), strict=True
        )
        for field in dataclasses.fields(req):
            choices = field.metadata.get("choices")
            if choices is not None and getattr(req, field.name) not in choices:
                raise ValueError(f"{name}.{field.name} must be one of {choices}")
        full = dataclasses.asdict(req)
        sections[name] = {
            key: value for key, value in full.items() if key not in MODEL_FIELDS
        }
        models = {key: value for key, value in full.items() if key in MODEL_FIELDS}
    return Profile(models=models, **sections)


def profile_path(name: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
        raise ValueError(
            "Profile name must contain 1–64 letters, digits, underscores or hyphens"
        )
    root = resolve_under_home("configs/custom/moss").resolve()
    path = root / f"{name}.toml"
    if path.is_symlink():
        raise ValueError("MOSS profiles cannot be symbolic links")
    return path


def list_profiles() -> list[str]:
    root = profile_path("default").parent
    return sorted(
        {
            "default",
            *[path.stem for path in root.glob("*.toml") if not path.is_symlink()],
        }
    )


def read_profile(name: str) -> Profile:
    path = profile_path(name)
    if not path.exists():
        if name == "default":
            return normalize_profile(Profile())
        raise FileNotFoundError(f"MOSS profile does not exist: {name}")
    return normalize_profile(Profile.model_validate(toml.load(path), strict=True))


def save_profile(name: str, profile: Profile) -> Profile:
    normalized = normalize_profile(profile)
    path = profile_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(toml.dumps(normalized.model_dump(exclude_none=True)))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return normalized


def inspect_cache(directory: str) -> CacheStatus:
    root = resolve_under_home(directory).resolve()
    if root.exists() and not root.is_dir():
        raise ValueError(f"Cache path must be a directory: {root}")
    latents: dict[str, tuple[Path, int, int]] = {}
    text = {
        path.name.removesuffix(TE_SUFFIX): path
        for path in root.glob(f"*{TE_SUFFIX}")
        if not path.name.startswith("_empty")
    }
    for path in root.glob(f"*{LATENT_SUFFIX}"):
        match = LATENT_RE.fullmatch(path.name)
        if match is None:
            continue
        stem = match["stem"]
        old = latents.get(stem)
        if old is None or path.stat().st_mtime > old[0].stat().st_mtime:
            latents[stem] = (path, int(match["w"]), int(match["h"]))
    errors = []
    pairs = 0
    for stem in sorted(latents.keys() & text.keys()):
        try:
            path, width, height = latents[stem]
            if path.resolve().parent != root or text[stem].resolve().parent != root:
                raise ValueError("cache files must stay inside the selected directory")
            latent = read_header(path)
            encoded = read_header(text[stem])
            if width <= 0 or height <= 0 or width % 16 or height % 16:
                raise ValueError(
                    "cached width and height must be positive multiples of 16"
                )
            if latent.tensors["latent"].shape != [128, height // 16, width // 16]:
                raise ValueError("expected packed latent [128, H/16, W/16]")
            if (
                int(latent.metadata["width"]) != width
                or int(latent.metadata["height"]) != height
            ):
                raise ValueError("latent dimensions differ from its filename")
            shape = encoded.tensors["embeds"].shape
            if len(shape) != 2 or shape[0] < 1 or shape[1] != 4096:
                raise ValueError("expected unpadded text embeddings [tokens, 4096]")
            if latent.tensors["latent"].dtype not in {
                "F16",
                "BF16",
                "F32",
            } or encoded.tensors["embeds"].dtype not in {"F16", "BF16", "F32"}:
                raise ValueError("latent and text cache tensors must be floating point")
            pairs += 1
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f"{stem}: {exc}")
    missing_text = len(latents.keys() - text.keys())
    missing_latents = len(text.keys() - latents.keys())
    return CacheStatus(
        directory=str(root),
        pairs=pairs,
        missing_text=missing_text,
        missing_latents=missing_latents,
        errors=errors,
        ready=pairs > 0 and not errors and not missing_text and not missing_latents,
    )
