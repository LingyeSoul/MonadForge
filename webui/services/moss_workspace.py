"""Torch-free MOSS profiles and cache-header inspection."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from library.env import resolve_under_home
from library.moss.requests import (
    CacheRequest,
    GenerateRequest,
    TrainRequest,
    TuneSwapRequest,
)
from library.moss.scan import LATENT_RE, LATENT_SUFFIX, TE_SUFFIX
from webui.services.model_workspace import ProfileStore
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


# `resolve_under_home` is resolved through this module's globals on every call,
# so redirecting the profile tree stays a one-line patch in tests.
_STORE = ProfileStore(
    root="configs/custom/moss",
    label="MOSS",
    section_types=SECTION_TYPES,
    model_fields=MODEL_FIELDS,
    profile_cls=Profile,
    empty_factory=Profile,
    root_resolver=lambda value: resolve_under_home(value),
)


def normalize_profile(profile: Profile) -> Profile:
    return _STORE.normalize(profile)


def profile_path(name: str) -> Path:
    return _STORE.path(name)


def list_profiles() -> list[str]:
    return _STORE.list()


def read_profile(name: str) -> Profile:
    return _STORE.read(name)


def save_profile(name: str, profile: Profile) -> Profile:
    return _STORE.save(name, profile)


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
