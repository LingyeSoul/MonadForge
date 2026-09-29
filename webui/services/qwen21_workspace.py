"""Persistent Qwen workspaces and bounded, tensor-free cache inspection.

Profiles use the project's custom TOML tree. Missing optional keys restore
request defaults; model components and hardware auto-sizing stay in workers.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import tempfile
from pathlib import Path

import toml
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

from library.env import resolve_under_home
from library.qwen21.requests import CacheRequest, GenerateRequest, TrainRequest
from webui.services.qwen21_service import Scalar

MODEL_FIELDS = frozenset(
    {"model_dir", "dit", "text_encoder", "vae", "processor", "scheduler"}
)


class Profile(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore", allow_inf_nan=False)
    models: dict[str, Scalar]
    cache: dict[str, Scalar]
    train: dict[str, Scalar]
    generate: dict[str, Scalar]


def normalize_profile(profile: Profile) -> Profile:
    models = {
        key: value for key, value in profile.models.items() if key in MODEL_FIELDS
    }
    sections: dict[str, dict[str, Scalar]] = {}
    for name, cls in (
        ("cache", CacheRequest),
        ("train", TrainRequest),
        ("generate", GenerateRequest),
    ):
        values = {
            key: value
            for key, value in getattr(profile, name).items()
            if key not in MODEL_FIELDS
        }
        req = TypeAdapter(cls).validate_json(
            json.dumps({**values, **models}), strict=True
        )
        # Editing a profile may be incomplete; action validation happens at submission.
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
    root = resolve_under_home("configs/custom/qwen21").resolve()
    path = root / f"{name}.toml"
    if path.is_symlink():
        raise ValueError(f"Qwen profile must not be a symbolic link: {path}")
    return path


def list_profiles() -> list[str]:
    root = profile_path("default").parent
    return sorted({"default", *(path.stem for path in root.glob("*.toml"))})


def read_profile(name: str) -> Profile:
    path = profile_path(name)
    if name == "default" and not path.exists():
        return normalize_profile(Profile(models={}, cache={}, train={}, generate={}))
    return normalize_profile(
        Profile.model_validate(
            toml.loads(path.read_text(encoding="utf-8")), strict=True
        )
    )


def save_profile(name: str, profile: Profile) -> Profile:
    normalized = normalize_profile(profile)
    path = profile_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        section: {key: value for key, value in values.items() if value is not None}
        for section, values in normalized.model_dump().items()
    }
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(toml.dumps(data))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return normalized


class TensorHeader(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")
    dtype: str
    shape: list[int]
    data_offsets: list[int]


class SafeHeader(BaseModel):
    tensors: dict[str, TensorHeader]
    metadata: dict[str, str]


def read_header(path: Path) -> SafeHeader:
    """Read at most 16 MiB of header, never mmap or materialize tensor data."""
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Truncated safetensors prefix: {path}")
        length = int.from_bytes(prefix, "little")
        if not 2 <= length <= 16 * 1024 * 1024 or 8 + length > path.stat().st_size:
            raise ValueError(f"Invalid safetensors header length {length}: {path}")
        raw = TypeAdapter(dict[str, JsonValue]).validate_json(
            handle.read(length), strict=True
        )
    metadata = TypeAdapter(dict[str, str]).validate_python(
        raw.pop("__metadata__", {}), strict=True
    )
    tensors = TypeAdapter(dict[str, TensorHeader]).validate_python(raw, strict=True)
    payload_size = path.stat().st_size - 8 - length
    for key, tensor in tensors.items():
        offsets = tensor.data_offsets
        if (
            len(offsets) != 2
            or not 0 <= offsets[0] <= offsets[1] <= payload_size
            or any(dim < 0 for dim in tensor.shape)
        ):
            raise ValueError(f"Invalid tensor header in {path}: {key}")
    return SafeHeader(tensors=tensors, metadata=metadata)


class WeightIndex(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")
    weight_map: dict[str, str]


def check_component_headers(path: Path) -> None:
    if path.is_file():
        files = [path]
        expected: dict[str, str] = {}
    else:
        indexes = sorted(path.glob("*.safetensors.index.json"))
        if len(indexes) > 1:
            raise ValueError(
                f"Multiple safetensors indexes in component directory: {path}"
            )
        expected = (
            WeightIndex.model_validate_json(indexes[0].read_text()).weight_map
            if indexes
            else {}
        )
        files = (
            [path / name for name in sorted(set(expected.values()))]
            if indexes
            else sorted(path.glob("*.safetensors"))
        )
        if not files:
            raise FileNotFoundError(f"No safetensors weights found in {path}")
    seen: dict[str, str] = {}
    for file in files:
        if path.is_dir() and file.resolve().parent != path.resolve():
            raise ValueError(f"Weight shard escapes its component directory: {file}")
        header = read_header(file)
        for key, tensor in header.tensors.items():
            if key.endswith(".comfy_quant") or tensor.dtype.startswith(
                ("F8", "I8", "U8")
            ):
                raise ValueError(
                    f"Quantized Qwen checkpoint is unsupported: {file}; select BF16 weights ({key}, {tensor.dtype})"
                )
            if key in seen:
                raise ValueError(
                    f"Duplicate tensor {key} across weight shards in {path}"
                )
            seen[key] = file.name
    for key, file in expected.items():
        if seen.get(key) != file:
            raise ValueError(
                f"Weight index entry is missing or points to the wrong shard: {path}, {key}, {file}"
            )


class CacheStatus(BaseModel):
    directory: str
    pairs: int
    missing_text: int
    missing_latents: int
    errors: list[str]
    ready: bool


def inspect_cache(directory: str) -> CacheStatus:
    root = resolve_under_home(directory).resolve()
    if root.exists() and not root.is_dir():
        raise ValueError(f"Cache path must be a directory: {root}")
    latent_files = {
        p.name.removesuffix(".latent.safetensors"): p
        for p in root.glob("*.latent.safetensors")
    }
    text_files = {
        p.name.removesuffix(".te.safetensors"): p for p in root.glob("*.te.safetensors")
    }
    pairs = 0
    errors: list[str] = []
    for stem in sorted(latent_files.keys() & text_files.keys()):
        try:
            latent = read_header(latent_files[stem])
            text = read_header(text_files[stem])
            grid = (int(latent.metadata["latent_h"]), int(latent.metadata["latent_w"]))
            if any(edge <= 0 or edge % 2 for edge in grid) or latent.tensors[
                "latents"
            ].shape != [grid[0] * grid[1], 64]:
                raise ValueError(
                    "expected packed latents [H*W, 64] with positive even H/W"
                )
            embed = text.tensors["prompt_embeds"].shape
            if (
                len(embed) != 2
                or embed[0] <= 0
                or embed[1] != 4096
                or text.tensors["prompt_embeds_mask"].shape != [embed[0]]
            ):
                raise ValueError(
                    "expected unpadded text [tokens, 4096] and matching token mask"
                )
            if latent.tensors["latents"].dtype not in {
                "BF16",
                "F16",
                "F32",
            } or text.tensors["prompt_embeds"].dtype not in {"BF16", "F16", "F32"}:
                raise ValueError(
                    "latent and text caches must contain floating-point tensors"
                )
            pairs += 1
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f"{stem}: {exc}")
    missing_text = len(latent_files.keys() - text_files.keys())
    missing_latents = len(text_files.keys() - latent_files.keys())
    return CacheStatus(
        directory=str(root),
        pairs=pairs,
        missing_text=missing_text,
        missing_latents=missing_latents,
        errors=errors,
        ready=pairs > 0 and not errors and not missing_text and not missing_latents,
    )
