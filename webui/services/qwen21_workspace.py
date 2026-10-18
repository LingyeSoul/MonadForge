"""Persistent Qwen workspaces and bounded, tensor-free cache inspection.

Profiles use the project's custom TOML tree. Missing optional keys restore
request defaults; model components and hardware auto-sizing stay in workers.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

from library.env import resolve_under_home
from library.qwen21.requests import CacheRequest, GenerateRequest, TrainRequest
from webui.services.model_workspace import ProfileStore
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


# `resolve_under_home` is resolved through this module's globals on every call,
# so redirecting the profile tree stays a one-line patch in tests.
_STORE = ProfileStore(
    root="configs/custom/qwen21",
    label="Qwen",
    section_types={
        "cache": CacheRequest,
        "train": TrainRequest,
        "generate": GenerateRequest,
    },
    model_fields=MODEL_FIELDS,
    profile_cls=Profile,
    empty_factory=lambda: Profile(models={}, cache={}, train={}, generate={}),
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


class TensorHeader(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")
    dtype: str
    shape: list[int]
    data_offsets: list[int]


class SafeHeader(BaseModel):
    tensors: dict[str, TensorHeader]
    metadata: dict[str, str]


_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "F64": 8,
    "I64": 8,
    "U64": 8,
}


def read_header(path: Path) -> SafeHeader:
    """Read at most 16 MiB of header, never mmap or materialize tensor data."""
    with path.open("rb") as handle:
        file_size = os.fstat(handle.fileno()).st_size
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Truncated safetensors prefix: {path}")
        length = int.from_bytes(prefix, "little")
        if not 2 <= length <= 16 * 1024 * 1024 or 8 + length > file_size:
            raise ValueError(f"Invalid safetensors header length {length}: {path}")
        raw = TypeAdapter(dict[str, JsonValue]).validate_json(
            handle.read(length), strict=True
        )
    metadata = TypeAdapter(dict[str, str]).validate_python(
        raw.pop("__metadata__", {}), strict=True
    )
    tensors = TypeAdapter(dict[str, TensorHeader]).validate_python(raw, strict=True)
    payload_size = file_size - 8 - length
    for key, tensor in tensors.items():
        offsets = tensor.data_offsets
        if (
            len(offsets) != 2
            or not 0 <= offsets[0] <= offsets[1] <= payload_size
            or any(dim < 0 for dim in tensor.shape)
        ):
            raise ValueError(f"Invalid tensor header in {path}: {key}")
        item_size = _DTYPE_BYTES.get(tensor.dtype)
        if item_size is None:
            raise ValueError(f"Unsupported tensor dtype in {path}: {tensor.dtype}")
        if offsets[1] - offsets[0] != math.prod(tensor.shape) * item_size:
            raise ValueError(f"Tensor payload size differs from shape in {path}: {key}")
    # A readable safetensors file covers the payload exactly once. Bounds alone
    # miss truncated tensor storage, overlapping tensors and unused trailing data.
    cursor = 0
    for tensor in sorted(tensors.values(), key=lambda item: item.data_offsets):
        start, end = tensor.data_offsets
        if start != cursor:
            raise ValueError(f"Tensor payload has overlapping ranges or gaps: {path}")
        cursor = end
    if cursor != payload_size:
        raise ValueError(f"Tensor payload has trailing data: {path}")
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
            if (
                latent_files[stem].resolve().parent != root
                or text_files[stem].resolve().parent != root
            ):
                raise ValueError("cache files must stay inside the selected directory")
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
