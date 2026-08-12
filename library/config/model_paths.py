"""Model checkpoint path configuration shared by CLI, API, and WebUI.

The shipped defaults live in ``configs/model.toml``. Machine-local overrides
live in gitignored ``configs/custom/model.toml``. Model keys left in
``configs/base.toml`` are still read as a compatibility layer for older
installations, but new code must write only the custom model file.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

MODEL_CONFIG_KEYS = frozenset({"pretrained_model_name_or_path", "qwen3", "vae"})
MODEL_CONFIG_DEFAULTS = {
    "pretrained_model_name_or_path": "models/diffusion_models/anima-base-v1.0.safetensors",
    "qwen3": "models/text_encoders/qwen_3_06b_base.safetensors",
    "vae": "models/vae/qwen_image_vae.safetensors",
}


def read_model_config_file(path: Path) -> dict[str, Any]:
    """Read and validate recognized model-path keys from one TOML file."""
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"{path}: invalid TOML: {exc}") from exc
    result = {key: data[key] for key in MODEL_CONFIG_KEYS if key in data}
    for key, value in result.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{path}: {key} must be a non-empty string")
    return result


def load_model_config(
    configs_dir: Path | str,
    *,
    include_legacy_base: bool = True,
) -> dict[str, Any]:
    """Return effective model paths, ordered from defaults to user overrides.

    Precedence is ``model.toml -> base.toml (legacy) -> custom/model.toml``.
    Missing files are allowed so embedders can fall back to built-in defaults.
    Parse and I/O errors are intentionally surfaced to configuration callers.
    """
    root = Path(configs_dir)
    paths = [root / "model.toml"]
    if include_legacy_base:
        paths.append(root / "base.toml")
    paths.append(root / "custom" / "model.toml")

    merged: dict[str, Any] = {}
    for path in paths:
        if path.is_file():
            merged.update(read_model_config_file(path))
    return merged
