"""Comfy-Org Qwen-Image-2.1 single-file checkpoint adapters.

Comfy fuses SwiGLU gate/up rows, flattens Qwen3-VL's language-model prefix,
and uses the original Wan-style VAE names. These are structural conversions,
not a change to Qwen's forward. Use the Comfy-Org BF16 components. Quantized
checkpoints are rejected from their headers before materializing weights;
this loader never expands an INT8/FP8 checkpoint into a floating-point model.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open


class CheckpointFormatError(ValueError):
    """A checkpoint does not satisfy the selected Qwen component's contract."""


def transformer_keys(key: str) -> tuple[str, ...]:
    key = key.removeprefix("model.diffusion_model.")
    if key.endswith(".img_mlp.gate_up.weight"):
        return (
            key.replace(".gate_up.", ".gate_layer."),
            key.replace(".gate_up.", ".proj."),
        )
    return (key,)


def text_encoder_keys(key: str) -> tuple[str, ...]:
    if key.startswith(("model.layers.", "model.embed_tokens.", "model.norm.")):
        return (key.replace("model.", "model.language_model.", 1),)
    if key.startswith("visual."):
        return ("model." + key,)
    return (key,)


def vae_keys(key: str) -> tuple[str, ...]:
    if key.startswith("conv1."):
        return (key.replace("conv1.", "quant_conv.", 1),)
    if key.startswith("conv2."):
        return (key.replace("conv2.", "post_quant_conv.", 1),)
    key = re.sub(r"^(encoder|decoder)\.conv1\.", r"\1.conv_in.", key)
    key = re.sub(r"^(encoder|decoder)\.head\.0\.", r"\1.norm_out.", key)
    key = re.sub(r"^(encoder|decoder)\.head\.2\.", r"\1.conv_out.", key)
    key = key.replace(".middle.0.", ".mid_block.resnets.0.")
    key = key.replace(".middle.2.", ".mid_block.resnets.1.")
    key = key.replace(".middle.1.", ".mid_block.attentions.0.")
    key = re.sub(
        r"encoder\.downsamples\.(\d+)\.downsamples\.2\.",
        r"encoder.down_blocks.\1.downsampler.",
        key,
    )
    key = re.sub(
        r"encoder\.downsamples\.(\d+)\.downsamples\.(\d+)\.",
        r"encoder.down_blocks.\1.resnets.\2.",
        key,
    )
    key = re.sub(
        r"decoder\.upsamples\.(\d+)\.upsamples\.3\.",
        r"decoder.up_blocks.\1.upsampler.",
        key,
    )
    key = re.sub(
        r"decoder\.upsamples\.(\d+)\.upsamples\.(\d+)\.",
        r"decoder.up_blocks.\1.resnets.\2.",
        key,
    )
    for before, after in (
        ("residual.0.", "norm1."),
        ("residual.2.", "conv1."),
        ("residual.3.", "norm2."),
        ("residual.6.", "conv2."),
        ("shortcut.", "conv_shortcut."),
    ):
        key = key.replace(before, after)
    return (key,)


@dataclass(frozen=True)
class CheckpointPlan:
    mapping: dict[str, tuple[str, ...]]


def inspect_checkpoint(
    path: Path,
    expected: Mapping[str, torch.Tensor],
    key_map: Callable[[str], tuple[str, ...]],
) -> CheckpointPlan:
    """Validate the entire component without allocating its multi-GB weights."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Qwen checkpoint not found: {path}; select a Comfy-Org safetensors file"
        )
    with safe_open(path, framework="pt", device="cpu") as source:
        quant_keys = sorted(
            key for key in source.keys() if key.endswith(".comfy_quant")
        )
        if quant_keys:
            raise CheckpointFormatError(
                f"Quantized Qwen checkpoint is unsupported: {path}; found {quant_keys[0]}. "
                "Select the Comfy-Org BF16 safetensors component; automatic dequantization is disabled"
            )
        mapping: dict[str, tuple[str, ...]] = {}
        seen: set[str] = set()
        for key in source.keys():
            destinations = key_map(key)
            if not any(name in expected for name in destinations):
                continue
            if any(name not in expected or name in seen for name in destinations):
                raise CheckpointFormatError(
                    f"Conflicting tensor mapping in {path}: {key} -> {destinations}"
                )
            shape = list(source.get_slice(key).get_shape())
            wanted = list(expected[destinations[0]].shape)
            if len(destinations) > 1:
                wanted[0] = sum(expected[name].shape[0] for name in destinations)
            if len(shape) == 5 and len(wanted) == 4 and shape[2] == 1:
                shape = shape[:2] + shape[3:]
            if shape != wanted:
                raise CheckpointFormatError(
                    f"Wrong tensor shape in {path}: {key} has {shape}, expected {wanted}"
                )
            stored_dtype = source.get_slice(key).get_dtype()
            if stored_dtype not in ("BF16", "F16", "F32"):
                raise CheckpointFormatError(
                    f"Unsupported weight dtype in {path}: {key} = {stored_dtype}; "
                    "select the Comfy-Org BF16 safetensors component"
                )
            seen.update(destinations)
            mapping[key] = destinations
        missing = sorted(set(expected) - seen)
        if missing:
            raise CheckpointFormatError(
                f"Incomplete or wrong Qwen checkpoint {path}: missing {len(missing)} required tensors, first: {missing[:8]}"
            )
        return CheckpointPlan(mapping)


def read_checkpoint(
    path: Path,
    expected: Mapping[str, torch.Tensor],
    key_map: Callable[[str], tuple[str, ...]],
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Validate all keys/shapes first, then materialize one tensor at a time on CPU."""
    if dtype not in (torch.float32, torch.bfloat16):
        raise CheckpointFormatError(
            f"Qwen-Image-2.1 loading requires BF16 or FP32, got {dtype}; Anima's FP16 guards do not apply"
        )
    plan = inspect_checkpoint(path, expected, key_map)
    with safe_open(path, framework="pt", device="cpu") as source:
        state: dict[str, torch.Tensor] = {}
        for key, destinations in plan.mapping.items():
            weight = source.get_tensor(key)
            if weight.ndim == 5 and expected[destinations[0]].ndim == 4:
                weight = weight.squeeze(2)
            weight = weight.to(dtype)
            pieces = (
                weight.split([expected[name].shape[0] for name in destinations], dim=0)
                if len(destinations) > 1
                else (weight,)
            )
            state.update(
                (name, piece.contiguous())
                for name, piece in zip(destinations, pieces, strict=True)
            )
        return state
