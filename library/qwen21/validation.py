"""Torch-free validation shared by Qwen entry points and the WebUI boundary."""

from __future__ import annotations

import dataclasses
import math

from library.qwen21.requests import CacheRequest, GenerateRequest, TrainRequest


def validate_request(req: CacheRequest | TrainRequest | GenerateRequest) -> None:
    for field in dataclasses.fields(req):
        value = getattr(req, field.name)
        choices = field.metadata.get("choices")
        if choices is not None and value not in choices:
            raise ValueError(f"--{field.name} must be one of {choices}; got {value!r}")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"--{field.name} must be finite; got {value!r}")
    if isinstance(req, (TrainRequest, GenerateRequest)):
        _validate_swap("blocks_to_swap", req.blocks_to_swap, 30)
    if isinstance(req, (CacheRequest, GenerateRequest)):
        _validate_swap("te_blocks_to_swap", req.te_blocks_to_swap, 34)
        if req.resolution < 32:
            raise ValueError(f"--resolution must be at least 32; got {req.resolution}")
    if isinstance(req, TrainRequest):
        if req.epochs < 1 or req.rank < 1 or req.lr <= 0 or req.max_grad_norm <= 0:
            raise ValueError(
                "Training requires positive epochs, rank, lr and max_grad_norm"
            )
        if (
            not 0 <= req.warmup_ratio <= 1
            or req.save_every_epochs < 0
            or req.logit_std < 0
        ):
            raise ValueError(
                "warmup_ratio must be within [0, 1]; save_every_epochs and logit_std must be non-negative"
            )
        if req.activation_reserve_gb is not None and req.activation_reserve_gb < 0:
            raise ValueError(
                "--activation_reserve_gb must be non-negative or omitted for auto-sizing"
            )
    if isinstance(req, GenerateRequest):
        if req.steps < 1 or req.true_cfg_scale < 0:
            raise ValueError(
                "--steps must be positive and --true_cfg_scale non-negative"
            )
        if (req.width is None) != (req.height is None):
            raise ValueError("--width and --height must be specified together")
        for name, value in (("width", req.width), ("height", req.height)):
            if value is not None and (value <= 0 or value % 32):
                raise ValueError(
                    f"--{name} must be a positive multiple of 32; got {value}"
                )
        multipliers = [float(value) for value in req.multipliers.split(",")]
        if not all(math.isfinite(value) for value in multipliers):
            raise ValueError(
                "--multipliers must contain finite comma-separated numbers"
            )
        if not req.lora and multipliers != [0.0]:
            raise ValueError(
                "Without --lora, set --multipliers to 0.0 for base-only generation"
            )


def _validate_swap(name: str, value: int | None, maximum: int) -> None:
    if value is not None and not 0 <= value <= maximum:
        raise ValueError(
            f"--{name} must be within [0, {maximum}], or omitted for auto-sizing; got {value}"
        )
