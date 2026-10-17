"""Cheap MOSS request checks shared by CLI and browser submission; torch-free."""

from __future__ import annotations

import dataclasses
import math

from .requests import CacheRequest, GenerateRequest, TrainRequest, TuneSwapRequest


def validate_request(
    req: CacheRequest | GenerateRequest | TrainRequest | TuneSwapRequest,
) -> None:
    for field in dataclasses.fields(req):
        value = getattr(req, field.name)
        choices = field.metadata.get("choices")
        if choices is not None and value not in choices:
            raise ValueError(f"{field.name} must be one of {choices}")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{field.name} must be finite")
    if isinstance(req, CacheRequest):
        if req.resolution < 16 or req.max_text_tokens < 1:
            raise ValueError(
                "resolution must be at least 16 and max_text_tokens positive"
            )
    if isinstance(req, TrainRequest):
        if req.epochs < 1 or req.rank < 1 or req.lr <= 0 or req.max_grad_norm <= 0:
            raise ValueError(
                "epochs, rank, learning rate and gradient limit must be positive"
            )
        if (
            req.max_steps < 0
            or req.save_every_epochs < 0
            or req.blocks_to_swap < -1
            or req.blocks_to_swap > 40
        ):
            raise ValueError("invalid steps, checkpoint interval or block swap count")
        if req.factor != -1 and req.factor < 1:
            raise ValueError("factor must be -1 or positive")
        if (
            not 0 <= req.warmup_ratio <= 1
            or not 0 <= req.caption_dropout <= 1
            or not 0 <= req.activation_memory_budget <= 1
        ):
            raise ValueError(
                "warmup_ratio, caption_dropout and activation_memory_budget must be in [0, 1]"
            )
        if (
            req.sigma_s < 0
            or not 0 < req.sigma_cutoff <= 1
            or not 0 <= req.sigma_uniform <= 1
            or req.log_every < 1
        ):
            raise ValueError("invalid sigma distribution or logging interval")
    if isinstance(req, GenerateRequest):
        if req.width <= 0 or req.height <= 0 or req.width % 16 or req.height % 16:
            raise ValueError("width and height must be positive multiples of 16")
        if req.steps < 1 or req.cfg < 0 or req.shift <= 0:
            raise ValueError(
                "steps and shift must be positive; cfg must be non-negative"
            )
        if not req.decode_only and not (
            req.prompt.strip() or req.prompts_file or req.example
        ):
            raise ValueError("a prompt, prompts file or gallery example is required")
        try:
            skip = [int(x.strip()) for x in req.uncond_skip.split(",") if x.strip()]
        except ValueError as exc:
            raise ValueError(
                "uncond_skip must contain comma-separated layer indexes"
            ) from exc
        if any(x < 0 or x > 41 for x in skip):
            raise ValueError("uncond_skip layers must be in [0, 41]")
        if (
            req.spectrum_window_size < 1
            or req.spectrum_flex_window < 0
            or req.spectrum_warmup < 0
            or req.spectrum_m < 1
            or req.spectrum_lam < 0
        ):
            raise ValueError("invalid Spectrum forecast parameters")
    if isinstance(req, TuneSwapRequest):
        if req.rank < 1 or req.patience < 1 or req.steps < 1 or req.warmup < 0:
            raise ValueError(
                "rank, patience and timed steps must be positive; warmup non-negative"
            )
        if (
            req.margin_gb < 0
            or not 0 <= req.tolerance < 1
            or not 0 < req.budget_step <= 1
        ):
            raise ValueError("invalid tuning margin, tolerance or budget step")
        try:
            counts = [
                int(value.strip())
                for value in req.swap_grid.split(",")
                if value.strip()
            ]
        except ValueError as exc:
            raise ValueError("swap_grid must contain comma-separated integers") from exc
        if not counts or any(count < 0 or count > 40 for count in counts):
            raise ValueError("swap_grid must contain counts in [0, 40]")
