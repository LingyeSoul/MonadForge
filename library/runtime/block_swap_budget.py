"""Preflight memory budget for non-zero DiT block swapping.

The block scheduler can reduce resident frozen-block weights, but it cannot
remove adapter parameters, gradients, optimizer state, or the activation
workspace needed by the largest token family.  This module performs a
conservative check before ``accelerator.prepare``/the first optimizer step so a
configuration that cannot fit fails with an actionable explanation.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import torch

logger = logging.getLogger(__name__)

MiB = 1024 * 1024


@dataclass(frozen=True)
class BlockSwapBudget:
    blocks_to_swap: int
    max_tokens: int
    trainable_params: int
    adapter_param_bytes: int
    gradient_bytes: int
    optimizer_state_bytes: int
    activation_workspace_bytes: int
    fixed_safety_bytes: int
    estimated_required_bytes: int
    free_bytes: int | None
    total_bytes: int | None
    device: str
    strict: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _max_tokens(token_budget: Any) -> int:
    """Accept train.py's ``(count, (min,max))`` and simple test values."""

    if token_budget is None:
        return 0
    if isinstance(token_budget, int | float):
        return max(0, int(token_budget))
    if isinstance(token_budget, (tuple, list)):
        if len(token_budget) >= 2 and isinstance(token_budget[1], (tuple, list)):
            return max(0, _int(token_budget[1][-1], 0))
        values = [_int(v, 0) for v in token_budget]
        return max(values or [0])
    return 0


def _trainable_params(network: Any) -> tuple[int, int, int]:
    params: Iterable[Any] = ()
    if network is not None:
        try:
            params = (p for p in network.parameters() if getattr(p, "requires_grad", False))
        except Exception:
            params = ()
    count = param_bytes = fp32_bytes = 0
    for param in params:
        numel = _int(getattr(param, "numel", lambda: 0)(), 0)
        element_size = _int(getattr(param, "element_size", lambda: 2)(), 2)
        count += numel
        param_bytes += numel * max(1, element_size)
        # Adam-family optimizers keep fp32 moments even when adapter weights are
        # fp16/bf16.  The caller uses this as the gradient/param estimate too.
        fp32_bytes += numel * 4
    return count, param_bytes, fp32_bytes


def _optimizer_state_bytes(optimizer: Any, trainable_params: int) -> int:
    name = str(type(optimizer).__name__).lower()
    if any(token in name for token in ("8bit", "paged", "bnb")):
        # Two quantized moment buffers plus a small fp32 scale overhead.
        return int(trainable_params * 3.0)
    if any(token in name for token in ("sgd", "adagrad")):
        return int(trainable_params * 4.0)
    # Adam/AdamW/Prodigy and unknown adaptive optimizers: two fp32 moments.
    return int(trainable_params * 8.0)


def _free_total(device: torch.device) -> tuple[int | None, int | None]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None, None
    try:
        free, total = torch.cuda.mem_get_info(device)
        return int(free), int(total)
    except Exception:
        try:
            total = int(torch.cuda.get_device_properties(device).total_memory)
            used = int(torch.cuda.memory_reserved(device))
            return max(0, total - used), total
        except Exception:
            return None, None


def estimate_block_swap_budget(
    args: Any,
    *,
    model: Any = None,
    network: Any = None,
    optimizer: Any = None,
    token_budget: Any = None,
    device: torch.device | str | None = None,
) -> BlockSwapBudget:
    """Estimate the non-swappable memory cost for one training step."""

    if device is None:
        device = getattr(model, "device", None) or "cpu"
    device = torch.device(device)
    blocks = max(0, _int(getattr(args, "blocks_to_swap", 0), 0))
    max_tokens = _max_tokens(token_budget)
    if max_tokens <= 0:
        max_tokens = max(1, _int(getattr(args, "max_tokens", 0), 0))

    trainable, param_bytes, fp32_param_bytes = _trainable_params(network)
    # Gradients are kept in fp32 for the conservative estimate on Volta and in
    # the parameter dtype elsewhere. ``fp32_param_bytes`` is a safe upper bound.
    gradient_bytes = fp32_param_bytes
    optimizer_bytes = _optimizer_state_bytes(optimizer, trainable)
    hidden = max(
        1,
        _int(
            getattr(model, "model_channels", None)
            or getattr(args, "model_channels", None),
            2048,
        ),
    )
    batch = max(1, _int(getattr(args, "train_batch_size", 1), 1))
    dtype_bytes = 2 if str(getattr(args, "mixed_precision", "fp16")).lower() in {"fp16", "bf16"} else 4
    # This covers the live block input/output, attention/MLP temporaries, and
    # allocator fragmentation. Gradient checkpointing lowers the multiplier,
    # but never makes it zero because the adapter projection remains live.
    activation_factor = 10 if getattr(args, "gradient_checkpointing", False) else 18
    activation_bytes = max_tokens * hidden * batch * dtype_bytes * activation_factor
    fixed_safety = 512 * MiB
    required = param_bytes + gradient_bytes + optimizer_bytes + activation_bytes + fixed_safety
    free, total = _free_total(device)
    strict = os.environ.get("ANIMA_BLOCK_SWAP_BUDGET_STRICT", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    return BlockSwapBudget(
        blocks_to_swap=blocks,
        max_tokens=max_tokens,
        trainable_params=trainable,
        adapter_param_bytes=param_bytes,
        gradient_bytes=gradient_bytes,
        optimizer_state_bytes=optimizer_bytes,
        activation_workspace_bytes=activation_bytes,
        fixed_safety_bytes=fixed_safety,
        estimated_required_bytes=required,
        free_bytes=free,
        total_bytes=total,
        device=str(device),
        strict=strict,
    )


def check_block_swap_budget(
    args: Any,
    *,
    model: Any = None,
    network: Any = None,
    optimizer: Any = None,
    token_budget: Any = None,
    device: torch.device | str | None = None,
) -> BlockSwapBudget:
    """Log and, when measurable, enforce the preflight budget."""

    budget = estimate_block_swap_budget(
        args,
        model=model,
        network=network,
        optimizer=optimizer,
        token_budget=token_budget,
        device=device,
    )
    setattr(args, "block_swap_budget", budget.as_dict())
    logger.info(
        "block swap budget: blocks=%d max_tokens=%d trainable_params=%d "
        "required_mib=%.1f free_mib=%s total_mib=%s",
        budget.blocks_to_swap,
        budget.max_tokens,
        budget.trainable_params,
        budget.estimated_required_bytes / MiB,
        f"{budget.free_bytes / MiB:.1f}" if budget.free_bytes is not None else "n/a",
        f"{budget.total_bytes / MiB:.1f}" if budget.total_bytes is not None else "n/a",
    )
    if (
        budget.blocks_to_swap > 0
        and budget.strict
        and budget.free_bytes is not None
        and budget.estimated_required_bytes > budget.free_bytes
    ):
        raise RuntimeError(
            "block-swap preflight exceeded the available CUDA memory: "
            f"estimated {budget.estimated_required_bytes / MiB:.0f} MiB required, "
            f"only {budget.free_bytes / MiB:.0f} MiB free (max_tokens={budget.max_tokens}, "
            f"trainable_params={budget.trainable_params:,}). "
            "Reduce resolution/token count or network rank, increase "
            "blocks_to_swap, disable preview/extra conditioning, or free other "
            "GPU processes before retrying. The first optimizer step was not run."
        )
    return budget

