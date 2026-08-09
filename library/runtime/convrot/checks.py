"""Runtime checks for ConvRot apply safety."""

from __future__ import annotations

from typing import Any

from torch import nn

from library.runtime.convrot.rht import assert_group_divides


def raise_if_compiled(root: Any, *, context: str = "apply_convrot") -> None:
    """Raise if any submodule already has a compiled forward / dynamo mark."""
    if root is None:
        return
    # torch.compile wraps modules; also Anima may set a flag after compile_blocks.
    if bool(getattr(root, "_anima_blocks_compiled", False)):
        raise RuntimeError(
            f"{context}: refused — blocks already compiled "
            "(_anima_blocks_compiled=True). Apply ConvRot before compile_blocks."
        )
    for name, module in getattr(root, "named_modules", lambda: [])():
        # OptimizedModule from torch.compile
        cls_name = type(module).__name__
        if cls_name in {"OptimizedModule", "OptimizedModuleWithMetadata"}:
            raise RuntimeError(
                f"{context}: refused — found compiled module at '{name}' "
                f"({cls_name}). Apply ConvRot before torch.compile."
            )
        if getattr(module, "_orig_mod", None) is not None and cls_name.startswith(
            "Optimized"
        ):
            raise RuntimeError(
                f"{context}: refused — found compiled module at '{name}'. "
                "Apply ConvRot before torch.compile."
            )


def validate_base_linear_for_convrot(
    linear: nn.Linear,
    *,
    group_size: int,
    name: str = "<linear>",
) -> None:
    if not isinstance(linear, nn.Linear):
        raise TypeError(f"{name}: expected nn.Linear, got {type(linear).__name__}")
    if linear.bias is not None:
        raise ValueError(f"{name}: ConvRot requires bias=None")
    if linear.weight.requires_grad:
        raise ValueError(f"{name}: ConvRot requires frozen weight (requires_grad=False)")
    assert_group_divides(int(linear.in_features), group_size)


def is_dora_module(lora: Any) -> bool:
    cls_name = type(lora).__name__
    if "DoRA" in cls_name or "Dora" in cls_name:
        return True
    if getattr(lora, "dora_scale", None) is not None:
        return True
    if getattr(lora, "magnitude", None) is not None and hasattr(lora, "lora_down"):
        # Heuristic: DoRA modules expose magnitude alongside lora_down.
        if "dora" in cls_name.lower():
            return True
    return False


def normalize_base_compute(value: Any) -> str:
    text = str(value or "bf16").strip().lower()
    aliases = {
        "bf16": "bf16",
        "fp16": "bf16",  # compute flag is orthogonal to mixed precision
        "none": "bf16",
        "off": "bf16",
        "w8a16": "w8a16_convrot",
        "w8a16_convrot": "w8a16_convrot",
        "w8a8": "w8a8_convrot",
        "w8a8_convrot": "w8a8_convrot",
    }
    if text not in aliases:
        raise ValueError(
            f"unknown base_compute={value!r}; expected bf16 | w8a16_convrot | w8a8_convrot"
        )
    return aliases[text]


def convrot_mode_from_base_compute(base_compute: str) -> str | None:
    mode = normalize_base_compute(base_compute)
    if mode == "w8a16_convrot":
        return "w8a16"
    if mode == "w8a8_convrot":
        return "w8a8"
    return None


def assert_convrot_block_swap_mutex(
    *,
    base_compute: str,
    block_swap_transfer_dtype: str,
) -> None:
    mode = normalize_base_compute(base_compute)
    transfer = str(block_swap_transfer_dtype or "bf16").strip().lower()
    if mode != "bf16" and transfer == "int8":
        raise ValueError(
            "base_compute ConvRot paths are mutually exclusive with "
            "block_swap_transfer_dtype=int8 (double dequant / confused semantics). "
            "Use block_swap_transfer_dtype=bf16 when enabling ConvRot."
        )


def warn_convrot_blocks_to_swap(
    *,
    base_compute: str,
    blocks_to_swap: int | None,
) -> str | None:
    """Return an informational warning when ConvRot free-base stacks with swap.

    Free-base marks patched ``Linear.weight`` as meta / ``_convrot_weight_freed``.
    Block-swap skips those tensors (see ``is_weight_swap_excluded``), while the
    runtime-only quantized weight and scale carriers are owned by the base
    Linear inside each DiT block and follow that block's residency. This saves
    GPU memory at the cost of transferring the payload on every swap. Int8
    transfer dtype remains hard-mutex with ConvRot.
    """
    mode = normalize_base_compute(base_compute)
    n = int(blocks_to_swap or 0)
    if mode == "bf16" or n <= 0:
        return None
    return (
        f"base_compute={mode} with blocks_to_swap={n}: ConvRot quantized weight "
        "and scale payloads follow their DiT blocks between CPU and GPU. This "
        "reduces GPU residency but adds transfer/step latency; "
        "block_swap_transfer_dtype=int8 remains incompatible "
        "(docs/experimental/convrot_int8_training.md §G.23)."
    )
