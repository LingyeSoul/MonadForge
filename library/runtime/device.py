import functools
import gc
from typing import Optional, Union

import torch
import torch.nn as nn


def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()


def clean_memory_on_device(device: Optional[Union[str, torch.device]]):
    r"""
    Clean memory on the specified device, will be called from training scripts.
    """
    gc.collect()
    if device is None:
        return
    if isinstance(device, str):
        device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()


def synchronize_device(device: Optional[Union[str, torch.device]]):
    if device is None:
        return
    if isinstance(device, str):
        device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize()


def is_weight_swap_excluded(module: nn.Module) -> bool:
    """Return whether ``module.weight`` has no movable backing storage."""
    if bool(getattr(module, "_convrot_weight_freed", False)):
        return True
    weight = getattr(module, "weight", None)
    if weight is None:
        return False
    device = getattr(getattr(weight, "data", weight), "device", None)
    return getattr(device, "type", None) == "meta"


def should_move_weight_to_device(
    module: nn.Module,
    device: torch.device,
    *,
    include_trainable: bool = True,
) -> bool:
    weight = getattr(module, "weight", None)
    if weight is None or is_weight_swap_excluded(module):
        return False
    target_device = torch.device(device)
    if (
        not include_trainable
        and target_device.type == "cpu"
        and getattr(weight, "requires_grad", False)
    ):
        return False
    return True


def weighs_to_device(
    layer: nn.Module, device: torch.device, *, include_trainable: bool = True
):
    """Move parameter storage while optionally keeping trainable adapters resident.

    Block swapping must never move a LoRA/adapter parameter during an autograd
    graph.  The default remains the historical all-weights behavior for callers
    outside the offloader; swap preparation passes ``include_trainable=False``.
    """
    for module in layer.modules():
        if should_move_weight_to_device(
            module, device, include_trainable=include_trainable
        ):
            module.weight.data = module.weight.data.to(device, non_blocking=True)


def str_to_dtype(
    s: Optional[str], default_dtype: Optional[torch.dtype] = None
) -> torch.dtype:
    """Convert a string to a torch.dtype."""
    if s is None:
        return default_dtype
    if s in ["bf16", "bfloat16"]:
        return torch.bfloat16
    elif s in ["fp16", "float16"]:
        return torch.float16
    elif s in ["fp32", "float32", "float"]:
        return torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {s}")


@functools.lru_cache(maxsize=None)
def get_preferred_device() -> torch.device:
    r"""
    Do not call this function from training scripts. Use accelerator.device instead.
    """
    device = torch.device("cuda")
    print(f"get_preferred_device() -> {device}")
    return device
