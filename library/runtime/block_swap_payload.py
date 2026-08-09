"""Non-persistent tensor payloads managed by the block-swap weight path."""

from __future__ import annotations

from collections import defaultdict
from typing import Callable

import torch
from torch import nn


class BlockSwapManagedTensor(nn.Module):
    """Expose a frozen tensor through the offloader's ``module.weight`` protocol.

    ``weight`` is intentionally neither a Parameter nor a registered buffer, so
    adapter/model state dicts do not persist runtime-only payloads. Normal
    ``Module.to`` calls still move it until block swap marks the carrier active;
    after that, the offloader is the sole owner of placement and can reuse the
    same CPU-master and GPU-storage machinery as frozen weights.
    """

    def __init__(self, tensor: torch.Tensor) -> None:
        super().__init__()
        self.weight = tensor.detach().contiguous()
        self._block_swap_placement_active = False

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True):
        if not self._block_swap_placement_active:
            original_dtype = self.weight.dtype
            moved = fn(self.weight)
            # These runtime payloads intentionally mix int8, bf16 and fp32.
            # Parent ``model.to(dtype=...)`` calls may move them, but must not
            # erase the precision selected by the ConvRot quality path.
            if moved.dtype != original_dtype:
                moved = moved.to(dtype=original_dtype)
            self.weight = moved
        return self

    def extra_repr(self) -> str:
        return (
            f"shape={tuple(self.weight.shape)}, dtype={self.weight.dtype}, "
            f"device={self.weight.device}"
        )


def set_block_swap_payload_placement(module: nn.Module, *, active: bool) -> None:
    """Enable or release offloader-owned placement below ``module``."""

    for child in module.modules():
        if isinstance(child, BlockSwapManagedTensor):
            child._block_swap_placement_active = bool(active)


def block_swap_payload_residency(module: nn.Module) -> dict[str, object]:
    """Return byte/count residency for managed payloads, grouped by device."""

    bytes_by_device: dict[str, int] = defaultdict(int)
    tensors_by_device: dict[str, int] = defaultdict(int)
    total_bytes = 0
    total_tensors = 0
    for child in module.modules():
        if not isinstance(child, BlockSwapManagedTensor):
            continue
        tensor = child.weight
        device = str(tensor.device)
        nbytes = int(tensor.numel()) * int(tensor.element_size())
        bytes_by_device[device] += nbytes
        tensors_by_device[device] += 1
        total_bytes += nbytes
        total_tensors += 1
    return {
        "total_bytes": total_bytes,
        "total_tensors": total_tensors,
        "bytes_by_device": dict(sorted(bytes_by_device.items())),
        "tensors_by_device": dict(sorted(tensors_by_device.items())),
    }
