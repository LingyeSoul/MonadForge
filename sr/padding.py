"""Padding helpers shared by ResShift and RSD inference entry points."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def safe_spatial_pad(
    tensor: torch.Tensor, pad: tuple[int, int, int, int]
) -> torch.Tensor:
    """Reflect-pad when valid, falling back to replication for tiny images."""

    left, right, top, bottom = pad
    height, width = tensor.shape[-2:]
    can_reflect = left < width and right < width and top < height and bottom < height
    return F.pad(tensor, pad, mode="reflect" if can_reflect else "replicate")
