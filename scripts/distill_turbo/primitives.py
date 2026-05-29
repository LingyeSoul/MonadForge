"""Re-noising, τ samplers, and shared loop-side helpers."""

from __future__ import annotations

import torch


def renoise(
    x_pred: torch.Tensor, tau: torch.Tensor, eps: torch.Tensor
) -> torch.Tensor:
    """``x_τ = (1 - τ)·x_pred + τ·ε`` — flow-matching forward path at level τ.

    ``tau`` is per-batch; broadcast to ``x_pred``'s shape.
    """
    tau_e = tau.view(-1, *([1] * (x_pred.dim() - 1)))
    return (1.0 - tau_e) * x_pred + tau_e * eps


def sample_t_above(t: torch.Tensor, min_gap: float = 0.05) -> torch.Tensor:
    """Sample τ ~ U(t + min_gap, 1.0) per batch element.

    Clamps the lower bound so very-late steps (t ≈ 1) don't collapse to a
    near-empty interval (proposal R5).
    """
    lower = (t + min_gap).clamp(max=1.0 - 1e-4)
    u = torch.rand_like(t)
    return lower + u * (1.0 - lower)


def sample_t(
    B: int,
    *,
    distribution: str,
    sigmoid_scale: float,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Sample generator-t on ``device``.

    ``uniform`` and ``sigmoid`` are the only two strategies accepted by
    :func:`scripts.distill_turbo.config.resolve_config`.
    """
    if distribution == "uniform":
        return torch.rand(B, device=device, dtype=dtype)
    return torch.sigmoid(sigmoid_scale * torch.randn(B, device=device, dtype=dtype))


def make_scheduler(
    opt: torch.optim.Optimizer, total_steps: int, lr: float
) -> torch.optim.lr_scheduler.LRScheduler:
    """Warmup (2% of ``total_steps``) → cosine annealing to ``0.1·lr``."""
    warmup_steps = max(1, int(0.02 * total_steps))
    warmup = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=1e-6 / lr, total_iters=warmup_steps
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=total_steps - warmup_steps, eta_min=lr * 0.1
    )
    return torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[warmup, cosine], milestones=[warmup_steps]
    )


class PadCache:
    """Per-spatial-shape zero-pad tensor, cached across forwards.

    Constant-token bucketing keeps the shape stable within a step (and
    constant in single-prompt mode), so we recycle instead of re-allocating.
    """

    def __init__(self, dtype: torch.dtype):
        self._dtype = dtype
        self._cache: dict[tuple[int, int, int], torch.Tensor] = {}

    def get(self, x: torch.Tensor) -> torch.Tensor:
        key = (x.shape[0], x.shape[-2], x.shape[-1])
        pad = self._cache.get(key)
        if pad is None or pad.dtype != self._dtype or pad.device != x.device:
            pad = torch.zeros(
                x.shape[0],
                1,
                x.shape[-2],
                x.shape[-1],
                dtype=self._dtype,
                device=x.device,
            )
            self._cache[key] = pad
        return pad


def make_collate(use_masked_loss: bool):
    """Stacking collate that optionally appends the per-image mask.

    Pooled-text is unused by turbo but ``CachedDataset`` always returns it.
    """

    def _collate(batch):
        out = [
            [b[0] for b in batch],
            torch.stack([b[1] for b in batch]),
            torch.stack([b[2] for b in batch]),
            torch.stack([b[3] for b in batch]),
        ]
        if use_masked_loss:
            out.append(torch.stack([b[4] for b in batch]))  # [B, 1, H, W] mask
        return tuple(out)

    return _collate
