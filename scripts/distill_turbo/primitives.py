"""Re-noising, τ samplers, and shared loop-side helpers.

These were the first copy of the distillation per-step primitives; they have
since been promoted to ``library/training`` and ``library/datasets`` so the
other distillation loops (``scripts/distill_mod``, ``scripts/distill_spd``)
share one implementation. This module is now a thin compatibility shim — the
turbo loop imports the same names from here as before.
"""

from __future__ import annotations

from library.datasets.cache import make_cached_collate as make_collate
from library.training.forward import PadCache, renoise
from library.training.forward import sample_sigma as sample_t
from library.training.schedulers import make_warmup_cosine_scheduler

__all__ = ["renoise", "sample_t", "make_scheduler", "PadCache", "make_collate"]


def make_scheduler(opt, total_steps: int, lr: float):
    """Warmup (2% of ``total_steps``, ≥1 step) → cosine annealing to ``0.1·lr``."""
    warmup_steps = max(1, int(0.02 * total_steps))
    return make_warmup_cosine_scheduler(opt, total_steps, lr, warmup_steps=warmup_steps)
