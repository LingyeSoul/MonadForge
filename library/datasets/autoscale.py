"""Resolution-curriculum schedule (``autoscale_mode``).

Trains the bulk of a run at a cheap low tier and switches to the expensive high
tier for the final ``finish`` fraction of steps — a *temporal* resolution
curriculum, orthogonal to the adapter/method axis. The premise (low-res training
points the adapter the same way high-res does) is Phase-0 validated in
``bench/res_curriculum/`` (see ``docs/proposal/autoscale_resolution_curriculum.md``).

This module holds only the pure progress→active-tier policy; the dataset
(``library/datasets/base.py``) owns the bucket/position bookkeeping and applies
the policy as an index remap inside ``__getitem__``. Keeping the policy pure
makes it unit-testable without a dataset (see ``tests/test_autoscale_schedule.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

# Ramp modes. "step" = a hard low→high switch (the simple, defensible v1, exact
# for a 2-tier ladder); "stairs" = a progressive climb across ≥3 tiers, with the
# top tier reserved for the last ``finish`` fraction and the lower tiers sharing
# the remaining bulk equally. For a 2-tier ladder the two modes coincide.
RAMP_MODES = ("step", "stairs")


@dataclass
class AutoscaleSchedule:
    """Progress→active-tier policy for the resolution curriculum.

    ``finish`` is the fraction of ``max_train_steps`` spent at the top tier;
    ``warmup_batches`` leading steps are left unscheduled so the natural shuffle
    touches *every* tier up front and warms all ``torch.compile`` block graphs
    before the curriculum masks the high tier away (no recompile / VRAM climb at
    the switch — cf. ``_largest_bucket_first`` and project_compile_context_vram_climb).
    """

    finish: float = 0.15
    ramp: str = "step"
    warmup_batches: int = 8

    def __post_init__(self) -> None:
        self.finish = float(self.finish)
        if not 0.0 < self.finish < 1.0:
            raise ValueError(f"autoscale_finish must be in (0, 1), got {self.finish}")
        self.ramp = str(self.ramp).strip().lower()
        if self.ramp not in RAMP_MODES:
            raise ValueError(
                f"autoscale_ramp must be one of {RAMP_MODES}, got {self.ramp!r}"
            )
        self.warmup_batches = max(0, int(self.warmup_batches))

    def active_rank(self, step: int, max_steps: int, n_ranks: int) -> int | None:
        """Which tier rank (0 = cheapest … n_ranks-1 = top) trains at ``step``.

        Returns ``None`` to mean "do not remap" — either there is nothing to
        schedule (single tier, or ``max_steps`` not yet known) or we are inside
        the warmup window where every tier should flow through to warm compile.
        """
        if n_ranks <= 1 or max_steps <= 0:
            return None
        if step < self.warmup_batches:
            return None
        frac = step / max_steps
        if frac >= 1.0:
            frac = 0.999999
        top = n_ranks - 1
        # Top tier owns the trailing ``finish`` fraction in both modes.
        if frac >= 1.0 - self.finish:
            return top
        if self.ramp == "step":
            return 0
        # stairs: split the leading (1 - finish) bulk equally across the lower
        # ``top`` ranks, lowest first.
        band = (1.0 - self.finish) / top
        return min(int(frac / band), top - 1)
