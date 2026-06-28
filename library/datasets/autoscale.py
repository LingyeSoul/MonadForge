"""Resolution schedule (``autoscale_mode``).

A *resolution* training schedule orthogonal to the adapter/method axis. Two
non-trivial modes share one ``autoscale_highres_ratio`` knob (the fraction of
training spent at the expensive top tier):

* ``curriculum`` — train the bulk of the run at a cheap low tier, then switch to
  the top tier for the trailing ``highres_ratio`` fraction of steps (a *temporal*
  curriculum). The premise (low-res training points the adapter the same way
  high-res does) is Phase-0 validated in ``bench/res_curriculum/`` (see
  ``docs/proposal/autoscale_resolution_curriculum.md``).
* ``random`` — interleave high- and low-res batches throughout the whole run,
  drawing the top tier for a ``highres_ratio`` fraction of each epoch's batches
  (and the cheapest tier otherwise). No hard phase boundary — the same long-run
  compute split as curriculum, spread uniformly instead of back-loaded.

``none`` disables the schedule entirely (and collapses each stem's cached tier
emits to its highest-res copy so they don't train as duplicates — see
``DreamBoothDataset.collapse_autoscale_tiers``).

This module holds only the pure policy + value object; the dataset
(``library/datasets/base.py``) owns the bucket/position bookkeeping and applies
the policy as an index remap inside ``__getitem__``. Keeping the policy pure
makes it unit-testable without a dataset (see ``tests/test_autoscale_schedule.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

# Schedule modes selectable via ``autoscale_mode``. ``none`` builds no schedule
# (the dataset collapses tier emits and trains exactly like a single-res run);
# ``curriculum`` and ``random`` are the two real schedules (see module docstring).
AUTOSCALE_MODES = ("none", "curriculum", "random")

# Curriculum ramp modes. "step" = a hard low→high switch (the simple, defensible
# v1, exact for a 2-tier ladder); "stairs" = a progressive climb across ≥3 tiers,
# with the top tier reserved for the last ``highres_ratio`` fraction and the lower
# tiers sharing the remaining bulk equally. For a 2-tier ladder the two coincide.
# Ignored by ``random`` mode (which is always a top-vs-cheapest draw).
RAMP_MODES = ("step", "stairs")


def normalize_autoscale_mode(value) -> str:
    """Coerce a config/CLI ``autoscale_mode`` value to a canonical mode string.

    Accepts the three canonical strings plus legacy booleans (the knob used to be
    a ``true``/``false`` flag): truthy → ``curriculum``, falsy → ``none``. Raises
    on an unrecognized string so a typo fails loudly rather than silently
    disabling the schedule.
    """
    if value is None:
        return "none"
    if isinstance(value, bool):  # legacy true/false flag
        return "curriculum" if value else "none"
    s = str(value).strip().lower()
    if s in ("true", "1", "yes", "on"):
        return "curriculum"
    if s in ("", "false", "0", "no", "off"):
        return "none"
    if s in AUTOSCALE_MODES:
        return s
    raise ValueError(
        f"autoscale_mode must be one of {AUTOSCALE_MODES} (or a bool), got {value!r}"
    )


@dataclass
class AutoscaleSchedule:
    """Resolution-schedule policy (a no-op ``none`` mode never builds one).

    ``highres_ratio`` is the fraction of training spent at the top tier — the
    trailing-step fraction in ``curriculum`` mode, the per-epoch batch fraction in
    ``random`` mode. ``warmup_batches`` leading steps are left unscheduled so the
    natural shuffle touches *every* tier up front and warms all ``torch.compile``
    block graphs before the schedule masks the high tier away (no recompile / VRAM
    climb at the switch — cf. ``_largest_bucket_first`` and
    project_compile_context_vram_climb). ``ramp`` applies to ``curriculum`` only.
    """

    highres_ratio: float = 0.15
    mode: str = "curriculum"
    ramp: str = "step"
    warmup_batches: int = 8

    def __post_init__(self) -> None:
        self.highres_ratio = float(self.highres_ratio)
        if not 0.0 < self.highres_ratio < 1.0:
            raise ValueError(
                f"autoscale_highres_ratio must be in (0, 1), got {self.highres_ratio}"
            )
        self.mode = str(self.mode).strip().lower()
        if self.mode not in ("curriculum", "random"):
            raise ValueError(
                f"AutoscaleSchedule.mode must be 'curriculum' or 'random', "
                f"got {self.mode!r} (use mode='none' upstream to disable)"
            )
        self.ramp = str(self.ramp).strip().lower()
        if self.ramp not in RAMP_MODES:
            raise ValueError(
                f"autoscale_ramp must be one of {RAMP_MODES}, got {self.ramp!r}"
            )
        self.warmup_batches = max(0, int(self.warmup_batches))

    def active_rank(self, step: int, max_steps: int, n_ranks: int) -> int | None:
        """Which tier rank (0 = cheapest … n_ranks-1 = top) trains at ``step``.

        ``curriculum`` policy only — ``random`` mode picks per-batch (per-epoch)
        in the dataset, not per-step here. Returns ``None`` to mean "do not remap"
        — either there is nothing to schedule (single tier, or ``max_steps`` not
        yet known) or we are inside the warmup window where every tier should flow
        through to warm compile.
        """
        if n_ranks <= 1 or max_steps <= 0:
            return None
        if step < self.warmup_batches:
            return None
        frac = step / max_steps
        if frac >= 1.0:
            frac = 0.999999
        top = n_ranks - 1
        # Top tier owns the trailing ``highres_ratio`` fraction in both ramps.
        if frac >= 1.0 - self.highres_ratio:
            return top
        if self.ramp == "step":
            return 0
        # stairs: split the leading (1 - highres_ratio) bulk equally across the
        # lower ``top`` ranks, lowest first.
        band = (1.0 - self.highres_ratio) / top
        return min(int(frac / band), top - 1)
