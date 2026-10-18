# Minimal training runtime boundaries

This CPU benchmark exercises the production warmup/cosine scheduler with real
AdamW updates and the production stage selector. It needs no model weights,
dataset, GPU, or daemon.

```bash
uv run python bench/training_runtime/run_minimal.py
uv run python bench/training_runtime/run_minimal.py --output-root /tmp/training-runtime
uv run pytest -o addopts= -q tests/test_distill_runtime.py tests/test_stage_schedule.py
```

Each run writes a standard `bench._common` result envelope. The scheduler cases
use `(total_steps, warmup_steps)` of `(1, 1)`, `(2, 500)`, and `(1000, 20)`, with
one scalar parameter, AdamW, and peak learning rate `0.001`. They require a real
parameter update and final learning rate `0.0001`. The stage case requires three
equal stages to select subsets `0`, `1`, and `2` on three consecutive steps.

The 2026-10-18 tiny training audit reproduced these failures before the fixes:

| Boundary | Before | After |
| --- | --- | --- |
| One training step with one warmup step | Turbo updates parameters, then cosine `T_max=0` raises `ZeroDivisionError` | Warmup is capped at `total_steps - 1`; the one-step update and scheduler finish |
| Warmup longer than the run | No positive cosine interval | At least one cosine step remains |
| Three equal stages over three steps | `100/3%` and `1/3` round differently, delaying the stage transition | The selector uses the existing stage compiler's `1e-12` boundary tolerance and visits `0, 1, 2` |

Normal schedules with a warmup shorter than the total keep their previous
behavior; the 1000-step case checks the existing cosine floor. Nonpositive
total steps now fail explicitly. Stage intervals remain half-open apart from
the existing inclusive final stage; only floating-point boundary agreement
changes. These tests establish execution at minimal step counts, not model
convergence or GPU performance.

The continuation regression also compares the former global-RNG sampler replay
with a dedicated loader generator saved before the epoch permutation. On the
ten-sample, three-consumed fixture, the former path repeats samples 2 and 7;
the saved-permutation path reproduces the remaining samples exactly and visits
all ten once. The invariant tests additionally exercise the real training loop
at mid-epoch and epoch-end interruption, and checkpointing at a stage boundary.

Fresh runs now use their own seeded loader generator, so their order can differ
from older versions. New checkpoints carry its epoch and state. Legacy ordinary
checkpoints cannot recover a permutation they never stored and warn explicitly.
Legacy stage-boundary checkpoints start the next stage with cursor zero; they
never reuse a cursor from the completed stage. New boundary checkpoints record
the next stage and its pending generator state before save/stop returns.
