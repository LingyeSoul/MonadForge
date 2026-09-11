# Continuation scheduler state

Run `.venv/bin/python bench/continuation/run_bench.py` from the repository root.

Three scenario groups, each comparing an interrupted-then-continued AdamW
trajectory against the uninterrupted reference (bit-equal parameters and
identical LR sequences):

- **extend** — `constant_with_warmup`, budget 10 → 16, restarting before
  warmup ends, after warmup, and at the original training limit. Extending the
  stop budget must preserve the original proportional warmup.
- **cosine_resume** — `cosine` at its original target: pinning the horizon
  keeps the decay curve identical across a mid-run save/load.
- **chained_extend** — 10 → 16 → 24 in two hops: the second extension must
  inherit the first hop's horizon, matching a single 10 → 24 extension.

Results use `results/<timestamp>-scheduler-state/result.json` (the directory is
gitignored; tracked result sets are force-added). This is a small numerical
invariant benchmark; actual Anima training validation is recorded separately
under `output/validation/continuation/`.
