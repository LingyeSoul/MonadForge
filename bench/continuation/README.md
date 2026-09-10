# Continuation scheduler state

Run `.venv/bin/python bench/continuation/run_bench.py` from the repository root.
The benchmark compares an uninterrupted AdamW trajectory with a fresh optimizer
and scheduler restored before warmup ends, after warmup, and at the original
training limit. Extending the stop budget must preserve the original proportional
warmup and produce exactly equal learning rates and final parameters.

Results use `results/<timestamp>-scheduler-state/result.json`. This is a small
numerical invariant benchmark; actual Anima training validation is recorded
separately under `output/validation/continuation/`.
