# bench/lokr

Tier 1.5 numerics gate for the LoKR decomposition threshold, independent
full-factor mode, and SVD rank-cap behavior in `networks/lora_modules/lokr.py`
and `networks/lora_save.py`.

Pure parameter math — no DiT load (`bench/_anima.py` is opt-in per
`CONTRIBUTING.md`, this is an analytical simulator). CPU-friendly; runs in
under a second.

## Run

```bash
python bench/lokr/run_bench.py
python bench/lokr/run_bench.py --label <date> --lora_dim 4 8 16 32
```

## What it quantifies (before → after)

| Change | Headline numbers |
|---|---|
| Decomposition threshold `max(out,in)/2` → `min(out,in)` | effective rank (`rank(w1)·rank(w2)`) and trainable param count, per `(shape, lora_dim)` |
| Rank-controlled factors → `lokr_full_factor=true` | effective rank, trainable parameter count, and recommended-vs-legacy training scale |
| SVD rank cap global `128` → per-module `alpha` | saved param count + fixed-seed relative-Frobenius SVD-truncation drift |

Drops a `result.json` envelope (schema from `bench/_common.py`) into
`bench/lokr/results/<ts>[-<label>]/`.

The stdout summary table is sized for copy-paste into the PR description (per
the Tier 1.5 rule: before/after numbers must be in the PR).
