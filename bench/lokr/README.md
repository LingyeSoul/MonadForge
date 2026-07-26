# bench/lokr

Tier 1.5 numerics gate for the LoKr migration from the former local threshold
to the official `lycoris-lora==3.4.0` decomposition/full-matrix behavior, plus
the SVD rank-cap path in `networks/lora_save.py`.

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
| Local threshold `min(out,in)` → official `max(out,in)/2` | effective rank (`rank(w1)·rank(w2)`) and trainable param count, per `(shape, lora_dim)` |
| Rank-controlled factors → official full-matrix mode | effective rank, trainable parameter count, and explicit-vs-legacy unit scaling |
| SVD rank cap global `128` → per-module `alpha` | saved param count + fixed-seed relative-Frobenius SVD-truncation drift |

Drops a `result.json` envelope (schema from `bench/_common.py`) into
`bench/lokr/results/<ts>[-<label>]/`.

The stdout summary table is sized for copy-paste into the PR description (per
the Tier 1.5 rule: before/after numbers must be in the PR).

## Eager memory path

`run_eager_memory_bench.py` compares official LyCORIS eager bypass autograd
with MonadForge's chunked/rematerialized FP32 path. It reports saved-tensor
bytes on CPU and, when CUDA is selected, peak allocated/reserved memory and
forward+backward time.

```bash
python bench/lokr/run_eager_memory_bench.py
python bench/lokr/run_eager_memory_bench.py \
  --device cuda --rows 4200 --shape 9216 3072 \
  --chunks 512 1024 2048 3072 --label v100
```

The default `--shape 9216 3072 --factor 8` represents a wide Anima FFN
projection. CPU defaults to 128 rows so the saved-tensor comparison remains
quick; use the actual bucket row count for a hardware peak-memory run.
