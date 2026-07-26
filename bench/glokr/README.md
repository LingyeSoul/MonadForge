# bench/glokr

Tier 2 gate for **GLoKr** — Kronecker-factored delta (LoKr layout) + BoRA
bi-dimensional weight decomposition
([arXiv 2412.06441](https://arxiv.org/abs/2412.06441)).

Pure CPU analytics — no DiT load (`bench/_anima.py` is opt-in per
`CONTRIBUTING.md`; every number here computes from synthetic Linears at real
Anima DiT shapes). Runs in under a minute.

## Run

```bash
python bench/glokr/run_bench.py
python bench/glokr/run_bench.py --label <date> --lora_dim 4 16 32 --factor 8
```

## What it measures / what "good" looks like

| Surface | Headline numbers | Good |
|---|---|---|
| Parameter accounting | GLoKr vs kron-only vs plain-LoRA trainable params per `(shape, rank)`; BoRA overhead is exactly `out + in` scalars per Linear | GLoKr ≪ LoRA at equal rank (~2.5–8× fewer at DiT shapes) |
| Invariant errors | init-identity `max\|W′−W0\|`, merge parity, fuse round-trip | all ≈ 1e-9 (fp32 noise); anything ≥ 1e-5 is a regression |
| Forward overhead | weight-decomposed forward vs plain GEMM at 4096 tok (CPU-indicative) | ratio ≈ 1.0 — W′ construction is elementwise over `out×in`, small next to the token GEMM |
| Row-rescale capability (BoRA §3.3 framing) | final loss fitting `y = x @ (d_row ⊙ W0)^T` for `glokr_bora` vs a DoRA-style column-only arm vs plain kron, identical direction params + Adam LR; ΔM (eq. 5) reported as context | see falsifier below |

**What would falsify the method**: `bora_wins_row_rescale_task` must be
`True` — a pure per-output-row gain is exactly what BoRA's `m_row`
parameterizes, while the DoRA-style column-only arm has no row magnitudes and
the additive kron arm is rank-capped far below the full-rank `(d−1)⊙W0`
target. If BoRA does not beat both baselines by ≥2× loss (or the invariant
errors blow past 1e-5), the bi-dimensional decomposition is not delivering the
capability the paper claims and `glokr_bora` should not ship enabled by
default.

## Baseline run (checked in)

`results/20260726-1817-baseline/` — exact CLI:

```bash
python bench/glokr/run_bench.py --label baseline --steps 400
```

Headlines from that run: invariant errors all ≤ 5.6e-9; row-rescale task
final loss `0.0000` (BoRA, exact fit) vs `0.0630` (DoRA-col) vs `0.0857`
(plain kron); CPU forward overhead ~1.3× (interleaved min-of-N; GPU-relative
overhead is far smaller since the token GEMM dominates there).

## Output

`result.json` envelope (schema from `bench/_common.py`) in
`bench/glokr/results/<YYYYMMDD-HHMM>[-<label>]/`.

Method doc: [`docs/methods/glokr.md`](../../docs/methods/glokr.md).
