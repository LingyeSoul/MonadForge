# bench/moss_rfe

2:4 activation sparsity (RT-Lynx, arXiv 2605.26632) on MOSS-Image: does a top-2-of-4
mask on a block linear's input keep the GEMM output? Tier 1 §2 of
`docs/proposal/moss_anima_negatives_revisit.md`; run 2026-10-18. **Closed: the gate
fails.**

```bash
make daemon-run ARGS="bench/moss_rfe/probe_rfe.py --label base"
```

## Motivation

On Anima the plain-GELU MLP kept `mlp_down` dense: output RFE 0.25 against the paper's
0.01 (`_archive/rt_lynx/`). MOSS's FFN is gated (SiTU-GLU), the setting the paper's low
error comes from. Inference lever only: training is PCIe-bound under block swap.

## Probe

`probe_rfe.py` (docstring has the details). Training-path forward over
`post_image_dataset/moss/_calib_mixed` (8 samples, evenly spaced by stem) at σ 0.1 / 0.3 /
0.5 / 0.7 / 0.9, Base, cond only. A pre-hook on each of the six linears per block
recomputes the GEMM on the 2:4 input and leaves the live activation dense.
RFE = `‖y − ŷ‖_F / ‖y‖_F` per call with norm compensation, the archived Anima
definition.

**Gate**: `ffn_down` RFE ≤ 0.05 on most blocks.

## Results

**2026-10-18 — base** (`results/20261018-1846-base`; smoke at
`results/20261018-1842-smoke`, 1 sample at σ 0.5, the same numbers to ±0.005). **Gate fails:
`ffn_down` is ≤ 0.05 on 1 / 42 blocks (block 0, 0.043).**

| target | RFE mean | block range | image rows | text rows | input dropped (image) |
|---|---|---|---|---|---|
| `ffn_down` | 0.185 | 0.043–0.317 | 0.205 | 0.118 | 0.21 |
| `ffn_up_gate` | 0.288 | 0.225–0.334 | 0.288 | 0.297 | 0.31 |
| `q` | 0.206 | 0.129–0.239 | 0.201 | 0.253 | 0.33 |
| `k` | 0.193 | 0.074–0.241 | 0.190 | 0.223 | 0.33 |
| `v` | 0.369 | 0.157–0.472 | 0.372 | 0.328 | 0.33 |
| `o` | 0.390 | 0.167–0.539 | 0.393 | 0.358 | 0.36 |

- `ffn_down` by depth: 0.04 at block 0, 0.11–0.18 on 1–24, 0.21–0.23 on 25–33, rising to
  0.31 on 36–39, 0.12 on 41. Only block 0 is under 0.10.
- σ barely matters: `ffn_down` 0.188 at σ 0.1 down to 0.180 at σ 0.9. The other
  targets move by ≤ 0.05 across σ.
- Norm compensation changes RFE by ≤ 0.01 on every target.
- Text rows are sparser than image rows on `ffn_down` (0.118 against 0.205), the only
  target where the two differ by much.

Reading: the gated FFN brings `ffn_down` from Anima's 0.25 to 0.19, still ~4× the
gate. SiTU's gate (`4·tanh(g/4)·sigmoid(g)`) is smooth and has no hard zeros, so
the post-gate activation is not near 2:4. The other five targets are no better. With
the SpTC ceiling at ~1.75× on sm120, no kernel is worth writing.
