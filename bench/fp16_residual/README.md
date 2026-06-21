# bench/fp16_residual — fp16-safe residual accumulation

## Why this exists

The Anima DiT residual stream exceeds fp16's 65504 ceiling late in the block
stack (`docs/findings/selfflow.md`). Under fp16 autocast the Block residual
adds (`x + gate * sublayer_result`) and the FinalLayer AdaLN modulate
(`norm(x) * (1 + scale) + shift`) overflow to `inf → NaN` from step 0.

`Anima.enable_fp32_residual()` (flipped on by `train.py` when
`--mixed_precision fp16`) keeps the residual adds + final-layer modulate in
fp32; the sublayer matmuls still run fp16 because autocast re-casts fp32
activations into fp16 at each `Linear`/attention. Values >65504 have no fp16
representation, so the residual must stay fp32 (not cast back) until a
downstream matmul picks it up. V100 / sm_70 has no native bf16, so the `bf16`
default silently runs fp32 autocast — users pick fp16 for the matmul speedup.

This is a **Tier 1.5 numerics change** (`CONTRIBUTING.md` §1): it revises the
DiT compute path's numerics, so it carries a bench script + invariant test.

## `run_bench.py` — correctness + speed across dtype configs

Builds a tiny synthetic Anima (CPU-runnable, no real weights) and runs N
forward steps under three configs:

| config | what it tests |
|---|---|
| `bf16` | the production default — baseline correctness + speed |
| `fp16` flag off | the broken path on a trained model (NaN past 65504) |
| `fp16` flag on | the fix — fp32 residual accumulation, matmuls still fp16 |

Headline metrics in `result.json`:

- `*.finite` — output stays finite across N steps (correctness)
- `*.ms_per_step` — per-step wall-time
- `speedup_fp16_on_vs_bf16` — does the fp16 path (with the flag's fp32
  promotion overhead) still beat bf16? This is the speedup the flag exists to
  unlock on V100.
- `flag_overhead_fp16_on_vs_off_pct` — the fp32 promotion's cost relative to
  raw fp16.

```bash
uv run python -m bench.fp16_residual.run_bench [--steps N] [--num_blocks N]
# or: .venv/Scripts/python.exe -m bench.fp16_residual.run_bench
```

### Result (2026-06-21, CPU, 8 blocks, 10 steps)

| config | finite | ms/step | out dtype |
|---|:---:|---:|---|
| bf16 | ✓ | 9.688 | bfloat16 |
| fp16 flag off | ✓ | 9.602 | float16 |
| fp16 flag on | ✓ | 7.360 | float16 |

- `speedup_fp16_on_vs_bf16`: **1.32×** — fp16 + flag beats bf16 even with the
  fp32 promotion overhead (the matmul speedup the flag exists to unlock).
- All three configs finite here because the tiny untrained fixture has
  default-zero AdaLN gates that don't accumulate the residual past 65504.

**Caveat — CPU timing caveat:** the `flag_overhead` is *negative* on CPU
(-23.3%) because CPU fp16 dispatch has its own overhead that dominates the
fp32 promotion cost. On a V100/GPU this inverts — the fp32 promotion shows a
real (positive) overhead, but the fp16 matmul speedup still wins overall. Run
on GPU for representative flag-overhead numbers.

### Overflow reproduction — not in this bench (by design)

The >65504 overflow only bites on a **trained** model at **large resolution**
with **deep** block stacks. A tiny untrained CPU fixture can't reproduce it
(CPU fp16 dispatch, default-zero AdaLN gates that don't accumulate the residual
past 65504), so this bench stays finite on all three configs by design.

The guard has two layers, each pinned by a dispositive unit test:

1. **The residual add** (`Block._residual_add`):
   `test_residual_add_unit_overflow_guard` injects a >65504 sum directly into
   `_residual_add` and asserts finiteness with the flag on vs inf with it off.
2. **The gated product** (`Block._gated_residual_add`): the actual V100 fp16
   NaN from step 0 that survived the original `_residual_add` guard. `gate *
   branch` is materialized in fp16 under autocast *before* it reaches the
   residual add, so a product past 65504 has already collapsed to `inf` — the
   add guard runs too late. `_gated_residual_add` pulls the product into the
   fp32 region. `test_gated_residual_add_guards_the_product_overflow` injects a
   >65504 `gate*branch` and asserts finiteness with the flag on; it also pins
   the negative control that the old `_residual_add` cannot recover an
   already-`inf` product.

This bench validates the *mechanism wires through a real multi-block forward*
(flag propagation, dtype contract, finite output) and quantifies the speed
tradeoff — not the overflow itself.

For real overflow reproduction + GPU timing, run on a V100 with a trained
checkpoint at training resolution.

> Note — the unpatchify `linear` cast is *still* outside the guard's scope:
> if the residual stream itself is past 65504 *after* the last block, the
> autocast fp32→fp16 cast at `FinalLayer.linear`'s matmul overflows. In
> practice the FinalLayer `LayerNorm` collapses the residual to O(1) before
> the modulate, so this only bites if the AdaLN `scale` is pathologically
> large. Not observed on the trained V100 checkpoint after the gated-product
> fix.

## Related

- `tests/test_fp16_residual_safe.py` — the invariant tests (unit overflow,
  inert-on-default parity, propagation, compile-ordering regression,
  backward, end-to-end FinalLayer forward).
- `docs/findings/selfflow.md` — the residual-stream magnitude analysis that
  motivates the guard.
- `library/anima/models.py::Anima.enable_fp32_residual` — the canonical
  rationale docstring.
