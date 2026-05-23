# bench/static_padding

Does static-shape zero-padding leak into the real-token output under
`attn_mode="flash"`? (issues.md #1 / CLAUDE.md "constant-token bucketing".)

`probe_pad_leak.py` — for real buckets (token counts just under 4096) and
synthetic large-gap latents, runs the base DiT with identical seeded inputs
four ways and compares the real-token velocity (fp32 rel-L2 vs the
native-no-padding forward, which is exact ground truth since flash is
deterministic):

| variant | static count | attn | role |
|---|---|---|---|
| ref / ref2 | None | flash | native seq_len, no pad — truth + noise floor |
| pad_flash | 4096 | flash | production path, unmasked padding |
| pad_flex (`--flex`) | 4096 | flex | padding masked out |

```
python -m bench.static_padding.probe_pad_leak [--flex] [--seed N] [--sigmas …]
```

## Finding (2026-05-23, 3 seeds, random inputs)

**The attention-sink assumption is FALSIFIED.** Padded tokens aren't zero into
attention (σ-dependent AdaLN shift + Q/K/V bias), so they leak:

- gap=0 control → bit-exact (reshape itself is clean).
- gap 16 (bucket 4080) → ~0.6–0.75% rel-L2, steady.
- gap 40–64 → ~1% baseline, **spikes to 4–6.5%** at σ=0.5–0.8 on some inputs.
- synthetic gap 1024/3072 → ~5–11% / up to 27%. Grows with gap and σ.
- flex stays flat ~0.6% across all gaps (that 0.6% is a flex-vs-flash *kernel*
  difference seen even at gap=0, not masking residual) → masking removes the leak.

Caveat: random inputs (not real cached latents); single-forward velocity, not
end-to-end sample. See memory `project_static_flash_padding_leaks`.

## The no-pad fix (`--no_static_pad`, 2026-05-23)

`set_static_token_count(count, pad=False)` keeps constant-token-bucket mode
(block-compile, not whole-model) but runs each bucket at its native token count
instead of zero-padding to `count`. No padding → no leak. Crucially it still
**flattens** the patch grid to `(B,1,L,1,D)` (just with zero pad), so the block
stack keys on token count, not `(H,W)` — otherwise dynamo guards on H and W
separately and recompiles per *resolution* (17) instead of per token-count.
The 17 buckets collapse to **5** distinct token-counts {4032,4050,4056,4080,4096}.
Inference never padded, so this is a **training-only** change that also removes a
pre-existing train/infer mismatch. Default stays `pad=True`; opt in with
`train.py --no_static_pad`.

`probe_pad_leak.py` now also runs a `nopad_flash` variant. Verified clean:

- no-pad verdict **CLEAN** — `nopad_flash_rel_l2 = 0.0`, cos `1.000000` at every
  gap (bit-exact to the no-pad ground truth), while `pad_flash` LEAKS up to 27%.

`bench_compile_mem.py` measures the cost of holding 5 graphs vs 1 (forward-only,
inductor default, no CUDAGraphs):

- pad (1 shape): peak **4.12 GiB**, 2 unique graphs.
- no-pad: running all **17 resolutions** yields only **10 unique graphs**
  (= 5 token-counts × fwd/bwd) — confirms the collapse — peak **7.77 GiB** →
  **+3.66 GiB**.
- recompiles bounded (10 < bumped `cache_size_limit = 2·5+8 = 18`); no eager
  fallback. `compile_mode='full'` (compile_core) is rejected — not shape-invariant.

So no-pad is correct and fits the default 16 GB preset (7.8 GiB fwd) but the
~+3.7 GiB resident overhead argues against it for low_vram/8 GB. Keep it opt-in.
`reduce-overhead` (per-shape CUDAGraph pools) is expected worse — run
`bench_compile_mem.py --inductor_mode reduce-overhead` before pairing them.
