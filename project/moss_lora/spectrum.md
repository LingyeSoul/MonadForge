# Phase 1 — Spectrum on MOSS-Image

`library/moss/spectrum.py`, wired into `generate.denoise` and switched on by
`--spectrum` (`GenerateRequest`). Bench: `bench/moss_spectrum/run_bench.py`.

## What it forecasts

The image-token hidden state after block 41 (before `final_norm_video`), `(R, H·W, 3072)`.
A forecast step runs only `time_conditioning` + `head`; the 42 blocks and the text
refiner are skipped. Cond and uncond rows are stacked in one forecaster. The forecast is
a weighted sum of each element's own history, so the skip-22 uncond trajectory never
mixes into cond (`test_rows_are_independent`).

Schedule: Anima's growing window (warmup 6, last 3 steps full, a full step every
`floor(N)` steps, `N` from 2 growing by `flex_window`), fit over the step index.
Estimator: the Chebyshev-ridge + Taylor blend of `networks/spectrum_forecast.py`
(m 3, λ 0.1, w 0.3), solved in its linear form. The core module's own predictor
`torch.cat`-grows its history and casts it to fp32 on every fit. On the first run that
cost **+3 GiB peak (13.0 vs 10.1 GiB)** and allocator OOM retries. The linear form
costs +1.0 GiB (22 stored observations × 2 rows × 4096 × 3072 bf16) and has no retries.

## Numbers (5070 Ti, 1024², CFG 3, uncond skip 22, seed 42, 5 prompts, r2)

`bench/moss_spectrum/results/20261016-2203-r2/`; PNGs in
`/media/sorryhyun/data/moss_image/bench/20261016-2203-r2/`.

| arm | full fwd | Base s/img | SFT s/img | speed-up | peak GiB | latent rel-L2 vs p50 (Base / SFT) |
|---|---|---|---|---|---|---|
| p50 (reference) | 50 | 71.1 | 71.6 | 1.00 | 10.06 | 0 / 0 |
| p25 | 25 | 36.0 | 35.4 | 1.98 | 10.06 | 0.48 / 0.63 |
| s50 (fw 0.25) | 22 | 31.8 | 31.7 | 2.23 | 11.05 | 0.37 / 0.27 |
| **s50 fw 0.5** | 19 | 27.4 | 27.2 | **2.60** | 10.90 | 0.40 / 0.30 |
| s50 fw 1.0 | 16 | 23.0 | 22.9 | 3.09 | 10.76 | 0.42 / 0.38 |
| s50 batched (b=1) | 22 | 39.5 | 39.8 | 1.80 | 11.60 | 0.39 / 0.27 |

Base-only first run (`20261016-2129-base3`, 3 prompts, old predictor): p30 42.2 s
(1.68×), p20 28.1 s (2.52×), s30 24.0 s (2.94×, 17 forwards).

A forecast step costs ≈ 0.02 s against ≈ 1.42 s for a full CFG step: s50 = 22 × 1.42 +
28 × ~0.02.

**The batched cond+uncond forward is slower**: 39.5 s vs 31.8 s at the same 22 forwards.
The masked SDPA leaves the flash kernel. The unbatched pair stays the default.

## By eye

I viewed all ten r2 grids at 512 px, plus full-res crops of SFT anime_sunset (face and
bow) and Base zh_fridge (the note), and the Base ja_manga bubble from the first run.

- **Spectrum tracks the 50-step composition far better than plain fewer steps.** p25
  rearranges the image: SFT ja_manga changes the boy, pose and bubble placement, and
  SFT anime_sunset changes the pose. s50 at every fw keeps p50's layout, pose and
  palette. The latent rel-L2 agrees: s50 fw 1.0 (16 forwards) is closer to p50 than
  p25 (25 forwards) on both models.
- Text: the quoted string stays correct in every Spectrum arm viewed (待ってくれ！,
  오늘도 수고했어, the three Chinese lines). Base zh_fridge is the one prompt where
  every Spectrum arm departs from p50 the same way: a larger note with larger
  characters, all correct.
- **fw 0.25 and fw 0.5** look indistinguishable from p50 at grid scale. In the SFT face
  crop they match it closely (eyes, hair clips, bow).
- **fw 1.0 starts to lose fine detail**: the SFT bow is smeared and a pink blob
  appears on the jacket. On the first run, Base ja_manga showed grainier hatching on
  the shirt at fw 1.0, and s30 put a stray mark in front of 待.

## Default

`--spectrum` ships off. When on, it defaults to 50 steps, window 2, **flex_window 0.5**
(19 full forwards, 2.6×, +0.85 GiB). fw 0.25 (2.2×) is the conservative setting, and
fw 1.0 (3.1×) visibly costs detail. These are 10 images per model at one seed, viewed
at grid scale with three crops: enough to pick a default, not a quality measurement.

## torch.compile (2026-10-17)

`--compile` (default on) on `scripts/moss/generate.py` (and `run_bench.py --compile`) wraps each
block's `forward` in `torch.compile(dynamic=True)` after the LoRA load, the same
`compile_blocks` training uses (now in `loader.py`). Spectrum is untouched by it: a
forecast step skips the blocks, a full step runs the compiled ones. Base, 1024², 3
prompts (`results/20261017-0020-cmp_eager`, `…-0026-cmp_compile`):

| arm | eager s | compiled s | peak eager / compiled GiB |
|---|---|---|---|
| p50 | 71.7 | 52.2 (1.37×) | 10.06 / 9.84 |
| s50 fw 0.5 | 27.2 | 20.0 (1.36×) | 10.90 / 10.69 |

Spectrum × compile = 3.6× over eager p50. The two-step warm-up took 7.2 s compiled
vs 3.3 s eager, with the inductor cache already warm from training runs; a cold
compile is closer to training's ~30 s. Eager vs compiled final latents differ by
rel-L2 0.16–0.32 (p50) and 0.23–0.53 (s50). Composition and character are the same in
every pair; strands of hair, tie pattern and background lines differ. bf16 kernel
differences grow over 50 CFG-3 steps, and manga hatching drifts the most.

`--batched` stays slower when compiled: p50 55.0 vs 52.2 s, s50 fw 0.5 20.9 vs 20.0 s,
peak +0.3 GiB (`results/…-cmp_batched_cudnn`, first image includes its compile).
Inductor ignores an `sdpa_kernel` priority, so the compiled masked SDPA stays on the
mem-efficient kernel. Eager does honour it, but cuDNN masked at batch 2 (5.9 ms) is
still slower than two unmasked flash calls (2 × 2.46 ms), so batching cannot win at
1024². CFG costs the same compute either way.

## Not done

- Spectrum at non-square sizes and ~2 MP (the history grows to 2× per observation:
  ~2.2 GiB at 8214 tokens).
- The SEA schedule (`networks/spectrum_sea.py`) and residual calibration are not
  wired in.
- No sweep of `w` / `m` / warmup.
