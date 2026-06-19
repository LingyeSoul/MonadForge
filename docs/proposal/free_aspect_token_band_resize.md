# Free-aspect token-band resize ("free-fit")

**Status:** proposal · **Owner:** —  · **Tier:** plumbing + new preprocess mode (no numerics change)

## Summary

Stop snapping every image to one of the ~12 discrete `(W, H)` buckets per tier.
Instead, **preserve the native aspect ratio** and resize so the patch-grid token
count `Wp·Hp` lands *anywhere inside the tier's token band* (`[4032, 4200]` for
the 1024 tier), picking the grid that minimizes aspect distortion and crop. Each
forward already runs at its true token count with no padding; under
`compile_dynamic_seq` the whole band is **one** compiled block graph, so finer
shape granularity is free at compile time.

Pair it with a **GUI max-aspect-ratio limiter** (default 1:4 / 4:1) that clamps
degenerate ratios (1:5, 1:6 …). This is not just a quality guard — it also keeps
the token band *solvable* for elongated images (see §4), so the two features are
deliberately co-designed.

## 1. Motivation

The discrete `CONSTANT_TOKEN_BUCKETS` table
(`library/datasets/buckets.py`) exists for exactly one reason: under the **static**
compile path each distinct token count is its own dynamo graph, so collapsing all
shapes onto two counts (4032 = 63·64, 4200 = 60·70) keeps the 1024 tier at 2
graphs instead of ~24. Everything else about the table — the hand-picked aspect
ladder, the cover-then-crop in `process_image` (`library/preprocess/images.py:189`)
— is a *consequence* of that constraint.

But `compile_blocks(dynamic_seq=True)` (`library/anima/models.py:1614`,
`_make_dynamic_seq_forward`) marks **only** the seq axis dynamic over `[lo, hi]`
and produces a **single graph for the entire band**. RoPE is built per actual
`(Hp, Wp)` outside the block and flattened in, so the block forward sees only
`seq_len` vary — it is indifferent to whether 4096 tokens is 64×64 or 32×128.
Once `dynamic_seq` is the trusted path, the table's reason-to-exist evaporates:
any token count in `[4032, 4200]` costs nothing extra.

What we pay today for that 2-graph collapse is **crop loss**. `process_image`
does cover-then-crop to the nearest-aspect bucket; the aspect ladder jumps
(e.g. 1.29 → 1.34 → 1.68 → 1.75 once both families interleave), so an image at
ar≈1.5 snaps to 1.34 or 1.68 and loses ~6–13% of one dimension. The 2:3 / 3:2
and 16:9 bands are where this bites hardest. Free-fit drives crop to ~zero
(only the sub-patch <16px residual remains, as a negligible <1.6% stretch).

## 2. Design

### 2.1 The free-fit solver (`library/preprocess/`)

Pure function, deterministic in its inputs:

```
freefit_bucket(w, h, band=(lo, hi), max_ratio=R, patch=16, rope_cap=256) -> (W, H)
```

1. **Clamp aspect.** `a = w/h`; `a' = clamp(a, 1/R, R)`. If `a` was outside, the
   image is cover-cropped to `a'` (the *only* place crop survives — and only for
   ratios the user explicitly allowed). With the default `R = 4.0` this matches
   the current table's most-elongated reach (2016×512 = ar 3.94), so the default
   crops nothing the old path didn't.
2. **Seed the grid.** `Hp = round(sqrt(N/a'))`, `Wp = round(Hp·a')`, with `N` the
   band target (default the count minimizing `|log(cover_scale)|` so we neither
   up- nor down-scale more than necessary — same spirit as `choose_edge`).
3. **Snap into the band.** Search a small neighborhood `(Wp±k, Hp±k)` for the pair
   that minimizes `|Wp/Hp − a'|` subject to `Wp·Hp ∈ [lo, hi]` **and**
   `max(Wp, Hp) ≤ rope_cap`. Return `(Wp·patch, Hp·patch)`.

Aspect error is sub-patch by construction; crop is zero unless the ratio clamp
fired. Reuses `_cover_scale` / `choose_edge` for tier assignment, so multi-tier
(`--target_res 512 768 …`) keeps working — free-fit operates *within* the tier
`choose_edge` already picks.

### 2.2 Batching: aspect-bin granularity knob

Images batch within an **identical** `(W, H)`, not within a token family (the
4032 family already holds 12 incompatible shapes). Pure per-image free-fit gives
nearly every image a unique shape → buckets of size ~1 → effective `batch_size=1`.

- **`bs = 1`** (typical at 16 GB with block swap): non-issue, pure win.
- **`bs > 1`:** quantize `a'` to `N_bins` aspect bins (default 32), each bin →
  one shared free-fit shape. Bin count is a pure *batching-coherence* knob,
  fully decoupled from compile (all bins live in the same band → still 1 graph
  under `dynamic_seq`). `N_bins → ∞` is the pure free-fit limit.

This is the recommended default: 32 bins ≈ 3× finer than today's ladder, crop
loss largely gone, batches still coherent.

### 2.3 Compile coupling (mandatory)

Free-fit **requires** `compile_dynamic_seq` (or `torch_compile` off). Under the
static path the extra shapes would explode the graph count. The train side is
already compatible: `_derive_token_budget` (`train.py:1521`) computes
`seq_range = (min, max)` from the buckets the dataset actually populates, and
`dynamic_seq` keys one graph off exactly that range. Free-fit just populates more
distinct `(W, H)` *within the same `[lo, hi]`* — `seq_range` is unchanged, graph
count stays 1. Action: when free-fit is on, **auto-enable `dynamic_seq`** (or hard
error if the user forced static).

### 2.4 Train-side bucket registration (the one real load-bearing change)

`make_buckets(constant_token_buckets=True)` currently sets the predefined reso set
to `all_constant_token_buckets()` and exact-matches each cached latent
(`buckets.py:402`). Free-fit shapes are **not** in that table, so `select_bucket`
would AR-snap-and-resize them at load — defeating the purpose. Fix: a `freefit`
bucket mode whose predefined set is the **union of the actual on-disk cached
`(W, H)`** (scan the cache dir). The docstring already calls the caches "the
source of truth"; this makes that literal. `add_if_new_reso` machinery exists.
The size-aware re-resize skip in `process_image` (`images.py:178`) stays valid
because `freefit_bucket` is deterministic — but `_resize_metadata_signature`
(`images.py:80`) must fold in `band`, `max_ratio`, `N_bins` so changing any of
them re-resizes correctly.

### 2.5 GUI (`gui/tabs/preprocess_tab.py`)

- **Fit-mode toggle:** `Snap to buckets` (current) vs `Free-fit (band)`.
- **Max aspect ratio** control (spinbox/slider, default `4.0`): clamps both
  portrait and landscape; beyond-clamp images cover-crop to the limit honoring
  the existing `resize_crop_anchor`. Blocks 1:5 / 1:6 inputs.
- **Aspect bins** (advanced, default 32) — visible only in free-fit mode.
- **Live preview:** extend `compute_resize_preview` (`resize_preview.py:182`,
  already wired to the GUI) to free-fit so the user sees the exact resulting
  `W×H` + crop rect, plus a one-line "free-fit ⇒ dynamic_seq auto-enabled; N
  distinct shapes" note.
- i18n + field help: add `preprocess_max_ratio` / `preprocess_fit_mode` /
  `preprocess_aspect_bins` strings and `_preprocess_fields.json` entries across
  en/ja/cn/ko (use the `translator` agent after the English source lands).

## 3. What does *not* change

- **Numerics / bit-exactness.** Each forward already runs at its true token count
  with zero padding; free-fit only changes *which* counts appear. `dynamic_seq`
  is bit-exact vs the static path (memory: `project_static_pad_nopad_fix`,
  `project_compile_context_vram_climb`). No loss/sampler/adapter change.
- **The frozen 1024 table** stays as the `Snap` default and for DCW (see §4).
- **`--target_res` tiering** — `choose_edge` still assigns the tier; free-fit
  works inside it.

## 4. Risks & known limitations

- **RoPE per-axis cap (256 patches).** The solver hard-caps `max(Wp, Hp) ≤ 256`.
  Safe for 1024 (`R≤4` ⇒ ≤126). Only the 1536 tier (band 8640) with a high
  `max_ratio` can approach the cap — the clamp covers it; add a test.
- **Band solvability ↔ max_ratio (the co-design point).** As `a'` grows, the band
  `[lo, hi]` admits fewer integer grids (small `Hp` ⇒ coarse steps). At `R=4`,
  1024-tier min `Hp≈32`, plenty of room. Past ~1:5 the band can't be hit without
  a visible aspect error *and* the image is degenerate for training — which is
  exactly why the GUI limiter both improves quality and keeps free-fit well-posed.
- **DCW calibration.** `DCW_ASPECT_BUCKETS` is a frozen 5-bucket table keyed by
  exact `(H, W)`; free-fit shapes won't match its `aspect_id` lookup. Inference is
  bucket-agnostic post-cleanup (memory: `project_dcw_bucket_prior_cosmetic`), so
  generation is unaffected, but `make dcw` recalibration should keep running over
  a `Snap`-mode dataset (or map free-fit shapes to the nearest DCW bin). Flag in
  docs; don't block.
- **Compile-cache guard poisoning.** More distinct seq values pushed through one
  dynamic graph is precisely the regime that hit
  `project_compile_cache_guard_poisoning`. The per-signature
  `isolate_compile_cache()` fix should cover it; Phase 1 bench must confirm graph
  count stays 1 and no `ConstraintViolationError` across a fully-populated band.

## 5. Plan

- **Phase 0 — solver + test (no training).** `freefit_bucket` + unit test: every
  test aspect lands in band, respects `max_ratio` clamp and rope cap, aspect error
  < 1 patch, deterministic. Wire `compute_resize_preview` so the GUI/CLI preview
  agree with `process_image`.
- **Phase 1 — train side.** `freefit` bucket mode (predefined resos from cache),
  auto-enable `dynamic_seq`, signature bump. Smoke-train `bs=1` on one artist set;
  assert **1** block graph over the populated band (`TORCH_LOGS=recompiles`), peak
  VRAM via `mem_get_info`, no ConstraintViolation.
- **Phase 2 — GUI.** Fit-mode toggle, max-ratio control, preview note, i18n.
- **Phase 3 — batching + A/B.** Aspect-bin knob for `bs>1`; CMMD val (memory:
  `project_cmmd_val_signal`) snap vs free-fit + visual edge-crop inspection
  (no scalar quality reward exists for Anima — `project_null_tta_phase0...`).

## 6. Touch list

| Area | File |
|---|---|
| Solver + preview | `library/preprocess/resize_preview.py`, `library/preprocess/images.py` |
| Bucket mode | `library/datasets/buckets.py` (`BucketManager.make_buckets`, new `freefit` path) |
| Compile coupling | `train.py` (`_derive_token_budget`, auto-enable `dynamic_seq`), `library/config/cli_args.py` |
| GUI | `gui/tabs/preprocess_tab.py`, `gui/i18n/*`, `gui/explanations/guides/*/preprocess.*` |
| Tests | `tests/` (solver invariants), bench graph-count check |
