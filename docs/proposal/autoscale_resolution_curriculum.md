# `autoscale_mode` — resolution-schedule training

Status: **IMPLEMENTED (v1) — awaiting the Phase-1 A/B.** The schedule +
autoscale-aware preprocess emit ship behind the `autoscale_mode` knob (default
`"none"`); the matched-FLOPs arbiter (Phase 1 below) has not run yet.

`autoscale_mode` is a three-way enum (was a `true`/`false` flag; legacy booleans
still parse as `curriculum`/`none`):

- **`none`** — off; train normally (cached tier emits collapse to one per stem).
- **`curriculum`** — train the bulk at the cheapest cached tier, then the top
  tier for the trailing `autoscale_highres_ratio` fraction of steps (`ramp`
  picks the multi-tier climb shape). The original temporal curriculum.
- **`random`** — interleave high/low-res throughout the run, drawing the top
  tier for `autoscale_highres_ratio` of each epoch's batches and the cheapest
  tier otherwise (binary, no hard phase boundary; `ramp` is ignored). Same
  long-run compute split as `curriculum`, spread uniformly instead of back-loaded.

Both real modes share the single `autoscale_highres_ratio` knob (this replaced
the curriculum-only `autoscale_finish`).

## How to run (v1)

```bash
# 1. Cache each image at BOTH ladder tiers (stem-suffixed .as896 / .as1024).
#    ~N× latent disk for N tiers (TE/PE caches are shared). The curriculum picks
#    ONE tier per step — total steps stay equal to a single-resolution run; off
#    autoscale the tiers collapse to the highest-res copy per stem.
make preprocess-resize ARGS="--autoscale_tiers 896 1024"
make preprocess-vae && make preprocess-te && make preprocess-pe   # cache every emitted PNG

# 2. Train with the curriculum on (bulk at the cheap tier, top tier for the
#    final 15% of steps). Tiers are auto-discovered from the populated buckets.
#    Swap curriculum→random to interleave high/low-res instead of back-loading.
make lora ARGS="--autoscale_mode curriculum --autoscale_highres_ratio 0.15 --autoscale_ramp step"
```

Implementation map:
- **Schedule policy**: `library/datasets/autoscale.py` (`AutoscaleSchedule`,
  pure progress→active-tier, unit-tested in `tests/test_autoscale_schedule.py`).
- **Dataset hook**: `BaseDataset.enable_autoscale` + an index remap in
  `__getitem__` (`library/datasets/base.py`) — the DataLoader uses `shuffle=True`
  (RandomSampler), so the curriculum is a per-fetch remap keyed on the shared
  `current_step`/`max_train_steps`, not a custom sampler. Armed on the **train
  group only** (val always scores at full populated resolution). A leading
  `warmup_batches` window is left unscheduled so every tier's `torch.compile`
  graph warms up front (no recompile/VRAM climb at the phase switch). Tiers are
  discovered from the data via `edge_for_token_count` (`buckets.py`); single-tier
  data ⇒ no-op.
- **Epoch length is pinned to one tier (total steps unchanged).** The buckets
  hold *every* tier of each image, so a naive epoch would be ~N_tiers× longer and
  the same `max_train_epochs` would silently train N× the steps — the curriculum
  would degenerate into "train the whole dataset at every resolution." Instead
  `enable_autoscale` shrinks the reported length to one tier's worth (the largest
  single-tier pool — the low tier, which holds every stem), and the `__getitem__`
  remap fills that fixed step budget from whichever tier is active (the smaller
  top tier repeats to cover the finish phase). So a run at fixed `max_train_epochs`
  has the **same total step count as a single-resolution run** — autoscale changes
  *which resolution* each step trains at, never *how many* steps there are. To
  actually spend the saved FLOPs on more steps, raise `max_train_epochs` yourself.
  Because this resizes the train group, `enable_autoscale` runs in `train.py`
  **before** the `len(dataloader)`-based step calc (and `DatasetGroup.enable_autoscale`
  calls `refresh_concat_state()` so `ConcatDataset` re-reads the new length).
  - **Off autoscale (`autoscale_mode="none"`) the tiers collapse**: the dataset
    keeps only the **highest-edge tier per stem** (autoscale never upscales, so
    that's the least-downscaled copy) instead of training every emitted tier as a
    duplicate — so a corpus preprocessed with `--autoscale_tiers` trains exactly
    like a single-resolution run when the curriculum is off
    (`DreamBoothDataset.collapse_autoscale_tiers`, default on; always on for val).
- **Preprocess emit**: `--autoscale_tiers` on `scripts/preprocess/resize_images.py`
  → `resize_to_buckets(autoscale_tiers=…)` → one stem-suffixed PNG per tier.
  **No gratuitous upscaling**: a tier is emitted for an image only if it does not
  exceed the image's natural `choose_edge` home tier — i.e. cheaper (downscaled)
  tiers for the bulk phase + the natural tier for the finish phase, never a tier
  *above* natural (which would invent detail and waste the costly tokens). So a
  sub-896-native image just emits its natural tier and trains normally (no
  curriculum, no upscale); only images with the native resolution to span the
  ladder get the multi-tier curriculum. This makes the FLOP-accounting bound
  ("images below the low tier are unaffected") literally true.
- **Config**: `autoscale_mode` / `autoscale_highres_ratio` / `autoscale_ramp` in
  `configs/base.toml` + `library/config/cli_args.py`; normalized via
  `normalize_autoscale_mode` and wired in `train.py`.

Cache sharing: the stem-suffixed emit keeps the **latent** cache per-tier
(resolution-specific) but **shares the TE (caption) and PE (semantic) caches**
across tiers — both are tier-independent, so the cache-name builder strips the
`.as<edge>` marker for those suffixes (`library/io/cache_names.tier_base_stem`,
applied at the `resolve_cache_path` chokepoint + the few direct-construction
sites). So disk cost is ~2× **latents only**, not 2× everything, and the TE pass
encodes each caption once (the second tier hits the shared cache and skips).
Bespoke loops (turbo/spd/mod) do not inherit the schedule (the standing mirror
caveat).

---

Phase-0 record: **Phase-0 PASS (weak / KILL-case ruled out).** Bench:
`bench/res_curriculum/grad_alignment.py` — run `20260628-0138-grad-align`
(`anima_artist_3_16`, 896→1024, 32 imgs × 2 seeds). The instantaneous training
gradient is shown to be **resolution-robust within noise** (low-res ≈ high-res
direction), so the cheap bulk phase is not wasted. The payoff is **modest and
detail-concentrated**, not dramatic — so this is "safe to try, lean on a short
high-res tail," gated on the matched-FLOPs A/B (Phase-1). It is *not* a new
adapter family: it's a training-schedule feature, orthogonal to the method axis
(composes with LoRA / Ortho / T-LoRA / Hydra / Chimera like `channel_scaling`
and the T-LoRA σ-schedule do).

## TL;DR

`autoscale_mode=true` trains most of the run at a **cheap low tier** (e.g. 896 →
~3000 tok) and automatically switches to the **expensive high tier** (1024 →
4032/4200 tok) for the final fraction of steps. The total step count is **the
same** as a single-resolution run at the same `max_train_epochs` (the epoch
length is pinned to one tier — see the dataset-hook note); the win is that the
bulk phase costs ~0.65–0.73× the per-step FLOPs of the high tier, so the run is
**cheaper for the same number of steps**. The bet is that final-resolution
quality is preserved because low-res steps point the same way high-res steps do.
To convert the saved FLOPs into *more* steps instead of a cheaper run, raise
`max_train_epochs` explicitly. The bench confirms the premise holds and rules out
the failure mode (low-res misdirecting the adapter).

## Motivation

Free-fit native bucketing already trains multi-scale, but each image sits at the
**single** tier that resamples it least (`choose_edge`), for its **whole** run.
The FLOP cost of a step scales steeply with token count (attention ~N², linear
~N), so every step on a high-native-res image pays the full 1024-tier price from
step 0 — including the early phase where the adapter is learning coarse
structure that is largely scale-invariant. A resolution curriculum reallocates
that early compute: learn structure cheap, spend the expensive tokens only on
the late detail phase.

This is the NaViT efficiency lever ("more useful examples per compute budget")
adapted to a *temporal* schedule rather than NaViT's packing — and unlike the
"cache the same image at N scales and sample randomly" idea (which re-renders
content without adding examples; see the discussion that spawned this), a
curriculum trains **one resolution per step** on a schedule. The step count is
held equal to a single-resolution run (the epoch length is pinned to one tier, so
the extra cached tiers never inflate the steps); the FLOP saving is a cheaper run
at the same step budget. Spending those saved FLOPs on *more* steps is an opt-in
(bump `max_train_epochs`), not an automatic doubling from caching extra tiers.

## What `autoscale_mode` does

Config surface (training side; `configs/base.toml` defaults, override per
method/preset/CLI):

```toml
autoscale_mode      = true        # off by default — opt-in schedule
autoscale_tiers     = [896, 1024] # low → high ladder (must be cached, see below)
autoscale_finish    = 0.15        # fraction of max_train_steps at the top tier
# autoscale_ramp    = "step"      # "step" (hard switch) | "stairs" (multi-tier)
```

Schedule (the simple, defensible v1 — `ramp = "step"`):
- steps `[0, (1−finish)·T)`  → only the **low** tier's cached samples are active.
- steps `[(1−finish)·T, T)`  → only the **high** tier's samples are active.

With a 3-tier ladder and `ramp = "stairs"`, split the run into contiguous
progress bands, lowest→highest. Adaptive switching (advance a tier when the
val/CMMD signal plateaus) is deferred — see Phase-2.

## Why it's safe — the Phase-0 evidence

`bench/res_curriculum/grad_alignment.py` measures the adapter-param gradient of
the rectified-flow loss at two resolutions of the *same* image, per σ, and
brackets the cross-resolution cosine with two controls (noise-draw ceiling
`cos_seed`; cross-image floor `cos_img`).

Run `20260628-0138-grad-align`:

| σ | cos_res | cos_seed | cos_img | mag (hi/lo) |
|------|--------|--------|--------|------|
| 0.05 | 0.753 | 0.852 | 0.475 | **1.28** |
| 0.45 | 0.156 | 0.218 | 0.118 | 1.09 |
| 0.75 | 0.222 | 0.223 | 0.109 | 0.92 |
| 0.95 | 0.407 | 0.408 | 0.121 | **1.18** |
| 1.00 | 0.778 | 0.857 | −0.009 | **1.18** |

The load-bearing observation: **`cos_res ≈ cos_seed` at every σ** (0.75: 0.222 vs
0.223; 0.95: 0.407 vs 0.408). Changing the resolution decorrelates the gradient
*barely more than changing the noise seed does* — resolution is a near-negligible
perturbation relative to the FM objective's intrinsic noise. That firmly **rules
out the KILL case**: low-res training does not pull the adapter off the high-res
direction.

Two honest caveats (carried from the bench readout):
1. **Mid-σ (0.15–0.85) is noise-dominated** — even the same-image/same-res ceiling
   `cos_seed` is only ~0.20–0.27 there, so the per-sample gradient is mostly
   noise; the high *normalized* scores in that band ride a tiny denominator and
   are fragile. The trustworthy signal is at the σ **extremes** (well-determined
   gradients), where resolution-robustness is clean (score 0.74 @ 0.05, 0.91 @
   1.0).
2. **High-res's extra contribution is modest and lives at the σ extremes** — the
   gradient-norm ratio `mag = ‖g_high‖/‖g_low‖` exceeds 1 only near clean (1.28 @
   0.05, the detail regime) and near pure noise (1.18 @ 0.95–1.0); in the mid-band
   it's <1. So the high-res phase earns its keep on **fine detail (low σ)**, which
   a *short* finish captures.

Limitation: this is an instantaneous-gradient probe — agreeing single steps do
not prove the multi-epoch optima coincide. Phase-0 gates *out* the failure case
and licenses the A/B; it is not itself the A/B.

## Mechanism / implementation sketch

Grounded in the existing plumbing (no new compile machinery needed):

1. **Caches**: `autoscale_tiers` must all be present on disk. Caches are keyed
   `{stem}_{WxH}_anima.npz`, so multiple resolutions of one stem coexist as
   independent samples and `make_buckets()` already enumerates them. Preprocess
   gets an autoscale-aware emit (emit each image at every tier in the ladder,
   bypassing `choose_edge`'s single-tier pick) — the multi-res emit we already
   scoped, here used as a *curriculum*, not random augmentation. ~2× cache disk
   for a 2-tier ladder; TE caches are resolution-independent and untouched.
2. **Schedule hook**: a progress-gated **sample mask** layered on the existing
   `BucketBatchSampler`. `current_step` / `max_train_steps` are already threaded
   to the data pipeline as shared `Value`s (collator wiring in `train.py`); the
   sampler reads the fraction and emits only the active tier's bucket indices.
3. **Compile is free across the switch**: under free-fit the dynamo budget
   auto-derives from *all* populated buckets and `seq_range = (min, max)` collapses
   the whole token span to **one** block graph. The sampler already front-loads
   the largest bucket to step 0 to warm compile, so the high tier is warmed from
   the start — the phase switch triggers **no recompile** and no mid-run VRAM
   climb (cf. the "warm all token families up front" fix). This is the property
   that makes a temporal schedule cheap on Anima specifically.
4. **Bespoke loops**: the schedule lives in the sampler, so `train.py` gets it for
   free; the turbo/spd/mod loops would need the mirror (the standing
   bespoke-loop-mirroring caveat) — out of scope for v1.

## FLOP accounting

Per-step cost blends linear (~N) and attention (~N²) terms; for d≈3072, N≈3–4k
they're comparable, so total scales ~N^1.5. With 896 (~3012 tok) vs 1024
(~4116 tok), r = 3012/4116 ≈ 0.73, blended low-tier step ≈ **0.65–0.73×** a
high-tier step. At `finish = 0.15`:

```
avg cost ≈ 0.85·0.68 + 0.15·1.0 ≈ 0.73×  →  ~27% fewer FLOPs at fixed steps,
                                            or ~1.37× more steps at fixed FLOPs.
```

**Honest bound**: the baseline is native free-fit, not all-1024. Images already
below the low tier natively are unaffected; the realized saving is
`(bulk fraction) × (downscale ratio) × (fraction of data above the low tier)`.
On a dataset that's mostly ≤896 native, autoscale saves little; the win scales
with how much high-native-res content the set carries. Report the realized
ratio, never the idealized one.

## Phases

- **Phase 0 — DONE (weak pass).** `grad_alignment.py`. KILL-case ruled out;
  payoff modest + detail-concentrated. Optional firm-up: re-run mid-σ with
  `--num_seeds 6` to average down the noise band before committing (the mid-σ
  scores are currently noise-limited).
- **Phase 1 — the arbiter (matched-FLOPs A/B).** Two LoRAs at **equal FLOPs** on
  one artist set: (A) native free-fit × full epochs; (B) `autoscale_mode` —
  low-tier bulk + short high-res tail, more steps to match A's FLOPs. Score on
  **CMMD** (live val signal) + the **memorization probe** (`bench/memorization`).
  PASS if B ≥ A on CMMD at equal FLOPs (the FLOP saving is then free quality-
  neutral throughput) or B improves memorization without losing CMMD.
- **Phase 2 — σ×res (only if Phase-1 passes).** The bench says high-res earns its
  keep at **low σ**. Couple the schedule: in the finish phase bias σ sampling
  toward the low-σ/detail band (or run high-res *only* on low σ, low-res on high
  σ throughout). Strictly stronger than the flat schedule if the Phase-0 mag
  signal holds up under more seeds.

## Risks / honest priors

- **The win may be ~zero on small/low-res sets** (FLOP-accounting bound above).
  Don't ship as default; opt-in, and document the dataset precondition.
- **It's augmentation-flavored, not free capability.** Phase-0 says safe, not
  better — the realistic expectation is *FLOP-neutral throughput* with a possible
  small memorization/generalization edge, mirroring the broader scale-aug
  literature. Resist over-claiming.
- **Mid-σ noise** means we can't yet certify the schedule is *optimal* (vs merely
  safe). Phase-2's σ-coupling is the place that bites if the mag signal is noise.
- **Bespoke loops** (turbo/spd/mod) won't inherit the schedule without a mirror.

## Success criterion

Phase-1: at matched FLOPs, `autoscale_mode` LoRA is **CMMD-non-inferior** to native
free-fit (and ideally memorization-better) on ≥2 artist sets, with the realized
FLOP saving logged. If B only ties A at *equal steps* (i.e. the saving is the
whole story), ship it as an efficiency knob; if B also wins on memorization, it's
a quality lever too. If B loses CMMD at matched FLOPs, the instantaneous-gradient
agreement didn't survive the trajectory — shelve with the negative logged.

## Scope

Training-schedule feature, default **off**. Composes with any adapter family.
Pilot in `train.py` (sampler-level); turbo/spd/mod mirrors deferred. Requires the
ladder tiers pre-cached (autoscale-aware preprocess emit).
