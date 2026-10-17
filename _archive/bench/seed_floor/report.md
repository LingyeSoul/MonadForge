# Seed-lottery floor probe → CMMD is too fragile to measure it at n=24/96

**Run:** 2026-07-06 · **Verdict: the between-seed "floor" we set out to measure is
inseparable from CMMD sampling noise at the eval size we use. The median-heuristic
does NOT fix it — the fragility is sample-size, not bandwidth. Any seed / recipe
ranking from paired-holdout CMMD at 24 prompts / 96 refs is untrustworthy.**

## What we set out to do

Deliver the `docs/proposal/seed_lottery_noise_floor.md` idea cheaply: instead of
building a harness, *extract* the between-seed CoV floor from checkpoints we
already have. The uncond-soup bench trained a seed trio from one shared uncond
init on identical data (`bench_sincos_half_uncondpoolft{,_s1002,_s1003}`), which
is a ready-made between-seed sample. Score the trio + their soup with the reft
paired-holdout CMMD harness (`bench/reft/eval_cmmd.py`) and read the spread.

The number was supposed to gate our banked single-run A/Bs ("is this gap real or
inside the floor?"). Instead the probe surfaced that the *metric itself* can't
resolve differences at this scale.

## The trail (each step overturned the previous read)

### 1. cfg-1 said seed s1002 was a catastrophic bad draw

First pass reused the existing `bench/memorization/generalize_soup3` numbers
(20 steps, **CFG 1.0** — the trainer's CMMD convention). holdout-CMMD:

| seed | cfg-1 holdout ↓ |
|---|---|
| s1001 | 0.4545 |
| **s1002** | **1.0214** ← "the bad draw" |
| s1003 | 0.4859 |
| soup-of-3 | 0.5300 |

CoV 39.8%. Story: a reseed 2.2×'d the metric; soup rescued the disaster. This
matched the banked `project_uncond_soup_bench` narrative.

### 2. cfg-4 flipped the ranking — s1002 went from worst to best

The reft finding's own GOTCHA is that **cfg-1 CMMD misranks** (documented
cross-family; `docs/findings`/`bench/reft/report.md`). Re-ran the trio at the
trusted protocol — **28 steps / CFG 4.0** (`bench/reft/results/20260706-1210-seed_floor_cfg4`):

| model | cfg-4 holdout ↓ | (cfg-1 was) |
|---|---|---|
| base | 0.2793 | 0.5932 |
| s1001 | 0.3521 | 0.4545 |
| **s1002** | **0.1370** ← now BEST | 1.0214 (worst) |
| s1003 | 0.4730 | 0.4859 |
| soup_r48 | 0.2980 | 0.5300 |
| plain (ref) | 0.1816 | 0.9612 |

s1002 went **worst → best**. So the "1.02 disaster draw" was largely a cfg-1
protocol artifact — and the within-family misrank shows cfg-1 reshuffles even
same-recipe seeds, not just across adapter families. CoV was still ~43% at cfg-4,
but s1002's cfg-4 holdout (0.137) landed **below the real-vs-real noise floor
(0.155)** — the first tell that we were reading noise.

### 3. A rank confound in the soup comparison

`bench_sincos_half_uncondpoolsoup3` reports `network_dim=16` in its snapshot but
its stored `lora_down` is **rank 48** (3×16, un-truncated) — the SVD-truncated
deployable is the separate `..._r16` (rank 16). So "soup vs ingredients" compared
an r48 soup against r16 ingredients. Two distinct trios also exist:

- **r16**: `bench_sincos_half_uncondpoolft{,_s1002,_s1003}` (the memorization-bench
  reduced-rank set — what we scored).
- **r32**: `anima_soup_sincos_s1001/2/3` (the shipped `make soup` recipe;
  `anima_sincos` is the r32 production LoRA — NOT scored here).

The trio CoV is still clean (all three ingredients are r16); only the soup
number carries the rank confound.

### 4. The metric can't resolve any of it — and median-σ doesn't help

`library/training/cmmd.py` uses PE-Core (CLIP-L/14) features, **L2-normalized to
unit vectors**, then a Gaussian-kernel MMD² with the paper constant **σ=10**,
×1000 scale. On unit vectors pairwise squared distances are ≤4, so
`gamma=1/(2·10²)=0.005` puts every kernel value in **0.98–1.00** — the flat,
least-discriminative regime. σ=10 was tuned for *raw* CLIP features, not
unit-normalized ones.

Natural fix to try: **median-heuristic bandwidth** (σ = median pairwise distance
of the real holdout pool = **0.678**, model-independent so the kernel stays fixed
across comparisons). Recomputed from the already-saved render PNGs — no re-render
(`bench/seed_floor/median_sigma_recompute.py`):

| model | σ=10 h/floor | median-σ (0.678) h/floor |
|---|---|---|
| s1002 | 1.08 | 1.12 |
| plain | 1.18 | 1.13 |
| base | 2.14 | 1.99 |
| soup_r48 | 2.39 | 2.16 |
| s1001 | 2.57 | 2.26 |
| s1003 | 3.39 | 2.87 |

**The ranking is identical at both bandwidths, and the floor-ratios barely move.**
On L2-normalized features MMD ordering is essentially bandwidth-invariant (every
kernel is a monotone function of the same pairwise distances). The fragility is
therefore **not** a bandwidth problem — it is signal-vs-noise at n=24/96:
**s1002 and plain sit ~1.1× the noise floor**, i.e. statistically
indistinguishable from a real-vs-real split. No bandwidth can manufacture
resolution the sample size doesn't contain.

(Aside: the σ=10 recompute values differ slightly from the stored run because the
saved PNGs go through an 8-bit round-trip before re-encoding — the noise floor
matches to the 4th decimal since refs load from the same originals, so the delta
is entirely the gen-side quantization. A minor fragility datum of its own.)

## What this means

- **The seed lottery may be real (spread 0.17→0.53 is large) but is inseparable
  from metric noise here.** Several models — plain, s1002, and the floor itself —
  cluster within ~1.1× of each other. We cannot say which seed "won."
- **Prior banked reads are demoted:** the cfg-1 "1.02 bad draw", the cfg-4 rank
  flip, and "soup ≈ best ingredient / insures the tail" (`project_uncond_soup_bench`)
  are all downstream of a metric that can't rank these checkpoints at this size.
  With plain included, cfg-4 even puts plain LoRA ≈ best seed < base < soup < the
  other two seeds — a muddy, near-floor ordering, not a clean signal.
- **The reft GOTCHA generalizes:** cfg-1 CMMD misranks *within* an adapter family
  across seeds, not only across families.

## What a real measurement needs

CMMD at 24 prompts / 96 refs is a noise generator for effect sizes this small.
To actually measure the seed floor:

1. **Far more samples** — prompts/refs in the hundreds, so the MMD variance drops
   below the between-seed spread; or
2. **A per-item paired signal** — same prompt + same seed across two models, PE
   cosine (or LPIPS) per pair, then a paired test. Paired differencing cancels
   the per-prompt variance that dominates the unpaired MMD.

Until then, do **not** gate A/Bs on a single paired-holdout CMMD number at this
size, and treat any sub-2×-floor gap as unresolved.

## Reproduction

- cfg-4 render + score: `bench/reft/results/20260706-1210-seed_floor_cfg4/`
  (`eval_cmmd.py --adapters <trio+soup> --membership_adapter bench_sincos_half_plain
  --steps 28 --cfg 4.0 --num_prompts 24 --num_refs 96`).
- plain cfg-4 (same protocol/seeds/refs — floor & base match to 4 decimals):
  `bench/reft/results/20260705-2242-reft_phase1p_cfg4/`.
- median-σ recompute from saved PNGs: `bench/seed_floor/median_sigma_recompute.py`.
