# paired_gram_eval — replace CMMD with paired per-item PE-Core token-Gram scoring

Status: **CLOSED 2026-07-12 — Phase 0 FAILED (gate 3), bench archived to
`_archive/bench/paired_eval/`; see the closing section at the bottom.**
Phase 0 run 2026-07-06: PGS does not clear the
known-difference gate: plain-r16 vs base reads TIE with base ≥ plain, reversing
the CMMD-vindicated ranking (CMMD at this CFG-4 set: plain 0.182 < base 0.279).
The whitened-Gram remedy did not fix it. Gates 1/4/5 pass, gate 2 is a
near-miss. Diagnosis: PGS is content/palette-dominated (~0.14 gap) and
style-weak (~0.025 artist gap). Full adjudication:
`bench/paired_eval/results/20260706-1354-phase0/VERDICT.md`. Pooled follow-up
(`pooled_gram_probe.py`) shows the failure is the **Gram feature, not the
pairing**: pooled *or* paired, token-Gram ranks base ≥ plain at CFG-4, while
mean-pooled **gist** (= CMMD's feature) ranks plain closer correctly. The
original probe dropped gist for being content-dominated, but that was *paired*
gist — the fix was **pooling** gist, not Gram. So the live options are **CSD**
(new style feature) or **keep CMMD's pooled gist and fix the convention**
(CFG-4, larger n), NOT the token-Gram. Bench code (`bench/paired_eval/phase0.py`,
`cross_artist_probe.py`, `whiten_probe.py`) and metric helpers
(`library/training/cmmd.py::gram_of` / `paired_gram_score`) landed.

- Planned bench: `bench/paired_eval/` (new; render phases lift from
  `bench/reft/eval_cmmd.py`, item/prompt machinery from
  `bench/memorization/generalize.py`)
- Premise sources: `bench/seed_floor/report.md` (CMMD at n=24/96 cannot rank
  checkpoints; median-σ refuted as fix), `bench/reft/report.md` verdict 3
  (CFG-1 CMMD misranks across adapter families), probe run 2026-07-06 on the
  saved CFG-4 renders (memory `project_paired_eval_probe`), and the standing
  metric-trap priors: FM-val-loss uninformative, eyeball-before-trusting
  (`docs/findings/spectral_fraction_metric_inverts.md`), real captions
  mandatory.

## Objective

Retire pooled CMMD as a decision metric. Replace it on both surfaces where it
currently decides things:

- **(A) Cross-model A/B gate** (bench harness): does adapter/recipe X beat Y?
- **(B) In-training validation signal** (`library/training/validation.py`):
  is this checkpoint better than the last one?

CMMD survives only as a shadow column until Phase 2 flips the default, then
deprecates.

## Why CMMD fails (condensed from the seed_floor audit)

1. **Sample size.** MMD² variance at 24 gens / 96 refs dwarfs the effect
   sizes we care about: a same-recipe reseed trio spans holdout-CMMD
   0.137–0.473 (2.2× the noise floor) while its renders are 0.95–0.97
   pairwise-cosine identical.
2. **Unpaired pooling.** The gen pool's dominant variance axis is *which
   prompts were drawn*, not which model rendered them (measured prompt:model
   variance ratio ≈ 2:1 on PE-Core gist). MMD mixes that prompt variance into
   every model comparison; the paired design subtracts it exactly.
3. **Estimator calibration.** `mmd_gaussian` is the biased V-statistic; its
   H0 offset scales with (1/m + 1/n), so the 48-vs-48 floor never matched the
   96-vs-24 measurements (25% offset mismatch), and the floor is a single
   split with no CI.
4. **Feature space.** σ=10 on unit-norm features puts every kernel value in
   [0.98, 1.0]; what survives is dominated by global tone (the CMMD "winner"
   matched the real pool's mean luma/sat almost exactly: 208.5/41.2 vs
   209.2/40.4). Near-floor values move ~20% under a PNG 8-bit round-trip.

## The metric

Per image, from PE-Core (`load_pe_encoder(name="pe")`, the encoder CMMD
already uses — sidecars stay valid):

```
tokens  = last_hidden_state[1:]          # [T, D] — drop CLS
tokens  = tokens - tokens.mean(0)        # center
G       = tokens.T @ tokens / T          # [D, D] second moment ("style Gram")
G       = G / ||G||_F                    # Frobenius-normalize, flatten
PGS(gen, real) = cos(G_gen, G_real)      # paired Gram similarity — PRIMARY
gist(gen, real) = cos(pool_and_normalize(gen), pool_and_normalize(real))
                                         # content-drift COVARIATE, not a score
```

Pairing: every generated image is scored against **its own real counterpart**
— the holdout image whose caption produced it (`Prompt.ref_path`, same stem,
same per-stem seed across all models). Model comparisons are then paired
statistics over per-stem differences, never pool-vs-pool distances.

Decision rule for the A/B gate (two models, n stems):

- **WIN**: Wilcoxon signed-rank p < 0.05 on per-stem PGS deltas **and** sign
  consistency ≥ 70%.
- **TIE** otherwise. No verdict may cite the mean delta alone.
- Montage gate stays mandatory (headroom arm-A lesson) — a WIN with a failed
  montage is a metric bug report, not a result.

Why Gram and not alternatives (probed 2026-07-06, n=4 SFW stems, existing
CFG-4 renders — see `project_paired_eval_probe`):

| candidate | verdict |
|---|---|
| PE-Core gist cosine | content-dominated — base beats adapters because adapters drift content; kept as covariate only |
| PE-Spatial (final layer) | **blind to the model axis** — deltas ±0.003, no sign consistency, prompt:model ratio 4.4×; encodes layout (seed-determined, identical across models) |
| PE-Core token-Gram | smallest content contamination of the three; resolved s1002 > s1003 at 4/4 sign consistency while correctly reporting the seed trio as near-identical |
| CSD | purpose-built style embedding but new weights + granularity unproven on same-community artists; raw cosine known uncalibrated (arXiv:2605.09030). Fallback if Gram fails Phase 0, not the default |

## Phase 0 — validation gates (offline, reuses existing renders; ~zero GPU)

All five must pass before any harness code ships:

1. **Aliveness**: right-pair vs wrong-pair PGS gap > 0 with margin (probe
   measured +0.086 on PE-Spatial; establish the PE-Core Gram number).
2. **Seed-trio null**: the uncondpoolft trio must read as statistically TIE
   (any WIN between same-recipe reseeds at n≈50 = metric fails; the tiny
   4/4-consistent s1002 edge should stay sub-threshold or vanish).
3. **Known-difference detection**: plain-r16 vs base must produce a verdict
   consistent with the eyeball read (plain visibly restyles).
4. **Quantization stability**: PGS moves < 2% under a PNG 8-bit round-trip
   (CMMD moved ~20%; this is the regression test for that failure).
5. **Tone-confound check**: synthetically shift render luma/sat (±10 luma) —
   PGS must move less than gist cosine does. If Gram inherits the tone
   confound, center per-channel *and* whiten before the Gram, re-gate; if it
   still fails, escalate to CSD.

## Phase 1 — bench harness (surface A)

`bench/paired_eval/run_eval.py`:

- Prompts: holdout items via the `generalize.py` membership replay; default
  n ≥ 48 stems (power for Wilcoxon at the probe's effect sizes), optional
  `--rating` filter on the caption's leading rating tag.
- Renders: 28 steps / CFG 4.0 (the trusted convention), paired seeds, eager,
  fresh model load per arm — all lifted from `eval_cmmd.py`.
- Output: per-stem table (PGS + gist per model), pairwise verdict matrix
  (WIN/TIE + p + sign consistency + bootstrap CI on the mean delta),
  montages, `result.json` envelope. CMMD reported as a shadow column for
  continuity, explicitly labeled non-decisional.
- Runtime: rendering dominates (~identical to `eval_cmmd.py` per arm);
  scoring is seconds. Gram memory: D=1024 → 4MB/image fp32, trivial.

## Phase 2 — in-training swap (surface B)

`_compute_cmmd_validation` in `library/training/validation.py` is **already
paired by construction** — gen order is preserved so `gen_pooled[i]` pairs
with `ref_pool[i]`, and the ref sidecars (`{stem}_anima_pe.safetensors`)
store raw `[T, D]` `image_features` (pooling happens at load). The swap is
local:

- Compute PGS per val item from `feats_list` (pre-pooling, already in hand at
  PHASE 3) vs the item's ref sidecar tokens; log `val_pgs_mean` (and keep the
  20-step/CFG-1 render convention — within-run tracking never crosses model
  families, so the CFG-1 misrank doesn't apply; pairing removes the prompt
  draw as a variance source that pooled CMMD paid for even in-run).
- **Shadow mode first**: log both `cmmd` and `pgs_mean` for ≥3 real training
  runs; flip the default (`use_cmmd` → `val_metric = pgs|cmmd`) only if PGS
  checkpoint-ranking agrees with rendered-sample eyeball at least as well as
  CMMD does.
- `library/training/cmmd.py` grows `gram_of(feats)` + `paired_gram_score()`
  next to `pool_and_normalize` — one home for both metrics during the shadow
  period.

## Non-goals / limits

- **Reference-free quality scoring.** PGS needs a per-item real counterpart;
  it measures style fidelity *to an artist's held-out work*, not absolute
  image quality. (No quality reward exists for Anima — standing guard.)
- **Cross-run absolute comparability.** PGS values depend on the stem set;
  compare only within a run's paired design. Never bank a raw PGS scalar the
  way CMMD scalars were banked (that habit is what seed_floor demoted).
- **Memorization detection** — unchanged, stays with `bench/memorization/`.

## Invariant tests (CONTRIBUTING Tier 1.5)

- Gram invariance: token permutation ⇒ identical PGS; global feature scale ⇒
  identical PGS (Frobenius normalization); T-mismatch between gen/ref buckets
  handled (Gram is [D,D], T-free by construction).
- Paired-vs-pooled separation on synthetic data: two "models" = same feature
  distribution + per-stem offsets ⇒ pooled MMD fails to separate at n=24
  while the paired test succeeds (regression-pins cause #2).
- Decision rule: seed-trio fixture ⇒ TIE; plain-vs-base fixture ⇒ WIN.

## Closing (2026-07-12) — line archived without pursuing CSD

Archived together with `seed_floor` and `seed_lottery` (the whole
eval-instrument cluster): `bench/paired_eval/` → `_archive/bench/paired_eval/`
(phase0 + cross_artist / whiten / pooled_gram / fm_holdout probes + results).
The Phase-0 verdict stands as the scientific output:

- **The token-Gram is the wrong feature** — content/palette-dominated (~0.14
  gap) vs style-weak (~0.025 artist gap), pooled *or* paired; it reverses the
  CMMD-vindicated plain<base ranking, and whitening doesn't fix it.
- **Pooled gist at CFG-4 (= CMMD at the right protocol) was right** on the one
  known difference — so the cheap protocol fix ("render at 28-step/CFG-4,
  raise n") remains the first thing to try if a ranking metric is ever needed
  again, before any new feature.
- **Holdout FM confidence is anti-correlated** with the vindicated ranking
  (Spearman −0.40) — not a covariate on the style/quality axis.
- `gram_of` / `paired_gram_score` stay in `library/training/cmmd.py` only
  because the archived probes import them; nothing live calls them.

**Why closed rather than escalated to CSD:** the consumer this instrument was
being built for — seed/recipe ranking for soup and banked single-run A/Bs —
lost its motivation. Soup's live axis moved to memorization
(`bench/memorization` — e.g. adapter-interference × memorization-amplification
correlation), which has its own working instruments (loss-gap MIA, xerox
probe) and doesn't need a paired style metric. **Reopening gate:** a concrete
decision that needs sub-noise-floor model ranking; the designated next step is
still CSD as the paired feature, re-run through these exact five gates.
