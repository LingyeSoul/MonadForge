# Item 2 (FEI band-deficit CA reweighting): why it was plausible, why it falsified

Item 2 (`item2_plan.md`, `proposal.md` §2) reweights the CFG-uplift `δ_cfg`
in turbo distillation by *where the student's FEI lags the teacher's*:

```python
δ_cfg' = w_low · LP(δ_cfg) + w_high · HP(δ_cfg)
w_high = 1 + β · relu(e_high_T − e_high_S)
w_low  = 1 + β · relu(e_low_T  − e_low_S)
```

Phase 0 said **GO**. The live training run said **the validated lever isn't
there** — the active arm is the inverse one, at ~identity strength. This note
records the gap between the two so we don't re-wire it the same way.

## Why it was plausible (the case for GO)

1. **A real failure mode.** The 4-step turbo student over-blurs. FEI
   (`e_low + e_high ≡ 1`, a partition of unit-norm latent energy) is the
   natural axis to describe "too much low-frequency, not enough high."
2. **A structurally safe loss.** Because the two bands sum to 1, the two
   `relu`s can never both be positive — at most one weight is amplified, so
   total CA magnitude is bounded by `1 + β·max(Δ)`. Unlike the prior direct
   FEI-statistic matching attempts (which destroyed image quality, see
   `project_fera_probe_2band_decision`), this is *feedback control on
   `δ_cfg`*, not a target on the image.
3. **Phase 0 measured a clean lever.** Probe on `anima_turbo_C` (n=90 = 30
   artists × 3 seeds, `bench/fera_artist/results/20260528-1902-turbo_C_phase0/`)
   found, at the student's 4-step rollout stages:

   | stage | t_S | d=16 SNR | sign% | Δ_low |
   |---|---|---|---|---|
   | 1 | 0.90 | 1.51 | 92% | −0.0011 |
   | 2 | 0.75 | **2.25** | **100%** | −0.0118 |
   | 3 | 0.50 | 1.81 | 94% | −0.0493 |

   Direction `student_over_low` everywhere (student's trajectory latent
   carries *more* LF than the teacher's at the same t) → the `w_high` arm
   should fire, `w_low` inert. Divisor D/16 dominated D/8 and D/4. All four
   decision-rule thresholds cleared. A textbook GO.

## How it was wired (Phase 1)

`scripts/distill_turbo/ca_band.py` + call site `distill.py:296–330`.
`[ca_band_weight] enabled=true, beta=0.2, divisor=16, window=[0.30, 0.95]`.
Trained into `anima_turbo_E_1k` and `anima_turbo_F`.

A diagnosis pass during wiring caught one σ-mismatch — measuring the gap at
τ_ca (the teacher's denoise of the renoised x_pred) injects a ~0.08 LP shift
that masks the ≈0.012 lever and inverts the TB sign. The fix: one extra
no-grad teacher forward at the student's `(x_t, t, c)` so both x0 estimates
live at the same σ:

```python
x0_T = x_t − t · v_teacher_at_t      # extra no-grad forward
x0_S = x_pred = x_t − t · v_student  # already computed
e_T, e_S = FEI(x0_T), FEI(x0_S)
```

## Why it falsified (the live run inverts Phase 0)

Live band scalars on `anima_turbo_F` (`output/logs/turbo/20260528-212337`;
the same pattern holds in `20260528-194036` → turbo_E_1k):

| scalar | head → tail | max | reading |
|---|---|---|---|
| `band_w_high` | 1.000 → 1.000 | 1.001 | **the Phase-0 arm is dead** |
| `band_w_low`  | 1.005 → 1.011 | 1.038 | the arm that fires |
| `band_dh_pos` | ~0 | 0.006 | HF deficit ≈ never positive |
| `band_dl_pos` | 0.027 → 0.054 | 0.188 | **LF** deficit is the live signal |

The active arm is `w_low`, the exact inverse of Phase 0's `w_high`, and it
reweights δ_cfg by ≤4% (β=0.2 × Δ≤0.19) — the loss is very nearly a no-op.

### Root cause: Phase 0 and the loss measure different things

Two mismatches, either of which alone breaks the transfer:

1. **`x_t` distribution.** Phase 0 probed the student's own **4-step rollout
   states** — where the *integrated* output is over-blurred (`student_over_low`
   → `w_high`). The loss measures at `x_t = (1−t)·x₀ + t·ε` (`distill.py:284`),
   a DMD2 renoise of **real clean data**. At a single renoised point the
   student's one-shot x0 estimate is *under*-denoised / noisier than the
   teacher's denoise of the same point → `student_under_low` → `w_low`. The
   two distributions carry opposite-sign gaps.

2. **Measured quantity.** Phase 0 computed FEI on the **trajectory latent
   `z_t`** itself (the curve where `e_low` rises 0→0.57 along the rollout).
   `ca_band.py` computes FEI on **x0 estimates** (`x_pred`, `x_t − t·v_teacher`).
   The σ-match fix correctly removed the τ_ca operator mismatch, but in doing
   so it pinned the measurement to x0-estimates-at-t — which Phase 0 never
   validated.

**The deeper reason:** the over-blur failure lives on the student's *own
inference trajectory*, and DMD2 single-call training never visits those
states. At the renoised-data points training does visit, the only FEI gap is
the near-trivial "student's one-shot x0 is noisier than the teacher's," which
fires `w_low` weakly — and boosting `LP(δ_cfg)` (coarse CFG bake) is, if
anything, mildly *antagonistic* to the over-blur it was meant to fix.
`band_dl_pos` even *grows* over training (0.027→0.054), i.e. the student is
trending toward more HF and `w_low` is pushing coarse structure back in.

This is also why **bumping β is the wrong response**: it amplifies an
unvalidated arm in a plausibly-counterproductive direction. The fix is to
move the measurement, not to turn up the gain.

## What survives

- **The mechanism is sound, just mis-sited.** The LP/HP DoG split + bounded
  per-sample deficit (`e_low+e_high≡1` ⇒ at most one arm fires) is correct and
  reusable. `ca_band.py` is kept; `enabled` is flipped to `false` (β=0 is
  bit-identical to off, and it reclaims the ~14% wall-clock of the no-op
  teacher forward).
- **The right way to revive it** is to measure the band deficit *on the
  student's rollout states*, which only exist if training visits them — i.e.
  merge item 2 with item 4 (front-loaded on-trajectory anchors). At anchor
  states the student genuinely over-blurs, so `w_high` would fire where the
  failure actually is. Item 5 (curation) first would also clean the
  artifact-chasing component out of the gap signal before re-measuring.
- **Phase 0 wasn't wrong** about turbo_C's rollout — it was right about a
  distribution the loss never touches. The lesson: a go/no-go probe must
  measure the *same quantity at the same distribution* the loss will see.

## Pointers

- Memory: `project_turbo_fei_gap_phase0` (now annotated with this inversion).
- `item2_plan.md` §"the w_high branch is the active arm in practice" — marked
  falsified.
- Context: `docs/findings/sigma_signal_where_anima_resolves.md` (the σ-window
  motivation), `project_turbo_dmd_x0_norm_wins` (the seed-diversity / over-blur
  history), `project_fera_probe_2band_decision` (why direct FEI matching fails).
