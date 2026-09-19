# Merge-interference behavioral probe — the Δ_member gauge for LoRA merges

Status: **DRAFT 2026-07-12 — premise validated the same day
(`memorization_uncond_gap.md` Phase 0.5: the statistic reproduces the June
visual merge verdict from loss-side numbers alone). Nothing built yet;
Phase 0 (calibration) gates Phase 1 (ship).**

Ships a *behavioral* merge check next to the shipped *weight-geometry* check:
`scripts/merge_loras.py --analyze` (`library/anima/merge_analysis.py`)
predicts interference from pairwise ΔW cosines and subspace-overlap bands,
but nothing measures what a merge actually *does* to each ingredient's fit.
Phase 0.5 showed that measurement exists, is cheap, and is enormous when the
merge is bad.

## Premise sources

- **Phase 0.5 ground-truth ladder** (2026-07-12, caption-Δ, artist1
  perspective, `results/20260712-19*`): member Δ solo **+0.011** → norm
  merge **−0.008** (healthy, mild dilution) → raw 3-way sum **−0.157**
  (overdriven — the merged model fits its own artists *worse than base*,
  15× the solo spread). Matches the June visual verdict cell for cell.
- **Memorization inheritance** (same campaign, uncond-Δ): rw_measure ⊕ ama2
  raw sum keeps the overfit ingredient's member signal ~intact (σ0.5
  0.777→0.733, σ0.7 0.735→0.714, p 1e-4) — merges don't launder
  memorization; a merged artifact inherits its worst ingredient
  (consistent with soup AUC ≈ max, `bench/memorization/report.md`).
- **June Phase 0** (`project_artist_lora_merge_interference_phase0`,
  `bench/lora_merge_interference/`): artist ΔWs near-orthogonal
  (|cos| 0.06–0.08); raw-sum failure is *magnitude overdrive* (√N energy),
  fixed by norm control — now `merge_loras.py --normalize global` (default).
  Residual worry the geometry can't settle: similar artists (p90 |cos|
  0.13–0.20) and "weight-cosine ≠ perceptual collision".
- **Probe mechanics**: `bench/memorization/loss_gap.py` — paired
  base-vs-adapted confidence, membership replay from `.snapshot.toml`;
  Phase 0.5 established the **snapshot-sidecar trick** (copy an
  ingredient's snapshot beside the merged file → probe the merge *from that
  ingredient's perspective*) and that per-σ values are bit-comparable
  across runs with different σ sets (same per-item seeded noise).
- **G3b caveat** (uncond-gap Phase 0): scalar/mean statistics are the
  trustworthy output; per-item nomination still needs a render check.

## The statistic

Per ingredient `i` with a replayable membership (snapshot):

    interference_i = Δ_member_i(merged) − Δ_member_i(solo_i)

caption-Δ mode (interference is caption-coupled style fit — uncond-Δ drops
it by design), σ grid {0.5, 0.7}, K=4 antithetic probes, **mean over
members** (the mean, not AUC, is the interference readout; the instance-gate
AUC rides along when `sample_ratio < 1`). Secondary columns:

- `spread_i` = member − cross-artist Δ (dilution shows as spread
  compression at healthy absolute levels);
- optional `--uncond` arm: member-vs-holdout uncond-Δ AUC per ingredient =
  memorization inheritance ("which ingredient's xerox risk did this merge
  keep").

Provisional bands from Phase 0.5 (to be calibrated in Phase 0): ≈0 healthy ·
mildly negative + compressed spread = dilution · ≲ −0.1 = overdriven/broken.

## Phase 0 — calibration bench (blocks the ship)

~8–10 probe runs ≈ 1.5 h GPU on existing checkpoints, one orchestrator in
`bench/lora_merge_interference/` (it owns the merge-bench envelope already):

1. **Noise band**: re-probe one solo twice with different `--seed` item
   draws → paired spread of Δ_member. The band must sit ≤ ~0.01 for the
   dilution readout to be resolvable (overdrive at −0.1+ is safe either way).
2. **N-scaling**: 2-way raw + norm merges (artist1 ⊕ artist2) next to the
   existing 3-way results — raw overdrive should scale ~√N; norm should stay
   in band at both N.
3. **The information cell (G3)**: an *overlapping-artist* norm-controlled
   merge — e.g. ama2 ⊕ artist1 (artist1 contains ama_mitsuki) — where
   Tier-1 energy is fine but directional collision is plausible. Probed
   from both perspectives, cross-referenced against
   `merge_analysis.analyze` bands on the same inputs.

Gates:

- **G1 (monotone)**: solo ≈ norm > raw separation reproduces at every N.
- **G2 (resolution)**: noise band from (1) small enough to resolve
  dilution-tier effects, not just catastrophe.
- **G3 (added value)**: at least one grid cell where the behavioral verdict
  *quantifies or contradicts* what the Tier-1 bands say — i.e. the GPU probe
  carries information the free geometry check doesn't.

## Phase 1 — ship (only if Phase 0 passes)

- **Promote the probe core out of `bench/`** (the `build_anima` precedent):
  `confidence()` + item scoring + membership replay from
  `bench/memorization/loss_gap.py` into a `library/` home, with
  `loss_gap.py` re-importing (bench scripts stay thin entry points per the
  tooling layering contract).
- **`scripts/check_merge.py`** (thin shell) + `make check-merge`:
  - Input: the merged file. Perspectives auto-resolve from the
    `merged_from` metadata `merge_loras.py` already writes (extend it to
    record full ingredient *paths*, so snapshots are found without user
    input; explicit `--ingredients` overrides).
  - **Shared-base scheduling**: one base pass over the union of all
    perspectives' item sets, one pass per solo, one merged pass over the
    union — `N+2` model passes instead of the manual `4N`. A 3-ingredient
    check lands ~15–20 min (Phase 0.5 full runs were 7–9 min each).
  - Output: per-ingredient table (interference, spread, band verdict,
    optional uncond inheritance AUC) + result envelope + an
    `ANALYZE_RESULT`-style JSON marker line (same convention as
    `merge_loras.py`) so the GUI/daemon can consume it; the report embeds
    the Tier-1 `merge_analysis` summary for the same inputs so both tiers
    read side by side.
  - Unit test: statistic + perspective/snapshot resolution on synthetic
    factors (CONTRIBUTING Tier 1.5: this bench + that test).
- Docs: a section in the merge workflow docs; cross-link from
  `memorization_uncond_gap.md` Phase 0.5.

## Phase 2 (deferred candidates — not scoped here)

- Soup-pipeline auto-audit: soup ingredients share one membership, so the
  perspective table collapses; the useful soup column is the **uncond
  inheritance** arm (fold into `make soup` as an opt-in post-step).
- Snapshot-less ingredients (downloaded LoRAs): no membership → no Δ_member.
  A style-fit-only fallback over user-supplied reference images is
  plausible but unvalidated — open question, deliberately out of scope.

## Cost

Phase 0 ≈ 1.5 h GPU, no training. Shipped check: ~15–20 min for a 3-LoRA
merge (opt-in; the free Tier-1 analyze remains the always-on default).
A `--fast` preset (fewer members/probes) is only offered if Phase 0's noise
band shows it still resolves the dilution tier.

## Non-goals

- **Not a merge fixer** — norm control ships in `merge_loras.py`; the
  content-router escape from the strength↔noise curve is its own line.
- **Not per-item xerox nomination** (G3b: render/`probe.py` stays the
  conviction step).
- **No new metric surface elsewhere** — CMMD and the training-time
  `_memgap.json` tracker are untouched.

## Kill criteria

- **G2 fails** (noise band swallows everything below catastrophe) → the
  Tier-1 energy summary already catches catastrophe for free; don't build
  Phase 1; bank the band numbers in the June bench's README.
- **G3 fails** (Tier-1 bands + norm ratio predict every behavioral verdict
  in the grid) → the GPU probe adds nothing over shipped geometry; document
  the equivalence in the merge docs and close.
- Phase 1's shared-base scheduling turns out to change per-item values vs
  the validated per-run probe (it must not — same seeded noise per stem) →
  ship the naive 4N version instead; correctness over wall-clock.
