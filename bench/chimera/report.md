# ChimeraHydra Phase-0 probes — findings & future actions

Two questions, prompted by comparison with MedQwen ("Sparse Spectral LoRA: Routed
Experts for Medical VLMs", Nejati Manzari et al., CVPR'26), plus the pre-existing
expert-capacity probe. All three live under `bench/chimera/` (CPU/GPU, drop the
standard `bench/_common` envelope). Run: `python bench/chimera/run_bench.py <probe>`.

| Probe | Question | Status |
|---|---|---|
| `expert_capacity` | Do the frozen-Cayley levers deepen experts without freeing bases? | settled (prior) |
| `full_spectrum` | Does MedQwen full-spectrum SVD seeding have headroom over our top-slice? | **INCONCLUSIVE** |
| `content_specificity` | Does the shared-A content half learn content-specific reps? | **answered** |

---

## 1. content_specificity — ANSWERED, has an actionable win

**Finding.** The shared-A content half *does* encode content (routing NMI z=26 vs
shuffle) but it is **low-amplitude** (per-prompt gate std 0.004) and **over-provisioned**:
the K_c=6 pool collapses to an effectively binary live gate (experts 2 & 3), **3 of 6
experts dead**, mean_gate `[0.002, 0.057, 0.238, 0.237, 0.235, 0.232]`. Experts are
genuinely distinct (cross-expert cos ≈ 0), so routing *can* matter — it's just barely
driven. Verdict: **content-LOCKED but LOW-AMPLITUDE**.

(Probe: `anima_chimera-0616`, 400 real cached captions from TE caches, grouped by artist.)

### Future actions
1. **[easy win] Drop `num_experts_content` 6 → 2–3** in `configs/methods/chimera.toml`.
   3 experts are dead weight; the live router is already binary. Near-zero risk, smaller
   checkpoint. → re-run `content_specificity` on the retrained ckpt to confirm
   utilization recovers (0 dead experts at K_c=3).
2. **[if the content half should matter more] Amplify, don't re-discriminate.** It already
   tracks content; it's just soft. Levers: raise `network_content_router_lr_scale`
   (currently 10), or investigate whether `balance_w_content`/centered-gate is flattening
   it toward uniform. CMMD is the only judge ([[project_cmmd_val_signal]]); FM val is
   uninformative.
3. **[closes a doc open question]** This is the offline half of the doc's "is the content
   pool redundant?" (C-fei) — it's alive but weak. A C-fei training A/B (feed FEI into the
   content router) would confirm whether the thin content signal beats a freq-only pool.

---

## 2. full_spectrum — INCONCLUSIVE, needs a sharper Part B before any decision

**Part A (mechanism, analytic — SOLID).** Chimera's frozen-Cayley adapter is caged:
`colspace(ΔW) ⊆ span(P_bases)` (the r×r rotation is a no-op on an r-column span). Analytic
best-capture `‖PᵀTQ‖²/‖T‖²` confirms top-slice captures an easy top band (1.00) but **0.00
of a deep band**, while full-spectrum (bands relocated across the spectrum) captures the
deep band (**1.00**). The lever works — it unlocks directions top-slice structurally
cannot reach.

**Part B (where uncaged ΔW lands — CONFOUNDED).** Fraction of a free LoRA's ΔW energy
beyond the top slice (left `(K_c+K_f)·r=256` of U; right `2r=64` of V). The answer depends
entirely on the proxy:

| Proxy | rank / regime | beyond top-slice (left / right) | reading |
|---|---|---|---|
| `anima_channel`, freefit_quarter (`use_ortho_init`) | 32, top-seeded | 0.02 / 0.11 | **invalid** — seeded *into* the top slice |
| `anima_sincos` (plain, converged) | 48, full data | 0.04 / 0.21 | concentrated in W's top |
| `anima_byg` (plain) | 64 | 0.81 / 0.95 | ≈ random baseline |
| `anima_repa_freefit_tenth` (plain) | 16, 1/10 data | 0.81 / 0.97 | ≈ random baseline |

Random-orientation baseline: `(2048−256)/2048 = 0.875` (left), `(2048−64)/2048 = 0.969`
(right). The high-beyond proxies sit *at* that baseline → their ΔW is uncorrelated with
W's spectrum (under-concentrated: low rank + low data), not evidence of a deep-band
preference.

**Two unresolved confounds — why Part B can't gate the decision yet:**
1. **Convergence/rank.** Undertrained ≈ random ≈ "beyond top"; converged (sincos) → top.
   The metric partly measures *how much training happened*, not *where it wants to go*.
2. **Diffuse ≠ deep.** "Beyond top-slice" does **not** imply full-spectrum captures it.
   Full-spectrum is 8 narrow strided strips; it only helps if ΔW concentrates in a *few
   deep bands*. A diffuse ΔW (the random-baseline case) is missed by the strips as much as
   by the top — that needs more **rank**, not relocated bands.

### Future actions
1. **[required before any verdict] Sharpen Part B with a concentration metric.** Of the
   beyond-top energy, measure how much sits *in the strided spectrum bands* vs spread
   uniformly across the tail. Only band-concentrated energy is full-spectrum-capturable.
   This separates "needs rank" from "needs full-spectrum" — the actual decision.
2. **[required] Control the proxy.** Measure on a **converged, full-data, plain** (random-
   init, no ortho/ortho_init) LoRA that targets the same Linears chimera does, ideally
   trained in the **freefit+repa regime** chimera uses. Report against the random baseline,
   not against an absolute 0.3 threshold (the current bench verdict over-claims "HEADROOM").
3. **[only if 1+2 show band-concentrated deep energy] Training A/B.** Add a
   `chimera_full_spectrum` seeding flag (frozen `P_bases`/`Q_basis` carved from strided
   non-overlapping bands across the full spectrum instead of the top slice; `_seed_bands`
   in `full_spectrum.py` already has the exact index construction). A/B vs top-slice on
   CMMD. Note this is orthogonal to — and may compose with — `use_ortho_init` (free basis)
   and `chimera_expert_basis_mult` (over-complete top slice).

### Do NOT
- Ship full-spectrum on the current Part B (proxy-dependent, metric conflates
  diffuse/deep). The bench's printed "HEADROOM" verdict is not yet trustworthy.
- Use an `use_ortho_init` checkpoint as the "free training" proxy — it's top-32-seeded and
  structurally biased into the top slice.

---

## Bench provenance
- `bench/chimera/results/*-content_specificity/` — K_c=6 utilization + NMI.
- `bench/chimera/results/*-full_spectrum*/` — Part A capture + Part B per-proxy.
- Memory: [[project_chimera_content_half_weak_overprovisioned]],
  [[project_chimera_full_spectrum_proxy_dependent]].
- Related: [[project_orthoinit_variant]], [[project_chimera_expert_capacity_levers]].
