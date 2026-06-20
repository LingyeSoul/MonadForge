# ChimeraHydra Phase-0 probes — findings & future actions

Two questions, prompted by comparison with MedQwen ("Sparse Spectral LoRA: Routed
Experts for Medical VLMs", Nejati Manzari et al., CVPR'26), plus the pre-existing
expert-capacity probe. All three live under `bench/chimera/` (CPU/GPU, drop the
standard `bench/_common` envelope). Run: `python bench/chimera/run_bench.py <probe>`.

| Probe | Question | Status |
|---|---|---|
| `expert_capacity` | Do the frozen-Cayley levers deepen experts without freeing bases? | settled (prior) |
| `full_spectrum` | Does MedQwen full-spectrum SVD seeding have headroom over our top-slice? | **answered — NO (don't ship)** |
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

## 2. full_spectrum — ANSWERED (Part B sharpened): do NOT ship full-spectrum

**Part A (mechanism, analytic — SOLID).** Chimera's frozen-Cayley adapter is caged:
`colspace(ΔW) ⊆ span(P_bases)` (the r×r rotation is a no-op on an r-column span). Analytic
best-capture `‖PᵀTQ‖²/‖T‖²` confirms top-slice captures an easy top band (1.00) but **0.00
of a deep band**, while full-spectrum (bands relocated across the spectrum) captures the
deep band (**1.00**). The lever works — it unlocks directions top-slice structurally
cannot reach.

**Part B — SHARPENED (2026-06-18).** The old metric (one-sided "fraction of ΔW energy
beyond the top slice") was confounded: it rewards *any* tail energy, including energy
spread uniformly across the tail that no narrow strided band can ever capture. Replaced
with the only decision-relevant quantity — the **two-sided analytic cage capture** of the
*real* ΔW (Part A's `‖PᵀΔW Q‖²/‖ΔW‖²`, now with the trained ΔW as the target) under three
cages: `top` (chimera today), `spectrum` (MedQwen relocation), and a same-width **random
null**. Plus a per-axis **band-concentration enrichment** (`in-band beyond-top frac ÷ band
coverage of the tail`; ≫1 concentrated, ≈1 diffuse). The verdict is now **relative** —
full-spectrum earns the A/B only if it beats *both* the top-slice and the random null *and*
the tail energy is band-concentrated.

Two valid proxies, one consistent answer — **neither supports full-spectrum**:

| Proxy | regime | cap_top / cap_spectrum / cap_random | enrichment L/R | reading |
|---|---|---|---|---|
| `anima_sincos` | plain, converged, full data | **0.755** / 0.425 / 0.003 | 2.0 / 0.01 | top-slice captures 75% of ΔW; full-spectrum **loses −0.34**. ΔW lives in W's top. |
| `anima_repa_freefit_tenth` | plain, freefit+repa, 1/10 data (undertrained) | 0.0016 / 0.0012 / **0.0010** | 1.02 / 0.99 | **all three cages ≈ random null**; enrichment ≈ 1.0 = uniformly diffuse → needs **rank**, not relocated bands. |

The `freefit_tenth` row is the proof the sharpening works: under the **old** metric it read
0.81/0.97 "beyond top-slice" and the bench printed **"HEADROOM — ship the A/B"**. The
sharpened metric exposes that as a pure **diffuseness artifact** — top, spectrum, *and*
random cages all reconstruct ~0.1% of ΔW, and enrichment is exactly 1.0. That is confound
#2 ("diffuse ≠ deep") caught in the act: the undertrained ΔW is spread uniformly, so no
narrow band set helps; only more rank would. (`use_ortho_init` proxies — `anima_channel` /
`freefit_quarter` — remain structurally invalid here: top-32-seeded.)

**Conclusion.** Full-spectrum seeding has **no demonstrated headroom** on any valid proxy.
Where free training concentrates (converged sincos), it lands in W's **top** — exactly
where chimera already seeds — and the top-slice cage dominates full-spectrum by 33 points.
Where the metric *looked* promising (undertrained), the energy is diffuse and uncapturable
by *any* band relocation. Don't build `chimera_full_spectrum`.

### Future actions
1. **[optional, only path that could revive it] A converged full-data freefit+repa plain
   LoRA.** `sincos` is converged+full-data but not the freefit+repa regime; `freefit_tenth`
   is the right regime but undertrained. A converged freefit+repa plain LoRA would be the
   ideal single proxy. Given sincos already shows top-slice dominance and the analytic Part
   A shows the only directions full-spectrum adds are deep bands free training avoids, this
   is low-priority confirmation, not a live decision.
2. **[shelved] Training A/B.** `_seed_bands` in `full_spectrum.py` still carries the exact
   strided-band index construction if a future proxy ever shows band-concentrated deep
   energy (enrichment ≫ 1 *and* cap_spectrum > cap_top). Not warranted on current evidence.

### Do NOT
- Build / ship `chimera_full_spectrum` — sharpened Part B shows top-slice dominates on the
  converged proxy and the "headroom" on undertrained proxies is diffuseness, not deep bands.
- Trust the one-sided "beyond top-slice" marginal — it conflates diffuse with deep. Use the
  two-sided cage capture vs the random null (now the bench default).
- Use an `use_ortho_init` checkpoint as the "free training" proxy — it's top-32-seeded and
  structurally biased into the top slice.

---

## Bench provenance
- `bench/chimera/results/*-content_specificity/` — K_c=6 utilization + NMI.
- `bench/chimera/results/*-full_spectrum*/` — Part A capture + Part B per-proxy.
- `bench/chimera/results/*-sharp_*/` — sharpened Part B (two-sided cage capture + band
  enrichment): `sharp_sincos` (top dominates), `sharp_repa_freefit_tenth` (diffuse null).
- Memory: [[project_chimera_content_half_weak_overprovisioned]],
  [[project_chimera_full_spectrum_proxy_dependent]].
- Related: [[project_orthoinit_variant]], [[project_chimera_expert_capacity_levers]].
