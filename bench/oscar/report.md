# OSCAR on Anima — Phase-0 results

**Verdict: NO.** On Anima, pushing a same-prompt seed-set apart in **bare-latent**
space buys ~none of the *semantic* (PE-feature) diversity that the decode+PE push
delivers. The VAE decode + PE tower is load-bearing — there is no free latent
shortcut. This matches OSCAR's own thesis (latent/Particle-Guidance repulsion
gives low-level/texture spread, not semantic mode coverage); it holds on Anima.

OSCAR (arXiv 2510.09060v2) descends a log-det set-volume energy
`E(Z) = −½ log det(I + τ ZZᵀ)` on endpoint features `Z = φ(x̂0)` to spread a
batch of trajectories, projecting the push ⊥ to `v_θ`. Phase 0 asked the cost
question: can `φ` be the **identity on the bare latent** (no decode, no tower),
or must it be the faithful **PE-Core** image tower?

## Run

`probe_feature_space.py` — self-contained batched Euler rollout (bare DiT, no
LoRA); at probe σ's it recomputes the volume-energy push in {latent, PE} and
compares.

| | |
|---|---|
| run | `results/20260601-2040-phase0-report/` |
| resolution / set size | 768², m=6 seeds × 2 prompts |
| CFG / flow_shift / steps | 1.0 / 3.0 / 28 |
| τ (volume energy) | 1.0 |
| endpoint | Euler one-step `x̂0 = x_t − σ·v` |
| PE feature | PE-Core-L14-336, pooled + L2 (fp32) |

## Metrics

Two reads. **Validity** (is each push a real spread direction *in its own
space*): `lat_cen_cos` = cos(latent push, latent-centroid offset);
`feat_push_cos` = cos(−∇E, PE-feature-centroid offset). **Transfer** (the
decision, space-agnostic): step the set along each push and measure the change
in PE-space volume energy `ΔE_pe` (negative = more semantic spread);
`transfer = ΔE_pe(latent) / ΔE_pe(PE)` is the fraction of the PE push's
diversity gain the *free* latent push recovers.

| σ | feat_push | lat_cen | ΔE_pe (PE push) | ΔE_pe (latent push) | transfer | band |
|------:|----------:|--------:|----------------:|--------------------:|---------:|:--------|
| 0.844 | 0.92 | 0.69 | −1.12e-01 | −1.51e-02 | 0.13 | early |
| 0.660 | 0.90 | 0.68 | −1.31e-01 | −1.82e-02 | 0.14 | early |
| 0.450 | 0.86 | 0.67 | −1.17e-01 | −9.34e-03 | 0.07 | forming |
| 0.333 | 0.87 | 0.66 | −9.80e-02 | −2.91e-03 | 0.01 | forming |
| 0.188 | 0.89 | 0.66 | −1.79e-01 | −1.70e-02 | 0.07 | forming |

Band aggregates (forming = σ≤0.45): **transfer = +0.07**, feat_push = 0.88,
lat_cen = 0.66, cos⊥v ≈ 0.0. `frac_par_v` ≈ 0.10 (latent) / 0.00 (PE) — the ⊥v
safeguard removes almost none of either push.

## Reading

- **Both pushes are valid.** `feat_push ≈ 0.88` (PE energy gradient genuinely
  points off the PE-feature centroid) and `lat_cen ≈ 0.66` (latent push points
  off the latent centroid). The PE pullback delivers end-to-end:
  `ΔE_pe(PE) ≈ −0.12 < 0` at every σ — the decode+PE push really does spread the
  semantic features.
- **The latent push does not transfer.** Stepping along the free latent push
  barely moves PE-space volume energy (`ΔE_pe(latent)` ~10× smaller, and ~0 in
  the mid-forming band). It recovers only **~7%** of the PE push's semantic
  diversity gain.
- **cos⊥v ≈ 0 is a real geometric finding here, not a degenerate read.** Because
  both gradients are independently valid, their near-orthogonality means the
  latent-space spread direction and the semantic spread direction are essentially
  unrelated in the σ-window where x0 forms — exactly why transfer is near zero.

## Implication for later phases

If OSCAR is built on Anima, `φ` must be **PE-Core with per-step VAE decode**
(the priced path) — the bare-latent shortcut is off the table for *semantic*
set-diversity. The decode+PE per-step cost (and the VJP back through both) is the
real budget item to weigh against the diversity gain before Phase 1.

Not yet probed: CFG=4 (this run is CFG=1; DCW work shows Anima's sampler-boundary
behavior is CFG×aspect-dependent), non-square aspects, and the paper's Heun
endpoint (we used a one-step Euler `x̂0`).
