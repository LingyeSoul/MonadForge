# OSCAR on Anima — bench plan

OSCAR (arXiv 2510.09060v2, "Letting Trajectories Spread") is a training-free,
inference-time **set-level diversity** control for flow matching. For a batch of
seeds sharing one prompt it: (1) predicts each trajectory's endpoint x̂0, (2)
encodes endpoints with a feature tower φ, (3) descends a log-det **volume**
energy `E(Z) = −½ log det(I + τ ZZᵀ)` to push the set apart, and (4) projects
both the diversity push and an injected stochastic noise **orthogonal to the
base velocity v_θ** so diversity never fights quality. Baselines: CADS, Particle
Guidance, DiverseFlow (DPP), APG.

It's the FM-native diversity method (its backbone FM-SD3.5 is the closest public
analog to Anima: DiT + rectified flow + CFG). Anima has no diversity axis in its
training-free stack (DCW/SMC-CFG/CNS/SPD/Spectrum are quality/speed/bias).

## The load-bearing cost question (Phase 0)

OSCAR's φ is "a pretrained image tower." Faithful φ = PE-Core, but PE eats RGB,
so the endpoint must be `VAE.decode(x̂0) → RGB → PE` **every step, whole batch**,
and the gradient pulls back through both PE *and* the decoder (VJP). That decode
is the real per-step cost.

The free alternative is to run the volume energy directly on the **bare latent**
x̂0 — no extra model, no decode. OSCAR doesn't do this on purpose: latent-space
repulsion (their Particle-Guidance baseline) tends to give low-level/texture
spread + artifacts, not semantic mode diversity. But Anima's latent geometry may
differ, and our PE manifold is collapsed (PR=6.2, mean-centering is the lever —
see `project_pe_feature_diagnostics`).

**Phase-0 probe = `probe_feature_space.py`:** can the *free* bare-latent push
substitute for the decode+PE push, in the σ-window where x0 is forming (≲0.45,
see `project_sigma_signal_resolves_by_045`)?

Cosine between the two gradients is a weak read on its own — they live in
different spaces (147k-dim raw latent vs a pooled D-vector), so disagreement is
expected even when both are valid. The decisive metric is a **finite-difference
transfer test**: step the latent set along each push and measure ΔE_pe (the
change in PE-space volume energy). `transfer = ΔE_pe(latent)/ΔE_pe(PE)` is how
much of the PE push's diversity gain the free latent push recovers.

Before trusting transfer, two validity gates (each push in its *own* space):
`lat_cen_cos` (latent push points off the latent centroid) and `feat_push_cos`
(the PE energy gradient points off the PE-feature centroid); the PE pullback is
validated end-to-end by ΔE_pe(PE) < 0.

**Precision is load-bearing.** The PE path (VAE decode → PE ViT → VJP back
through both) runs in **fp32** — a bf16 backward through that depth turns the
pullback into quantization noise (every cosine ≈ 0). That bf16 asymmetry (latent
path fp32, PE path bf16) was the original bug that made the bench nonsensical.

Read (forming band):
- transfer ≥0.50 → latent ≈ PE for diversity → **skip PE + per-step decode** (big win).
- transfer <0.20 → semantic diversity needs decode+PE → price it before wiring.
- feat_push_cos<0.1 (COLLAPSED) / ΔE_pe(PE)≥0 (PULLBACK-BROKEN) → fix the probe, don't read transfer.
- High frac-parallel-to-v in the latent push → ⊥v safeguard nukes it → space unusable regardless.

OSCAR uses a **centered** Gram (matches our PE mean-centering finding); the
volume energy's +I trace-stabilizer (Eq. 3) plays that role here.

## Later phases (only if Phase 0 says build)
- Phase 1: wire a `--diverse` set-level sampler mode (batch = N seeds / prompt),
  endpoint-volume gradient (latent or PE per Phase-0), orthogonal-to-v projection
  + σ-gated stochastic noise. Compose with CFG.
- Phase 2: Vendi / PRD / mode-coverage vs CADS/PG/APG; CMMD/CLIP to confirm
  quality held. Reuse PE-Core (already loaded for CMMD val).
