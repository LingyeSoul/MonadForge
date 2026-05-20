# SPD integration — staged plan & go/no-go gates

Spectral Progressive Diffusion (Xiao et al., arXiv:2605.18736): grow spatial
resolution along the denoising trajectory, running early noise-dominated steps
at low resolution and injecting HF detail via spectral noise expansion only when
finer frequencies emerge. Training-free claim up to ~7×/2.5× (image/video).

This plan is **gated** — each phase is cheap relative to the next and can kill
the idea before we pay for integration. Don't skip ahead; the later phases are
worthless if an earlier gate fails.

---

## Phase 0 — Spectral statistics precondition ✅ DONE (2026-05-20)

Does Eq.(4) `P_ω ∝ |ω|^{-β}`, β∈[2,3] hold for Anima VAE latents?

- `measure_latent_spectrum.py`: **β = 2.26, R² = 0.9994** (200 imgs).
- `per_artist_spectrum.py`: **β = 2.26 ± 0.08**, 30/30 artists in [2,3].

**Verdict: PASS.** Premise #1 holds dataset-wide. → proceed to Phase 1/2.

---

## Phase 1 — Autoregression *dynamics* probe (Fig 2b)  ✅ DONE (2026-05-21) — WEAK

The static spectrum says HF *carries less signal*; it does NOT prove HF stays
*noise-dominated until late* in Anima's actual trajectory. That dynamic is what
justifies running early steps at low res.

**Built** `bench/spd/measure_autoregression.py` (smoke-validated; full run pending):
- **Half A (eyeball, no δ):** runs N full-res generations, captures `x_t` and the
  running clean estimate `x0_pred = x_t − σ·v` per step (own Euler loop — no hook
  needed since the loop owns `x_t`), per-channel-standardizes, radial-bins into K
  bands, plots **resolved fraction `R_band(σ) = P(σ)/P(σ→0)`** per band. Reports each
  band's `σ_half` and monotonicity. Autoregression ⇔ low bands hit R≈1 at high σ,
  high bands stay flat until small σ.
- **Half B (δ schedule):** measures `P_ω` from real latents (ortho-FFT on
  unit-variance latents so the paper's `x0^(ω)~N(0,P_ω)`, `ε^(ω)~N(0,1)` holds),
  runs Prop 1 `t_ω = 1/(1+sqrt(δ/(P_ω(1+P_ω−δ))))` → Prop 2 `t*_i = t_ω(k=s_i)`,
  maps onto the real 28-step σ schedule, reports token/attn-FLOP speedup for
  S∈{2,3}. Derived `t*_i` overlaid as vertical lines on the Half-A plot.

### Result (2026-05-21, 4 prompts × 2 seeds @ 1024², 28 steps; P_ω from 96 real imgs)

**Half A — autoregression CONFIRMED (PASS).** `σ_resolve` (the σ below which a band's
power stays ≥80% of final) is cleanly monotone in frequency:

| band k | 0.08 | 0.25 | 0.42 | 0.58 | 0.75 | 0.92 |
|--------|------|------|------|------|------|------|
| σ_resolve | 1.00 | 0.75 | 0.54 | 0.39 | 0.32 | 0.29 |

spread 0.71. Low frequencies lock in by σ≈0.75, the top band only by σ≈0.29 — exactly
the Fig-2b picture. (Metric note: a *first-crossing* σ_half is fooled by the first ~2
steps, where the one-shot CFG-amplified `x0_pred` from near-pure-noise spikes high bands
spuriously then falls back — the original run mis-verdicted FAIL on that artifact. The
shipped metric uses last-crossing, immune to it. Lowest band reads >1 early — `x0_pred`
DC over-estimate, ignore.)

**Half B — δ=0.01 schedule is CONSERVATIVE (the WEAK half).** Real `P_ω` is a clean
power law; `t_ω` spans only ~0.78→1.0 across all k, so the principled transitions land
high: `t*(s=0.5)=0.87`, `t*(s=0.75)=0.81`. Mapped onto 28 steps that's **S2 attn ×1.15,
S3 attn ×1.22** — far below the **×1.65** the Phase-2/3 *hand-tuned* `σ≈0.5` knee gets.

**Net: WEAK.** Autoregression is real (so early-low-res is justified in principle), but
the δ=0.01 δ-optimal schedule barely saves compute — the speedup lives in the *aggressive*
hand-tuned regime, which δ=0.01 does not sanction. This is the headroom the δ-sweep and
the fine-tune are aimed at.

### Next actions (curves are in → WEAK)

1. **δ-sweep — the immediate next experiment.** Half A confirms autoregression, so the
   open question is purely *how aggressive a schedule the spectrum sanctions*. Sweep
   `--delta 0.01 0.03 0.05 0.1` (one-line loop in `measure_autoregression.py`, or N
   invocations) and read `t*(s=0.5)` for each. Find the δ whose `t*` lands near the
   hand-tuned σ≈0.5 knee. Two outcomes, both informative:
   - a *moderate* δ reaches σ≈0.5 → the bench's hand-tuned knee is **principled**, and we
     get a single-knob schedule derivation for free (drop the hand-tuning).
   - only an *aggressive* δ (loose error bound) reaches σ≈0.5 → the hand-tuned knee is
     **over-aggressive** vs the paper's guarantee; this quantifies the quality risk that
     the ×1.65 training-free knee was already taking.
2. **Then decide the track:**
   - If a sane δ recovers a real saving (say best attn ≥×1.4): proceed to **Phase 3**
     training-free integration using the δ-derived schedule.
   - If even a swept δ stays thin (~×1.2, the likely outcome given `t_ω`'s narrow 0.78→1.0
     range): the training-free δ-schedule's speedup is mostly theoretical on Anima. Ship
     the hand-tuned Phase-2/3 schedule as-is and **stop deriving it** — *or* pivot to the
     fine-tune (`docs/proposal/spd_finetune_lora.md`), where a trained model can hold a
     more aggressive schedule than training-free tolerates. WEAK is exactly the regime
     where Case B earns its keep (see link below).
   - **FAIL** (no clean frequency-ordered resolution): early-low-res is unjustified;
     drop SPD, lean on Spectrum/Turbo. (Smoke suggests this won't happen.)

### Connection to the SPD fine-tuning LoRA (`docs/proposal/spd_finetune_lora.md`)

Phase 1 is the missing input that proposal explicitly defers ("v0 hand-sets the
schedule; the δ-optimal derived schedule requires the autoregression-dynamics probe,
not yet built"). Three concrete links:

- **Phase 1 picks the schedule the Case-B LoRA trains on.** Instead of hand-setting
  the `0.5→1.0 @ σ0.5` knee, the fine-tune can train on the δ-optimal `t*_i` derived
  here — a principled, single-knob (δ) schedule.
- **WEAK is Case B's whole reason to exist.** The gap Half B measures — between the
  conservative δ the *training-free* model tolerates (~×1.2) and the aggressive σ≈0.5
  knee — is precisely the headroom the fine-tuned LoRA aims to recover: having learned
  the multi-resolution trajectory, it should tolerate a lower transition σ (larger
  effective δ) without the handoff divergence that gates training-free. Phase 1
  **quantifies the prize** Case B is chasing; if that gap is ~0, Case B isn't worth it.
- **The probe is also Case B's verification tool.** Re-run `measure_autoregression.py`
  on the trained LoRA (add a `--lora_weight` arm) and the Half-A band curves should
  shift — HF bands resolving *earlier* at the handoff is the direct evidence the LoRA
  closed the train–inference gap.

---

## Phase 2 — Resolution-generalization smoke test  ✅ DONE (2026-05-20) — PASS

SPD training-free assumes the bare DiT can denoise a *lower-resolution* latent
for the early steps. Anima trains only at the ~4096-token bucket and the
inference loop pins one static shape (`library/inference/generation.py` reads
`h_latent/w_latent` once + builds `padding_mask` once; `library/datasets/buckets.py`
exists specifically to avoid recompiles). If the model produces mush at low res,
training-free SPD is **dead** and only the fine-tuning recipe (Phase 4) survives.

**Built** `bench/spd/probe_lowres_denoise.py` (throwaway, no pipeline changes):
loads the **bare** DiT (no LoRA) eager/dynamic-shape (no torch.compile), encodes
one prompt, and runs from the same seed noise a full-res Euler baseline vs an
SPD variant — DCT low-pass init → low-res steps → spectral-noise-expansion
handoff (Eq. i–iii, σ-scaled HF noise) + timestep alignment (Eq. 5–6) → full-res
finish. The SPD math mirrors the community `SamplerSPEED` node
(`../comfy/custom_nodes/comfyui-speed/`), so `--community` reproduces the exact
`GJ5Rt3Xz` workflow schedule. Verdict is visual (saves baseline/spd/montage
PNGs); auto-metrics flag only hard divergence (NaN/Inf, latent-std blow-up,
sharpness collapse/grain).

### Result (2026-05-20, 2 seeds × 2 schedules @ 1024², CFG=4, 28 steps, flow_shift=1)

**COHERENT — Phase 2 PASSES.** The bare DiT denoises low-res latents and accepts
the spectral-expansion handoff with no instability across both schedules:

- single-stage `0.5→1.0 @ 30% of steps`: std ×0.95, sharpness ×1.83, no NaN.
- community `0.5→0.75→1.0 @ σ 0.8/0.6` (GJ5Rt3Xz): std ×0.95, sharpness ×1.71, no NaN.

Visual: all 4 montages (`results/20260520-2302-single-stage-0p5/`,
`results/20260520-2304-community-GJ5Rt3Xz/`) show coherent subjects, intact
anatomy, sky/clouds/trees, **no smear or double-image at the handoff**. SPD
output is *sharper/higher-contrast* than baseline and diverges compositionally
(expected — fresh HF noise injected at expansion). Note: the single-stage 2-stage
schedule rendered the "ANIMA" sign text more legibly than the community 3-stage
(which garbled it to "ANMA") — early evidence the community schedule is **viable
but not obviously optimal**.

**Verdict: PASS → Phase 3.** Training-free SPD is viable on Anima; the
resolution-generalization risk that could have killed it does not materialize.

> Recommendation: run Phase 2 first if forced to pick — it's the cheapest test
> that can actually falsify the whole approach.

---

## Phase 3 — Training-free SPD integration  [conditional on Phase 2 PASS]

Tier-2 method (see `CONTRIBUTING.md`). Engineering notes:
- **Static-shape / compile conflict.** Each resolution stage = a distinct token
  count = a `torch.compile` graph. S=2–3 → 2–3 cached shapes (survivable, but
  breaks the single-static-shape guarantee). Per-block compile mode amortizes.
- Lift the once-only `h_latent/w_latent`/`padding_mask` assumption in
  `generation.py` to a per-stage rebuild; add bucket entries for the low-res
  stages or compute them on the fly.
- New module `networks/spd.py`: spectral noise expansion (DCT default), the
  δ-optimal schedule from Phase 1's `P_ω`, timestep alignment. Sampler-level, so
  it should compose with LoRA / T-LoRA (token-count-agnostic per-Linear).
- CLI: `--spd --spd_delta 0.01 --spd_scales 2`. Wire into `inference.py`.
- **Composition study:** SPD (token-reduction) is orthogonal to Spectrum
  (block-skipping on cached steps) and Turbo (few-step). The real value question
  is SPD∘Spectrum and SPD-vs-Turbo at matched quality — not SPD vs a naive
  50-step baseline. Bench both.
- Bench: `bench/spd/bench_speed_quality.py` — speedup + ImageReward/CLIP-IQA/CMMD
  at S∈{2,3}, δ sweep, vs baseline / Spectrum / Spectrum∘SPD. Standard envelope.

## Phase 4 — Spectral fine-tuning recipe  → promoted to `docs/proposal/spd_finetune_lora.md`

Paper §4.3 / Eq. (11)-(14): fine-tune `v_θ` (as a plain LoRA) on stage-specific
straight-line targets so the model sees the multi-resolution trajectory. Now has
its own gated proposal — see **`docs/proposal/spd_finetune_lora.md`** (the "Case B"
trajectory adapter).

**Reframed since Phase 2 PASSED.** The original "only if Phase 2 MUSH" gate is moot:
the bare DiT does *not* mush at low res, so the fine-tune is no longer a rescue. Its
new job (per the proposal) is twofold — (a) close the handoff divergence training-free
SPD shows (sharper/recomposed output), and (b) tolerate a **more aggressive schedule**
than training-free can, recovering the speedup headroom Phase 1's δ analysis quantifies.
Unlike Turbo it needs **no teacher and no fake-score net** — analytic MSE targets, one
adapter — so it is structurally far more stable than the DMD2 pipeline.

**Dependency:** ships only after Phase 3 builds `networks/spd.py` (the sampler the
LoRA both trains against and is evaluated with). Greenlit as a proposal; not started.

---

## Bonus (independent of the speedup track)

Frequency-based **image editing** (paper §5.5): add noise to the low-freq band
of `T_Φ(x_in)`, spectral-noise-expand, resume from the schedule's timestep with
an edit prompt. Potential complement/competitor to DirectEdit. Could be probed
standalone even if the speedup track stalls — but lower priority.
