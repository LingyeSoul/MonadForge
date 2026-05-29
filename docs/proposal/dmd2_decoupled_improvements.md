# Decoupled DMD2 — improvement proposal (diagnostics + validated levers)

> Improvement proposal for the shipped Decoupled DMD2 (Turbo) distillation. For the
> method itself see `docs/experimental/dmd2-decoupled.md` (usage / ops) and
> `docs/structure/dmd2-decoupled.md` (math / walkthrough). This doc is the live
> decision log: which levers survived contact with Anima's constraints, and the
> metric-driven rules for picking the next one.

Distilled from the turbo-distill review exchange (`suggestions.md` → `response.md`),
keeping only the ideas that survived contact with Anima's actual constraints: the
output must bake to a **plain standard LoRA** run at `--infer_steps 4 --cfg 1.0`, we
train at batch=1 on a single 16 GB GPU at native-token buckets (so no full-BPTT), and
the student-loss **sign fix landed 2026-05-27** — every pre-fix log reads like "base
4-step blur," not a blow-up.

Current status: the 1k-step checkpoint already produces coherent 4-step samples
(`output/tests/20260527-112920-536_40_.png` — clean anatomy, legible "ANIMA" sign), so
Phase 0 is effectively passing. The visible stress signal is **fake under-tracking**,
not a broken objective. This doc is deliberately not phased — pick levers by eyeballing
the metrics below.

## 0. Post-8k read (2026-05-28)

The `anima_turbo_B_8k` run changes the decision rule: **more same-config steps are not
the next lever.** The run restarted from scratch with the same objective/config family
(`student_rank=64`, `fake_rank=64`, `fake_steps_per_student_step=1`, `dm_x0_norm=true`)
and its first 2k steps matched the 2k B run almost exactly. The 8k tail did not show
better fake tracking; it mostly oscillated in the same band:

| run/window | `dm_rel_gap` | `dm_cos` | `dm_to_ca` | `v_student_rms` | read |
|---|---:|---:|---:|---:|---|
| B 2k, steps 1501-2000 | 0.0935 | 0.9930 | 0.356 | 1.226 | baseline plateau |
| B_8k, steps 1501-2000 | 0.0956 | 0.9932 | 0.381 | 1.231 | same early behavior |
| B_8k, steps 7001-8000 | 0.1098 | 0.9914 | 0.406 | 1.304 | no useful late improvement |

So the current candidates should be read as:

1. **Fake time-scale first:** a short 2k run with `fake_steps_per_student_step = 2`.
   If wall-clock is acceptable, pair it with `fake_lr = 3e-5` while keeping
   `student_lr = 2e-5`. This directly tests the observed under-tracking without
   changing student capacity.
2. **Capacity second:** `student_rank = fake_rank = 128` only after the time-scale
   check, or sooner if samples clearly look capacity-limited rather than fake-lagged.
   Use `fake_rank = 2 * student_rank` only if `dm_rel_gap` stays high / `dm_cos` drops.
3. **Objective change third:** if fake tracking is healthy but 4-step samples still
   plateau, promote section C (detached on-trajectory anchors). That is the direct test
   for the single-call DMD2 vs 4-step inference-trajectory mismatch.
4. **Do not schedule a plain "8-step training" run yet.** In the current script,
   `student_steps` is logged and saved as metadata, but it does not change the training
   objective. It only becomes a real training knob after anchors or schedule-jitter make
   the target sampler trajectory part of the data distribution.

## 1. Metrics (shipped)

The logger was trimmed and refocused. `student_loss` (a sign-random gradient vehicle,
~noise), `t_mean` (≈0.5 by construction), the raw `v_real_dm_rms`/`v_fake_dm_rms`
(subsumed by the ratios), and the three raw `delta_dm_rms_tau_{lo,mid,hi}` buckets
(misleading now — they're **pre-τ-weight**, so they show a ~7× low-τ skew that the
τ-weighting in the fixed update already rebalances) were removed.

**Health (keep an eye on, but they're mostly "is it alive"):**

| metric | healthy | bad |
|---|---|---|
| `grad_signal_rms` | bounded, slowly rising | runaway → NaN risk |
| `delta_dm_rms` | bounded | — (context for the ratios) |
| `delta_cfg_rms` | bounded | — (context) |
| `x_pred_std` | ~0.6–0.9, steady | →0 collapse / drifting up = explode |
| `v_student_rms` | steady (~1.2) | rising fast = runaway (leads `x_pred_std`) |
| `alpha_eff` | ramps 1→`teacher_cfg` over warmup | — (schedule sanity) |
| `fake_loss` | may rise — **not a trigger by itself** | — |

**Fake-tracking diagnostics (the real triggers — added):** all evaluated at the DM
eval point, τ-weighted to reflect the *effective* gradient, not the raw velocity gap.

| metric | definition | read |
|---|---|---|
| `dm_rel_gap` | `rms(τ·Δ_dm) / rms(τ·v_real_dm)` | **the fake-lag trigger.** ↑ over a window → fake is falling behind → bump fake. |
| `dm_mag_ratio` | `rms(v_fake_dm) / rms(v_real_dm)` | ≈1 healthy; collapsing toward 0 or exploding = fake mis-scaled. |
| `dm_cos` | `cosine(v_fake_dm, v_real_dm)` | →1 healthy; dropping = fake pointing the wrong way (worse than a magnitude miss). |
| `dm_to_ca` | `rms(τ_dm·Δ_dm) / rms(τ_ca·(α−1)·Δ_cfg)` | Decoupled DMD guard: CA is the *engine*, DM the *shield*. DM ≳ CA for long stretches = red flag. Logged only on `do_ca` steps. |

Decision rule (from the exchange): **trigger fake interventions on `dm_rel_gap` ↑ /
`dm_cos` ↓, not on `fake_loss` ↑.** A rising `fake_loss` against a moving, sharpening
student is expected equilibrium behavior.

## 2. Validated levers (loosely ordered, cheapest/safest first)

### A. Fake time-scale, then capacity
The observed failure is fake-lag, and the cheapest fix is **time-scale, not capacity**:

1. `fake_steps_per_student_step = 2` (already a first-class knob).
2. `fake_lr = 1.5–2× student_lr` — today both are `2e-5`, so this is an untried lever.
3. `fake_rank = 2× student_rank` — **only** if `dm_rel_gap` stays high after 1–2. Rank
   is capacity; reach for it last. (Note proposal R1 already says fake rank ≥ student.)

### B. DMD per-sample normalization — in x0 space ✅ VALIDATED 2026-05-28, (b) SHIPS
**Outcome:** the (a)-vs-(b) A/B below was run (knob `dm_x0_norm`, now defaulting **true**
in `configs/methods/turbo.toml`). **(b) won decisively on samples** — (a) collapsed to
near-identical outputs across seeds (DMD mode-seeking), while (b) restored per-prompt seed
diversity, and (b) clearly improved **text rendering**. Mechanism: (a)'s τ-damp × *raw*
magnitude over-weights high-`|v_real|` samples (high-frequency structure, text, off-mode
tails) and over-pulls them to the dominant mode; (b)'s per-sample normalization gives every
sample a unit-scale direction → even distribution-matching pressure → preserves the tails
(diversity) and fine structure (text). The effective-LR confound (b runs ~2× the DM-grad
magnitude at fixed `student_lr`) is *ruled out as the cause*: more DM pressure would mean
*more* mode-seeking, so the diversity gain happened despite the magnitude bump → the
scale-invariance is the driver. Corollary: the τ-weighting that co-landed with the sign fix
was **harmful, not inert** — (b) supersedes it. The per-step health scalars (`x_pred_std`,
`v_student_rms`, `dm_cos` dip count) all looked *calmer* under (b) and read like a possible
softening risk, but they're blind to between-seed diversity — the multi-seed 4-step sample
check was the only decisive signal. The original A/B framing is preserved below for context.

Original DMD normalizes the DM gradient per-sample for scale-invariance. The clean
transfer to Anima's velocity path (`u = ε − x0`, confirmed in
`docs/findings/asymflow_parameterization.md`):

```python
x0_real = x_renoised_dm - tau_dm_e * v_real_cond_dm
denom   = (x0_real - x_renoised_dm).abs().mean(dim=(1,2,3), keepdim=True)  # = τ·mean|v_real|
denom   = denom.clamp_min(norm_floor)        # norm_floor ≈ 0.05 in latent scale
grad_dm = (tau_dm_e * delta_dm) / denom
```

**Critical caveat — do not blindly stack this on the τ-weight.** Because
`denom ≈ τ·mean|v_real|` and `mean|v_real| ≈ 0.9` here, the τ cancels across the whole
bulk of the range: `(τ·Δ_dm)/(τ·0.9) = Δ_dm/0.9`. The `clamp_min(0.05)` floor only bites
for `τ < ~0.056`, a sliver. So "τ-weight **+** x0-norm" ≈ "no τ-weight, just
magnitude-normalize" — the normalization largely *reverts* the τ-weighting that the sign
fix introduced. These are three different policies, not additive:

- **(a)** τ-damping only (the pre-2026-05-28 default; lost seed diversity + blurred text)
- **(b)** DMD magnitude-normalization (τ roughly cancels) — **shipped default**
- **(c)** both ≈ (b)

The **(a) vs (b) A/B is settled (see banner above): (b) ships.** `dm_rel_gap`/`dm_cos` are
policy-invariant diagnostics (computed from the raw teacher↔fake gap) and stay near-identical
across the two — the decisive signal was multi-seed 4-step sample quality, not the scalars.
Still **do not ship (c)** believing it composes — it's just (b) with the τ re-multiplied in.

### C. Detached on-trajectory anchors — low cadence, sampled-index rollout
Single-anchor training vs 4-step inference is a real DMD2 mismatch, but anchors are a
*trajectory-distribution fix*, **not** a general quality booster — only worth the
wall-clock if 4-step eval plateaus or the trajectory degrades over its 4 steps while
single-step looks fine. Keep it cheap:

```text
sample k ∈ {0,1,2,3}
with inference_mode: roll the current student from noise to schedule state x_tk   # ~1.5 fwd avg
train one normal DMD update from (x_tk, t_k):
    CA: τ_CA > t_k   (anchor the NOISY inference state, not clean x_pred; keep the >t rule)
    DM: τ_DM uniform
```

Start `anchor_prob = 0.05`, raise to 0.10–0.20 only if drift remains. The sampled-index
rollout (~1.5 extra no-grad forwards on average) keeps this inside the 16 GB envelope —
no full-BPTT. **Verify `blocks_to_swap = 0` first**: the block-swap offloader desyncs on
a 2nd DiT forward; turbo runs `use_custom_down_autograd=true` to keep swap off, so this
is expected-safe but worth confirming before wiring rollout forwards.

### D. Consistency auxiliary — only if trajectory drift persists
Low-weight (0.01–0.05) output-space consistency after CA warmup. Redundant with C, so
defer behind it. Anima caveat: an EMA-LoRA teacher was inert for a *representation* loss
on the frozen backbone (`docs/findings/selfflow.md`); this is output-space MSE so that
doesn't kill it, but keep the weight low and don't lean on the EMA teacher.

### E. (Optional) schedule-jitter for solver robustness
If we want the baked LoRA less brittle to small solver/timestep perturbations: keep the
deployed schedule fixed at 4-step/cfg=1 but jitter the 4 anchor times slightly during
training. It will **not** become "2/4/8-step robust" — a plain LoRA must average
antagonistic per-`t` corrections. True multi-stride robustness needs Shortcut/MeanFlow
Δt-conditioning, which **cannot survive a plain-LoRA bake** (see skip list). If multi-
stride is ever a hard requirement, the only honest test is: train separate 2/4/8-step
LoRAs, measure delta-weight SVD/subspace overlap, and soup *only* if the deltas align.

## 3. Explicitly skipped (with reasons)

- **Share `τ_CA = τ_DM`** to save a teacher forward — breaks the load-bearing `τ_CA > t`
  schedule (the whole calibration thesis). 25% no-grad saving not worth it.
- **Timestep-conditioned student (T-LoRA / per-step scale)** — T-LoRA's mask is
  training-only; inference is full-rank at every t, so it gives *nothing* after the
  plain-LoRA bake (see `project_tlora_inference_full_rank`).
- **Shortcut / MeanFlow Δt-conditioning** — needs a step-size input at inference; doesn't
  survive the plain-LoRA export unless schedule-locked (→ lever E).
- **Adversarial / LADD feature discriminator for v1** — out of v1 scope, and the LADD
  "frozen generative features" variant has an Anima headwind: feature-distillation off the
  finished frozen denoiser exerts ~zero transferable pressure on a rank-r adapter
  (`docs/findings/selfflow.md`).
- **Lowering fake cadence / fake rank** (suggestion's cheap fake options) — backwards
  relative to the observed under-tracking; the lever points *up*, see A.
