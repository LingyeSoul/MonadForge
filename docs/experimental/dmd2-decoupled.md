# Decoupled DMD2 (Turbo Anima) — few-step distillation

Distills the 28-step CFG=4 Anima teacher into a **4-step LoRA student** via Liu et
al.'s Decoupled-Hybrid DMD2 (arXiv:2511.22677, *"CFG Augmentation as the Spear,
Distribution Matching as the Shield"*, Table 1 row 4). The output is a **plain
standard LoRA** — there is no inference-side turbo code; you load it through the
normal LoRA path and run `--infer_steps 4 --cfg 1.0` (CFG is baked into the student
during distillation).

> **For the structural walkthrough** (the DMD-in-practice gradient decomposition,
> why CA is the engine and DM the shield, the velocity→x0 conversion and sign
> convention, the τ_CA > t schedule, the co-LoRA capacity argument, and the
> x0-norm vs τ-damping policy), see **`docs/structure/dmd2-decoupled.md`**. This
> doc is the usage / ops / decision-log reference.

- **Training:** `scripts/distill_turbo.py` — bespoke single-GPU loop (bypasses
  `train.py`/accelerate, like `distill-mod` / `distill-spd`).
- **Harness:** `networks/methods/turbo_dmd.py::TurboDMDNetwork` — two `LoRANetwork`
  stacks (student + fake) view-toggled on one frozen DiT.
- **Config:** `configs/methods/turbo.toml` — **bespoke sectioned schema** read only
  by the script. Don't `print-config METHOD=turbo` (the flat method+preset merge
  doesn't apply here). CLI flags override TOML values.
- **Improvement proposal / decision log:**
  `docs/proposal/dmd2_decoupled_improvements.md` (live diagnostics + validated levers).

## Quick start

```bash
make exp-turbo                                       # configs/methods/turbo.toml defaults
make exp-turbo ARGS="--student_rank 128 --iterations 5000"
make exp-turbo ARGS="--single_prompt_idx 0"          # Phase 0 single-prompt overfit
make exp-turbo PRESET=low_vram                        # grad ckpt + offload + sample_ratio

make exp-test-turbo                                   # infer latest student LoRA @ 4 steps, cfg=1.0
```

`make exp-turbo` honors `PRESET` (translates `blocks_to_swap` /
`gradient_checkpointing` / `sample_ratio` from `configs/presets.toml` into CLI
flags) and appends `ARGS` last so user overrides win. The output is
`output/ckpt/<output_name>.safetensors` — a normal LoRA — plus the standard
`.snapshot.toml`.

At inference the student LoRA loads through the existing LoRA adapter path; the
caller just sets 4 steps and CFG=1. It composes with concept LoRAs the way LCM-LoRA
composes with style LoRAs (linear LoRA composition, ranks add).

## How it works (one screen)

Three roles share one frozen DiT; the LoRA stacks toggle per forward via
`TurboDMDNetwork.set_view` (each `LoRAModule` short-circuits on `not self.enabled`,
so a view switch is an O(num_modules) flag flip, negligible vs a DiT forward):

```
teacher view  — both LoRA stacks off  → base velocity (CFG'd at α=teacher_cfg)
student view  — student on, fake off  → v_student, gives x_pred
fake   view   — fake on, student off  → v_fake_cond_dm (the score tracker)
```

Per training step (single-call DMD2 — **no inference-sampler unroll at train
time**; the gradient is one ODE step from the sampled generator-t):

1. **Student forward (grad):** `v_student = student(x_t, t, c)`, then the
   velocity→endpoint conversion `x_pred = x_t − t·v_student`.
2. **CA branch (no-grad, τ_CA > t):** two teacher forwards (cond + T5("") uncond)
   at a renoised `x_pred`; `Δ_cfg = v_real_cond − v_real_uncond`. This is the
   CFG-bake **engine**. Skipped when generator-`t` is very late (`> tau_ca_skip_above_t`).
3. **DM branch (no-grad teacher + no-grad fake, τ_DM ∈ [0,1]):** `Δ_dm =
   v_real_cond_dm − v_fake_cond_dm`. The distribution-matching **shield**.
4. **Assemble + backward into student.** α_eff ramps 1→`teacher_cfg` over
   `alpha_warmup_steps`. The grad signal in x0 space (per-branch τ factor, sign
   per the structure doc) is `grad_signal = τ_dm·Δ_dm + τ_ca·(α_eff−1)·Δ_cfg`,
   applied via the DMD2 grad trick `loss = (grad_signal · x_pred).mean()`.
5. **Fake update:** `fake_steps_per_student_step` plain flow-matching MSE steps on
   the student's `x_pred.detach()` distribution (resampling τ_fake, ε_fake each).

The teacher uncond is the **T5("") sidecar** (`library/inference/uncond.py`), *not*
a zero tensor — a zero crossattn is fed-out-of-distribution, and the resulting
`v_real_uncond_ca` amplified at (α−1)=3× drives the student off-manifold (saturated
white output). Staged by `make distill-prep` / `make preprocess-te`; shared with the
mod-guidance distill.

## Config surface (`configs/methods/turbo.toml`)

Sectioned, bespoke. Every key has a matching CLI override flag (see
`scripts/distill_turbo.py` argparse). The shipped defaults:

| Section | Key | Default | Notes |
|---|---|---|---|
| top | `output_name` | `anima_turbo_C` | output stem under `output/ckpt/` |
| top | `iterations` | `1000` | |
| top | `use_custom_down_autograd` | `true` | keeps activation memory down so swap can stay 0 (see [[project_custom_down_autograd_distill_lever]]) |
| top | `use_masked_loss` | `true` | **student-only** mask on the DMD2 grad; fake/critic stays full-frame |
| `[network]` | `student_rank` / `fake_rank` | `64` / `64` | `fake_rank ≥ student_rank` (fake is a score *tracker*, capacity ceiling on DM strength) |
| `[dmd]` | `student_steps` | `4` | sampler steps baked into the student |
| `[dmd]` | `teacher_cfg` (α) | `4` | matches Anima production CFG; baked in |
| `[dmd]` | `tau_ca_strategy` | `above_t` | τ_CA > t (only mode in v1) |
| `[dmd]` | `tau_dm_strategy` | `uniform` | τ_DM ∈ [0,1] (only mode in v1) |
| `[dmd]` | `tau_ca_min_gap` | `0.05` | clamp τ_CA ≥ t + gap (late-step stability) |
| `[dmd]` | `tau_ca_skip_above_t` | `0.95` | skip CA when generator-t exceeds this |
| `[dmd]` | `dm_x0_norm` | `true` | **DM-grad policy (b)** — DMD per-sample x0-norm; **ships** (see below) |
| `[dmd]` | `norm_floor` | `0.05` | clamp_min for the (b) denominator; inert under (a) |
| `[optim]` | `student_lr` / `fake_lr` | `2e-5` / `3e-5` | fake runs hotter |
| `[optim]` | `fake_steps_per_student_step` | `2` | keep the fake ahead of the moving x_pred |
| `[optim]` | `alpha_warmup_steps` | `500` | linear α_eff ramp 1→teacher_cfg |
| `[optim]` | `grad_clip` | `1.0` | grad-norm cap (both nets) |
| `[sampling]` | `t_distribution` | `uniform` | or `sigmoid` |

## Status (2026-05-28)

Phase 0 is effectively passing — the ~1k-step checkpoint produces coherent 4-step
samples (clean anatomy, legible "ANIMA" sign). Two landed correctness fixes and one
A/B settled the current default config:

- **Student-loss sign fix (2026-05-27).** The student gradient was inverted
  (anti-distill). Pre-fix logs read like "base 4-step blur / never trained," **not**
  a blow-up — that's the tell. See [[project_turbo_dmd_sign_fix]].
- **DM-grad policy: x0-norm (b) beats τ-damping (a)** — shipped as the default
  (`dm_x0_norm=true`). (a) collapsed to near-identical outputs across seeds (DMD
  mode-seeking); (b) restored per-prompt seed diversity and fixed text rendering.
  The τ-weighting that co-landed with the sign fix was **harmful, not inert**. Do
  **not** stack (b) on the τ-weight (that's policy "c" — just (b) with τ
  re-multiplied). See [[project_turbo_dmd_x0_norm_wins]] and `docs/proposal/dmd2_decoupled_improvements.md §2B`.
- **Post-8k read: more same-config steps are not the next lever.** An 8k run
  oscillated in the same `dm_rel_gap`/`dm_cos` band as the 2k run. The next lever
  is **fake time-scale** (`fake_steps_per_student_step=2`, `fake_lr` > `student_lr`
  — both already on), then **capacity** (rank 128), then **on-trajectory anchors**.

### Reading the metrics

Trigger fake interventions on **`dm_rel_gap` ↑ / `dm_cos` ↓**, *not* on
`fake_loss` ↑ (a rising fake loss against a moving, sharpening student is expected
equilibrium). The live triggers, all τ-weighted at the DM eval point:

| TB scalar | Read |
|---|---|
| `dm_rel_gap` | `rms(τ·Δ_dm)/rms(τ·v_real_dm)` — fraction of the teacher score the gap still is. ↑ = fake lagging. |
| `dm_cos` | `cos(v_fake_dm, v_real_dm)` — →1 healthy; ↓ = fake pointing the wrong way (worse than a magnitude miss). |
| `dm_mag_ratio` | `rms(v_fake)/rms(v_real)` — ≈1 healthy. |
| `dm_to_ca` | effective DM vs CA magnitude; DM ≳ CA for long stretches = red flag (CA must stay the engine). Logged only on `do_ca` steps. |
| `x_pred_std` / `v_student_rms` | collapse → 0 or runaway up = student exploding (`v_student_rms` leads). |

## Limitations & composition

- **Plain-LoRA bake is the hard constraint.** Anything needing a step-size or
  per-t input at inference (Shortcut / MeanFlow Δt-conditioning, timestep-conditioned
  T-LoRA — its mask is training-only, see [[project_tlora_inference_full_rank]])
  gives nothing after the bake. True multi-stride robustness is out of scope.
- **Spectrum:** incompatible by construction — Spectrum's Chebyshev cache assumes
  ≥16 steps. Don't stack.
- **DCW:** the v4 fusion head was calibrated on 28-step trajectories; a turbo
  student needs its own 4-step recalibration (out of scope for v1).
- **Mod guidance:** tunable — the distilled `pooled_text_proj` may still help, but a
  turbo student may have re-learned the modulation pathway implicitly. Test, don't
  assume.
- **Block swap must stay off** (`blocks_to_swap=0`) if on-trajectory anchors are
  ever wired in — the offloader desyncs on a 2nd DiT forward
  ([[project_blockswap_extra_forwards_gradcache]]). The CA+DM branches today are
  all no-grad single forwards, which is fine; `use_custom_down_autograd=true` keeps
  swap off in the default path.

## References

- Liu et al., arXiv:2511.22677 — Decoupled DMD2 ("Spear / Shield").
- `docs/structure/dmd2-decoupled.md` — structural walkthrough + math.
- `docs/proposal/dmd2_decoupled_improvements.md` — live diagnostics, validated levers,
  decision rules.
- `docs/findings/asymflow_parameterization.md` — Anima's `u = ε − x0` velocity path
  (the conversion the renoise/grad-assembly relies on).
