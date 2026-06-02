# DP-DMD (Turbo Anima) — diversity-preserved few-step distillation

Distills the CFG=4 Anima teacher into a **few-step LoRA student** via
**Diversity-Preserved Distribution Matching Distillation** (Wu, Li, Zhang, Ma —
arXiv:2602.03139). The output is a **plain standard LoRA** — there is no
inference-side turbo code; you load it through the normal LoRA path and run
`--infer_steps 2 --cfg 1.0` (CFG is baked into the student during distillation).

> **History.** This replaced the CA-decoupled DMD2 ("CFG-as-Spear, Distribution-
> Matching-as-Shield", Liu et al. arXiv:2511.22677) objective on **2026-05-30**.
> The whole turbo program had been spent managing the CA branch's standing CFG
> bias (it never reaches a fixed point — see [[project_turbo_alpha4_overdistill]]),
> and every CA-side lever came back inert or harmful
> ([[project_turbo_fei_gap_phase0]], `ca_band`). DP-DMD removes the CA branch
> entirely. The structural walkthrough (diversity-anchor / DMD gradient split,
> flow-matching velocity↔x0 math, the per-step schedule) lives at
> `docs/structure/dpdmd.md`; the CA-era decision log survives at
> `docs/proposal/dmd2_decoupled_improvements.md`. The original migration proposal
> is archived at `_archive/proposals/dpdmd.md`.

- **Training:** `scripts/distill_turbo/distill.py` — bespoke single-GPU loop
  (bypasses `train.py`/accelerate, like `distill-mod` / `distill-spd`).
- **Harness:** `networks/methods/turbo_dmd.py::TurboDMDNetwork` — two `LoRANetwork`
  stacks (student + fake) view-toggled on one frozen DiT.
- **Config:** `configs/methods/turbo.toml` — **bespoke sectioned schema** read only
  by the script. Don't `print-config METHOD=turbo` (the flat method+preset merge
  doesn't apply here). CLI flags override TOML values.

## Quick start

```bash
make exp-turbo                                       # configs/methods/turbo.toml defaults
make exp-turbo ARGS="--student_rank 128 --iterations 5000"
make exp-turbo ARGS="--single_prompt_idx 0"          # Phase 0 single-prompt overfit
make exp-turbo PRESET=low_vram                        # grad ckpt + offload + sample_ratio

make exp-test-turbo                                   # infer latest student LoRA @ 2 steps, cfg=1.0
```

`make exp-turbo` honors `PRESET` (translates `blocks_to_swap` /
`gradient_checkpointing` / `sample_ratio` from `configs/presets.toml` into CLI
flags) and appends `ARGS` last so user overrides win. The output is
`output/ckpt/<output_name>.safetensors` — a normal LoRA — plus the standard
`.snapshot.toml` (and a per-run resolved-config snapshot in the TB log dir).

At inference the student LoRA loads through the existing LoRA adapter path; the
caller just sets the step count and CFG=1. It composes with concept LoRAs the way
LCM-LoRA composes with style LoRAs (linear LoRA composition, ranks add).

## How it works (one screen)

The student is a **genuine N-step Euler rollout**, role-separated by step. Linear
flow path `z_t = (1−t)·x + t·ε`, velocity `v = ε − x` — Anima's native schedule.
Three roles share one frozen DiT; the LoRA stacks toggle per forward via
`TurboDMDNetwork.set_view` (each `LoRAModule` short-circuits on `not self.enabled`,
so a view switch is an O(num_modules) flag flip, negligible vs a DiT forward):

```
teacher view  — both LoRA stacks off  → base velocity (CFG'd at α=teacher_cfg)
student view  — student on, fake off  → v_student (the rollout)
fake   view   — fake on, student off  → v_fake_cond_dm (the score tracker)
```

Per training step:

1. **Teacher K-step CFG anchor (no-grad).** From a shared noise `ε`, roll the
   *teacher* (CFG-guided, `v_u + α·(v_c − v_u)`) for `k_anchor` Euler steps on the
   `teacher_anchor_steps` grid to an intermediate latent `z_tk` at continuous time
   `t_k`. The first-step diversity target is `v_target = (ε − z_tk) / (1 − t_k)`.
   `t_k` is read from the **teacher** grid, not the student grid — a σ mismatch
   silently mis-scales `v_target`. This anchor is what de-collapses pose/composition
   diversity (the DMD mode-seeking collapse the old `dm_x0_norm` band-aid was
   fighting at the *symptom*).
2. **Student N-step rollout.** Step 0 (from `ε`, t=1) is **diversity-supervised**:
   `div_loss = ‖v_first − v_target‖²`. Under `detach_after_first` (load-bearing)
   the diversity term is backwarded immediately and the step-0 graph is severed —
   the DMD reverse-KL from later steps must **not** flow back into the diversity
   mapping (their Fig 5: preference rises while diversity falls without it). Steps
   2..N then roll on with grad to produce `x_θ`.
3. **DMD on `x_θ` (no-grad teacher + no-grad fake, τ_DM ∈ [0,1]).** The real score
   is **CFG-guided** (`v_u + α·(v_c − v_u)`) — *not* cond-only. This is the one
   un-decoupling vs the CA-era code: the old DM branch was deliberately unguided
   because CFG lived in the separate CA branch; with CA gone, guidance has to ride
   the single DMD real score (matches the reference `compute_dmd_loss`). Without it
   `v_real ≈ v_fake` (`dm_cos ≈ 0.9999`) and the quality gradient is noise. The
   fake stays cond-only. `Δ_dm = v_real_cond_dm − v_fake_cond_dm`; the x0-space
   grad is `τ_dm·Δ_dm` (optionally per-sample x0-norm), applied via the DMD2 grad
   trick `loss = (grad_signal · x_θ).mean()`.
4. **Assemble + backward.** `loss = loss_dmd (+ div_weight·div_loss if not
   detached) (+ mean_var_weight·L_mv)`. `grad_clip` runs once on the accumulated
   student grad (diversity + DMD) either way.
5. **Fake update.** `fake_steps_per_student_step` plain flow-matching MSE steps on
   the student's `x_θ.detach()` distribution (resampling τ_fake, ε_fake each) —
   keeps the fake score tracker ahead of the moving x_θ.

The teacher uncond is the **T5("") sidecar** (`library/inference/uncond.py`), *not*
a zero tensor — a zero crossattn is fed-out-of-distribution and the resulting
`v_real_uncond` amplified at (α−1)=3× drives the student off-manifold (saturated
white output). Staged by `make distill-prep` / `make preprocess-te`; shared with the
mod-guidance distill.

A **fake (critic) head-start** runs `fake_warmup_steps` fake-only updates before the
main loop, calibrating the zero-init fake against the student's (init ≈ teacher)
`x_θ` distribution so the critic is ready before the student LR warmup ramps — this
kills the early `grad_signal_rms` spike (~step 50). The student is untouched during
it.

## Config surface (`configs/methods/turbo.toml`)

Sectioned, bespoke. Every key has a matching CLI override flag (see
`scripts/distill_turbo/config.py` argparse). The shipped defaults:

| Section | Key | Default | Notes |
|---|---|---|---|
| top | `output_name` | `anima_turbo_I` | output stem under `output/ckpt/` |
| top | `iterations` | `2000` | |
| top | `use_custom_down_autograd` | `true` | keeps activation memory down so swap can stay 0 (see [[project_custom_down_autograd_distill_lever]]) |
| top | `use_masked_loss` | `true` | **student-only** mask on the DMD grad; fake/critic stays full-frame |
| `[network]` | `student_rank` / `fake_rank` | `64` / `64` | `fake_rank ≥ student_rank` (fake is a score *tracker*, capacity ceiling on DM strength) |
| `[dmd]` | `student_steps` (N) | `2` | Euler steps the student rolls; inference matches (`--infer_steps 2`) |
| `[dmd]` | `teacher_cfg` (α) | `4` | CFG scale baked into the teacher anchor + DMD real score (Anima prod CFG=4) |
| `[dmd]` | `dm_x0_norm` | `true` | per-sample x0-space magnitude normalization of the DM grad ([[project_turbo_dmd_x0_norm_wins]]) |
| `[dmd]` | `norm_floor` | `0.05` | clamp_min for the `dm_x0_norm` denominator (latent scale) |
| `[dpdmd]` | `k_anchor` (K) | `4` | teacher steps rolled to the diversity anchor |
| `[dpdmd]` | `teacher_anchor_steps` | `8` | teacher σ-grid the K is counted against |
| `[dpdmd]` | `div_weight` (λ) | `0.05` | weight on the first-step diversity MSE |
| `[dpdmd]` | `detach_after_first` | `true` | **load-bearing** stop-grad after step 1; keep True (A/B only) |
| `[optim]` | `student_lr` / `fake_lr` | `2e-5` / `3e-5` | fake runs hotter |
| `[optim]` | `fake_steps_per_student_step` | `4` | keep the fake ahead of the moving x_θ |
| `[optim]` | `fake_warmup_steps` | `50` | fake (critic) head-start before the main loop — kills the early grad_signal_rms spike (~step 50); `0` = off |
| `[optim]` | `grad_clip` | `1.0` | grad-norm cap (both nets) |
| `[sampling]` | `t_distribution` | `uniform` | τ sampling for the fake update + warmup (or `sigmoid`) |
| `[sampling]` | `flow_shift` | `3.0` | σ-schedule shift for the student/teacher Euler grids (matches inference) |
| `[mean_var]` | `weight` | `0.0` (off) | optional Eq.7 mean-variance KL shield (lever B); `~0.01–0.05` to enable |

Validation enforces `student_steps ≥ 2` (step 1 is diversity-supervised + detached,
so at least one further step must carry the DMD loss) and
`1 ≤ k_anchor < teacher_anchor_steps`.

**Anchor fidelity — why the defaults dropped `14/28 → 4/8` (2026-06-01).** Both
ratios anchor at the *same* σ-fraction (`14/28 = 4/8 = 0.5`), so the diversity
anchor lands at the same continuous time; the only change is how coarsely the
teacher integrates to it (4 Euler forwards vs 14 — the anchor rollout gets ~3.5×
cheaper). A/B'd at 500 steps, sigmoid τ, `div_weight=0.05`, every other knob
identical (logs `20260531-144835` k14/t28 vs `20260601-121104` k4/t8): training
metrics are a **wash**. `dm_cos` (~0.979), `dm_mag_ratio` (~0.99), and `dm_rel_gap`
(~0.18–0.19) are flat within run-to-run noise; `div_loss` is equal-to-marginally
*lower* under k4/t8 (tailμ 0.093 vs 0.095); no instability spike. The only
systematic difference is `v_student_rms` / `x_pred_std` sitting ~1–2% higher in the
low-k run — the variance-inflation / over-bake lean ([[project_turbo_alpha4_overdistill]],
[[project_turbo_dmd_x0_norm_wins]]) — but well inside noise at this length.

Caveat before reading this as "lower K is free": **`div_loss` measures how well the
student *hits* the anchor, not how *diverse* the anchor is.** A k4 anchor is a
coarser, smoother target, so equal-or-lower `div_loss` does not prove the diversity
injection survived — a less-faithfully-integrated anchor can land off the teacher's
true trajectory, which these scalars can't see. The anchor's whole job is pose
de-collapse on real captions, and that only shows in sample grids (the PE-pooled
metric is blind to pose — [[project_dpdmd_pivot_phase0]]). Verdict on the lowered
defaults is therefore metrics-green / grid-pending: A/B `anima_turbo_J500_500`
(k4/t8) vs `anima_turbo_I_sigmoid_500` (k14/t28) at `--infer_steps 2 --cfg 1.0` and
read pose diversity + saturation, not the scalars.

## Inference: step count

The student is trained at `student_steps=2`, so `--infer_steps 2 --cfg 1.0` is the
matched schedule. **However**, an under-trained / lightly-distilled student behaves
like a continuous velocity field (the DMD quality loss is trained at *random* τ, not
on the 2-step grid; only step 0 is grid-anchored), so it can integrate **better** at
more Euler steps — at 2 steps the entire detail-forming band below σ≈0.5
([[project_sigma_signal_resolves_by_045]]) is crossed by a single `0.75→0` Euler
jump, while at 4 steps it gets a function evaluation at σ=0.5 *and* preserves the
σ=0.75 anchor. If a checkpoint looks better at 4 steps than 2, that's the tell that
distillation hasn't reached a true 2-step map yet — train longer or raise
`student_steps`. Always keep `--cfg 1.0` regardless of step count (CFG is baked;
don't double-guide).

## Reading the metrics

Trigger fake interventions on **`dm_rel_gap` ↑ / `dm_cos` ↓**, *not* on `fake_loss`
↑ (a rising fake loss against a moving, sharpening student is expected
equilibrium). Watch `div_loss` fall as the student's step-1 velocity converges on
the teacher anchor. The live TB scalars:

| TB scalar | Read |
|---|---|
| `div_loss` | `‖v_first − v_target‖²` — first-step diversity MSE (pre-weight). Falling = step-1 velocity converging on the diverse teacher anchor. |
| `dm_rel_gap` | `rms(τ·Δ_dm)/rms(τ·v_real_dm)` — fraction of the teacher score the gap still is. ↑ = fake lagging. |
| `dm_cos` | `cos(v_fake_dm, v_real_dm)` — →1 healthy; ↓ = fake pointing the wrong way (worse than a magnitude miss). |
| `dm_mag_ratio` | `rms(v_fake)/rms(v_real)` — ≈1 healthy. |
| `x_pred_std` / `v_student_rms` | collapse → 0 or runaway up = student exploding (`v_student_rms` leads). |
| `mean_var_kl` | Eq.7 KL (pre-weight); 0 when the reg is off. |

## Limitations & composition

- **Plain-LoRA bake is the hard constraint.** Anything needing a step-size or
  per-t input at inference (Shortcut / MeanFlow Δt-conditioning, timestep-conditioned
  T-LoRA — its mask is training-only, see [[project_tlora_inference_full_rank]])
  gives nothing after the bake.
- **Spectrum:** incompatible by construction — Spectrum's Chebyshev cache assumes
  ≥16 steps. Don't stack.
- **DCW:** the v4 fusion head was calibrated on 28-step trajectories; a turbo
  student needs its own few-step recalibration (out of scope).
- **Mod guidance:** tunable — the distilled `pooled_text_proj` may still help, but a
  turbo student may have re-learned the modulation pathway implicitly. Test, don't
  assume.
- **Block swap.** The student rolls `N` forwards and the teacher `K` anchor
  forwards per step (multi-forward); the offloader desyncs on a 2nd DiT forward
  ([[project_blockswap_extra_forwards_gradcache]]). The loop calls
  `prepare_block_swap_before_forward(free_cache=False)` before each forward, but the
  default path keeps `blocks_to_swap=0` (`use_custom_down_autograd=true` keeps
  activation memory low enough to run full-res on 16 GB without swap). Audit the
  multi-forward offloader path before turning swap on.

## References

- Wu, Li, Zhang, Ma — arXiv:2602.03139, *Diversity-Preserved Distribution Matching
  Distillation*. Reference impl: `dpdmd/train_sd35_dpdmd.py` (SD3.5-M, flow-matching).
- `_archive/proposals/dpdmd.md` — the migration proposal (Phase 0 GO, the
  pose-vs-pooled-cosine metric caveat, the depth-m fallback).
- `docs/structure/dpdmd.md` — structural walkthrough: the diversity-anchor / DMD
  gradient split, the flow-matching velocity↔x0 conversion, and the sign convention.
- `docs/proposal/dmd2_decoupled_improvements.md` — CA-era decision log; the record
  of why the CA branch was abandoned.
- `docs/findings/asymflow_parameterization.md` — Anima's `u = ε − x0` velocity path
  (the conversion the renoise/grad-assembly relies on).
