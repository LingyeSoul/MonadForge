# SPD fine-tuning LoRA — a trajectory adapter for progressive-resolution inference

A proposal to train a **plain LoRA** that closes the train–inference gap of
Spectral Progressive Diffusion (SPD, Xiao et al., arXiv:2605.18736, §4.3) on
Anima. The output is a *sampler/trajectory* adapter — the SPD analog of an
LCM-LoRA — not a concept adapter: it teaches the frozen DiT to follow the
stage-specific straight-line velocity targets of the multi-resolution SPD
trajectory, so that running early steps at low resolution gains *quality* on
top of the speed the training-free path already delivers.

This is the "Case B" of the SPD investigation. **Case A** (existing concept
LoRAs run unchanged under training-free SPD inference) needs no training and
is the prerequisite integration work — see `bench/spd/plan.md` Phase 3.

## Why this is worth doing

1. **Training-free SPD already works on Anima, but opens a quality gap.**
   `bench/spd/` confirmed the spectral premise (β = 2.26, R² = 0.9994,
   30/30 artists in [2,3]) and that the bare DiT denoises low-res latents and
   accepts the spectral-expansion handoff coherently (Phase 2 PASS) at
   ×1.65–1.73. But the bench also noted a behavioral signature: SPD output
   runs *sharper / higher-contrast* than baseline and diverges
   compositionally at the expansion (fresh HF noise). That divergence is the
   train–inference gap §4.3 exists to close — the frozen DiT was never trained
   on the multi-resolution trajectory, so it handles the handoff "viable but
   not obviously optimal."

2. **It is cheap and stable, unlike turbo.** The §4.3 target velocity (Eq. 12)
   is **analytic** — computed from the data sample and the constructed stage
   endpoints. There is **no teacher forward, no fake-score network, no
   adversarial loop, no CFG-bake**. It is plain flow-matching MSE (Eq. 14) on
   a single trainable adapter over a frozen base. Per-step cost ≈ ordinary
   LoRA training (one forward + backward). Compared to
   `docs/proposal/turbo_anima_dmd_lora.md` this removes every one of turbo's
   structural failure modes (R1 fake under-tracking, R2 CA divergence, R3
   step-capacity collapse). If turbo "isn't working at all," this is the
   stable path to a *training-improved* fast inference recipe — though note it
   buys **token efficiency, not few-step generation** (it does not compete
   with turbo's payload; it competes with turbo's *trainability*).

3. **The paper's own result is a quality win, not just a speed win.** Table 2:
   on Z-Image at S=2 the rank-32 LoRA fine-tune scores ImageReward **0.982**,
   beating both training-free SPD (0.904) *and* the full-resolution baseline
   (0.965) at the same ×1.65 speedup. The promise is "faster *and* better,"
   conditional on the gap being closeable by a low-rank delta.

## Background: the §4.3 recipe

Given `S` resolution scales `s_{1:S}` (`s_S = 1`) and δ-optimal transition
times `t_{1:S-1}` (`1 > t_1 > … > t_{S-1} > 0`), stage `i` runs at resolution
`(s_i H, s_i W)` for `t ∈ (t_i, t_{i-1})`. Per training step:

1. Sample `x_0 ~ p_data`, `t ~ U(0,1)`; assign `t` to the stage `i` with
   `t ∈ (t_i, t_{i-1})`. Sample `ε^{s_i} ~ N(0,I)` at scale `s_i`.
2. **Stage-entry state** `x̃^{s_i}_{t̃_{i-1}}` — apply spectral noise expansion
   from `s_{i-1} → s_i` at transition `t_{i-1}` exactly as at inference
   (Eq. i–iii), filling the newly representable HF slots with
   `t_{i-1} · T_Φ(ε^{s_i})`, then timestep-align (Eq. 5–6) to `t̃_{i-1}`.
3. **Stage endpoint** (standard FM state at the next transition), using the
   *same* `ε^{s_i}` so the two ends are correlated:
   `x^{s_i}_{t_i} = (1 - t_i) x_0^{s_i} + t_i ε^{s_i}`   (Eq. 11)
4. **Stage velocity target** — the straight line between them:
   `v^{s_i} = (x̃^{s_i}_{t̃_{i-1}} − x^{s_i}_{t_i}) / (t̃_{i-1} − t_i)`   (Eq. 12)
5. **Training sample** on that segment:
   `x^{s_i}_t = x^{s_i}_{t_i} + (t − t_i) v^{s_i}`   (Eq. 13)
6. **Loss:** `L(θ) = E ‖ v_θ(x^{s_i}_t, t) − v^{s_i} ‖²`,
   `v_θ = base + LoRA`, base frozen.   (Eq. 14)

So the only difference from ordinary Anima LoRA training is the *noising
process*: instead of one straight line from `x_0` to `ε` at full resolution,
the model is regressed onto the per-stage segment of the SPD multi-resolution
trajectory, at that stage's resolution.

## Anima-specific design decisions

### Stage-homogeneous batches (the static-shape concession)

The paper samples `t ~ U(0,1)` then routes to a stage, which would mix
resolutions within a batch. Anima pins one ~4096-token shape for
`torch.compile` (`library/datasets/buckets.py`,
`library/inference/generation.py` reads `h_latent/w_latent` once). To preserve
sanity:

- **Sample the stage first**, then `t ∈ (t_i, t_{i-1})` within it. The whole
  batch runs at one resolution per step. Across training there are only `S`
  distinct shapes → at most `S` compile graphs (per-block compile mode
  amortizes), or run **eager** for v0 (the probe already runs the bare DiT
  eager/dynamic-shape with no instability).
- Weight the stage sampling by the `t`-interval widths `(t_{i-1} − t_i)` so
  the marginal over `t` stays uniform (matches the paper's `U(0,1)`).

### No new caching — reuse the LoRA data layout

Low-res stage states are produced by **DCT low-pass of the cached full-res VAE
latent** (`dct_lowpass_init` in `bench/spd/probe_lowres_denoise.py:108`), and
spectral expansion runs on those — so the existing
`post_image_dataset/lora/{stem}_{WxH}_anima.npz` latents and
`{stem}_anima_te.safetensors` text caches are sufficient. No teacher cache, no
new sidecar. Text embeddings are token-count-agnostic and reused verbatim.

### The schedule it trains on is the schedule it must infer with

The LoRA is **schedule-coupled**: it learns the segment geometry of a specific
`(s_{1:S}, t_{1:S-1})`. v0 trains on the bench's recommended knee — the
**single-late `0.5 → 1.0 @ σ≈0.5`** schedule (`SamplerSPEED` widgets
`[0.5, 0.5, 0.99, 0.5]`), which the sweep found matches a 2-stage ramp in
quality while being simplest and fastest. The δ-optimal *derived* schedule
(Prop 1/2) requires the autoregression-dynamics probe (bench Phase 1, not yet
built); v0 hand-sets the schedule and defers derivation.

### The weights are shape-agnostic; only the learned behavior is coupled

Worth stating because it removes a false worry: the LoRA is per-Linear
(`A: r×d_in`, `B: d_out×r`, applied per-token), so it **loads and runs at any
resolution** through the standard inference path. Nothing about the *file* is
multi-resolution. Only *what it learned* is tuned to a schedule. This is also
why it can later stack on a concept LoRA on the same frozen base.

### Plain LoRA for v0 (paper-faithful)

The paper uses rank-32 plain LoRA. Default v0 = plain LoRA rank 32, **not** the
Anima default T-LoRA+OrthoLoRA+Hydra stack — fewer moving parts for the first
falsification, and T-LoRA's timestep mask interacts with the per-stage
`t`-routing in ways that should be studied separately, not bundled into Phase 0.

## Dependency: this is downstream of the training-free integration

Case B cannot ship without the SPD sampler that Case A builds —
`networks/spd.py` (spectral noise expansion + timestep alignment + the
per-stage shape rebuild lifted out of `generation.py`'s once-only assumption).
The training loop and the evaluation both call the same primitives. **Build
`networks/spd.py` first** (bench Phase 3); promote the probe's
`dct_lowpass_init` / `spectral_expand` (`bench/spd/probe_lowres_denoise.py:108`,
`:119`) into it. This proposal assumes that module exists.

## Phasing & gates

### Phase 0: single-prompt-set overfit sanity (1 day)

- 8 prompts, rank-32 plain LoRA, single-late `0.5→1.0 @ σ0.5` schedule, eager,
  ~2k iters. Inference via the SPD sampler at the *same* schedule.
- **Pass:** SPD-LoRA output at the trained schedule is at least as coherent as
  training-free SPD on those prompts, with the handoff divergence visibly
  reduced (less HF-noise grain, composition closer to the full-res baseline).
- **Fail:** NaN / latent-std blow-up / quality below training-free SPD →
  almost certainly the stage-entry construction (Eq. 11/12 sign or alignment)
  is wrong; diff the train-time `x̃^{s_i}_{t̃_{i-1}}` against the sampler's
  expanded state on the same seed — they must match bit-for-bit at `t_{i-1}`.

### Phase 1: quality vs training-free SPD at matched speed (3 days)

- Full LoRA run on the standard dataset, single-late schedule.
- Bench at the production env (CFG=4, the 5 DCW aspect buckets): **ImageReward,
  CLIP-IQA, CMMD** for (full-res baseline) vs (training-free SPD) vs
  (SPD-LoRA), all at equal speedup. CMMD is our live val signal
  (`project_cmmd_val_signal`).
- **Pass:** SPD-LoRA beats training-free SPD on aggregate ImageReward/CMMD and
  closes to within noise of the full-res baseline at ×1.6+ speedup —
  reproducing the paper's Table 2 *direction* on Anima.
- **Weak:** SPD-LoRA only *matches* training-free SPD → the gap on Anima is not
  low-rank-closeable at rank 32; bump to rank 64–128 once, else the §4.3
  recipe doesn't earn its keep here and we ship Case A (training-free) only.

### Phase 2: composition with a concept LoRA (2 days)

The deployment story, identical in spirit to turbo Phase 3.

- Pick 3 existing concept LoRAs from `output/ckpt/`. Test (concept @ full-res
  28-step) vs (SPD-LoRA + concept, SPD sampler). Both deltas stack on the
  frozen base.
- **Pass:** concept survives and the SPD-LoRA's trajectory benefit persists.
- **Fail:** concept washed out or handoff artifacts return → train the SPD-LoRA
  with **concept-LoRA-on augmentation** a fraction of the time (so it learns
  the trajectory correction *with* a content delta present), or restrict the
  SPD-LoRA to a disjoint module set. Defer to a v1.

**Skip rule:** if Phase 1 is WEAK after one rank bump *and* training-free SPD
already composes fine with concept LoRAs (Case A), kill Case B — the §4.3
recipe is only worth a training pipeline if it demonstrably beats free.

## Risks and failure modes

### R1: low-res stage is too far OOD for a low-rank delta to fix
Anima trained only at the ~4096-token bucket; the `s=0.5` stage is a token
count it never saw. *Mitigant:* Phase 2 of the bench already showed the bare
DiT denoises low-res coherently, so the residual the LoRA must absorb is small,
not a from-scratch capability. If it's still too large, that surfaces as Phase
1 WEAK and the rank-bump branch applies.

### R2: schedule coupling makes the LoRA brittle
A LoRA trained for `0.5→1.0 @ σ0.5` may degrade on other schedules.
*Mitigant:* train on the production schedule and ship the schedule alongside
the weights (snapshot it into the `.safetensors` metadata). For robustness,
optionally jitter the transition σ over a small range during training (a few %
around 0.5) so the segment geometry is learned as a band, not a point.

### R3: composition with concept LoRAs breaks
Two low-rank deltas on the same Linears interfere. Lower-likelihood than turbo
(SPD doesn't collapse steps or bake CFG — it's a gentle modification of the
same multi-step path), but untested. *Mitigant:* concept-on augmentation (R3
branch above).

### R4: the Anima gap is bigger than Z-Image's
The paper's Table 2 win was on models with *some* multi-resolution exposure in
pretraining (Z-Image / PixelGen). Anima's single-bucket training means the
LoRA has more to fix, so the honest expected outcome may be "matches
training-free SPD and improves the handoff" rather than "beats the full-res
baseline." That's still a ship if it stabilizes the divergence the bench saw —
but the framing should not promise the paper's exact margin.

## Out of scope

- **Few-step generation.** This is not turbo. It keeps the full step count;
  the win is per-step token cost. No CFG-bake.
- **Video / temporal frequencies.** Anima is image-only.
- **δ-optimal derived schedule.** Depends on the autoregression-dynamics probe
  (bench Phase 1). v0 hand-sets the schedule; derivation is a follow-up.
- **Frequency-based editing** (paper §5.5) — tracked separately as a DirectEdit
  complement in `bench/spd/plan.md` Bonus.
- **DCW recalibration for the SPD trajectory.** The v4 fusion head was trained
  on 28-step full-res trajectories; an SPD-trajectory DCW is its own `make dcw`
  run if needed. Don't conflate.

## File-level plan

New files:
- `scripts/distill_spd.py` — training loop. Modeled on
  `scripts/distill_mod/distill.py` (frozen-DiT + adapter-only + MSE), **not**
  on `scripts/distill_turbo.py` — there is no teacher/fake here, so it's
  strictly simpler: one adapter, one optimizer, the stage-sampler + Eq. 11–14
  target construction, single MSE backward. Reuses `networks/spd.py` for the
  stage-entry / endpoint construction.
- `configs/methods/spd.toml` — schedule (`s_{1:S}`, transition σ), rank,
  LR/epochs, output_name. Bespoke-ish like turbo (the schedule block is the
  novel surface), read by `distill_spd.py`.
- `bench/spd/bench_finetune.py` + extend `bench/spd/README.md` — the Phase 1
  three-way bench (baseline / training-free / SPD-LoRA). Standard
  `bench/_common.py` envelope.
- `docs/methods/spd.md` — promote here from `proposal/` once Phase 1 passes.

Touched files:
- `networks/spd.py` — **shared with Case A**; must exist first.
- `tasks.py` + `scripts/experimental_tasks/training.py` — `exp-spd-finetune`
  (+ `exp-test-spd` reuses the Case-A SPD inference path).
- `Makefile` — same.
- `inference.py` — none; the SPD-LoRA loads through the existing LoRA adapter
  path, the SPD sampler is the Case-A `--spd` flag.

## Open questions before kicking off

1. **Eager vs per-block compile for training.** v0 eager (matches the probe,
   simplest). Per-block compile of the `S` stage shapes is the speed
   optimization if training throughput hurts — decide after Phase 0 timing.
2. **Schedule in metadata vs config-only.** Snapshotting the trained schedule
   into the `.safetensors` so inference can't silently mismatch it (R2) — do
   this from v0; it's cheap insurance.
3. **Concept-on augmentation from the start?** Default off for Phase 0/1
   (clean trajectory-correction signal), turn on only if Phase 2 fails.

None are blocking; defaults above hold until the gates say otherwise.
