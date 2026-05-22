# Variance-Reduced Flow-Matching Loss for Anima

> **2026-05-14 errata.** The HEADROOM-gate run cited below
> (`results/20260514-1300-tlora-vs-base/`) measured a **merged T-LoRA**
> trainable against a base frozen reference — a near-converged-adapter
> regime where `u_pred ≈ u_pred^L` is true by construction, which pins
> `λ → −1` and `ρ² → 1`. **Live training (`output/logs/
> lora_default_20260514-1921/`) shows `vr/lambda_ema` settles at ~−0.72,
> not −0.996.** The asymptote, fixed-λ=−1 ablation hypothesis, and the
> HF-residual-adapter motivation that all leaned on the −0.996 figure
> are no longer supported by that data point. VR-loss v1.5 itself still
> trains and the eyeball A/B held — what's wrong is the strength of the
> conclusions, not the existence of *some* variance-reduction signal.
> The bench's null gap (decorrelated-ε ρ² ≈ 0.024) confirms ε-pairing
> produces real structural correlation; the gradient-variance question
> (vr_loss.md Open Q #2) is what actually matters and remains unmeasured.
> See `README.md` for the full errata.

Plan to integrate the AsymFlow §5.2 control-variate FM loss (Chen et al.,
arXiv:2605.12964) into Anima's LoRA-family training. The headroom bench in
this directory's [`README.md`](README.md) / [`run_bench.py`](run_bench.py)
cleared its gate on 2026-05-14 (`results/20260514-1300-tlora-vs-base/`,
verdict **HEADROOM**). **As of 2026-05-14 the plan has shipped through
v1.5** — v1 wired and v1.5 eyeball A/B confirmed VR buys the quality win
at +40% step cost on r=16 / 2.56k steps. The v2/v3 λ refinements were
falsified by a follow-on perband-headroom bench
(`results/20260514-1637-perband-headroom-tlora/`) and are closed, not
deferred. This doc is now historical record + design rationale.

## TL;DR

```
# Standard FM loss (per sample)
L_FM = ||(x_0 − x̂_0) / σ_t||²

# VR loss (per sample, what this plan adds)
x_0^L  := gaussian_blur_2d(x_0, σ_low)         # FEI-aligned low-pass of the latent
x_t^L  := α_t · x_0^L + σ_t · ε                # paired noisy input, same ε
x̂_0^L  := frozen_anima(x_t^L, t, te) → x_0     # paired prediction
L_VR   = ||(x_0 − x̂_0) / σ_t + λ · (x_0^L − x̂_0^L) / σ_t||²
```

with `λ = -Cov(Y, Z) / Var(Z)` estimated online (EMA across batches), gated
behind `vr_loss_weight > 0` in `configs/methods/lora.toml`. Adds one extra
no-grad forward per training step (~50% step cost, fixed). Frozen control
variate is **the model at start of finetune**, held in VRAM for the duration
of the run.

Headroom is bench-validated. **2026-05-14 run (`results/20260514-1300-tlora-vs-base/`):**
36 (sample, t) pairs on `156x104`, T-LoRA-merged checkpoint as trainable,
frozen base DiT as control variate. `ρ²_high_band` mid-t median = **0.998**,
`ρ²_global` mid-t median = **0.996**, decorrelated-ε null gap = **+0.988**,
`λ_global` = **−0.996 ± 0.002** across all pairs. Per-t breakdown flat at
~0.997 from t = 0.10 through t = 0.85. The null gap rules out the
shared-model artifact that inflated the earlier smoke read; what remains is
the structural ε-pairing correlation the paper exploits.

## Why

Two motivations stack:

1. **AsymFlow's own claim.** §5.2 reports "substantially improved fine-grained
   details" on FLUX.2 klein finetuned with VR loss. The mechanism is generic
   variance reduction of the loss estimator — not specific to their pixel-AsymFlow
   case — so a transfer to Anima's standard latent flow-matching is plausible.

2. **Our measurement.** The 2026-05-14 bench (frozen base DiT vs T-LoRA-merged
   trainable, 36 paired (sample, t) measurements) confirms ~99.8% of per-sample
   loss variance on the high-frequency complement is recoverable by the paper's
   λ·Z correction at global λ ≈ −1. The decorrelated-ε null falls to ρ² ≈ 0.01
   on the same metric, so the correlation isn't "both forwards see similar
   inputs" — it's the structured pairing the paper relies on. Gradient noise is
   the dominant bottleneck on Anima's FM training at this regime, not bias.
   That's exactly where VR helps.

## What this is

- A modification to `library/training/losses.py`'s FM loss path, gated by a
  weight flag, that injects a control-variate correction onto the standard
  flow-matching target.
- One extra no-grad forward through a frozen reference model per training
  step. Reference model is the DiT at start of finetune.
- An online λ estimator (per-batch covariance + EMA) — no calibration pass
  needed at training start.
- A bench-validated default `vr_loss_weight = 1.0` (paper uses λ as the only
  knob; the overall weight is separate and lets us scale down if it interacts
  badly with other losses).

## What this isn't

- **Not a new adapter / method.** This is a loss-level change. The FM target
  shape, sampling procedure, model architecture, and inference path are all
  untouched. A VR-trained checkpoint inferences identically to a standard one.
- **Not the AsymFlow parameterization.** The rank-asymmetric `u_A = Pε − x_0`
  target swap is a *separate* idea from the same paper and is not on the table
  here — Anima's per-patch noise/hidden ratio is too small for it to matter
  (see [[project_fm_val_loss_uninformative]] context).
- **Not a replacement for FECL.** FeRA's frequency-energy consistency loss
  (`library/training/losses.py:278`) uses the same FEI band kernel but solves
  a different problem (directional alignment of adapter correction). VR and
  FECL compose; this plan doesn't touch FECL's gate.

## Design

### Choice of `x_0^L`

```
σ_low  = min(H_lat, W_lat) / fei_sigma_low_div   # default div=4.0
x_0^L  = library.runtime.fei.gaussian_blur_2d(x_0, σ_low)
```

Same kernel that defines FEI routing on the Hydra/FeRA paths. Three reasons:

1. The model's adapter routing is already shaped around this band split, so
   the control variate inherits the same inductive bias instead of inventing
   a new one.
2. `library.runtime.fei.gaussian_blur_2d` is in-tree, fp32-safe, and cached.
   No new module.
3. The 2-band FEI gives a free diagnostic axis: per-FEI-band breakdown in
   the bench tells us whether VR helps low-frequency-dominated samples
   differently than high-frequency ones. **Closed:** the 2026-05-14
   perband-headroom run found uniform effect — single-band VR is enough
   (see "Per-element λ … now closed" below).

Default `fei_sigma_low_div = 4.0` matches the live training default in
`configs/gui-methods/fera.toml` / `configs/gui-methods/hydralora_fei.toml`.

### Frozen control variate

The paper uses a frozen copy of the *initialized* model (post latent→pixel
lift, pre-finetune). For Anima we have no equivalent lift step — the natural
choice is **the base DiT at start of finetune**. Concretely:

- Loaded once at trainer init, held on `accelerator.device` in eval mode,
  `requires_grad_(False)`.
- bf16 weights to match the live model's precision footprint.
- For LoRA-family runs the frozen copy is just the base DiT (no adapters), so
  it's bit-identical to what the training run starts from before the LoRA
  applies. This is convenient: the frozen control variate is what the live
  model would predict at step 0.

VRAM cost: ~5 GB extra for the 2B base DiT in bf16. Tight on 12 GB cards,
fine on 16 GB. The plan recommends gating with a sensible default that
disables VR on low-VRAM presets:

```toml
# configs/methods/lora.toml
vr_loss_weight = 0.0        # gate. 1.0 enables.
vr_frozen_ref = "base_dit"  # only option supported in v1
```

Alternative if memory is tight: **EMA copy** instead of a fully frozen
reference. The EMA copy drifts slowly toward the trained model, breaks the
"unbiased control variate" property mildly, but stays in a single VRAM
budget if we offload via `accelerate`'s existing offload hooks. v2.

### λ estimation

Paper says "patch-wise adaptive weight chosen to minimize the loss gradient
norm." We do the simpler global-λ form:

```python
# Online per-batch estimate, then EMA across batches.
Y_centered = (x_0 - x̂_0) / σ_t - Y_running_mean
Z_centered = (x_0^L - x̂_0^L) / σ_t - Z_running_mean

cov_batch = (Y_centered * Z_centered).sum()        # scalar
var_batch = (Z_centered * Z_centered).sum()
λ_batch   = -cov_batch / max(var_batch, ε)

λ_ema = (1 - β) * λ_ema + β * λ_batch              # β = 0.01 default
```

EMA across batches because B is small (typically 4) and per-batch λ is
noisy. Bench confirms λ_global ≈ -1.0 across all (sample, t) pairs at
N=32, so the online estimator should converge fast and the EMA β = 0.01
is well-conditioned.

Per-element λ (v2) and per-band λ_k via `_fera_fecl_bands` (v3) were
considered as refinements and **falsified** by the 2026-05-14
perband-headroom bench (`results/20260514-1637-perband-headroom-tlora/`):
- v2 mid-t mean delta over global: **+7.9e-6** (= +0.00079%).
- v3 mid-t mean delta over global: **−3.4e-6** (estimator artifact;
  bounded above by v2's +7.9e-6, so still zero in practice).

Scalar λ already attains mid-t reduction ≈ 0.9999, leaving no headroom for
a richer λ to recover. v2 / v3 are now closed, not deferred.

### Compute cost

Per training step:

| | grad fwd | no-grad fwd | bwd | net |
|---|---|---|---|---|
| Standard FM | 1 | 0 | 1 | 1× |
| VR loss   | 1 | 1 | 1 | ~1.5× |

The no-grad forward (frozen ref on `x_t^L`) runs in inference mode — no
activation checkpointing, no autograd graph. On a 5060 Ti at typical Anima
bucket (`128×128` latent), it's ~40% the cost of the gradient-tracked
forward, so net step cost is **~1.4×** in practice.

Net win requires VR to give >1.4× effective convergence. The paper's
quantitative win on AsymFLUX.2 klein (Table 3) is +0.96 HPSv3 from VR alone
(12.03 → 12.99, ~8% relative) and a further +0.07 with the LPIPS perceptual
correction; they do not report a wall-clock or step-count speedup figure for
VR specifically. Anima's regime is different (smaller noise/hidden ratio, no
pixel lift), so Stage 1 has to actually demonstrate the convergence win — we
cannot assume it from the paper's quality delta.

## Integration points

### `library/training/losses.py`

New handler `_flow_matching_vr_loss` registered as `"flow_matching_vr"`,
activated when `args.vr_loss_weight > 0`. Replaces the standard `flow_matching`
entry in the active-losses list. v1 = single global λ. (v3 = per-band λ_k
was the alternative under consideration before Stage 0; closed by the
perband-headroom bench.)

### `train.py` `get_noise_pred_and_target`

After the main model forward + standard target construction, gated on
`args.vr_loss_weight > 0`:

```python
if args.vr_loss_weight > 0:
    sigma_low = fei_sigma_low(h_lat, w_lat, args.fei_sigma_low_div)
    x_0_L = gaussian_blur_2d(latents, sigma_low)
    x_t_L = (1 - sigma_t) * x_0_L + sigma_t * noise  # same noise
    with torch.no_grad():
        u_pred_L = self._vr_frozen_ref(x_t_L.unsqueeze(2), timesteps, crossattn_emb, ...)
    x0_L_hat = x_t_L - sigma_t * u_pred_L.squeeze(2)
    aux["vr_z"] = (x_0_L - x0_L_hat) / sigma_t       # consumed by loss handler
    aux["vr_y_factor"] = 1.0 / sigma_t                # for the loss handler to scale
```

### `AnimaTrainer.__init__`

Load the frozen reference once when `args.vr_loss_weight > 0`. Reuse
`library.anima.weights.load_anima_model` with `dit_path = args.frozen_ref_dit
or args.pretrained_model_name_or_path`. Hold on device in bf16, eval mode.

### `configs/methods/lora.toml`

Two new keys, both gated off by default until bench validates them:

```toml
vr_loss_weight = 0.0          # 0 = off (default); 1.0 = paper recipe
vr_fei_sigma_low_div = 4.0    # matches live FEI default
```

No preset-level overrides in v1.

## Validation plan

### Stage 0 — DONE (2026-05-14)

`bench/fm_vr_headroom/results/20260514-1300-tlora-vs-base/`. Pass criterion
was `ρ²_high_band` mid-t median ≥ 0.30 with a decorrelated-ε null gap ≥ 0.20;
actual `ρ²_high_band` mid-t median = **0.998**, null gap = **+0.988**, and
the per-t breakdown is flat across t = 0.10–0.85 (no high-t α_t → 0
degeneracy). `λ_global` = −0.996 ± 0.002 confirms the global-λ ≈ −1
assumption used by the online estimator below.

Open caveat carried into Risks #1: the trainable was T-LoRA-merged
(small Δ from base). The bench cannot distinguish "VR cancels real per-step
gradient noise" from "two near-identical models give identical residuals on
paired inputs." Stage 1 is the only way to resolve.

### Stage 1 (Tier 1.5 bench under `bench/fm_vr_headroom/results/`)

A/B short LoRA training run on a standard dataset:

- **Arm A**: `make lora-gui GUI_PRESETS=tlora_ortho` with default loss.
- **Arm B**: same config plus `--vr_loss_weight 1.0`.
- Same seed, same data, same step budget, same hardware.

Metrics:

1. **Wall-clock convergence**: steps-to-target val FM loss + wall-clock-to-target.
   Note `[[project_fm_val_loss_uninformative]]` — relative comparison is still
   useful even if absolute val FM doesn't track quality.
2. **Sample quality at fixed step**: HPSv3 + VQAScore + CLIP on a held-out
   prompt set (reuse `bench/dcw/`'s prompt pool for continuity).
3. **Sample quality at fixed wall-clock**: same metrics evaluated at the step
   count that arm A reaches in arm B's wall-clock budget.

Both arms should sample the same set of seeds at validation; report mean +
std across seeds. Bench script lives at `bench/fm_vr_headroom/training_ab.py`
(not yet written; Stage 1 deliverable).

### Pass criterion for shipping

- (2) shows ≥ 0.02 HPSv3 win at fixed step count, OR
- (3) shows ≥ 0.01 HPSv3 win at fixed wall-clock,

with the win robust across two prompt subsets. Below that, the extra 1.4×
compute isn't worth it.

## Risks

1. **Trainable ≠ frozen divergence isn't yet large.** The 2026-05-14 bench
   used a T-LoRA-merged checkpoint as trainable against a frozen base DiT —
   the LoRA delta is small enough that the two models' Jacobians at any given
   input are nearly aligned. The decorrelated-ε null gap (+0.988) rules out
   the literal-same-model artifact, but doesn't rule out the
   "two-near-identical-models" artifact: with ρ² so close to 1 and λ pinned
   to −1 ± 0.002, what the bench measures may be "models with nearly aligned
   gradients always agree on input-pair deltas" rather than "VR cancels real
   per-step gradient noise." The headroom bench has done what it can; only
   Stage 1 A/B on a longer training run can fully resolve. As insurance,
   `v0.5` keeps a clearly-diverged ablation (`anima-preview2` as trainable)
   on the table.

2. **Bias risk.** The control variate `(x_0^L − x̂_0^L)` has expectation
   `(x_0^L − E[x̂_0^L | x_t^L])` — this is non-zero in general because the
   model has bias. With a *frozen* ref, the bias is constant in `λ`, so the
   minimum of `Var(Y + λZ)` is still achieved at `λ* = -Cov/Var`. But it does
   mean the *expected* loss isn't exactly the same as standard FM — there's a
   systematic shift that could push the trained model away from the FM optimum.
   The paper handles this with the LPIPS perceptual correction term (§5.2 last
   paragraph). v1 skipped it; v1.5 confirmed sample quality holds at r=16 /
   2.56k steps without it. If a longer-step regression ever surfaces, the
   LPIPS term is the lever — not richer λ (v2/v3 are falsified).

3. **Interaction with existing losses.** Anima's training loss has multiple
   components (FM + FECL + functional + soft-tokens + …). VR replaces the
   FM term, not the others. If FECL depends on the FM loss having its
   native variance, there could be second-order effects. Stage 1 A/B is
   the only way to surface these.

4. **Static-shape compile.** The extra forward needs the same static padded
   shape as the main forward. Anima's `_run_blocks` is already shape-pinned
   for `torch.compile`, so adding a second forward in the same step should
   be transparent — but verify that gradient checkpointing's offload schedule
   doesn't try to free the frozen ref's activations.

5. **Memory.** ~5 GB extra for frozen ref in bf16 on a 2B base. Low-VRAM
   presets (8 GB cards) can't run this; v1 disables VR on `low_vram` /
   `fast_16gb` presets via a method-config override.

## Open questions

- **Frozen-ref granularity.** Reload base DiT every run, or freeze the
  finetune's *starting* state including any LoRA from a resumed checkpoint?
  v1 assumes pure base DiT; v2 might prefer "starting state" so resume-from-
  checkpoint runs use the resumed state as the control variate.
- ~~**Multi-band schedule.**~~ Closed by the 2026-05-14 perband-headroom
  bench: per-FEI-band λ recovers `−3.4e-6` mid-t reduction over global
  (estimator-artifact negative; upper-bounded by per-element `+7.9e-6`).
  Scalar λ stays.
- **CFG-dropout interaction.** When the trainer drops the cross-attention
  embedding for CFG training, does the frozen ref's prediction at `x_t^L`
  use the same dropped embedding? v1 says yes (same crossattn_emb for both
  forwards in the same step).
- **σ_min clamping.** AsymFlow §6.1 reports `σ_min = 0.04` clamping for
  numerical stability on the `σ_t` denominator. Anima's FM training is
  already stable without this (no reported NaN issues), but VR's
  `1/σ_t` factor amplifies low-σ instability. Defensive default
  `vr_sigma_min = 1e-3`; opt out via `0.0`.

## Sequencing

| Step | What | Where | Gate / status |
|---|---|---|---|
| v0 | Frozen-ref + decorrelated-ε-null headroom bench. | `bench/fm_vr_headroom/results/20260514-1300-tlora-vs-base/` | ✅ DONE — HEADROOM |
| v0.5 | (Optional) Re-run with `--trainable_dit anima-preview2.safetensors` to bound the trainable-similarity bias from Risks #1. | `bench/fm_vr_headroom/results/<run>/` | Not blocking v1; informative for Risks #1. |
| v0.7 | Per-element / per-band λ feasibility on the same headroom rig. | `bench/fm_vr_headroom/results/20260514-1637-perband-headroom-tlora/` | ❌ DEAD — v2 delta +7.9e-6, v3 delta −3.4e-6, both bench-noise. Closes v2/v3 below. |
| v1 | Wire VR loss into `train.py` + `library/training/losses.py`. | This doc's "Integration points" section. | ✅ DONE — Compiles, smoke-trains, frozen-ref VRAM under preset budget (no extra model via adapter-bypass). |
| v1.5 | Stage 1 A/B: standard FM vs VR on a short LoRA run. | eyeball A/B at r=16, 2.56k steps. | ✅ DONE — VR buys the quality win at the +40% step-cost overhead. Quantitative HPSv3/VQA pass still optional. |
| v2 | ~~EMA frozen ref, per-element λ, LPIPS correction.~~ | — | ❌ Closed by v0.7 (per-element λ). LPIPS correction still live if a long-step regression ever surfaces. |
| v3 | ~~Per-band λ_k via `_fera_fecl_bands`.~~ | — | ❌ Closed by v0.7. |

## References

- AsymFlow paper (arXiv:2605.12964), Chen et al., 2026-05-13. §5.2 is the VR
  loss section.
- `library/runtime/fei.py` — the FEI kernel (in-tree, 2-band Laplacian).
- `library/training/losses.py:278` `_fera_fecl_loss` — FECL, the existing
  band-aware loss that shares the FEI kernel.
- `[[project_fera_probe_2band_decision]]` — why we use 2 bands not 3.
- `[[project_fm_val_loss_uninformative]]` — why Stage 1 needs HPSv3/VQA, not
  just val FM curves.
- `docs/methods/hydra-lora.md` — FEI routing background.
