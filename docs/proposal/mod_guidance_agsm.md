# Mod Guidance — AGSM: a contrastive term on the modulation path

Status: **proposal** (2026-05-29, rev. 2026-05-29). Builds on
`docs/methods/mod-guidance.md` and **directly reuses the soft-tokens contrastive
machinery** — negative sourcing, AGSM target-shift, EMA shadow — re-targeted from
the soft-token bank onto `pooled_text_proj`.

Reference: Lee, Hong, Kwon, Ye, *Alignment-Guided Score Matching for Text-to-Image
Alignment in Diffusion Models* (ICML 2026; https://jaayeon.github.io/AGSM/), in
`_archive/Alignment-Guided Score Matching…`. Mod guidance: Starodubcev et al., ICLR
2026, arXiv:2602.09268. Sibling proposal that built the machinery:
`docs/proposal/soft_tokens_agsm.md`.

## TL;DR

Mod guidance's **inference** mechanism is already a preference direction in
modulation space (`delta = proj(pool(p₊)) − proj(pool(p₋))`), but its **training** is
a single-pair distillation MSE — for each image, one caption, regress student
(text-through-modulation) onto teacher (text-through-cross-attention). The projection
is only ever taught to *faithfully carry* text; nothing teaches it to **discriminate**
captions in modulation space, which is the geometry the steering delta rides on.

**The proposal:** add the soft-tokens **contrastive/AGSM term** to the distillation,
re-targeted onto `pooled_text_proj`. Same image latent, the **matched caption** plus
**k real mismatched captions** (soft-tokens' existing cached-TE negatives), each
injected through the projection; the bounded AGSM target-shift trains the projection
so the matched caption's modulation is more on-manifold (lower FM-error) than the
mismatched ones. This makes the modulation path **text-discriminative**, which is
mod-guidance's whole reason to exist ("make AdaLN text-aware") — and the bet is that a
sharper, less-entangled modulation geometry gives cleaner steering directions, so the
per-block `w` schedule that exists to fight pink/DC collapse becomes optional rather
than load-bearing.

**No synthetic negatives.** An earlier draft fabricated `worst quality, score_1`
strings as negatives; that invents a distribution the model never sees and is dropped.
Negatives are real mismatched captions from the same plumbing soft tokens use
(`setup_contrastive_negatives` / `IdentityPairSampler`), so they are in-distribution
and B=1-safe.

## The mapping (mod guidance gives the AGSM structure for free)

AGSM's structure is *"same noised latent, several conditionings, rank by FM-error,
shift the target to prefer the matched one."* The distill loop already supplies it on
the exact pathway we want to shape:

| AGSM ingredient | Where it comes from |
|---|---|
| same `x_t`, several conditionings | one `noisy_input`, several pooled vectors through the **same** MLP via `forward_mini_train_dit(..., pooled_text_override=…)` (`distill.py:653`) |
| matched conditioning (D⁺) | the image's own caption pooled vector (`pooled_text`), already per-batch |
| negatives (D⁻) | **real mismatched captions** — soft-tokens' `setup_contrastive_negatives` / `_load_te_for_stem` cached-TE swap (`library/datasets/base.py`); pool each negative's `crossattn_emb` the same way the base path pools the matched one |
| reward `r(x_t,c)` | `−‖v_student(pool_c) − v_teacher‖²` — which injection best reproduces the gold teacher velocity |
| v_target | `teacher_pred` (already computed, `distill.py:619/628`) — the gold |
| PL weights / Δ / γ⁺/γ⁻ | `_agsm_pl_weights` / `agsm_targets` / `agsm_losses` (`soft_tokens.py`) — pure tensor ops, network-agnostic |
| EMA shadow | EMA of the ~8M-param MLP weights — same `decay·ema + (1−decay)·cur` |

**No dual ψ⁺/ψ⁻ bank** (soft-tokens §3a) is needed or possible: there is one MLP, and
matched/mismatched are different *inputs* to it, not different parameter banks. The
single open question that soft-tokens AGSM is still falsifying does not exist here.

### Loss

Keep the distillation MSE as the carry term; add the contrastive term on the same
batch (only the injected pooled vector differs, so the extra forwards are the same
`(k+1)×` soft tokens already pays):

```
L_carry  = ‖ v_student(pool(caption))    − v_teacher ‖²                  # unchanged distillation
L⁺       = ‖ v_student(pool(caption))    − (v_teacher + γ⁺·Δ⁺)   ‖²       # matched
L⁻ⱼ      = ‖ v_student(pool(neg_capⱼ))   − (v_teacher − γ⁻·Δ⁻ⱼ) ‖²       # k real mismatched
L        = L_carry + λ_pref · ( L⁺ + mean_j L⁻ⱼ )
```

with `Δ_j = v̂_ema_j − Σ_k w_k v̂_ema_k`, `w_j = softmax_j(−‖v̂_ema_j − v_teacher‖²/τ)`
exactly as in `agsm_targets`. ε→v is free (Anima is velocity FM, `v=ε−x₀`, `x₀`
constant — an ε-target shift is a v-target shift). Start `Ã(t)=1`. `λ_pref` and a
`--mod_agsm` switch default **off**, so the shipped MSE-only distillation stays the
default until the Phase-2 A/B clears.

## Why no Phase-0 premise probe (deliberately skipped)

The soft-tokens version gated on a no-training reward-premise probe because AGSM's
reward is the denoising likelihood and `[[project_fm_val_loss_uninformative]]` says
generic FM-MSE doesn't track quality on Anima. Two reasons that gate is **not** needed
here:

1. **The premise is already proven for real-caption negatives.** With mismatched
   captions (not fabricated quality strings), the premise is just "matched ranks above
   mismatched by FM-error" — which the soft-tokens Phase-0 probe **already passed**
   (rank@1 0.993 shuffled / 0.958 hard, margin grows with σ;
   `[[project_agsm_reward_premise_holds]]`). The discriminative signal lives in the
   frozen DiT's response to conditioning; re-running it for mod guidance would re-prove
   a held result.
2. **Mod guidance's val loss is a different, informative quantity.** The
   `project_fm_val_loss_uninformative` finding is about generic data-FM-MSE. Mod
   guidance's distillation loss is `MSE(student, teacher)` — a *reconstruction-of-the-
   gold-teacher* metric — and on this method lower distillation val loss **has** tracked
   better samples (operator-confirmed; also why the synth-pool was added to floor the
   real-vs-teacher gap, mod-guidance.md §distill-prep Phase 2). So the loss the term
   builds on is trustworthy here.

The one residual unknown is *narrowness*: the conditioning enters through a pooled
vector → AdaLN, a tighter channel than soft tokens' per-layer splice, and the
projection starts near-zero. That is not a premise question (no probe can pre-answer
it) — it is exactly what the Phase-2 A/B measures. So we go straight to training.

## Phasing — gates, cheapest-first

- **Phase 1 — wire the contrastive term into distillation, single MLP.** Add the k
  negative-pool injections + EMA shadow + `agsm_targets`/`agsm_losses` call to
  `distill.py`'s loss (the `pooled_text_override` student forward already exists; the
  negative sourcing is the soft-tokens path). Knobs mirror soft tokens: `agsm_gamma`
  (γ⁺), `agsm_gamma_neg` (γ⁻, 0.1), `agsm_ema_decay` (0.99), PL τ, `k ∈ {1,2}`, plus a
  new `lambda_pref` and `--mod_agsm` (default off). Train against the synth pool
  (`--synth_data_dir`) so the real-vs-teacher gap doesn't confound the term. Watch the
  distillation val loss (informative here) doesn't regress.
- **Phase 2 — the A/B that decides ship-vs-revert.** AGSM-distilled projection vs the
  shipped MSE-only projection, **on the steering quality axis** (CMMD,
  `[[project_cmmd_val_signal]]`, + qualitative drift sweeps):
  - **Primary win condition — safer steering at uniform `w`.** Re-run the pink/DC
    collapse that forced the per-block schedule (the "channel" LoRA at `w=3`,
    mod-guidance.md §"Why schedule instead of uniform?"). If the contrastive-trained
    projection stays clean at `--mod_start_layer 0` (uniform) where the MSE-only one
    collapsed, the discrimination sharpening paid off and the schedule becomes optional.
  - **Secondary — CMMD quality-steering gain** at matched `w`, and the
    steering-direction-consistency metric (mod-guidance.md §"Quality direction
    consistency", 0.814 mean cosine) must not degrade.
  - **KILL → revert:** no uniform-`w` safety gain and no CMMD gain → the faithful-carry
    distillation already captured the modulation geometry well enough. Keep `--mod_agsm`
    off.

## Why this might help where it counts

The per-block `w` schedule exists because the learned quality direction is *entangled
with the early-block tonal-DC direction* — uniform `w` blows up the DC component (pink
collapse). That entanglement is the symptom of a modulation geometry trained only to
carry, never to discriminate. `[[project_mod_guidance_quality_tag_axis]]` already
showed the axis is real but **directionally double-counted** and the score ladder is a
*rotation*, not a clean axis — i.e. structured but entangled. A contrastive term that
pushes matched-caption modulation to higher likelihood than mismatched is a direct
attempt to disentangle that geometry. The σ-growth of the caption margin (Phase-0
finding) also lines up with mod guidance's own observation that the modulation path is
most sensitive at high noise (mod-guidance.md §"Modulation sensitivity") — AGSM's
natural band is where mod guidance has the most leverage.

## Costs to keep honest

- **Content-discrimination ≠ quality-steering (the real open risk).** Real mismatched
  captions sharpen *content* discrimination in the modulation path; the inference use
  is a *quality* direction. The bet is that a cleaner, more text-discriminative geometry
  yields cleaner steering — but that transfer is exactly what Phase 2 tests, not a given.
- **Double-count interaction.** `pooled_text_proj` feeds *both* the base modulation and
  the steering delta (`[[project_mod_guidance_quality_tag_axis]]`). Sharpening could
  amplify the base double-count; watch the steering-direction-consistency metric in
  Phase 2 — if it degrades, the term is over-rotating the base path.
- **Extra forwards.** `(k+1)×` student forwards/step + EMA value passes (same as
  soft-tokens AGSM). Teacher forward is cached (`teacher_cache_K`), unaffected. `k ∈ {1,2}`.
- **Redundancy risk.** Mod guidance ships and works; this is a *refinement bet*, not a
  rescue — it must earn its keep on uniform-`w` safety or CMMD, or revert.
- **EMA memory.** One shadow of the ~8M MLP. Negligible.

## What this does NOT do

- No external reward model, scorer, or teacher checkpoint — the reward is
  `−‖v_student − v_teacher‖²` off the frozen DiT (reward-free, in-house).
- No fabricated quality negatives — negatives are real mismatched captions from the
  soft-tokens plumbing.
- No dual ψ⁺/ψ⁻ bank — one MLP; matched/mismatched are inputs to it.
- No Phase-0 premise probe — the matched>mismatched premise is held from soft-tokens
  Phase 0, and mod guidance's distillation val loss is informative here.
- No inference-path change — the trained projection drops into the existing
  `--pooled_text_proj` / `--mod_w` / per-block surface unchanged (the *hypothesis* is
  the schedule becomes optional, not that the surface changes).

## Reference points

- Mod-guidance method + inference surface: `docs/methods/mod-guidance.md`
- Distillation loss site (where the term composes): `scripts/distill_mod/distill.py`
  (teacher forward `:628`, student forward w/ `pooled_text_override` `:653/658`, MSE
  loss `:662`)
- AGSM helpers to reuse (network-agnostic): `networks/methods/soft_tokens.py`
  (`_agsm_pl_weights`, `agsm_targets`, `agsm_losses`, `update_bank_ema`)
- Negative sourcing to reuse: `library/datasets/base.py::setup_contrastive_negatives`
  / `_load_te_for_stem`, `library/datasets/identity_pairs.py::IdentityPairSampler`
- Sibling proposal (built the machinery + ran Phase 0): `docs/proposal/soft_tokens_agsm.md`
- Context: `[[project_fm_val_loss_uninformative]]`, `[[project_cmmd_val_signal]]`,
  `[[project_mod_guidance_quality_tag_axis]]`, `[[project_agsm_reward_premise_holds]]`,
  `[[project_sigma_signal_resolves_by_045]]`
- Papers: AGSM (ICML 2026, https://jaayeon.github.io/AGSM/); Mod guidance (ICLR 2026,
  arXiv:2602.09268)
