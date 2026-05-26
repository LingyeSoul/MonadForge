# Soft Tokens — AGSM: bounded dual-token alignment guidance

Status: **proposal** (2026-05-22). Builds on `docs/experimental/soft_tokens.md`
and supersedes the InfoNCE direction in `docs/proposal/soft_tokens_contrastive.md`
*conditionally* — see "Relationship to the contrastive proposal". Reuses the
cached-TE negative plumbing already shipped for that proposal
(`library/datasets/base.py::setup_contrastive_negatives`, `IdentityPairSampler`).

Reference: Lee, Hong, Kwon, Ye, *Alignment-Guided Score Matching for Text-to-Image
Alignment in Diffusion Models* (ICML 2026 Spotlight; code "coming soon"). This is
the **direct sequel to SoftREPA** (Lee et al., arXiv:2503.08250) — the same paper
our soft-tokens parameterization is adapted from, same first author.

## TL;DR

SoftREPA's contrastive objective is unstable: its validation ImageReward degrades
even as training loss drops (the paper's own stability plot; mirrored by the SD3
FID regression we flag in `soft_tokens.md`). AGSM diagnoses the cause as
**unbounded contrastive divergence** — pushing negative pairs to *maximize* their
denoising error has no fixed point — and fixes it three ways:

1. **Shift the score-matching target instead of maximizing negative error.**
   Positives regress toward `v_t + γ⁺·Ã(t)·Δ`, negatives toward `v_t − γ⁻·Ã(t)·Δ`.
   Bounded by construction.
2. **Dual token banks ψ⁺ / ψ⁻** so positive and negative guidance don't fight
   over one shared token space.
3. **Reward-free Plackett–Luke normalization** of an *intrinsic* alignment reward
   read off the model's own denoising likelihood — **no external reward model**.

The headline for us: this is **implementable entirely from existing pieces**. The
extra-forward loop, cached-TE negatives, splice, and loss-compose site are already
in the tree from the contrastive proposal. The only genuinely new internal
machinery is a second token bank and an EMA shadow of predictions. Nothing
external — no ImageReward, no CLIP scorer, no teacher checkpoint.

## Why this is in-house (the "no external model" answer)

AGSM is *reward-free* by design. The alignment reward is

```
r(x_t, c) = −‖v_θ^ψ(x_t, c) − v_target‖²            # denoising likelihood proxy
p(z | x_t, c) = softmax_i r(x_t, c_i)                # Plackett–Luce over candidates
```

i.e. it asks "which candidate caption explains this noised latent best", scored by
*our own* flow-matching error. The guidance direction Δ is built from an **EMA of
our own soft-token-conditioned predictions** — a self-distillation target, not an
external model. So every term is computed from the frozen Anima DiT + the trainable
banks we already have. Concretely:

| AGSM ingredient | Where it comes from in our tree |
|---|---|
| ε_θ⁺ / ε_θ⁻ (pos/neg conditioned preds) | `SoftTokensMethodAdapter.extra_forwards` — already runs pos + k neg forwards |
| negative captions (D⁻) | `setup_contrastive_negatives` / `_load_te_for_stem` (cached-TE swap) |
| v_target | `v_target = noise − latents` (`soft_tokens.py:670`) — already computed |
| reward r(x_t,c) | the per-forward `−‖v−v_target‖²` already computed for the InfoNCE logit |
| ψ⁺ / ψ⁻ banks | **new**: second `SoftTokensNetwork`-style bank (or doubled `num_tokens` split in half) |
| Δ from EMA preds (ε̂⁺, ε̂⁻) | **new**: EMA shadow of the pos/neg velocity predictions |

## The flow-matching mapping (why ε→v is free here)

AGSM is written in ε-prediction. Anima is velocity flow-matching with
`v = ε − x₀` and a fixed data `x₀`, so a shift of the ε-target by `δ` is *exactly*
a shift of the v-target by `δ` (the `x₀` term is constant). The shipped FM target
is already `v_target = primary.noise − primary.latents`. Therefore AGSM's target
becomes, with no reparameterization:

```
positives (D⁺):   L⁺ = ‖ v_θ^{ψ⁺} − ( v_target + γ⁺·Ã(t)·Δ ) ‖²
negatives (D⁻):   L⁻ = ‖ v_θ^{ψ⁻} − ( v_target − γ⁻·Ã(t)·Δ ) ‖²
L_AGSM = E_{D⁺}[L⁺] + E_{D⁻}[L⁻]
```

Open derivation items (flagged, not free):

- **Ã(t)** — AGSM's bounded time-weight. SoftREPA/AGSM use an ε-noise schedule;
  ours is the FM σ∈[0,1] schedule already bucketed by `n_t_buckets`. Start with
  `Ã(t)=1` (constant, bounded) and only add t-shaping if a t-bucket sweep shows
  it matters. Do **not** import the ε-schedule weighting blindly.
- **Δ** — the guidance direction. AGSM: implicit-reward-weighted EMA of `(ε̂⁺,ε̂⁻)`.
  Velocity form: `Δ = v̂⁺_ema − v̂⁻_ema`, optionally reward-weighted by the
  Plackett–Luce `p`. Needs an EMA of the predictions (see below).

## New machinery (the only two non-existing pieces)

### 1. Second token bank ψ⁻

Today `SoftTokensNetwork` holds one bank (`tokens` + `t_offsets`). AGSM needs two
guidance regions. Cheapest: keep one `nn.Parameter` of shape
`(2, n_layers, K, D)` (or two banks) and splice ψ⁺ on positive forwards, ψ⁻ on
negative forwards, via the existing `apply_to` / splice path. Param count doubles
the bank term only (still ~1–2M total at our scale). The ablation in the paper
(dual > positive-only > shared at equal budget) is the thing to reproduce in
Phase 2.

### 2. EMA shadow of predictions

A bounded target needs a slow reference. Maintain an EMA of the pos/neg velocity
predictions (or of the bank weights, then forward through it — cheaper to EMA the
*outputs* per step). This is a standard `decay·ema + (1−decay)·current` update;
no external model. Lives next to the adapter, updated in the trainer step after
the optimizer step.

Everything else — extra forwards, negative sourcing, warmup gate, the
`aux["soft_tokens_contrastive"]` → `_soft_tokens_contrastive_loss` compose path —
is reused. The handler in `losses.py:318` already applies a warmup-gated weight to
an aux scalar; AGSM just puts a different scalar (`L_AGSM`) in that slot.

## Relationship to the contrastive proposal

`soft_tokens_contrastive.md` Phase 1 (plain InfoNCE, `shuffled` negatives) is
**already implemented but not yet benched**. That is the cheap probe that tells us
whether we even *have* the SoftREPA instability on Anima:

- **If Phase 1 A/B is stable and helps** → InfoNCE is fine here; AGSM's bounded
  reformulation buys little; shelve this proposal as "fallback if instability
  appears later."
- **If Phase 1 shows val-reward-degrades-while-loss-drops** (the SoftREPA pattern)
  → that is the empirical trigger to adopt AGSM. Don't tune τ/k; switch the
  objective in the *same* `extra_forwards`/compose seam.

So this proposal is **gated behind the Phase 1 result of the contrastive proposal,
not run in parallel.** They share all infrastructure; only the loss math differs.

## The premise risk to falsify first (cheap, no training)

AGSM's reward is the **denoising likelihood**, and we have a hard, repeatedly
confirmed finding that **FM-MSE does not track quality on Anima**
([[project_fm_val_loss_uninformative]]; why we moved to CMMD,
[[project_cmmd_val_signal]]). AGSM's entire alignment signal is built from exactly
that quantity.

The saving grace is that Plackett–Luce uses it as a **relative ranking across
candidate captions for the same latent**, not as an absolute quality score —
relative ordering can survive when absolute MSE is uninformative. But this must be
checked before any training:

- **Probe (Phase 0).** Extend `bench/soft_tokens_contrastive/negative_audit.py`
  with a no-training ranking test: for n anchors, score the matched caption and its
  cached-TE negatives by `−‖v−v_target‖²` (LoRA-off and with the current bank), and
  measure ranking accuracy (does matched beat mismatched?) and margin. **Gate:** if
  matched does not reliably out-rank `shuffled` negatives, AGSM's reward premise
  fails on Anima → stop. This is the single most important early kill-check.

## Phasing — gates, cheapest-first

- **Phase 0 — reward-premise probe (no training, runnable today).** The ranking
  test above. Gate: matched caption out-ranks `shuffled` negatives with positive
  margin on a meaningful fraction of anchors. Reuses the existing audit harness +
  cached TE.
- **Phase 1 — (deferred to the contrastive proposal).** Its plain-InfoNCE A/B is
  the instability detector. AGSM is only justified if that A/B exhibits the
  SoftREPA degrade-while-loss-drops pattern.
- **Phase 2 — AGSM target-shift, single bank first. ✅ IMPLEMENTED (2026-05-22,
  not yet benched).** The bounded target (`v_target ± γ·Ã(t)·Δ` with `Ã=1`, Δ from
  the bank-EMA shadow) lives in `SoftTokensMethodAdapter.extra_forwards`, selected
  by the network arg `contrastive_objective=agsm` and reusing the InfoNCE compose
  seam + `after_backward` grad-cache verbatim (only the loss math differs). Single
  bank (ψ⁺ = ψ⁻), constant `Ã(t)=1`. Knobs: `agsm_gamma` (γ, default 0.5),
  `agsm_ema_decay` (default 0.99). The EMA shadow is a plain tensor attribute
  (never saved; refreshed once per optimizer step in `after_backward` on
  `sync_gradients`). **Cost note:** the EMA value passes make this ~`(2k+1)` extra
  forwards/firing-step, above the proposal's headline `(k+1)×` — Δ is read off the
  shadow bank's *own* predictions (the load-bearing self-distillation decoupling),
  which costs the extra matched + mismatched EMA forwards; a cheaper live-Δ
  approximation was rejected because it reintroduces the moving-target dynamic AGSM
  exists to remove. Still to do: A/B vs (a) plain FM and (b) Phase-1 InfoNCE on the
  prompt-following / CMMD axis (FM-MSE val deltas are uninformative —
  `project_fm_val_loss_uninformative`). Keep `k ∈ {1,2}`.
- **Phase 3 — dual bank ψ⁺/ψ⁻ + ablation.** Add ψ⁻, reproduce the paper's
  dual > positive-only > shared ablation at equal token budget. Add Ã(t) shaping
  only if a t-bucket sweep shows the constant weight leaves signal on the table.
  Decide ship-on-default vs opt-in; update `soft_tokens.toml` + `soft_tokens.md`.

## Costs to keep honest

- **Extra forwards.** Same `(k+1)×` step cost as the contrastive path; AGSM does
  not reduce it. `k ∈ {1,2}` only.
- **EMA memory.** One shadow of the bank (small) or of per-step predictions
  (transient). Negligible vs the frozen DiT.
- **Reconstruction risk.** Code is "coming soon"; Ã(t) and the exact Δ
  reward-weighting are reconstructed from the website equations until the paper
  lands. The flow-matching mapping above is the load-bearing assumption.
- **Goal-mismatch.** AGSM optimizes COCO-style prompt *alignment* (counting, no
  repeated objects) on a general backbone. Our soft tokens train on a
  character/style dataset where the objective is closer to identity/style fidelity.
  Decide whether "alignment reward" points where we want before Phase 2 — the
  Phase 0 ranking probe partly answers this (does matched-caption preference even
  exist on our data).

## What this does NOT do

- Does not add any external reward model, scorer, or teacher checkpoint — every
  term is computed from the frozen Anima DiT + trainable banks (reward-free).
- Does not change `batch_size`, the dataloader batching, or the splice/block hook —
  negatives are cached-TE swaps (B=1-safe), same as the contrastive proposal.
- Does not run in parallel with the contrastive Phase 1 — it is gated on that
  result and reuses its plumbing.
- Does not claim parity with the paper's SD1.5/SDXL/SD3 numbers — it is a
  flow-matching, B=1-adapted reconstruction, gated on the Phase 0 reward probe.

## Reference points

- Module + extra-forward loop: `networks/methods/soft_tokens.py`
  (`SoftTokensMethodAdapter.extra_forwards` `:653`, `contrastive_loss` `:545`,
  `step_contrastive_warmup` `:524`, `v_target` `:670`)
- Loss compose site: `library/training/losses.py`
  (`_soft_tokens_contrastive_loss` `:318`, registered key `soft_tokens_contrastive`)
- Negative sourcing (reuse): `library/datasets/base.py::setup_contrastive_negatives`
  / `_load_te_for_stem`, `library/datasets/identity_pairs.py::IdentityPairSampler`
- Reward-premise probe to add: `bench/soft_tokens_contrastive/negative_audit.py`
- Sibling proposal (gates this one): `docs/proposal/soft_tokens_contrastive.md`
- Method doc: `docs/experimental/soft_tokens.md`
- Quality-signal context: [[project_fm_val_loss_uninformative]],
  [[project_cmmd_val_signal]]
- Papers: AGSM (ICML 2026, https://jaayeon.github.io/AGSM/); SoftREPA
  (arXiv:2503.08250, NeurIPS 2025)
