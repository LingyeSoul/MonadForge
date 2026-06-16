# REPA v2 — patchwise / relational alignment against PE-Spatial

Training-time auxiliary loss aligning a mid-block DiT feature to cached
PE-Spatial patch tokens (REPA, Yu et al., arXiv:2410.06940 — revived from the
archived v1 at the per-token granularity v1's own docstring pre-planned).
Design rationale and the pre-registered Phase-0 gate live in
`_archive/proposals/repa_v2_patchwise_pe_spatial.md` (implemented, archived);
this doc is the live operating reference. The Phase-1 operating-point plan
was implemented and its proposal archived
(`_archive/proposals/repa_phase1_operating_point.md`); the two levers still
open live in "Annealing plan" below.

This is a **loss-level regularizer**, not a new adapter — it rides the LoRA
network kwargs and composes with every variant in the LoRA family (the
lora.toml default stack included). The trained checkpoint inferences
identically to a plain run of the same variant: the only extra parameters
(Arm A's projection head) are stripped at save time.

## Status (2026-06-12) — Phase 0 CLOSED, relational wins

All three arms trained at the Phase-0 operating point (~tenth-scale, weight
0.05, block 8, PE-Spatial) and eyeballed:

- **REPA vs baseline: decisively better** — the line is live.
- **Arm B (relational) vs Arm A (absolute): B preferred ~6:4–7:3** on
  sample grids. Combined with B's structural advantages (no head, no resume
  wart, domain-offset cancellation), **relational is the shipped default**;
  the absolute head stays implemented for reference only.

### Phase-1 progress (2026-06-12 evening)

Levers run on the relational arm at the Phase-0 operating point (priority
order + closed-lever record in the archived proposal,
`_archive/proposals/repa_phase1_operating_point.md`):

- **Lever 2 — spatial norm: validated, ON by default** (`repa_spatial_norm
  = true` in lora.toml). iREPA target standardization; eyeball-validated and
  shipped.
- **Lever 3 — token-subset Gram: CLOSED without a training run.** The
  `repa_grad_heatmap` diagnostic (autograd grad of the Gram loss wrt pooled
  tokens, top-10% position recurrence on a quarter run) came back ~uniform:
  top-10% mean recurrence **1.51×** vs MaskAlign's ~21× pathology, max single
  position 3.4× (a corner/edge cluster, not interior content). The shortcut
  needs pretraining-scale iteration counts to form — don't re-propose unless
  the regime changes.
- **Lever 1 — anneal: A/B still open, curve-justified candidate 0.25.**
  align_loss plateaus by ~17–18% of the run (two quarter runs agree);
  weighted REPA settles at **~4% of total loss** (≈8% in the first decile).
  At plateau, w = 0.02 / 0.05 / 0.1 ≈ 1.7% / 4% / 8% of total loss — the
  frame for the eventual weight sweep (lever 4).
- Progress curves via `scripts/repa_progress_report.py <progress.jsonl>`
  (deciles + plateau; on annealed runs, the FM-only before/after-release
  delta — negative ⇒ REPA was fighting FM late).

### External validation (same-week literature)

Two papers independently corroborate the relational read:

- **iREPA** (Singh et al., arXiv:2512.10794) — across 27 vision encoders,
  the *spatial structure* of the target representation (pairwise patch-token
  cosine structure — exactly the affinity geometry the Gram form aligns)
  predicts generation FID at |r| > 0.85, while global semantic quality
  (linear probing) predicts at |r| ≈ 0.26; injecting global info into patch
  tokens monotonically hurts generation; and the standard MLP projection
  head (Arm A's mechanism) is shown to lose spatial contrast in transfer.
  Bonus: PE-Spatial-B beats PE-Core-G as a REPA target despite ~30pt worse
  ImageNet accuracy — our encoder choice, validated at scale.
- **MaskAlign** (Pang et al., arXiv:2606.08788) — full-token alignment
  concentrates alignment-gradient norm at stable spatial positions (a
  feature-fitting shortcut); random token-subset alignment regularizes it
  away. Their input-masking recipe is incompatible with our constant-token
  compile invariant, but the loss-side subset idea is a Phase-1 lever.
- **HASTE** (arXiv:2505.16792) — REPA helps early training and degrades it
  late; published support for the planned anneal/cutoff lever.

## The two arms

| | `repa_mode = "relational"` (Arm B — **default, Phase-0 winner**) | `repa_mode = "absolute"` (Arm A — reference only) |
|---|---|---|
| Loss | `MSE(G_dit, G_pe)` on Gram matrices `G = F̂F̂ᵀ` of per-token L2-normalized features | per-token cosine between head-projected DiT tokens and PE tokens, mean of `1 − cos` |
| Head | none — similarities live within each space, dims never need to match | 3-layer MLP `REPAHead` (DiT dim → DiT dim → encoder dim), last layer near-zero init |
| What transfers | only *which patches are alike* — layout, part structure, anatomy. Any global domain direction (the encoder's photo prior) cancels out of the pairwise structure by construction (relational KD, Park et al. 2019) | the encoder's actual feature directions — strictly more signal, but the target carries the photo-prior gestalt; the head can't filter it out (it maps DiT features *toward* the target, not the prior *out of* it) |
| Extra params | 0 | head — training-only, stripped from saved adapters; a warm start re-inits it (re-converges in a few hundred steps) |

Both arms align a **noisy-input** mid-block feature to the **clean-image**
encoder target at every sampled σ (paper-faithful σ-conditioning), captured
from the primary forward — no second DiT forward, so no block-swap offloader
desync.

## Quick start

```bash
# One-time prereq: cache PE-Spatial sidecars into the LoRA cache dir
# ({stem}_anima_pe_spatial.safetensors — disjoint from the PE-Core caches CMMD reads)
make preprocess-pe-spatial

# Toggle in configs/methods/lora.toml (or any gui-methods variant):
#   use_repa      = true
#   repa_mode     = "relational"   # or "absolute"
#   repa_weight   = 0.05           # v1's 0.5 was a pretraining default — do not revisit
#   repa_layer    = 8              # block to hook (of 28)
#   repa_encoder  = "pe_spatial"
#   repa_lr_scale = 1.0            # head LR multiplier (absolute mode only)
#   repa_spatial_norm = true       # iREPA target standardization — ON by default (lever 2)
# Phase-1 lever still open (one A/B at a time — see the proposal):
#   repa_anneal_steps = 0.25       # hard cutoff: (0,1] = fraction of run, >1 = opt steps
make lora
```

Missing sidecars don't crash: a batch without `repa_pe_features` skips the
term for that step (and `train.py` prints the preprocess hint at startup). If
PE features load but the block hook never fires, the adapter warns once —
REPA being silently inert is a logged condition, not a quiet no-op.

## Mechanics

- **Capture**: forward post-hook on `unet.blocks[repa_layer]`
  (`library/training/repa.py::REPAMethodAdapter`). The hook sits on
  `block.__call__`, *outside* the compiled `block._forward`, so it fires
  under `compile_blocks()`. Under native_flatten the captured shape is the
  fake-5D `(B, 1, seq, 1, D)` rather than eager `(B, 1, H, W, D)` — both
  flatten to the same row-major `(B, N_dit, D)`, so the reshape is
  layout-agnostic.
- **Grid match**: PE-Spatial tokens live on a per-aspect-bucket `(gh, gw)`
  grid (32×32 at square). The DiT side is adaptive-avg-pooled down to that
  grid so both sides are `N = gh·gw` tokens in the same row-major order.
  `(gh, gw)` is recovered from the encoder feature's own token count,
  disambiguated by the latent's orientation (the bucket table is
  aspect-symmetric, so token count alone is ambiguous).
- **Numerics**: Gram / cosine computed in fp32 (caches are bf16; low-norm
  cosine is precision-sensitive).
- **Loss attach**: scalar returned under `aux["repa"]`, weighted by
  `losses._repa_loss` in the stage-2 registry slot (same family as
  `fera_fecl`). Active iff the factory stamped `_repa_weight > 0`.
- **Gradient reach**: REPA gradient only flows into LoRA modules in blocks
  ≤ `repa_layer` (by design). Remember this when reading per-block deltas.
- **Arm A param group**: `repa_head` gets its own optimizer group at
  `repa_lr_scale × unet_lr` (`networks/lora_anima/network.py`); the head is
  deleted from the state dict at save so adapters stay inference-clean.
- **Config plumbing**: kwargs (`use_repa` / `repa_mode` / `repa_weight` /
  `repa_layer` / `repa_encoder` / `repa_lr_scale` / `repa_anneal_steps` /
  `repa_spatial_norm`) are parsed in `networks/lora_anima/factory.py` and
  stashed on the network; they are registered in the `NETWORK_KWARGS`
  allowlist (`networks/__init__.py`) — any new key must be added there or
  it's silently inert.
- **Anneal clock** (`repa_anneal_steps` > 0): the adapter counts train
  micro-batches in `prime_for_forward` and divides by
  `gradient_accumulation_steps` (the `step_contrastive_warmup` pattern) —
  validation passes don't tick it. Past the cutoff the term is skipped
  before the PE transfer (one log line at the cutoff step). Fractional
  values resolve against `args.max_train_steps`.
- **Spatial norm** (`repa_spatial_norm`): target-side only, relational mode
  only — `(pe − mean_tok) / (std_tok + ε)` across the token axis, applied
  after the CLS drop and before per-token L2-norm. Removes the shared global
  component that compresses pairwise cosines (iREPA). **On by default.**
- **Heatmap diagnostic** (`repa_grad_heatmap` = N > 0): read-only probe that
  every N micro-steps takes the autograd grad of the Gram loss wrt the pooled
  DiT tokens, accumulates a canonical-grid top-10% recurrence map, dumps
  `<output_name>_repa_grad_heatmap.npz` next to the checkpoint, and logs
  `repa/heatmap_conc`. This closed lever 3 (MaskAlign token-subset); it's a
  diagnostic, not a loss change.
- **Aux-loss dispatch caveat** (fixed 2026-06-12): `train.py` used to route
  method-adapter `extra_forwards` (where REPA attaches) through the
  cached-LLM-adapter path, which only exists when `crossattn_emb is not None`.
  EasyControl uses the in-model text path (`crossattn_emb=None`) so REPA was
  silently skipped there pre-fix — **failure signature: `repa/active=1.0` in
  the progress jsonl WITHOUT `repa/align_loss`.** The dispatch now runs on both
  text paths and relational REPA is validated + shipped as the EasyControl
  default for cond ≠ target tasks; still check both metrics at launch on any
  non-default method (closure: `_archive/proposals/easycontrol_repa_operating_point.md`,
  mechanism in `docs/experimental/easycontrol.md`).

## Annealing plan (open A/B — next)

Two operating-point levers are implemented and still open. Both run as
sequential short A/Bs (the Phase-0 recipe: ~tenth-scale, fixed seed/data,
current default stack — relational, w=0.05, block 8, spatial_norm on), one
knob at a time vs the current default. The shipped default stays no-anneal
until an A/B moves it.

**1. Anneal / cutoff — `repa_anneal_steps`, candidate `0.25`.** Hard cutoff
that drops the REPA term past a fraction of the run; carried from the v2
proposal ("alignment matters early; the late-run gradient is where style
drifts") with published support (HASTE: REPA helps early training, degrades
it late). It attacks the exact v1 failure axis (late-run style leak) for one
comparison run. The `0.25` candidate is curve-justified: align_loss plateaus
by ~17–18% of the run (two quarter runs agree), so the term has little left
to teach past the first quarter. Semantics (already wired): `(0,1]` =
fraction of `max_train_steps`, `>1` = absolute optimizer steps, `0` = off;
the adapter keeps its own micro-batch counter (validation excluded) and
converts via `gradient_accumulation_steps`; past the cutoff the term is
skipped before the PE transfer. Hard cutoff first (one knob); only if it
helps, compare linear decay-to-zero as a second variant.

- *Read the curve*: `scripts/repa_progress_report.py <progress.jsonl>` prints
  deciles + the plateau point, and on an annealed run the FM-only
  before/after-release delta — **negative ⇒ REPA was fighting FM late**,
  which is the signal the cutoff is paying off.

**2. Weight sweep — `repa_weight ∈ {0.02, 0.05, 0.1}`.** Run *after* the
anneal point settles, on the winning cutoff — weight interacts with the
window (a shorter window tolerates a higher weight). The effective-weight
frame for reading results: at plateau, `w = 0.02 / 0.05 / 0.1` ≈ `1.7% /
4% / 8%` of total loss (≈2× that in the first decile, before align plateaus).

**Gates (unchanged from Phase 0, pre-registered):** sample grids first
(style intact; anatomy — hands/limbs/eyes), CMMD-vs-own-PE-Core as the drift
tripwire, never FM val loss, never fraction-of-Δ readouts. **Any lever that
visibly moves style fails regardless of its anatomy delta.**

## Graduation (Tier 1.5, outstanding)

Gated on a settled operating point: a `repa_v2` bench envelope (to be added; `bench/_common.py`;
grids + CMMD in `result.json`), a flag-off byte-identity invariant test
(extend `tests/test_repa.py`), one full-length run at the final operating
point vs baseline, and a `configs/gui-methods/` variant + default-on
decision. Lever 3 (token-subset Gram) is **closed** — see Phase-1 progress
above; the `repa_token_keep` knob was never built and shouldn't be unless the
regime moves to pretraining-scale steps.

## Guardrails (from the v1 burn)

- **Never re-run v1's operating point** (global pooling + weight 0.5). v1's
  documented outcome: anatomy ↑ but anime style broken (vision-encoder
  photo-prior leak, `docs/experimental/soft_tokens.md`).
- **Style gate is non-negotiable** — any arm that visibly moves style fails
  regardless of its anatomy delta.
- Judge on **sample grids first**, CMMD-vs-own-PE-Core as the drift
  tripwire. Never FM val loss (uninformative on Anima), never fraction-of-Δ
  readouts.

## References

- `library/training/repa.py` — adapter + both loss forms (module docstring
  covers the grid/native_flatten mechanics in more depth).
- `networks/lora_anima/factory.py` (kwargs + head attach),
  `library/training/losses.py::_repa_loss`,
  `library/datasets/base.py::_try_load_repa_pe` (sidecar loading),
  `library/vision/buckets.py::PE_SPATIAL_B16_512_SPEC`.
- `_archive/proposals/repa_v2_patchwise_pe_spatial.md` — hypothesis, v1
  autopsy, pre-registered Phase-0 gate (implemented; archived on closure).
- `_archive/proposals/repa_phase1_operating_point.md` — Phase-1 plan
  (implemented; archived on closure — open levers migrated to "Annealing
  plan" above).
- `_archive/repa/` — v1 implementation (pooled-global, PE-Core, w=0.5).
- REPA: Yu et al., arXiv:2410.06940. Relational form cf. Park et al. 2019
  (relational knowledge distillation).
- iREPA: Singh et al., arXiv:2512.10794. MaskAlign: Pang et al.,
  arXiv:2606.08788. HASTE: arXiv:2505.16792 (PDFs for the first two in the
  repo root).
