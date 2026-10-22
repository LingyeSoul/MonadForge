# MOSS-Image — which Anima negatives are worth re-opening

Status: proposed 2026-10-18. Tier 1 is measured: §1 (sinks) has a finding, §2 (2:4
sparsity) is closed. Tier 2 is unmeasured. Tier 3 (added 2026-10-21, a second sweep
of `_archive/` and `docs/findings/`): §3 measured, the rest unmeasured. Each entry is
an Anima negative result whose recorded cause is checked against MOSS-Image's
architecture (`library/moss/CLAUDE.md`, `project/moss_lora/plan.md`). An entry is
re-opened only when the recorded cause depends on an Anima-specific fact that MOSS
does not share. Each re-opened entry gets a cheap premise probe and a gate before
any port.

## The MOSS facts that change a cause

| MOSS fact | Anima counterpart | Causes it can flip |
|---|---|---|
| Text is never padded; the DiT sees only valid tokens | 512-token max pad, zero pads act as cross-attn sinks | sink / register results |
| Single-stream: text is concatenated after the image tokens and runs all 42 blocks on every step | Cross-attn on a text KV computed once | text↔image attention probes |
| SiTU-GLU (gated) FFN | Plain GELU MLP | activation sparsity |
| Massive channels (1356, 42; ~1900× median at block 0, prompt-independent) | No comparable channel outliers recorded | sink / register results, adapter coupling |
| FLUX.2 VAE, 32 ch /8, packed 128 ch /16 | Anima VAE, 16 ch | latent-space objectives |
| σ sampling biased toward noise (final interval μ −1.09, median σ ≈ 0.75–0.79); sampler shift 5 | sigmoid(N(0,1)) | σ-band-limited premises |
| Modulation from the timestep alone, one `shared_block_modulation` for all 42 blocks | per-block AdaLN | — (makes AdaLN levers weaker, not stronger) |

## Tier 1 — the recorded cause does not hold on MOSS

### 1. Attention sinks / registers

Measured 2026-10-18: the 10 chat-template tokens are MOSS's attention sink
(`docs/moss/findings/attention_sink_chat_template.md`). Probe, gate and run log:
`bench/moss_sinks/README.md`.

### 2. 2:4 activation sparsity (RT-Lynx)

- **Anima**: `_archive/rt_lynx/`, `_archive/shelved_benches.md` (Numerics).
- **Cause on Anima**: "plain-GELU MLP keeps mlp_down dense (RFE 0.25 vs paper 0.01)";
  all-layer sparsity collapses to noise; SpTC ceiling ~1.75× on sm120.
- **Why MOSS differs**: the FFN is gated, which is the setting the paper's low RFE
  comes from. The SpTC ceiling is hardware and does not change. Training is
  PCIe-bound under block swap, so this is an inference lever only (71–73 s / image at
  1024², 50 steps, uncompiled).
- **Probe**: RFE of a 2:4 mask on the `ffn_down` input per block, from captured
  activations over a few prompts and σ.
- **Gate**: RFE ≤ ~0.05 on most blocks. Then a kernel-level timing before any
  quality arm.
- **Result (2026-10-18, closed)**: gate fails. `ffn_down` RFE 0.185 mean, ≤ 0.05
  on 1 / 42 blocks (Anima 0.25); the other five linears are 0.19–0.39. Flat over σ.
  `_archive/bench/moss_rfe/README.md`.

## Tier 2 — the cause was a measurement; re-measure on MOSS

| Item | Anima result | Why it may differ on MOSS | Probe |
|---|---|---|---|
| DCW SNR-t bias correction (`_archive/dcw/findings.md`) | Paper sign reverses (integrated gap −405.6 at CFG 1); learnable calibrator killed by per-prompt seed variance | Different σ training distribution (noise-heavy) and sampler (shift 5, skip-22 uncond) | The integrated-gap probe, CFG 1 and 3 |
| Oscar latent repulsion (`_archive/bench/oscar/report.md`) | "VAE decode + PE tower is load-bearing" — no latent shortcut | FLUX.2 latent may carry more semantics than Anima's; SFT's glossy pull suggests a diversity deficit worth fixing | Cross-seed latent distance vs decoded-feature distance, Base vs SFT |
| Spectral guidance (`docs/findings/spectral_guidance_no_subspace.md`) | Posterior collapse only in the far σ > 0.85 tail | Shift 5 and noise-heavy training spend more of the run at high σ | The Phase 0 subspace probe on MOSS's grid |
| CTCal (`docs/findings/ctcal_premise_inverted_on_anima.md`) | Cross-attn blurriest at the clean end, peak σ ≈ 0.85 | No cross-attn; the image→text block of joint attention is a different object | Same sharpness-vs-σ probe on the image→text attention block |
| SSC / MAFM velocity magnitude (`_archive/shelved_benches.md`) | Deficit orthogonal and U-shaped | Different target scaling and sampler; cheap to re-check | The velocity-deficit probe |
| MIST (`_archive/shelved_benches.md`) | Guidance delta decays front-loaded | CFG 3 + skip-22 is not plain CFG | Guidance-delta norm over steps |

Low expectation for CTCal, SSC and MIST; they are listed because the probes are cheap.

## Not worth re-opening — the cause holds on MOSS

| Item | Why it still holds |
|---|---|
| AdaLN / pooled-text content steering (`project/finished/mod_guidance/STATUS.md`, `docs/findings/mod_guidance_quality_tag_axis.md`) | "AdaLN can only write DC" is a property of per-channel scale / shift, not of Anima. MOSS's modulation is time-only and shared by all blocks, so it is a weaker lever |
| AsymFlow (`docs/findings/asymflow_parameterization.md`) | MOSS is also a compressed-latent DiT |
| L2P pixel transplant (`docs/findings/l2p_pixel_transfer.md`) | VAE-bound; nothing MOSS-specific helps |
| Self-Flow (`docs/findings/selfflow.md`) | MOSS Base is also a finished denoiser |
| DEFT (`docs/findings/deft_subtractive_coupling_slows_convergence.md`) | Coupling through W0's large singular values; the massive channels likely make it worse |
| HF-residual adapter (`_archive/proposals/hf_residual_adapter.md`) | MOSS round 2 (the LoRA training log, removed from `project/moss_lora/` on 2026-10-18) already found style lives at high σ, the same direction |
| HydraLoRA / chimera routing (`_archive/bench/hydralora/hydralora_proposal.md`) | No routed LoRA on the MOSS line; not a priority |
| FA4 / packed QKV / torch 2.14 (`docs/optimizations/fa4.md`, `docs/findings/torch214_no_win_on_sm120.md`) | Hardware (sm120), not model |
| CJK distillation-target, eval-instrument, reward and curriculum failures (`_archive/shelved_benches.md`, CJK findings) | Data / objective / instrument causes; the model change does not touch them |

## Tier 3 — second sweep (2026-10-21): causes measured on a cross-attn module

The first sweep checked each negative's recorded cause against MOSS's padding, FFN
and VAE. It skipped the findings whose *instrument* was Anima's cross-attention: a
module MOSS does not have (text rows run through all 42 joint blocks on every step).
Those results are not negatives on MOSS; they are unmeasured. Ordered by how much
else hangs off the answer.

### 3. σ staircase — where MOSS resolves the picture

- **Anima**: `docs/findings/sigma_signal_where_anima_resolves.md` — `x0_pred` is a
  recognisable picture at σ ≈ 0.75, essentially final below σ ≈ 0.45 (normalised
  latent-MSE < 0.2); the 20 % error crossover sits at σ ≈ 0.55.
- **Why MOSS differs**: training σ is noise-biased (median ≈ 0.75–0.79, about a
  third of the steps above 0.85) and the sampler runs shift 5 on a snapped grid, so
  the sampler spends a different share of its steps on each side of the knee.
- **What hangs off it**: every σ-gated lever on the line reads an Anima number —
  DAVE's τ window, Spectrum's `flex_window`, the REPA σ taper (`_archive/moss_lora/repa.md`
  § Open: "no target above σ ≈ 0.85" is Anima's shape), and §4's band.
- **Probe**: noise 16 cached latents at the sampler's own snapped σ grid
  (`model.snapped_sigmas`), one bare-DiT forward each (CFG 1, no LoRA), decode
  `x0_pred = x_σ − σ·v`, normalised latent-MSE and the decoded grid per σ. Base and
  SFT. One 1024² batch, minutes.
- **Gate**: none — it is a measurement. The reading moves the knee the other items
  use.
- **Result (2026-10-21)**: the σ knee carries over from Anima; the step it falls on
  does not. On Anima's normalisation, the full-latent error meets Anima's curve down
  to σ ≈ 0.75 ("recognisable" at σ 0.737) and is a little slower below it (20 % at
  0.57, Anima 0.63; "final" at 0.34, Anima 0.45).
  - Layout (4×4-pooled latent) resolves much earlier: recognisable at σ 0.87, 20 % at
    0.80.
  - SFT resolves where Base does: every knee is within 0.012.
  - On the 50-step shift-5 grid, σ 0.87 / 0.80 / 0.75 are steps 22 / 28 / 31 (Anima's
    σ 0.75 is step 14 of 28).
  - For the other items: DAVE τ and Spectrum are step-defined (nothing to move); the
    REPA σ 0.85 taper matches MOSS's shape; §4 reads its window over σ ≥ 0.80.
  - `docs/moss/findings/sigma_staircase.md`; run log `bench/moss_sigma_staircase/README.md`.

### 4. Text drive across σ, and an `xattn_boost` analogue

- **Anima**: `docs/findings/crossattn_self_attn_dominance.md` — the text pathway
  writes the plan early (gated cross-attn residual contribution and
  `‖v_cond − v_uncond‖/‖v_cond‖` both peak at σ = 1, ~0.02 floor below σ ≈ 0.85) and
  self-attn + MLP render it. Shipped lever: `--xattn_boost`
  (`docs/inference/xattn_boost.md`), a σ ≥ 0.85 scale on the cross-attn residual of
  the cond forward, norm-matched. The σ-gated CFG-up arm of the same proposal is
  closed (style collapse) and is not re-proposed here.
- **Why MOSS differs**: the measurement was of a cross-attn module. On MOSS the
  text's write into an image token is the text-key share of one joint attention,
  summed over all 42 blocks, and the text rows themselves change block by block.
  The sink finding (`docs/moss/findings/attention_sink_chat_template.md`) already
  shows that share is dominated by the 10 template tokens, so the caption's own
  share may be small and late rather than front-loaded.
- **Probe**: `bench/moss_sinks/probe_attention.py`'s hooks, split per block and per
  step into template / caption / image key mass on the image queries, plus the
  guidance-delta norm per step (which is also Tier 2's MIST probe). Base, the two
  Space prompts, 4 seeds.
- **Gate**: caption-key mass on image queries is front-loaded (falls by ≥ 3× from
  the first step to σ ≈ 0.5) on most blocks. Then one arm: scale the caption keys'
  attention logits (not the residual — there is no separate text residual) on the
  cond forward over the front-loaded window, norm-matched as `xattn_boost` is,
  judged on weak-tag adherence against the LoRA A/B grid (`src/lora_ab.py`). If the
  mass is flat over σ, the lever has no window and the item closes.

### 5. DCW SNR-t bias, moved up from Tier 2

Tier 2's cheapest probe and the one whose Anima result is most clearly a property
of the training σ distribution (`_archive/dcw/findings.md`: sign flips at CFG 1,
(CFG × aspect)-dependent at CFG 4). The integrated-gap probe is a sampler loop with
one extra forward-branch forward per step; run it first among the Tier 2 items,
at CFG 1 and 3, square and 3:4.

### 6. Postfix learned tokens

- **Anima**: `_archive/shelved_benches.md` (Postfix slot-collapse),
  `_archive/bench/postfix/` — `anima_postfix` learned K slots spliced into the
  zero-pad tail and collapsed to effectively K = 1, a domain prior rather than a
  prompt enhancer.
- **Why MOSS differs**: there is no pad tail to splice into; the only design is
  extra learned rows in the joint sequence, which also sets their RoPE frame
  position. The archive does not record whether the collapse came from the slots
  sharing the pad tail's null-attention role or from the objective itself, so
  this is a weaker entry: it needs a design before a probe.
- **Probe**: K = 4 learned rows after the caption (before the closing template),
  trained on the same set as the LoRA A/B, then the per-slot cosine of the learned
  rows and their per-slot attention mass from §4's probe.
- **Gate**: slot cosine < 0.9 and a measurable per-slot mass. Low expectation;
  listed because §4's probe makes it a one-day check.

### Not worth re-opening, found in the same sweep

| Item | Why it still holds |
|---|---|
| σ-demoted training (`docs/optimizations/sigma_lowres.md`, archived) | Cuts high-σ tokens; MOSS training is PCIe-bound under block swap, so fewer tokens do not shorten the step |
| Embedding inversion (`_archive/bench/inversion/analyze_inversion_and_proposal.md`) | The block-0 cross-attn null-space cause does flip, but the use (an image as condition) is already covered zero-shot by the vision tower (`docs/moss/findings/vision_tower_reference.md`) |
| `channel_stats` dominance metric (`docs/findings/channel_stats_content_independence.md`) | Already re-derived on MOSS with the massive channels tracked separately (`bench/moss_channel_stats/probe_lora_inputs.py`, `docs/moss/channel_scaling.md`) |
| Null-TTA, tag-readback reward, seed-floor / paired-eval, tagger ceiling, PE registers (`_archive/shelved_benches.md`, `docs/findings/`) | Causes are the objective, the instrument or the PE tower, not the DiT |

Anima ideas ported and closed on MOSS itself (not Anima negatives; logged with their
sub-line): relational REPA (`_archive/moss_lora/repa.md`), mixed-lr soup
(`project/moss_lora/soup.md`), caption dropout (`_archive/moss_lora/caption_dropout.md`),
the dual-LoRA corrector's speed claim (`project/moss_lora/correction.md`).

## Order

1. Tier 1: §1 measured (finding above), §2 closed.
2. Tier 3 §3 (σ staircase) first: it sets the σ knee the others read. Then §5, §4.
3. The rest of Tier 2 in any order; §6 only after §4 reads.

Each probe drops a `result.json` under `bench/moss_<item>/results/` and its reading is
logged here.
