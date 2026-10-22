# MOSS-Image — which Anima negatives are worth re-opening

Status (proposed 2026-10-18, second sweep 2026-10-21): §1 measured, §2 closed, §3
measured; §4–§6 and Tier 2 unmeasured. Each entry is an Anima negative whose recorded
cause is checked against MOSS-Image's architecture (`library/moss/CLAUDE.md`); it is
re-opened only when that cause depends on an Anima fact MOSS does not share, and gets a
cheap premise probe and a gate before any port.

## The MOSS facts that change a cause

| MOSS fact | Anima counterpart | Causes it can flip |
|---|---|---|
| Text is never padded; the DiT sees only valid tokens | 512-token max pad, zero pads act as cross-attn sinks | sink / register results |
| Single-stream: text is concatenated after the image tokens and runs all 42 blocks on every step | Cross-attn on a text KV computed once | text↔image attention probes |
| SiTU-GLU (gated) FFN | Plain GELU MLP | activation sparsity |
| Massive channels (1356, 42; ~1900× median at block 0, prompt-independent) | No comparable channel outliers recorded | sink / register results, adapter coupling |
| FLUX.2 VAE, 32 ch /8, packed 128 ch /16 | Anima VAE, 16 ch | latent-space objectives |
| σ sampling biased toward noise (final interval μ −1.09, median σ ≈ 0.75–0.79); sampler shift 5 | sigmoid(N(0,1)) | σ-band-limited premises |
| Modulation from the timestep alone, one `shared_block_modulation` for all 42 blocks | per-block AdaLN | none (AdaLN levers are weaker) |

## Tier 1 — the recorded cause does not hold on MOSS

### 1. Attention sinks / registers

Measured 2026-10-18: the 10 chat-template tokens are MOSS's attention sink
(`docs/moss/findings/attention_sink_chat_template.md`; run log `bench/moss_sinks/README.md`).

### 2. 2:4 activation sparsity (RT-Lynx), closed

- **Anima** (`_archive/rt_lynx/`, `_archive/shelved_benches.md`): the plain-GELU MLP keeps
  `mlp_down` dense (RFE 0.25 vs paper 0.01); SpTC ceiling ~1.75× on sm120.
- **MOSS premise**: the FFN is gated, the paper's setting. Inference-only lever (training
  is PCIe-bound under block swap; 71–73 s / image at 1024², 50 steps, uncompiled).
- **Gate**: 2:4-mask RFE on the `ffn_down` input ≤ ~0.05 on most blocks.
- **Result (2026-10-18)**: fails. `ffn_down` RFE 0.185 mean, ≤ 0.05 on 1 / 42 blocks; the
  other five linears 0.19–0.39; flat over σ. `_archive/bench/moss_rfe/README.md`.

## Tier 2 — the cause was a measurement; re-measure on MOSS

| Item | Anima result | Why it may differ on MOSS | Probe |
|---|---|---|---|
| Oscar latent repulsion (`_archive/bench/oscar/report.md`) | VAE decode + PE tower is load-bearing; no latent shortcut | FLUX.2 latent may carry more semantics; SFT's glossy pull suggests a diversity deficit | Cross-seed latent vs decoded-feature distance, Base vs SFT |
| Spectral guidance (`docs/findings/spectral_guidance_no_subspace.md`) | Posterior collapse only in the σ > 0.85 tail | Shift 5 and noise-heavy training spend more of the run at high σ | The Phase 0 subspace probe on MOSS's grid |
| CTCal (`docs/findings/ctcal_premise_inverted_on_anima.md`) | Cross-attn blurriest at the clean end, peak σ ≈ 0.85 | No cross-attn; the image→text block of joint attention is a different object | Same sharpness-vs-σ probe on the image→text block |
| SSC / MAFM velocity magnitude (`_archive/shelved_benches.md`) | Deficit orthogonal and U-shaped | Different target scaling and sampler | The velocity-deficit probe |
| MIST (`_archive/shelved_benches.md`) | Guidance delta decays front-loaded | CFG 3 + skip-22 is not plain CFG | Guidance-delta norm over steps (§4's probe covers it) |

Low expectation for CTCal, SSC and MIST; the probes are cheap. DCW moved to §5.

## Not worth re-opening — the cause holds on MOSS

| Item | Why it still holds |
|---|---|
| AdaLN / pooled-text content steering (`project/finished/mod_guidance/STATUS.md`, `docs/findings/mod_guidance_quality_tag_axis.md`) | "AdaLN can only write DC" is a property of per-channel scale / shift; MOSS's modulation is time-only and shared, a weaker lever |
| AsymFlow (`docs/findings/asymflow_parameterization.md`) | MOSS is also a compressed-latent DiT |
| L2P pixel transplant (`docs/findings/l2p_pixel_transfer.md`) | VAE-bound |
| Self-Flow (`docs/findings/selfflow.md`) | MOSS Base is also a finished denoiser |
| DEFT (`docs/findings/deft_subtractive_coupling_slows_convergence.md`) | Coupling through W0's large singular values; the massive channels likely make it worse |
| HF-residual adapter (`_archive/proposals/hf_residual_adapter.md`) | MOSS LoRA round 2 already found style lives at high σ, the same direction |
| HydraLoRA / chimera routing (`_archive/bench/hydralora/hydralora_proposal.md`) | No routed LoRA on the MOSS line |
| FA4 / packed QKV / torch 2.14 (`docs/optimizations/fa4.md`, `docs/findings/torch214_no_win_on_sm120.md`) | Hardware (sm120), not model |
| CJK distillation-target, eval-instrument, reward and curriculum failures (`_archive/shelved_benches.md`, CJK findings) | Data / objective / instrument causes |
| σ-demoted training (`docs/optimizations/sigma_lowres.md`, archived) | Cuts high-σ tokens; MOSS training is PCIe-bound, so fewer tokens do not shorten the step |
| Embedding inversion (`_archive/bench/inversion/analyze_inversion_and_proposal.md`) | The cause flips, but the use (image as condition) is covered zero-shot by the vision tower (`docs/moss/findings/vision_tower_reference.md`) |
| `channel_stats` dominance metric (`docs/findings/channel_stats_content_independence.md`) | Already re-derived on MOSS (`bench/moss_channel_stats/probe_lora_inputs.py`, `docs/moss/channel_scaling.md`) |
| Null-TTA, tag-readback reward, seed-floor / paired-eval, tagger ceiling, PE registers (`_archive/shelved_benches.md`, `docs/findings/`) | Causes are the objective, the instrument or the PE tower |

Ported and closed on MOSS itself (logged with their sub-line): relational REPA
(`_archive/moss_lora/repa.md`), mixed-lr soup (`project/moss_lora/done/soup.md`), caption
dropout (`_archive/moss_lora/caption_dropout.md`), the dual-LoRA corrector's speed claim
(`project/moss_lora/correction.md`).

## Tier 3 — results whose instrument was Anima's cross-attention

MOSS has no cross-attention module (text rows run all 42 joint blocks every step), so
these are unmeasured on MOSS rather than negative.

### 3. σ staircase, measured

- **Anima** (`docs/findings/sigma_signal_where_anima_resolves.md`): `x0_pred` recognisable
  at σ ≈ 0.75, final below σ ≈ 0.45; 20 % error crossover at σ ≈ 0.55.
- **Result (2026-10-21)**: the σ knee carries over, the step it falls on does not.
  Full-latent error meets Anima's curve to σ ≈ 0.75 (recognisable at 0.737), a little
  slower below (20 % at 0.57 vs Anima 0.63; final at 0.34 vs 0.45). Layout (4×4-pooled)
  resolves earlier: recognisable at σ 0.87, 20 % at 0.80. SFT within 0.012 of Base on every
  knee. On the 50-step shift-5 grid σ 0.87 / 0.80 / 0.75 are steps 22 / 28 / 31 (Anima's
  σ 0.75 is step 14 of 28).
- **Consequences**: DAVE τ and Spectrum are step-defined (nothing to move); the REPA σ 0.85
  taper matches MOSS's shape; §4 reads its window over σ ≥ 0.80.
  `docs/moss/findings/sigma_staircase.md`; run log `bench/moss_sigma_staircase/README.md`.

### 4. Text drive across σ, and an `xattn_boost` analogue

- **Anima** (`docs/findings/crossattn_self_attn_dominance.md`): the text pathway writes the
  plan early (cross-attn contribution and `‖v_cond − v_uncond‖/‖v_cond‖` peak at σ = 1,
  ~0.02 floor below σ ≈ 0.85). Shipped lever `--xattn_boost` (`docs/inference/xattn_boost.md`),
  a norm-matched σ ≥ 0.85 scale on the cond forward's cross-attn residual. The σ-gated
  CFG-up arm is closed (style collapse) and not re-proposed.
- **Why MOSS differs**: text writes into an image token through its key share of joint
  attention over 42 blocks, and the sink finding shows that share is dominated by the 10
  template tokens; the caption's own share may be small and late.
- **Probe**: `bench/moss_sinks/probe_attention.py`'s hooks, per block and step:
  template / caption / image key mass on image queries, plus the guidance-delta norm per
  step. Base, the two Space prompts, 4 seeds.
- **Gate**: caption-key mass falls ≥ 3× from the first step to σ ≈ 0.5 on most blocks. Then
  one arm: scale the caption keys' attention logits on the cond forward over that window,
  norm-matched, judged on weak-tag adherence against the LoRA A/B grid (`src/lora_ab.py`).
  Flat mass over σ closes the item.

### 5. DCW SNR-t bias (moved up from Tier 2)

- **Anima** (`_archive/dcw/findings.md`): paper sign reverses (integrated gap −405.6 at CFG
  1), (CFG × aspect)-dependent at CFG 4; learnable calibrator killed by per-prompt seed
  variance. Most clearly a property of the training σ distribution, which MOSS changes.
- **Probe**: the integrated-gap sampler loop (one extra forward per step) at CFG 1 and 3,
  square and 3:4.

### 6. Postfix learned tokens

- **Anima** (`_archive/shelved_benches.md`, `_archive/bench/postfix/`): `anima_postfix`
  K slots spliced into the zero-pad tail collapsed to effectively K = 1, a domain prior.
- **Why MOSS differs**: no pad tail; the only design is extra learned rows in the joint
  sequence (which also sets their RoPE frame position). The archive does not record
  whether the collapse came from the pad tail's null-attention role or the objective.
- **Probe**: K = 4 learned rows after the caption (before the closing template), trained
  on the LoRA A/B set; per-slot cosine and per-slot attention mass from §4's probe.
- **Gate**: slot cosine < 0.9 and a measurable per-slot mass. Low expectation.

## Order

§5, then §4, then the rest of Tier 2 in any order; §6 only after §4 reads. Each probe
drops a `result.json` under `bench/moss_<item>/results/` and its reading is logged here.
