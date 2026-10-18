# MOSS-Image — which Anima negatives are worth re-opening

Status: PROPOSED 2026-10-18. Nothing here has been measured on MOSS. Each entry is
an Anima negative result whose recorded cause is checked against MOSS-Image's
architecture (`library/moss/CLAUDE.md`, `project/moss_lora/plan.md`). An entry is
re-opened only when the recorded cause depends on an Anima-specific fact that MOSS
does not share. Each re-opened entry gets a cheap premise probe and a gate before
any port.

## The MOSS facts that change a cause

| MOSS fact | Anima counterpart | Causes it can flip |
|---|---|---|
| Text is never padded; the DiT sees only valid tokens | 512-token max pad, zero pads act as cross-attn sinks | sink / register results |
| Single-stream: text is concatenated after the image tokens and runs all 42 blocks on every step | Cross-attn on a text KV computed once | text-side compute, text↔image attention probes |
| SiTU-GLU (gated) FFN | Plain GELU MLP | activation sparsity |
| Massive channels (1356, 42; ~1900× median at block 0, prompt-independent) | No comparable channel outliers recorded | sink / register results, adapter coupling |
| FLUX.2 VAE, 32 ch /8, packed 128 ch /16 | Anima VAE, 16 ch | latent-space objectives |
| Trained on verbatim text transcription; renders quoted CJK out of the box | DiT composes glyphs in pixels; rare kanji fail | the whole CJK line |
| σ sampling biased toward noise (final interval μ −1.09, median σ ≈ 0.75–0.79); sampler shift 5 | sigmoid(N(0,1)) | σ-band-limited premises |
| Modulation from the timestep alone, one `shared_block_modulation` for all 42 blocks | per-block AdaLN | — (makes AdaLN levers weaker, not stronger) |

## Tier 1 — the recorded cause does not hold on MOSS

### 1. Attention sinks / registers

Started 2026-10-18. Motivation, probe, gate and readings: `bench/moss_sinks/README.md`.

### 2. CJK text rendering and OCR captions

- **Anima**: `project/finished/cjk_aware_anima/findings.md`,
  `project/finished/cjk_renderable_anima/findings.md`,
  `docs/findings/freetext_text_rendering.md`.
- **Cause on Anima**: "the DiT composes in pixels, not in row space"; rare kanji names
  smear (recovery_attn 0.90 → 0.45 under a rank-16 ext-gated LoRA); OCR quote rows in
  manga captions were inert ("the presence tag is the address").
- **Why MOSS differs**: the techniques do not need to be ported, because the goal is
  native. Quoted CJK in the main subject rendered correctly on day one (ja / zh / ko).
  The training captions transcribe visible text verbatim (report p.18), so the OCR
  text clause (`--combine_ocr`) that was inert on Anima is the format MOSS was trained
  on. Unverified: JA and KO are never mentioned in the report, and the day-one read is
  five prompts at one seed.
- **Probe**: (a) a JA / KO / rare-kanji prompt sheet (bubble text, signage, names from
  the Anima CJK eval set), Base and SFT, 4 seeds, read per glyph. (b) A manga LoRA
  A/B: the same set with and without the OCR text clause in the layout-JSON caption.
- **Gate**: (a) per-glyph read on held-out kana / hangul / rare kanji at least matches
  the zh rate seen so far. (b) the OCR-clause arm renders the set's bubble text more
  legibly and does not write captions into non-text regions.

### 3. 2:4 activation sparsity (RT-Lynx)

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

### 4. Text-token compute reuse across steps

- **Anima**: `docs/findings/traj_stats_front_loaded_commitment.md` (committed-token
  reuse, ceiling 25.5 % of token-steps).
- **Cause on Anima**: the ceiling was on image tokens; text was a cross-attn KV
  computed once, so text had no per-step cost to save.
- **Why MOSS differs**: every text token pays full QKV + FFN in all 42 blocks on every
  step. At 1776×1184 a 1.9k-token prompt adds ≈ +24 % linear and +53 % attention FLOPs
  (`project/moss_lora/plan.md`). A text token's inputs are the shared, time-only
  modulation and the joint attention over image tokens; if its hidden states drift
  little between steps, later steps could reuse the text rows' K / V and skip their
  FFN.
- **Probe**: per block, step-to-step relative change of the text-token hidden states
  (and of their K / V) over a 50-step run, cond and uncond (uncond skips block 22), at
  a short tag prompt and a long layout-JSON prompt.
- **Gate**: relative change below image-token drift by a wide margin over a contiguous
  late-step range. Then a reuse arm against the plain run at matched seeds, read by
  eye, and timed alongside Spectrum (forecast steps already skip all blocks, so the
  saving only lands on full steps).

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
| HF-residual adapter (`_archive/proposals/hf_residual_adapter.md`) | MOSS round 2 (`project/moss_lora/lora.md`) already found style lives at high σ, the same direction |
| HydraLoRA / chimera routing (`_archive/bench/hydralora/hydralora_proposal.md`) | No routed LoRA on the MOSS line; not a priority |
| FA4 / packed QKV / torch 2.14 (`docs/optimizations/fa4.md`, `docs/findings/torch214_no_win_on_sm120.md`) | Hardware (sm120), not model |
| CJK distillation-target, eval-instrument, reward and curriculum failures (`_archive/shelved_benches.md`, CJK findings) | Data / objective / instrument causes; the model change does not touch them |

## Order

1. Tier 1 §1 (sinks / registers) and §4 (text-token drift). Both are one probe job
   each on hooks the DAVE probe already uses, and §4's result also sizes the
   saving for §3.
2. Tier 1 §2(a), the CJK prompt sheet — no new code, `scripts/moss/generate.py` jobs through the daemon.
3. Tier 1 §3 (RFE probe), then Tier 2 in any order.

Each probe drops a `result.json` under `bench/moss_<item>/results/` and its reading is
logged here.
