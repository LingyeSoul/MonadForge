# Documentation

Index of the `docs/` tree. Each row is a one-line orientation; read the linked doc before working on the thing it describes.

- **Methods** — shipped training algorithms (adapter families).
- **Inference** — training-free runtime stacks: acceleration + sampler-boundary corrections.
- **Experimental** — wired and runnable, but not part of the default stack.
- **Structure** — architecture walkthroughs (how a thing is built, not how to use it).
- **Findings** — negative results and feasibility probes; methods evaluated and *not* adopted.
- **Optimizations** — compiler, kernel, and hardware setup.
- **Guidelines** — user-facing guides and references (multi-language).
- **Proposals** — active design docs for unbuilt work.
- **Architecture notes** — repo-wide planning docs at the top of `docs/`.

## Methods

Shipped training algorithms — adapter families.

| Doc | Description |
|-----|-------------|
| [methods/psoft-integrated-ortholora.md](methods/psoft-integrated-ortholora.md) | OrthoLoRA (Cayley) — SVD-informed init, structural orthogonality via tiny skew-symmetric seeds |
| [methods/hydra-lora.md](methods/hydra-lora.md) | HydraLoRA — MoE multi-head routing (shared-A experts), one cell of the three-axis routing surface in `configs/methods/lora.toml` |
| [methods/timestep_mask.md](methods/timestep_mask.md) | T-LoRA — timestep-dependent rank masking (full rank at noise, rank 1 at clean) |

## Inference

Training-free runtime stacks — acceleration and sampler-boundary corrections. See [inference/](inference/) for the section index.

| Doc | Description |
|-----|-------------|
| [inference/spectrum.md](inference/spectrum.md) | Spectrum — training-free inference acceleration via Chebyshev feature forecasting |
| [inference/spd.md](inference/spd.md) | SPD — training-free multi-resolution inference (low-res early, spectral noise-expansion handoff); Case B fine-tune wired |
| [inference/dcw.md](inference/dcw.md) | DCW — training-free post-step SNR-t bias correction at the sampler boundary (scalar or v4 learnable) |
| [inference/smc_cfg.md](inference/smc_cfg.md) | SMC-CFG — training-free α-adaptive sliding-mode CFG correction in velocity space |
| [inference/cns.md](inference/cns.md) | CNS — training-free SDE noise recolorer (`er_sde` only); shapes injected noise toward unresolved frequency bands |
| [inference/mod-guidance.md](inference/mod-guidance.md) | Modulation guidance — text-conditioned AdaLN steering via distilled `pooled_text_proj` MLP |
| [inference/invert.md](inference/invert.md) | Embedding inversion — optimize text embeddings (full and K-slot reference) |

## Experimental

Wired and runnable, but not part of the default stack — may break or change.

| Doc | Description |
|-----|-------------|
| [experimental/chimera-hydra.md](experimental/chimera-hydra.md) | ChimeraHydra — dual-pool additive MoE (content + freq routers) over disjoint SVD subspaces |
| [experimental/easycontrol.md](experimental/easycontrol.md) | EasyControl — extended self-attn image conditioning; frozen DiT, per-block cond LoRA + scalar gate |
| [experimental/soft_tokens.md](experimental/soft_tokens.md) | Soft Tokens — SoftREPA per-layer × per-t soft text tokens (~1M params); frozen DiT, optional B=1 contrastive |
| [experimental/dpdmd.md](experimental/dpdmd.md) | DP-DMD (Turbo) — diversity-preserved few-step distillation of the CFG=4 teacher into an N-step LoRA student |
| [experimental/directedit_editing_v3.md](experimental/directedit_editing_v3.md) | DirectEdit (v3) — flow-inversion image editing; what's actually wired and runnable |
| [experimental/anima_tagger.md](experimental/anima_tagger.md) | Anima Tagger — multi-label tagger emitting Anima-format tag strings (DirectEdit ψ_src source) |
| [experimental/vr_loss.md](experimental/vr_loss.md) | Variance-reduced FM loss — AsymFlow §5.2 control-variate correction at the loss level |

## Structure

Architecture walkthroughs — how a component is built.

| Doc | Description |
|-----|-------------|
| [structure/anima.md](structure/anima.md) | The Anima model end to end — text conditioning, VAE, DiT block stack, training-step flow |
| [structure/anima-optimizations.md](structure/anima-optimizations.md) | Non-obvious perf/compile decisions and the *why* behind each |
| [structure/lora.md](structure/lora.md) | Plain LoRA inside Anima — the scaffolding every variant stacks on |
| [structure/ortholora.md](structure/ortholora.md) | PSOFT-integrated OrthoLoRA — exactly-orthogonal bases from SVD + skew-symmetric seeds |
| [structure/hydralora.md](structure/hydralora.md) | HydraLoRA — layer-local MoE over LoRA up-heads |
| [structure/chimera-hydra.md](structure/chimera-hydra.md) | ChimeraHydra — dual-pool additive MoE on the OrthoHydra basis |
| [structure/timestep-mask.md](structure/timestep-mask.md) | T-LoRA — the one-line timestep→rank masking change |
| [structure/modulation.md](structure/modulation.md) | Pooled-text modulation — max-pooled caption summary into the AdaLN stack |
| [structure/spectrum.md](structure/spectrum.md) | Spectrum — Chebyshev feature forecasting at inference (run-or-predict per step) |
| [structure/dpdmd.md](structure/dpdmd.md) | DP-DMD — structural walkthrough of the diversity-preserved distillation |

## Findings

Negative results and feasibility probes — methods evaluated and not adopted.

| Doc | Description |
|-----|-------------|
| [findings/selfflow.md](findings/selfflow.md) | Self-Flow rep-loss — falsified on the frozen backbone (no information asymmetry to distill) |
| [findings/sigma_signal_where_anima_resolves.md](findings/sigma_signal_where_anima_resolves.md) | Where Anima's denoising signal lives — the σ ≈ 0.75 → 0.45 → 0 resolution staircase |
| [findings/spectral_guidance_no_subspace.md](findings/spectral_guidance_no_subspace.md) | Spectral Guidance — no low-rank guidable subspace on Anima (Phase 0 NO-GO) |
| [findings/asymflow_parameterization.md](findings/asymflow_parameterization.md) | AsymFlow rank-asymmetric velocity parameterization — assessed, not worth reviving |
| [findings/l2p_pixel_transfer.md](findings/l2p_pixel_transfer.md) | L2P latent→pixel transfer — shelved on the Anima budget |
| [findings/freetext_text_rendering.md](findings/freetext_text_rendering.md) | FreeText — localization works, native OOD text rendering doesn't |
| [findings/turbo_fei_band_deficit_falsified.md](findings/turbo_fei_band_deficit_falsified.md) | Turbo FEI band-deficit CA reweighting — why it was plausible, why it falsified |
| [findings/agsm_reward_premise_holds.md](findings/agsm_reward_premise_holds.md) | AGSM reward premise — relative FM-ranking survives where absolute FM-MSE doesn't |
| [findings/mod_guidance_quality_tag_axis.md](findings/mod_guidance_quality_tag_axis.md) | Mod-guidance pooled-text "quality axis" — demoted to geometry-only; replaced by image-space attribution |
| [findings/channel_stats_content_independence.md](findings/channel_stats_content_independence.md) | Channel-scaling calibration is content-agnostic on Anima (A/B result) |

## Optimizations

Compiler, kernel, hardware setup, and training-time optimizer geometry.

| Doc | Description |
|-----|-------------|
| [optimizations/for_compile.md](optimizations/for_compile.md) | Changes from sd-scripts for torch.compile / dynamo |
| [optimizations/channel_scaling.md](optimizations/channel_scaling.md) | Channel Scaling — SmoothQuant-style per-channel LoRA gradient rebalance (on by default, α=0.5; inert on frozen-basis ortho variants) |
| [optimizations/fa4.md](optimizations/fa4.md) | Flash Attention 4 — why it was evaluated and removed |
| [optimizations/adamw_fused.md](optimizations/adamw_fused.md) | AdamW8bit → fused AdamW — why bitsandbytes was dropped |
| [optimizations/hydra_analysis.md](optimizations/hydra_analysis.md) | HydraLoRA — nsys-driven optimization pass (2026-05-03) |

## Guidelines

User-facing guides and references.

| Doc | Description |
|-----|-------------|
| [guidelines/training.md](guidelines/training.md) | Training reference — LoRA variants, caption shuffle, masked loss, dataset config |
| [guidelines/inference.md](guidelines/inference.md) | Inference reference — flags, prompt files, LoRA format conversion |
| [guidelines/difference_between_comfy.md](guidelines/difference_between_comfy.md) | anima_lora vs ComfyUI implementation differences |
| [guidelines/guidebook.md](guidelines/guidebook.md) | Comprehensive guide (English) |
| [guidelines/가이드북.md](guidelines/가이드북.md) | 종합 가이드 (Korean) |
| [guidelines/ガイドブック.md](guidelines/ガイドブック.md) | 総合ガイド (Japanese) |
| [guidelines/指南书.md](guidelines/指南书.md) | 综合指南 (Chinese) |

## Proposals

Active design docs for unbuilt work.

| Doc | Description |
|-----|-------------|
| [proposal/postfix_residual_for_directedit.md](proposal/postfix_residual_for_directedit.md) | Image-conditional postfix as DirectEdit's ψ_src residual carrier |
| [proposal/postfix_residual_per_image_inversion.md](proposal/postfix_residual_per_image_inversion.md) | Per-image inversion as a probe of the residual manifold |

## Architecture notes

Repo-wide planning docs.

| Doc | Description |
|-----|-------------|
| [multi_model_support.md](multi_model_support.md) | Terrain map for adding a second image model (e.g. Z-Image-Base) alongside Anima — exploratory |
| [separation_plan.md](separation_plan.md) | Working plan to split inference into a standalone `../anima_inference` |
