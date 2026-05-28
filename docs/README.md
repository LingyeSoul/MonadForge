# Documentation

## Methods

Training and inference algorithms.

| Doc | Description |
|-----|-------------|
| [methods/hydra-lora.md](methods/hydra-lora.md) | HydraLoRA — MoE multi-head routing (shared-A experts), one cell of the three-axis routing surface in `configs/methods/lora.toml` |
| [experimental/fera.md](experimental/fera.md) | FeRA — independent-A stacked experts with a single global FEI router; another cell of the same three-axis surface |
| [methods/psoft-integrated-ortholora.md](methods/psoft-integrated-ortholora.md) | OrthoLoRA (Cayley) — SVD-informed init, structural orthogonality |
| [methods/timestep_mask.md](methods/timestep_mask.md) | T-LoRA — timestep-dependent rank masking |
| [methods/reft.md](methods/reft.md) | ReFT — block-level residual-stream intervention |
| [experimental/postfix.md](experimental/postfix.md) | Postfix (cond+ortho) — caption-conditional postfix with Cayley-rotated frozen SVD basis |
| [methods/mod-guidance.md](methods/mod-guidance.md) | Modulation guidance — text-conditioned AdaLN steering via distilled MLP |
| [methods/invert.md](methods/invert.md) | Embedding inversion — optimize text embeddings (full and K-slot reference) |
| [methods/spectrum.md](methods/spectrum.md) | Spectrum — training-free inference acceleration via Chebyshev forecasting |
| [experimental/dmd2-decoupled.md](experimental/dmd2-decoupled.md) | Decoupled DMD2 (Turbo) — few-step distillation of the 28-step teacher into a 4-step LoRA student (CA spear + DM shield); structural walkthrough in `structure/dmd2-decoupled.md` |

## Findings

Negative results and feasibility probes — methods evaluated and not adopted.

| Doc | Description |
|-----|-------------|
| [findings/selfflow.md](findings/selfflow.md) | Self-Flow rep-loss — falsified on the frozen Anima backbone (no information asymmetry to distill; alignment target trivially satisfiable) |

## Optimizations

Compiler, kernel, and hardware setup.

| Doc | Description |
|-----|-------------|
| [optimizations/for_compile.md](optimizations/for_compile.md) | Changes from sd-scripts for torch.compile / dynamo |
| [optimizations/fa4.md](optimizations/fa4.md) | Flash Attention 4 — why it was evaluated and removed |
| [optimizations/adamw_fused.md](optimizations/adamw_fused.md) | AdamW8bit → fused AdamW — why bitsandbytes was dropped |

## Guidelines

User-facing guides and references.

| Doc | Description |
|-----|-------------|
| [guidelines/training.md](guidelines/training.md) | Training reference — LoRA variants, caption shuffle, masked loss, dataset config |
| [guidelines/inference.md](guidelines/inference.md) | Inference reference — flags, P-GRAFT, prompt files, LoRA format conversion |
| [guidelines/graft-guideline.md](guidelines/graft-guideline.md) | GRAFT curation guideline |
| [guidelines/difference_between_comfy.md](guidelines/difference_between_comfy.md) | anima_lora vs ComfyUI implementation differences |
| [guidelines/가이드북.md](guidelines/가이드북.md) | 한국어 종합 가이드 (Korean comprehensive guide) |
