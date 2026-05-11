# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Anima — LoRA/T-LoRA training and inference pipeline for the Anima diffusion model (DiT-based, flow-matching). Supports several adapter families (LoRA / OrthoLoRA / T-LoRA / HydraLoRA / ReFT / postfix-prefix / IP-Adapter / EasyControl) selectable via method config + hardware preset.

## Setup

```bash
uv sync                    # Install dependencies (Python 3.13)
hf auth login              # Authenticate for model downloads
make download-models       # Download DiT, text encoder, VAE from HuggingFace
# Training images go in image_dataset/ with .txt caption sidecars
make preprocess            # Resize → post_image_dataset/resized/, cache → post_image_dataset/lora/
```

## Commands

Both `make` (Unix) and `python tasks.py` (cross-platform) are supported. The examples below show both forms. The `Makefile` is a thin catch-all dispatcher — every target forwards to `python tasks.py <target> $(ARGS)`. **`tasks.py` is the source of truth**; per-domain command implementations live in `scripts/tasks/{training,inference,preprocess,masking,gui,downloads,utilities,tagger,dcw}.py`, with the experimental commands (`exp-*`) in `scripts/experimental_tasks/{training,inference}.py`. Don't grep the Makefile for a target's recipe — look in `scripts/tasks/` (or `scripts/experimental_tasks/` for `exp-*`).

```bash
# Training (run from anima_lora/)
# Each training invocation selects a method + hardware preset. Method settings win
# over preset settings on overlap (e.g. postfix forces blocks_to_swap=0).
# Method files in configs/methods/: lora.toml, postfix.toml, ip_adapter.toml,
# easycontrol.toml, soft_tokens.toml. Variants are toggle blocks inside them —
# uncomment the target block to switch:
#   lora.toml             — classic LoRA / OrthoLoRA / T-LoRA / HydraLoRA / ReFT
#                           (default stacks LoRA + OrthoLoRA + T-LoRA + ReFT)
#   postfix.toml          — postfix / postfix_exp / postfix_func / postfix_sigma / prefix
#   ip_adapter.toml       — decoupled image cross-attention (PE-Core resampler)
#   easycontrol.toml      — extended self-attn image conditioning (per-block cond LoRA)
#   soft_tokens.toml      — SoftREPA-style per-layer × per-t soft text tokens (frozen DiT)
make lora                   # LoRA family (methods/lora.toml + presets.toml[default])
python tasks.py lora        # Same, works on Windows too
make lora PRESET=low_vram   # Override preset: methods/lora.toml + presets.toml[low_vram]
make lora PRESET=fast_16gb  # Override preset: methods/lora.toml + presets.toml[fast_16gb]
make lora PRESET=half       # Override preset: methods/lora.toml + presets.toml[half] (sample_ratio=0.5)
# Experimental methods are exposed under exp-* (postfix, ip-adapter,
# easycontrol). They may produce broken output, change without notice, or be
# removed.
make exp-postfix                # Postfix/prefix family (methods/postfix.toml)
make exp-ip-adapter             # IP-Adapter image cross-attention (methods/ip_adapter.toml)
                                # Reuses LoRA paths: source image_dataset/, cache post_image_dataset/lora/
make exp-ip-adapter-preprocess  # Alias for `make preprocess` + `make preprocess-pe`
make exp-easycontrol            # EasyControl image conditioning (methods/easycontrol.toml)
                                # Source: easycontrol-dataset/  Cache: post_image_dataset/easycontrol/
make exp-easycontrol-preprocess # Resize + VAE + text caches into post_image_dataset/easycontrol/
make exp-soft-tokens            # SoftREPA-style per-layer × per-t soft tokens (training-only v1)

# GUI-friendly per-variant path (configs/gui-methods/<variant>.toml — clean,
# self-contained, no toggle blocks). Intended for basic users who don't want
# to hand-edit methods/lora.toml's comment-toggle system.
make lora-gui GUI_PRESETS=tlora                                # gui-methods/tlora.toml + preset default
make lora-gui GUI_PRESETS=hydralora_experimental PRESET=low_vram  # override preset as usual
python tasks.py lora-gui hydralora_sigma                       # Windows; variant can also be 1st positional arg

# Modulation guidance distillation
make distill-mod           # Train pooled_text_proj MLP (text → AdaLN modulation)

# Embedding inversion (archived — run scripts directly under archive/inversion/)
# python archive/inversion/invert_embedding.py ...
# python archive/inversion/invert_reference.py ...
# python archive/inversion/interpret_inversion.py ...

# Inference (test with most recent output)
make test
make test-mod              # Test with modulation guidance (pooled_text_proj)
make test-hydra            # HydraLoRA router-live (anima_hydra*_moe.safetensors)
make test-merge            # Inference with a merged/baked DiT (no adapter loaded)
make test SPECTRUM=1       # Spectrum-accelerated inference (~3.75x speedup)
make test-dcw              # Latest LoRA + DCW scalar bias correction (--dcw, λ=−0.015)
make test-dcw-v4           # Latest LoRA + DCW v4 learnable calibrator (auto-resolves fusion_head.safetensors)
make test-spectrum-dcw     # Spectrum + DCW scalar composed
make test-dcw-v4-spectrum  # Spectrum + DCW v4 composed

# DCW v4 calibration (one-shot per LoRA checkpoint)
make dcw                   # Sample 5 aspect buckets + train fusion head (~3-5h on a 5060 Ti)
make dcw-train             # Train-only on existing pool under bench/dcw/results/ (~30s)

# Experimental inference (matched to make exp-* training)
make exp-test-prefix           # Test with prefix tuning
make exp-test-postfix          # Test with postfix tuning
make exp-test-postfix-exp      # Test with postfix tuning (exp variant)
make exp-test-postfix-func     # Test with postfix tuning (func variant)
make exp-test-ip REF_IMAGE=... # IP-Adapter inference (image-conditioned)
make exp-test-easycontrol REF_IMAGE=...  # EasyControl inference (image-conditioned)
make exp-test-ref              # Inference with a learned prefix-slot weight (--prefix_weight)
make exp-test-directedit PROMPT='double peace'  # DirectEdit on random source image
                                                # (anima_tagger if present, else wd-tagger seeds prompt_src;
                                                # PROMPT becomes the edit instruction)
make exp-test-directedit-dry                    # DirectEdit reconstruction sanity check
                                                # (ψ_tar == ψ_src; output should reconstruct the source)

# GUI (PySide6 — config editing, IP-Adapter / EasyControl preprocess+train, dataset browsing)
make gui
python tasks.py gui        # Windows
make gui-shortcut          # Create "Anima LoRA GUI.lnk" on the Windows desktop (no console window)

# Masking (for masked loss training)
# Outputs under masks/{sam,mit,merged}/. Subsets auto-pick masks/merged/ when
# it exists, falling back to masks/sam/ then masks/mit/.
make mask                  # Generate SAM3 + MIT masks under masks/{sam,mit}/, then merge → masks/merged/
make mask-sam              # SAM3 only → masks/sam/
make mask-mit              # MIT/ComicTextDetector only → masks/mit/
make mask-clean            # Remove masks/

# Merge LoRA into DiT (standalone ComfyUI-compatible checkpoint)
make merge ADAPTER_DIR=output/ckpt                    # bake latest bakeable LoRA in dir
make merge ADAPTER_DIR=output/ckpt MULTIPLIER=0.8     # scale strength
python scripts/merge_to_dit.py --adapter path/to/lora.safetensors --allow-partial
# Supports: LoRA / OrthoLoRA / DoRA / T-LoRA. Refuses ReFT / Hydra moe / postfix
# / prefix by default (they can't be folded into Linear weights); --allow-partial
# drops them and bakes only the LoRA portion.

# Batch
make comfy-batch           # Run ComfyUI batch workflow

# Debugging + tests
make print-config METHOD=lora PRESET=default   # Dump merged config chain (base→preset→method→CLI)
make test-unit                                  # pytest on tests/ (smoke, config, loss/network registries)
                                                # Use existing tests as templates: test_smoke.py, test_network_registry.py,
                                                # test_lora_custom_autograd.py, test_loss_registry.py, test_config.py.
make export-logs RUN=...                        # Export TensorBoard run to JSON (scripts/export_logs_json.py)

# Maintenance
make update                # Update from a GitHub release (preserves datasets/output/models, prompts on
                           # config conflicts, runs uv sync). Pass --dry-run / --version v1.0 / --no-sync.

# Linting
ruff check . --fix && ruff format .
```

All training invocations use `accelerate launch --mixed_precision bf16` with `train.py --method <name> --preset <name>`. Override any config value from CLI: `--network_dim 32 --max_train_epochs 64`. Override preset with `PRESET=low_vram make lora` or `python tasks.py lora` plus `PRESET` env.

On Windows, use `python tasks.py <command>` instead of `make <command>`. Extra args are forwarded: `python tasks.py lora --network_dim 32`.

## Key entry points

| File | Purpose |
|------|---------|
| `train.py` | `AnimaTrainer` class — main training loop via HF Accelerate |
| `inference.py` | Standalone image generation (`--help` for all flags) |
| `networks/spectrum.py` | Spectrum inference acceleration (Chebyshev feature forecasting) |
| `gui/` | PySide6 GUI package: config editing with presets, IP-Adapter / EasyControl preprocess+train tabs, dataset browser, training monitor |
| `tasks.py` | Cross-platform task runner (Windows-compatible Makefile alternative). Source of truth for every `make` target. |
| `scripts/tasks/` | Per-domain task implementations (`training`, `inference`, `preprocess`, `masking`, `gui`, `downloads`, `utilities`, `tagger`, `dcw`) — where command bodies actually live; `_common.py` holds shared helpers. |
| `scripts/experimental_tasks/` | Bodies for the `exp-*` commands (postfix, ip-adapter, easycontrol training + their `exp-test-*` inference). Reuses helpers from `scripts/tasks/_common.py`. |

Deep-dives in `docs/methods/` (shipped): `dcw.md`, `hydra-lora.md`, `invert.md`, `mod-guidance.md`, `psoft-integrated-ortholora.md`, `reft.md`, `spectrum.md`, `timestep_mask.md`. Experimental method docs live under `docs/experimental/`: `anima_tagger.md`, `directedit_editing_v3.md`, `easycontrol.md`, `ip-adapter.md`, `postfix-sigma.md`, `prefix-tuning.md`, `soft_token_idea.md`, `soft_tokens.md`. Active proposals are in `docs/proposal/`: `img2emb_plan.md`, `ip-adapter-0502.md`, `orthogonal_postfix.md`. Retired/superseded design docs (DCW v1–v4 proposals, DirectEdit v2, ip-adapter_plan) live under `archive/proposals/` and `archive/dcw-learnable-calibrator/proposals/`. APEX (proposal + bench + method + adapter) is archived under `archive/apex/`.

## Config flow

Training is config-driven via a three-layer chain: `base.toml → presets.toml[<preset>] → methods/<method>.toml → CLI args`. Method settings win over preset settings on overlap, so a method can force its own hardware requirements (e.g. postfix forces `blocks_to_swap=0`).

Layout:
- `configs/base.toml` — shared infrastructure (model paths, optimizer, compile flags, etc.) AND the default LoRA dataset blueprint (`[general]` + `[[datasets]]` + `[[datasets.subsets]]`). LoRA reads resized images from `post_image_dataset/resized/` with caches redirected to `post_image_dataset/lora/` via `cache_dir`. Captions live in `image_dataset/` (master) — TE caching reads `.txt` from there, training reads only the cached prompt embeddings. The dataset sections are consumed by `BlueprintGenerator` and skipped by the flat method+preset merge chain (see `_DATASET_CONFIG_SECTIONS` in `library/train_util.py`). Override with `--dataset_config <path>` when you need a different blueprint.
- `configs/presets.toml` — all hardware profiles in one file as TOML sections: `[default]`, `[fast_16gb]`, `[low_vram]` (also serves as Windows 8GB), `[half]` (experiment preset — sets `sample_ratio=0.5` for every subset via the global `--sample_ratio` override). Holds `blocks_to_swap`, `gradient_checkpointing`, `unsloth_offload_checkpointing`, etc.
- `configs/methods/` — one file per algorithm family. Holds rank, method flags (`use_hydra`, `add_reft`, …), and the method's opinionated learning rate / epochs / output_name. Eight files:
  - `lora.toml` — LoRA / OrthoLoRA / T-LoRA / HydraLoRA / ReFT. Variants are toggle blocks; default stacks classic LoRA + OrthoLoRA + T-LoRA + ReFT. Default ships `save_every_n_epochs = 2` / `checkpointing_epochs = 2` (mid-run snapshots, not just final).
  - `postfix.toml` — postfix / postfix_exp / postfix_func / postfix_sigma / prefix. Toggle blocks.
  - `ip_adapter.toml` — IP-Adapter image cross-attention (DiT frozen; trains resampler + per-block `to_k_ip`/`to_v_ip`). Reuses the LoRA pipeline's data layout (`post_image_dataset/resized/` + `post_image_dataset/lora/`). Defaults to PRE-CACHED PE features (`make preprocess-pe`).
  - `easycontrol.toml` — EasyControl image conditioning (DiT frozen; trains per-block cond LoRA on self-attn + FFN + scalar `b_cond` gate). Source: `easycontrol-dataset/`. Caches: `post_image_dataset/easycontrol/`. Reuses cached VAE latents — no new sidecar.
  - `soft_tokens.toml` — SoftREPA-style per-layer × per-t soft text tokens (DiT frozen; per-block `Block.forward` monkey-patch splices `s^(k,t)` into `crossattn_emb`). ~1M params. Training-only v1 — inference path not wired.
- `configs/gui-methods/` — GUI-friendly parallel tree. One self-contained TOML per **variant** instead of per family (`lora`, `lora-8gb`, `lora_longer`, `lora_repa`, `ortholora`, `tlora`, `tlora_ortho`, `tlora_ortho_reft`, `reft`, `hydralora_experimental`, `hydralora_sigma`, `postfix_ortho_cond`, `ip_adapter`, `easycontrol`, `soft_tokens`). No toggle blocks — what you see is what runs. Selected via `train.py --methods_subdir gui-methods` (wrapped by `make lora-gui GUI_PRESETS=<variant>` / `python tasks.py lora-gui <variant>`). Intended for basic users and as the eventual source of truth for the GUI's variant picker. The legacy `postfix*` / `prefix` GUI variants were removed in commit 990a82f (postfix/archive) — `postfix_ortho_cond` is the surviving GUI-exposed postfix path. Run `ls configs/gui-methods/` for the live list — variants get added/renamed.

Subsets accept an optional `cache_dir` key — when set, all VAE / text-encoder / PE caches are written to (and read from) that directory using stem-mirrored filenames, instead of sitting next to the source image. IP-Adapter and EasyControl method configs use this to keep `ip-adapter-dataset/` and `easycontrol-dataset/` purely user-facing source dirs while caches live under `post_image_dataset/`.

`library.train_util.load_method_preset(method, preset, methods_subdir="methods")` is the reusable merge helper. Pass `methods_subdir="gui-methods"` to resolve against the clean per-variant tree instead of the toggle-block method files. All paths in configs are relative to `anima_lora/` (e.g., `models/...`, `output/ckpt/`). Runtime outputs are split by kind: trained checkpoints (+ `.snapshot.toml` + `_moe` siblings) in `output/ckpt/`, inference images in `output/tests/`, embedding-inversion results in `output/inversions/`, img2emb artifacts in `output/img2embs/`.

## Architecture

- **Modular `library/`**: `train_util.py` is a re-exporting facade; actual code lives in domain subpackages:
  - `library/anima/` — anima-specific code: `models.py` (DiT class), `training.py` (training helpers, CLI args), `weights.py` (model/tokenizer loading + save), `strategy.py` (tokenization/encoding strategies), `configs/` (bundled Qwen3/T5 tokenizer configs).
  - `library/datasets/` — dataset classes, buckets, image utils.
  - `library/training/` — optimizer, scheduler, checkpoint, loss/sampler/metric registries (absorbs former `custom_train_functions`).
  - `library/inference/` — generation, sampling, output, plus `dcw_calibrator.py` (DCW v4 controller + scalar mode), `directedit.py` (DirectEdit invert+edit primitive), `mod_guidance.py`, `adapters.py`.
  - `library/models/` — ancillary model defs: `qwen_vae.py` (VAE), `sai_spec.py` (metadata spec).
  - `library/captioning/` — Anima Tagger + WD Tagger wrappers used by DirectEdit's case-1 ψ_src path (`anima_tagger.py`, `wd_tagger.py`, shared `tag_rules.py`, plus `anima_tagger_data.py` / `anima_tagger_model.py` for training).
  - `library/vision/` — shared vision tower / resampler / bucket helpers (extracted from `archive/img2emb`; live consumer is IP-Adapter).
  - `library/config/` — `schema.py` (validation), `loader.py` (TOML merge chain).
  - `library/io/` — `cache.py` (disk cache helpers), `safetensors.py`.
  - `library/runtime/` — `device.py`, `offloading.py`, `noise.py` (flow-matching sampling).
  - `library/env.py` — environment / path resolution helpers.
  - `library/log.py` — logging setup + `fire_in_thread`.
- **Strategy pattern** for model-specific tokenization/encoding (`library/anima/strategy.py`, `library/strategy_base.py`)
- **Pluggable adapters** under `networks/` — selected via `network_module` config key. Covers LoRA / OrthoLoRA / T-LoRA / HydraLoRA / ReFT (in `networks/lora_modules/`), postfix / IP-Adapter / EasyControl (in `networks/methods/`), the unified attention-backend dispatcher (`networks/attention_dispatch.py`), and Spectrum inference (`networks/spectrum.py`). See `networks/CLAUDE.md` for the per-module map, variant details, and dispatch invariants.

### Training flow (train.py)

1. Load text encoder -> cache text encoder outputs to disk -> unload text encoder
2. Load VAE -> cache latents to disk -> unload VAE
3. Load DiT lazily (after caching frees VRAM)
4. Create LoRA/Postfix network, apply to target modules via monkey-patching
5. Training loop: noise sampling -> DiT forward -> loss -> backward -> optimizer step
6. Optional validation: multi-timestep loss + sample generation

## Critical invariants

### Text encoder padding

The pretrained model expects max-padded text encoder outputs — zero-padded positions act as attention sinks in cross-attention softmax. Trimming to actual text length produces black images. Both training and inference must pad to `max_length` and must NOT mask out padding via `crossattn_seqlens`. Regenerate disk-cached `.npz` files after any tokenizer/padding changes.

### Constant-token bucketing

All bucket resolutions ensure `(H/16)*(W/16) ~ 4096` patches. Batch elements are zero-padded to exactly 4096 tokens, giving `torch.compile` a single static shape — no recompilation across aspect ratios.

### Lazy model loading

DiT is loaded AFTER text encoder/VAE caching and unloading to avoid OOM. The sequence is: text encoder -> cache -> free -> VAE -> cache -> free -> load DiT.

## Spectrum inference acceleration

Training-free speedup via Chebyshev polynomial feature forecasting (Han et al., CVPR 2026). `--spectrum` flag on `inference.py` enables it. On cached steps, all transformer blocks are skipped — only `t_embedder` + `final_layer` + `unpatchify` run. A `register_forward_pre_hook` on `final_layer` captures block outputs without monkey-patching the model. The adaptive window schedule (controlled by `--spectrum_window_size` and `--spectrum_flex_window`) concentrates actual forwards on early high-noise steps and increasingly predicts later refinement steps. See `networks/spectrum.py` for the Anima integration and `docs/methods/spectrum.md` for usage notes.

## DCW (post-step SNR-t bias correction)

Training-free, sampler-level correction that mixes each Euler step's `prev_sample` toward (or away from) the model's `x0_pred` to close the SNR-t bias of flow-matching DiTs (Yu et al., CVPR 2026; arXiv:2604.16044). Lives at the sampler boundary — composes with everything (LoRA, Spectrum, mod-guidance, IP-Adapter…). Two modes:

- **Scalar** (`--dcw`, default `--dcw_lambda −0.015`, `--dcw_band_mask LL`): one global constant. Default tuned at CFG=1; at production CFG=4 the bias direction is **(CFG × aspect)-dependent** — non-square aspects want small *positive* λ, square + CFG=1 wants negative. Shipped scalar default is therefore wrong-direction on CFG=4 non-square; gate on prompt intent (helps detail-dense, hurts intentionally flat styles). The Spectrum ComfyUI node ships `+0.01` as its scalar default.
- **v4 learnable** (`--dcw_v4 <fusion_head.safetensors|auto>`): MLP head fed `(aspect prior, prompt embedding, observed prefix gap over first k=7 steps)` predicts `(α̂, log σ̂²)` once at step k, then distributes `α_eff` across the remaining steps proportionally to the bucket's `μ_g[i]`. Per-step λ_i clamped to `±3·|λ_scalar[aspect]|`. `--dcw_v4_disable_shrinkage` is recommended until σ̂² calibration passes Gate B.

Calibration (`make dcw`) runs `scripts/dcw/measure_bias.py` against 5 aspect buckets at the production env (CFG=4, mod_w=3.0), then chains `train_fusion_head.py`. Incremental: each run drops a `manifest.json`, and subsequent runs grow the pool via `--exclude_stems`. End artifact `<run>/fusion_head.safetensors` (~285k params + per-aspect bucket profile + standardization stats). `make test-dcw-v4` auto-resolves the newest by mtime under `post_image_dataset/dcw/` then `bench/dcw/results/`. The trainer is **bucket-agnostic** by design (single population μ_g, aspect_emb pinned to zero — see `project_dcw_bucket_prior_cosmetic` memory); per-bucket sampling is kept only to balance the prompt pool. Final Euler step (σ_{i+1}=0) is always skipped — DCW is a numerical no-op there.

Code: `library/inference/dcw_calibrator.py`, `networks/dcw.py`, `scripts/dcw/`, `scripts/tasks/dcw.py`, `bench/dcw/`. See `docs/methods/dcw.md` and `archive/dcw-learnable-calibrator/proposals/dcw-learnable-calibrator-v4.md`.

## DirectEdit + Anima Tagger

Image-editing primitive that pairs an inversion (DDIM-style trajectory through the frozen DiT) with an edit-conditioning swap: ψ_src reconstructs the source image, ψ_tar = ψ_src + edit-instruction applies the change. Robust to ψ_src corruption for *reconstruction* but edit *leverage* collapses when ψ_src is structurally far from Anima's training-time embedding manifold — wd-tagger output was bad enough at this to be the live blocker, which is why **Anima Tagger** exists.

- **Anima Tagger** (`library/captioning/anima_tagger.py`): small classifier mapping image → comma-separated tag string in exactly Anima's training-time T5 format. Drop-in replacement for `WDTagger` (same `predict` surface). Frozen PE-Core-L14-336 trunk + LayerNorm + Linear head. Train with `python -m scripts.anima_tagger.cli` against `bench/wd_tagger_dataset/` (vocabulary + manifest + feature cache builders + threshold calibrator all wired). DirectEdit's case-1 ψ_src path picks Anima Tagger if a checkpoint is present, else falls back to WD Tagger.
- **DirectEdit core** (`library/inference/directedit.py`): the invert+edit primitive. Invoked from `scripts/edit.py` (CLI), `scripts/experimental_tasks/inference.py` (`make exp-test-directedit` / `exp-test-directedit-dry`), and `custom_nodes/comfyui-anima-directedit/` (ComfyUI nodes).

Use `make exp-test-directedit-dry` to verify ψ_tar == ψ_src reconstructs the source — gates whether the inversion is well-conditioned independent of edit semantics.

See `docs/experimental/directedit_editing_v3.md` (what's built) and `docs/experimental/anima_tagger.md` (tagger architecture). The earlier v2 proposal lives in `archive/proposals/directedit_editing_v2.md` (alongside `directedit_editing.md` and `directedit_gaps.md`).

## Modulation guidance

Text-conditioned AdaLN modulation via a learned `pooled_text_proj` MLP (Starodubcev et al., ICLR 2026). Distilled with `make distill-mod`: teacher uses real cross-attention, student uses zeroed cross-attention but receives pooled text through modulation. At inference, steers AdaLN coefficients toward quality-positive directions. See `docs/methods/mod-guidance.md`.

## IP-Adapter

Decoupled image cross-attention (Ye et al. 2023). DiT is frozen; trains only the Perceiver resampler and per-block parallel `to_k_ip`/`to_v_ip` projections (~150M params at default `K=16`, 28 blocks). Reference image → frozen vision tower (PE-Core-L14-336 by default) → resampler → K compact IP tokens → per-block KV → patched cross-attention adds `scale * SDPA(text_q, ip_k, ip_v)` to the existing text cross-attention. Reuses the LoRA pipeline's data layout — source images live under `post_image_dataset/resized/` and caches (latents, text-emb, PE features) live under `post_image_dataset/lora/`. Defaults to PRE-CACHED PE features (`{stem}_anima_pe.safetensors` from `make preprocess-pe`) so training never loads the vision encoder. CFG dropout (`image_drop_p`) zeros image conditioning so inference can do image-CFG independently of text-CFG. See `docs/experimental/ip-adapter.md`.

## EasyControl

Extended self-attention image conditioning. DiT is frozen; trains per-block cond LoRA on self-attn (q/k/v/o) + FFN (layer1/layer2) plus a per-block scalar logit-bias `b_cond` (init `-10`) that gates cond-position softmax mass. Reference is VAE-encoded and patch-embedded by the DiT's frozen `x_embedder` into condition tokens that flow through every block alongside the target stream; target self-attention attends to a key set extended with the cond stream's keys/values. Training uses a **two-stream block forward** (target + cond in one scope, no deferred-backward dance); inference prefills a per-block `(K_c, V_c)` cache once at setup and reuses it across every denoising step and every CFG branch (cond is deterministic — `cond_temb = t_embedder(0)`). Source images live in `easycontrol-dataset/`; caches go to `post_image_dataset/easycontrol/` via subset `cache_dir`. Reuses cached VAE latents — no new sidecar. See `docs/experimental/easycontrol.md`.

## Soft Tokens (SoftREPA parameterization)

Per-layer time-indexed soft text tokens (Lee et al., arXiv:2503.08250, NeurIPS 2025). DiT frozen; trains a `(n_layers, K, D)` token bank + `(n_t_buckets, n_layers, D)` t-offsets — ~1M params at default. For each of the first `n_layers` blocks, a `(layer, t-bucket)`-specific token slice is spliced into `crossattn_emb` via a per-block `Block.forward` monkey-patch (ReFT-pattern); end-of-sequence overwrite of zero-padding tail (or `front_of_padding` scatter) keeps `_run_blocks` torch.compile shape-static. Anima's cross-attention (not joint-stream MM-DiT) means each block independently sees a different `crossattn_emb` — no strip/re-prepend dance. Adopts only the parameterization from the SoftREPA paper; the contrastive InfoNCE objective is intentionally skipped (caused SD3 FID regression). Training-only v1: `inference.py` will refuse to load these checkpoints until per-step block hooks are wired into the denoising loop. See `docs/experimental/soft_tokens.md`.

## Embedding inversion

Optimizes text embeddings (post-T5, pre-DiT space) to minimize flow-matching loss for a target image through the frozen DiT. Reveals how the model interprets images in embedding space. Archived — run `python archive/inversion/invert_embedding.py` (or `invert_reference.py` / `interpret_inversion.py`) directly. See `docs/methods/invert.md`.

## Preprocessing

Data preparation scripts in `preprocess/`:
- `resize_images.py` — VAE-compatible image resizing (used by `make preprocess-resize`). Reads `image_dataset/`, writes resized PNGs to `post_image_dataset/resized/`. Drops images below `--min_pixels` (default 0.5MP). `--no_copy_captions` skips the `.txt` copy so captions stay only in `image_dataset/`.
- `cache_latents.py` — Cache VAE latents (used by `make preprocess-vae`). Reads `post_image_dataset/resized/`, writes `{stem}_{WxH}_anima.npz` into `post_image_dataset/lora/` via `--cache_dir`.
- `cache_text_embeddings.py` — Cache text encoder outputs (used by `make preprocess-te`). Reads `image_dataset/` (where `.txt` lives) and writes `{stem}_anima_te.safetensors` into `post_image_dataset/lora/` via `--cache_dir`. Mirrors `resize_images.py`'s `--min_pixels` filter so caches don't accumulate for images that would be dropped at resize.
- `cache_pe_encoder.py` — Cache PE-Core-L14-336 vision encoder features (`{stem}_anima_pe.safetensors`). Wrapped by `make preprocess-pe` (reads `post_image_dataset/resized/`, writes `post_image_dataset/lora/`). Both the LoRA / REPA pipeline and IP-Adapter consume the same sidecars from `post_image_dataset/lora/`.
- `make preprocess-pooled` — Cache pooled text-embedding sidecars (consumed by `make distill-mod`). CPU-only; reads existing `_anima_te.safetensors` and writes pooled companions next to them.
- `generate_masks.py` — SAM3-based text bubble mask generation
- `generate_masks_mit.py` — MIT/ComicTextDetector mask generation (manga-specific)
- `merge_masks.py` — Combine SAM3 + MIT masks into final mask set

## Scripts

Utility scripts in `scripts/`:
- `distill_modulation.py` — Train pooled_text_proj MLP for modulation guidance (used by `make distill-mod`)
- `convert_lora_to_comfy.py` — Convert LoRA key names between anima and ComfyUI formats
- `comfy_batch.py` — Run ComfyUI batch workflow from `workflows/` directory
- `merge_to_dit.py` — Bake a LoRA adapter into the base DiT (used by `make merge`)
- `bench_methods.py` — Benchmark inference across method configurations (writes to `bench/`)
- `compute_pe_centroid.py` — Compute PE-feature centroid for DCW v4's `cos(c_pool, μ_centroid)` channel.
- `export_logs_json.py` — Export TensorBoard run scalars to JSON/JSONL (used by `make export-logs`)
- `anima_tagger/cli.py` — Train the Anima Tagger checkpoint used by DirectEdit (invoke as `python -m scripts.anima_tagger.cli`). See `docs/experimental/anima_tagger.md`.
- `edit.py` — Standalone DirectEdit CLI entry (the `make exp-test-directedit` wrapper around it lives in `scripts/experimental_tasks/inference.py`).
- `scripts/dcw/` — DCW v4 calibration pipeline: `measure_bias.py` (per-aspect trajectory dump), `train_fusion_head.py` (fusion-head trainer), `trajectory.py`, `haar.py`, etc. Driven by `make dcw` / `make dcw-train` (`scripts/tasks/dcw.py`).

Archived utilities (legacy, no longer wired up): `archive/img2emb/` (resampler training + inference) and `archive/inversion/` (embedding/reference inversion). The shared resampler/encoder/bucket modules they used have been extracted into `library/vision/` for live IP-Adapter use.

## Custom nodes

Spectrum KSampler and mod guidance ComfyUI nodes live in a separate repo: https://github.com/sorryhyun/ComfyUI-Spectrum-KSampler (the Spectrum node also ships DCW: scalar default `+0.01` plus `auto` mode that runs the v4 fusion head in-node — see `reference_spectrum_node_dcw_defaults` memory).

In-tree under `custom_nodes/`:
- `comfyui-hydralora/` — **Anima Adapter Loader** node (unified LoRA / Hydra / ReFT + prefix/postfix). See `custom_nodes/comfyui-hydralora/CLAUDE.md` for code-level details and the `forward_hook`-not-`forward`-override invariant; `README.md` for user-facing docs and changelog.
- `comfyui-anima-directedit/` — `AnimaDirectEdit` node (invert + edit on a frozen Anima checkpoint, using stock MODEL/CLIP/VAE sockets). Consumes the `ANIMA_TAGGER` socket from `comfyui-anima-tagger`, or skips the tagger via the `prompt_src_override` STRING input. See its own `README.md`.
- `comfyui-anima-tagger/` — `AnimaTaggerLoader` (→ `ANIMA_TAGGER` socket) and `AnimaTaggerCaption` (`ANIMA_TAGGER` + `IMAGE` → `STRING`). Standalone captioner usable outside DirectEdit (LoRA caption pre-fill, prompt scaffolding, etc.).
- `comfyui-anima-trainer/` — In-ComfyUI training trigger nodes for Anima.

## External tools

ComfyUI, SAM3, and manga-image-translator live in the parent directory (`../comfy/`, `../sam3/`, etc.).

## Contributing

PRs are reviewed against a tier system spelled out in `CONTRIBUTING.md` (Tier 1 = bug fixes / typos; Tier 1.5 = efficiency or numerics revisions to existing methods — bench script + invariant test required; Tier 2 = new adapter method — paper citation + `bench/<method>/` subdir + docs entry + `make <name>` / `make test-<name>` targets; Tier 3 = new base-model support, currently not accepted). The `bench/<method>/` convention (README + runnable script + timestamped `results/`) is how method PRs prove their claims. Bench scripts share `bench/_common.py` (`make_run_dir` + `write_result`) — every run drops a standard `result.json` envelope (git SHA, env, args, metrics, artifacts) into `bench/<method>/results/<YYYYMMDD-HHMM>[-<label>]/` so cross-run indexing and reproduction are uniform.
