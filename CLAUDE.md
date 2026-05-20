# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Anima — LoRA/T-LoRA training and inference pipeline for the Anima diffusion model (DiT-based, flow-matching). Supports several adapter families (LoRA / OrthoLoRA / T-LoRA / HydraLoRA / FeRA / ReFT / ChimeraHydra / IP-Adapter / EasyControl) selectable via method config + hardware preset. The LoRA family is routed via a three-axis surface — `use_moe_style` / `route_per_layer` / `router_source` — see `configs/methods/lora.toml`.

## Setup

```bash
uv sync                    # Install dependencies (Python 3.13)
hf auth login              # Authenticate for model downloads
make download-models       # Download DiT, text encoder, VAE, SAM3, MIT, PE-Core, PE-Spatial
# Training images go in image_dataset/ with .txt caption sidecars
make preprocess            # Resize → post_image_dataset/resized/, cache → post_image_dataset/lora/
```

## Commands

Both `make` (Unix) and `python tasks.py` (cross-platform) are supported. The examples below show both forms. The `Makefile` is a thin catch-all dispatcher — every target forwards to `python tasks.py <target> $(ARGS)`. **`tasks.py` is the source of truth**; per-domain command implementations live in `scripts/tasks/{training,inference,preprocess,masking,gui,downloads,utilities,tagger,dcw}.py`, with the experimental commands (`exp-*`) in `scripts/experimental_tasks/{training,inference}.py`. Don't grep the Makefile for a target's recipe — look in `scripts/tasks/` (or `scripts/experimental_tasks/` for `exp-*`).

```bash
# Training (run from anima_lora/)
# Each training invocation selects a method + hardware preset. Method settings win
# over preset settings on overlap (e.g. a frozen-DiT method can force
# blocks_to_swap=0).
# Method files in configs/methods/: lora.toml, chimera.toml,
# ip_adapter.toml, easycontrol.toml, soft_tokens.toml, turbo.toml. Variants are
# toggle blocks inside them — uncomment the target block to switch:
#   lora.toml             — LoRA / OrthoLoRA / T-LoRA / HydraLoRA / FeRA / ReFT,
#                           routed via three-axis cfg (use_moe_style /
#                           route_per_layer / router_source). Default stacks
#                           LoRA + OrthoLoRA + T-LoRA + shared_A FEI-routed
#                           experts (ReFT block is commented). (DoRA retired —
#                           use_dora is no longer trained/saved/loaded.)
#   chimera.toml          — ChimeraHydra dual-pool additive MoE (content pool
#                           routed by per-Linear lx-router + freq pool routed
#                           by network-level FEI router; two A's per Linear)
#   ip_adapter.toml       — decoupled image cross-attention (PE-Core resampler)
#   easycontrol.toml      — extended self-attn image conditioning (per-block cond LoRA)
#   soft_tokens.toml      — SoftREPA-style per-layer × per-t soft text tokens (frozen DiT)
#   turbo.toml            — Turbo Anima (Decoupled-Hybrid DMD2 distillation).
#                           BESPOKE schema (sections, not flat kwargs); read by
#                           scripts/distill_turbo.py, NOT by train.py — don't
#                           `print-config METHOD=turbo`. Output is a normal LoRA.
make lora                   # LoRA family (methods/lora.toml + presets.toml[default])
python tasks.py lora        # Same, works on Windows too
make lora PRESET=low_vram   # Override preset: methods/lora.toml + presets.toml[low_vram]
make lora PRESET=fast_16gb  # Override preset: methods/lora.toml + presets.toml[fast_16gb]
make lora PRESET=half       # Override preset: methods/lora.toml + presets.toml[half] (sample_ratio=0.5)
# Experimental methods are exposed under exp-* (ip-adapter, easycontrol,
# soft-tokens, chimera, turbo). They may produce broken output, change
# without notice, or be removed.
make exp-ip-adapter             # IP-Adapter image cross-attention (methods/ip_adapter.toml)
                                # Reuses LoRA paths: source image_dataset/, cache post_image_dataset/lora/
make exp-ip-adapter-preprocess  # Alias for `make preprocess` + `make preprocess-pe`
make exp-easycontrol            # EasyControl image conditioning (methods/easycontrol.toml)
                                # Source: easycontrol-dataset/  Cache: post_image_dataset/easycontrol/
make exp-easycontrol-preprocess # Resize + VAE + text caches into post_image_dataset/easycontrol/
make exp-soft-tokens            # SoftREPA-style per-layer × per-t soft tokens (training-only v1)
make exp-chimera                # ChimeraHydra dual-pool additive MoE (methods/chimera.toml)
make exp-turbo                  # Turbo Anima DMD2 distillation (scripts/distill_turbo.py + methods/turbo.toml)
                                # Output anima_turbo.safetensors is a normal LoRA: infer with --infer_steps 4 --cfg 1.0

# GUI-friendly per-variant path (configs/gui-methods/<variant>.toml — clean,
# self-contained, no toggle blocks). Intended for basic users who don't want
# to hand-edit methods/lora.toml's comment-toggle system.
make lora-gui GUI_PRESETS=tlora                                # gui-methods/tlora.toml + preset default
make lora-gui GUI_PRESETS=hydralora PRESET=low_vram            # override preset as usual
python tasks.py lora-gui hydralora-8gb                         # Windows; variant can also be 1st positional arg
make lora-gui GUI_PRESETS=tlora_ortho_reft                     # T-LoRA + OrthoLoRA + ReFT stack
# Run `ls configs/gui-methods/` for the live variant list — files get added/renamed
# (current: lora, lora-8gb, hydralora, hydralora-8gb, tlora, tlora-8gb,
#  tlora_ortho_reft, reft, chimera_hydra, ip_adapter, easycontrol, soft_tokens).

# Modulation guidance distillation
make distill-prep          # Stage T5("") uncond sidecar (Phase 1) + teacher-synthetic
                           #   clean latents pool (Phase 2 — post_image_dataset/distill_mod_synth/)
make distill-mod           # Train pooled_text_proj MLP (text → AdaLN modulation).
                           # Add --synth_data_dir post_image_dataset/distill_mod_synth to fit on
                           # the teacher's own manifold (paper-faithful; removes real-vs-teacher gap).

# Inference (test with most recent output)
make test
make test MOD=1            # Latest LoRA + distilled pooled_text_proj (modulation guidance)
make test NOLORA=1         # Bare DiT (skip --lora_weight); compose with MOD=1 for mod-only path
make test-hydra            # HydraLoRA / FeRA router-live (anima_hydra*_moe.safetensors or
                           #   stacked-experts FeRA — both go through the same
                           #   `lora_ups.{i}.weight` safetensors sniff)
make test-merge            # Inference with a merged/baked DiT (no adapter loaded)
make test SPECTRUM=1       # Spectrum-accelerated inference (~3.75x speedup)
make test SPECTRUM=1 MOD=1 # Spectrum + modulation guidance composed
make test-dcw              # Latest LoRA + DCW scalar bias correction (--dcw, λ=−0.015)
make test-dcw-v4           # DCW v4 learnable calibrator on bare DiT (auto-resolves fusion_head.safetensors;
                           #   NOLORA=0 attaches the latest LoRA)
# SPECTRUM=1 / MOD=1 / NOLORA=1 all compose into test-dcw, test-dcw-v4, test-smc-cfg too.
make test-spectrum-dcw     # Spectrum + DCW scalar composed
make test-dcw-v4-spectrum  # Spectrum + DCW v4 composed

# DCW v4 calibration (one-shot per LoRA checkpoint)
make dcw                   # Sample 5 aspect buckets + train fusion head (~3-5h on a 5060 Ti)
make dcw-train             # Train-only on existing pool under bench/dcw/results/ (~30s)

# Experimental inference (matched to make exp-* training)
make exp-test-soft             # Soft-tokens inference (latest soft_tokens checkpoint)
make exp-test-turbo            # Turbo LoRA inference (4-step, CFG=1.0 baked-in student)
make exp-test-ip REF_IMAGE=... # IP-Adapter inference (image-conditioned)
make exp-test-easycontrol REF_IMAGE=...  # EasyControl inference (image-conditioned)
make exp-test-directedit PROMPT='double peace'  # DirectEdit on random source image
                                                # (Anima Tagger seeds prompt_src;
                                                # PROMPT becomes the edit instruction)
make exp-test-directedit-dry                    # DirectEdit reconstruction sanity check
                                                # (ψ_tar == ψ_src; output should reconstruct the source)

# Training daemon (local job queue — see plan.md, Phases 1–2)
# Single localhost process: FIFO serial queue + worker that spawns each job as a
# detached `accelerate launch` subprocess and follows it via the Phase-0
# progress.jsonl. The ComfyUI trainer node, the GUI Train button, and `--queue`
# all submit jobs; the daemon auto-starts on first submit.
make daemon                # Start (idempotent; detached, waits for /health)
make daemon-attach         # Follow events read-only (JOB=<id> tails that job's stdout); ctrl-C detaches only
make daemon-kill           # Abort the running job (or JOB=<id>); daemon stays up, advances the queue
make daemon-terminate      # Stop the daemon entirely (active job killed, GPU freed)
# `--queue` enqueues instead of running inline — the overnight sweep (submit ×N,
# drains serially). Works on `make lora` and `make lora-gui`:
make lora --queue                          # enqueue a default-preset LoRA run
make lora-gui GUI_PRESETS=tlora --queue    # enqueue a gui-methods variant
# The GUI Train button submits to the daemon too, so training survives closing
# the GUI; reopening re-attaches to the running job. Preprocessing (the
# Preprocessing tab's caching/mask Runs + the Train button's auto-chain cache
# build) also submits to the daemon — as a "command" job (a plain
# `python tasks.py <target>` with its knobs in extra_env) that queues serially
# behind training on the single GPU. Only the Test button stays in-process.

# GUI (PySide6 — config editing, IP-Adapter / EasyControl preprocess+train, dataset browsing)
make gui
python tasks.py gui        # Windows
make gui-shortcut          # Create "Anima LoRA GUI.lnk" on the Windows desktop (no console window)

# Masking (for masked loss training)
# Merged masks land in post_image_dataset/masks/<rel>/{stem}_mask.png. SAM +
# MIT intermediates run through a tempdir and are not persisted. Subsets
# auto-pick post_image_dataset/masks/, falling back to legacy
# masks/{merged,sam,mit}/ when the new tree is missing.
make mask                  # Run SAM3 + MIT (via tempdir), merge → post_image_dataset/masks/
make mask-clean            # rm -rf post_image_dataset/masks/

# Merge LoRA into DiT (standalone ComfyUI-compatible checkpoint)
make merge ADAPTER_DIR=output/ckpt                    # bake latest bakeable LoRA in dir
make merge ADAPTER_DIR=output/ckpt MULTIPLIER=0.8     # scale strength
python scripts/merge_to_dit.py --adapter path/to/lora.safetensors --allow-partial
# Supports: LoRA / OrthoLoRA / T-LoRA. Refuses ReFT / Hydra moe / postfix
# by default (they can't be folded into Linear weights); --allow-partial
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
| `scripts/experimental_tasks/` | Bodies for the `exp-*` commands (ip-adapter, easycontrol, soft-tokens, chimera training + their `exp-test-*` inference, plus the DirectEdit / postfix-tail inversion probes). Reuses helpers from `scripts/tasks/_common.py`. |

Method deep-dives in `docs/methods/` (shipped); experimental method docs in `docs/experimental/`; active proposals in `docs/proposal/`. Retired material lives under `_archive/`.

## Config flow

Training is config-driven via a three-layer chain: `base.toml → presets.toml[<preset>] → methods/<method>.toml → CLI args`. Method settings win over preset settings on overlap, so a method can force its own hardware requirements (e.g. a frozen-DiT method can force `blocks_to_swap=0`).

Layout:
- `configs/base.toml` — shared infrastructure (model paths, optimizer, compile flags, etc.) AND the default LoRA dataset blueprint (`[general]` + `[[datasets]]` + `[[datasets.subsets]]`). LoRA reads resized images from `post_image_dataset/resized/` with caches redirected to `post_image_dataset/lora/` via `cache_dir`. Captions live in `image_dataset/` (master) — TE caching reads `.txt` from there, training reads only the cached prompt embeddings. The dataset sections are consumed by `BlueprintGenerator` and skipped by the flat method+preset merge chain (see `_DATASET_CONFIG_SECTIONS` in `library/train_util.py`); use `--dataset_config <path>` for a wholly different blueprint, or drop a `[general]` / `[[datasets]]` block into the method TOML to shallow-override top-level scalars (e.g. `batch_size`) on the base blueprint — see `_apply_dataset_overrides` in `library/config/io.py`. Subset-level overrides are not supported via this path.
- `configs/presets.toml` — all hardware profiles in one file as TOML sections: `[default]`, `[fast_16gb]`, `[low_vram]` (also serves as Windows 8GB), `[half]` (experiment preset — sets `sample_ratio=0.5` for every subset via the global `--sample_ratio` override). Holds `blocks_to_swap`, `gradient_checkpointing`, `unsloth_offload_checkpointing`, etc.
- `configs/methods/` — one file per algorithm family. Holds rank, the three-axis routing knobs (`use_moe_style` / `route_per_layer` / `router_source`), other method flags (`add_reft`, `use_ortho`, `use_timestep_mask`, …), and the method's opinionated learning rate / epochs / output_name. Six files (`lora`, `chimera`, `ip_adapter`, `easycontrol`, `soft_tokens` are flat method+preset configs read by `train.py`; `turbo.toml` is the odd one out — a bespoke sectioned schema read only by `scripts/distill_turbo.py`, see the ChimeraHydra/Turbo notes below):
  - `lora.toml` — LoRA / OrthoLoRA / T-LoRA / HydraLoRA / FeRA / ReFT. Variants are toggle blocks; default stacks LoRA + OrthoLoRA + T-LoRA + Hydra (`use_moe_style="shared_A"` + `route_per_layer=False` + `router_source="fei"`). The σ-routed Hydra and ReFT blocks are present but commented. Default ships `save_every_n_epochs = 4` / `checkpointing_epochs = 4`. **Pre-three-axis checkpoints with `ss_use_hydra`/`ss_use_fei_router` metadata no longer load** — the legacy fallback was removed.
  - `chimera.toml` — ChimeraHydra dual-pool additive MoE (`networks/lora_modules/chimera.py`, registered as `chimera_hydra` in `networks/__init__.py::NETWORK_REGISTRY`). Two A's per Linear (one per pool): content pool with K_c B-heads routed by per-Linear lx-router, freq pool with K_f B-heads routed by the network-level FreqRouter on concat(FEI(z_t), sinusoidal-σ). Pool outputs are added; T-LoRA mask applies to the content branch only.
  - `ip_adapter.toml` — IP-Adapter image cross-attention (DiT frozen; trains resampler + per-block `to_k_ip`/`to_v_ip`). Reuses the LoRA pipeline's data layout (`post_image_dataset/resized/` + `post_image_dataset/lora/`). Defaults to PRE-CACHED PE features (`make preprocess-pe`).
  - `easycontrol.toml` — EasyControl image conditioning (DiT frozen; trains per-block cond LoRA on self-attn + FFN + scalar `b_cond` gate). Source: `easycontrol-dataset/`. Caches: `post_image_dataset/easycontrol/`. Reuses cached VAE latents — no new sidecar.
  - `soft_tokens.toml` — SoftREPA-style per-layer × per-t soft text tokens (DiT frozen; per-block `Block.forward` monkey-patch splices `s^(k,t)` into `crossattn_emb`). ~1M params. Inference is wired: standalone via `inference.py --soft_tokens_weight` (`library/inference/generation.py`) and in ComfyUI via the `AnimaSoftTokensLoader` node (`custom_nodes/comfyui-hydralora/soft_tokens.py`).
  - `turbo.toml` — Turbo Anima (Decoupled-Hybrid DMD2 distillation, Liu et al. arXiv:2511.22677; proposal `docs/proposal/turbo_anima_dmd_lora.md`). **Bespoke sectioned schema** (`[network]`, etc.) read by `scripts/distill_turbo.py`, NOT by `train.py` — don't `print-config METHOD=turbo`. Trains a student + a fake-score LoRA stack on one frozen DiT; output `anima_turbo.safetensors` is a normal LoRA loaded through the standard inference path with `--infer_steps 4 --cfg 1.0` (CFG baked into the student).
- `configs/gui-methods/` — GUI-friendly parallel tree. One self-contained TOML per **variant** instead of per family. No toggle blocks — what you see is what runs. Selected via `train.py --methods_subdir gui-methods` (wrapped by `make lora-gui GUI_PRESETS=<variant>` / `python tasks.py lora-gui <variant>`). Current variants: `lora`, `lora-8gb`, `hydralora`, `hydralora-8gb`, `tlora`, `tlora-8gb`, `tlora_ortho_reft`, `reft`, `chimera_hydra`, `ip_adapter`, `easycontrol`, `soft_tokens`. Run `ls configs/gui-methods/` for the live list — variants get added/renamed (e.g. the old `fera`/`ortholora`/`hydralora_*` variants were removed; FeRA now lives only on the `methods/lora.toml` `independent_A` toggle).

Subsets accept an optional `cache_dir` key — when set, all VAE / text-encoder / PE caches are written to (and read from) that directory using stem-mirrored filenames, instead of sitting next to the source image. IP-Adapter and EasyControl method configs use this to keep `ip-adapter-dataset/` and `easycontrol-dataset/` purely user-facing source dirs while caches live under `post_image_dataset/`.

`library.train_util.load_method_preset(method, preset, methods_subdir="methods")` is the reusable merge helper. Pass `methods_subdir="gui-methods"` to resolve against the clean per-variant tree instead of the toggle-block method files. All paths in configs are relative to `anima_lora/` (e.g., `models/...`, `output/ckpt/`). Runtime outputs are split by kind: trained checkpoints (+ `.snapshot.toml` + `_moe` siblings) in `output/ckpt/`, inference images in `output/tests/`.

## Architecture

- **Modular `library/`**: `train_util.py` is a re-exporting facade; actual code lives in domain subpackages:
  - `library/anima/` — anima-specific code: `models.py` (DiT class), `training.py` (training helpers, CLI args), `weights.py` (model/tokenizer loading + save), `strategy.py` (tokenization/encoding strategies), `configs/` (bundled Qwen3/T5 tokenizer configs).
  - `library/datasets/` — dataset classes, buckets, image utils.
  - `library/training/` — optimizer, scheduler, checkpoint, loss/sampler/metric registries (absorbs former `custom_train_functions`).
  - `library/inference/` — generation, sampling, output, plus `dcw_calibrator.py` (DCW v4 controller + scalar mode), `directedit.py` + `directedit_splice.py` + `edit_dispatcher.py` (DirectEdit invert+edit primitive + ψ-splice variant + multi-dispatch entry), `postfix_inversion.py` (postfix-tail per-image inversion — see `scripts/inversion/invert_postfix_tail.py`), `mod_guidance.py`, `adapters.py`.
  - `library/models/` — ancillary model defs: `qwen_vae.py` (VAE), `sai_spec.py` (metadata spec).
  - `library/captioning/` — Anima Tagger used by DirectEdit's case-1 ψ_src path (`anima_tagger.py`, shared `tag_rules.py`, plus `anima_tagger_data.py` / `anima_tagger_model.py` for training).
  - `library/vision/` — shared vision tower / resampler / bucket helpers (live consumer is IP-Adapter).
  - `library/config/` — `schema.py` (validation), `loader.py` (TOML merge chain).
  - `library/io/` — `cache.py` (disk cache helpers), `safetensors.py`.
  - `library/runtime/` — `device.py`, `offloading.py`, `noise.py` (flow-matching sampling).
  - `library/env.py` — environment / path resolution helpers.
  - `library/log.py` — logging setup + `fire_in_thread`.
- **Strategy pattern** for model-specific tokenization/encoding (`library/anima/strategy.py`, `library/strategy_base.py`)
- **Pluggable adapters** under `networks/` — selected via `network_module` config key plus (for the LoRA family) the three-axis routing cfg. Covers LoRA / OrthoLoRA / T-LoRA / HydraLoRA / FeRA / ReFT / ChimeraHydra (in `networks/lora_modules/` — including `stacked_experts.py` for FeRA's independent-A layout and `chimera.py` for the dual-pool MoE) coordinated by `networks/lora_anima/` (`network.py`, `factory.py`, `loading.py`, `config.py`, `attn_fuse.py`); IP-Adapter / EasyControl (in `networks/methods/`); the unified attention-backend dispatcher (`networks/attention_dispatch.py`); and Spectrum inference (`networks/spectrum.py`). See `networks/CLAUDE.md` for the per-module map, three-axis surface, variant details, and dispatch invariants.

## Critical invariants

### Text encoder padding

The pretrained model expects max-padded text encoder outputs — zero-padded positions act as attention sinks in cross-attention softmax. Trimming to actual text length produces black images. Both training and inference must pad to `max_length` and must NOT mask out padding via `crossattn_seqlens`. Regenerate disk-cached `.npz` files after any tokenizer/padding changes.

### Constant-token bucketing

All bucket resolutions ensure `(H/16)*(W/16) ~ 4096` patches. Batch elements are zero-padded to exactly 4096 tokens, giving `torch.compile` a single static shape — no recompilation across aspect ratios.

### Lazy model loading

DiT is loaded AFTER text encoder/VAE caching and unloading to avoid OOM. The sequence is: text encoder → cache → free → VAE → cache → free → load DiT → attach LoRA/adapter network → training loop.

## Spectrum inference acceleration

Training-free speedup via Chebyshev polynomial feature forecasting (Han et al., CVPR 2026). `--spectrum` flag on `inference.py` enables it. On cached steps, all transformer blocks are skipped — only `t_embedder` + `final_layer` + `unpatchify` run. A `register_forward_pre_hook` on `final_layer` captures block outputs without monkey-patching the model. The adaptive window schedule (controlled by `--spectrum_window_size` and `--spectrum_flex_window`) concentrates actual forwards on early high-noise steps and increasingly predicts later refinement steps. See `networks/spectrum.py` for the Anima integration and `docs/methods/spectrum.md` for usage notes.

## DCW (post-step SNR-t bias correction)

Training-free, sampler-level correction that mixes each Euler step's `prev_sample` toward (or away from) the model's `x0_pred` to close the SNR-t bias of flow-matching DiTs (Yu et al., CVPR 2026; arXiv:2604.16044). Lives at the sampler boundary — composes with everything (LoRA, Spectrum, mod-guidance, IP-Adapter…). Two modes:

- **Scalar** (`--dcw`, default `--dcw_lambda −0.015`, `--dcw_band_mask LL`): one global constant. Default tuned at CFG=1; at production CFG=4 the bias direction is **(CFG × aspect)-dependent** — non-square aspects want small *positive* λ, square + CFG=1 wants negative. Shipped scalar default is therefore wrong-direction on CFG=4 non-square; gate on prompt intent (helps detail-dense, hurts intentionally flat styles). The Spectrum ComfyUI node ships `+0.01` as its scalar default.
- **v4 learnable** (`--dcw_v4 <fusion_head.safetensors|auto>`): MLP head fed `(aspect prior, prompt embedding, observed prefix gap over first k=7 steps)` predicts `(α̂, log σ̂²)` once at step k, then distributes `α_eff` across the remaining steps proportionally to the bucket's `μ_g[i]`. Per-step λ_i clamped to `±3·|λ_scalar[aspect]|`. `--dcw_v4_disable_shrinkage` is recommended until σ̂² calibration passes Gate B.

Calibration (`make dcw`) runs `scripts/dcw/measure_bias.py` against 5 aspect buckets at the production env (CFG=4, mod_w=3.0), then chains `train_fusion_head.py`. Incremental: each run drops a `manifest.json`, and subsequent runs grow the pool via `--exclude_stems`. End artifact `<run>/fusion_head.safetensors` (~285k params + per-aspect bucket profile + standardization stats). `make test-dcw-v4` auto-resolves the newest by mtime under `post_image_dataset/dcw/` then `bench/dcw/results/`. The trainer is **bucket-agnostic** by design (single population μ_g, aspect_emb pinned to zero — see `project_dcw_bucket_prior_cosmetic` memory); per-bucket sampling is kept only to balance the prompt pool. Final Euler step (σ_{i+1}=0) is always skipped — DCW is a numerical no-op there.

Code: `library/inference/dcw_calibrator.py`, `networks/dcw.py`, `scripts/dcw/`, `scripts/tasks/dcw.py`, `bench/dcw/`. See `docs/methods/dcw.md` and `_archive/dcw-learnable-calibrator/proposals/dcw-learnable-calibrator-v4.md`.

## SMC-CFG (α-adaptive sliding-mode CFG)

Training-free CFG-combine modification that treats CFG as a control problem (Wang et al., arXiv:2603.03281). At each step, applies a bang-bang sliding-mode correction `Δe = −k_t · sign(s)` to the velocity-space residual `e = v_cond − v_uncond`, where `s = (e − e_prev) + λ·e_prev` is the sliding surface and `k_t = α·mean(|e_t|)` is the **α-adaptive gain** (paper's fixed `k=0.1` was ~14× off on Anima — see `bench/smc_cfg/analysis_and_proposal.md` §A; the α path replaces it entirely). Implementation ships `sign()` only — the tanh boundary-layer ε was tested and removed (concentrates correction into fewer voxels → grain). Defaults: λ=5, α=0.2. Observable effects: detail/fingers recover, outputs run slightly darker (anti-DC pull from clamping small-|e| brightness channels). Composes with DCW (post-step x-space), mod-guidance (AdaLN-side), Spectrum (cached-step combine still fires). Code: `library/inference/smc_cfg.py`, calls inside `library/inference/generation.py` + `networks/spectrum.py`. See `docs/methods/smc_cfg.md`.

## DirectEdit + Anima Tagger

Image-editing primitive that pairs an inversion (DDIM-style trajectory through the frozen DiT) with an edit-conditioning swap: ψ_src reconstructs the source image, ψ_tar = ψ_src + edit-instruction applies the change. Robust to ψ_src corruption for *reconstruction* but edit *leverage* collapses when ψ_src is structurally far from Anima's training-time embedding manifold — generic booru taggers were bad enough at this to be the live blocker, which is why **Anima Tagger** exists.

- **Anima Tagger** (`library/captioning/anima_tagger.py`): small classifier mapping image → comma-separated tag string in exactly Anima's training-time T5 format. Frozen PE-Core-L14-336 trunk + LayerNorm + Linear head. Train with `python -m scripts.anima_tagger.cli` against `$CAPTION_CORPUS_DIR` (vocabulary + manifest + feature cache builders + threshold calibrator all wired). DirectEdit's case-1 ψ_src path requires this checkpoint at `models/captioners/anima-tagger-v1/`.
- **DirectEdit core** (`library/inference/directedit.py`): the invert+edit primitive. Invoked from `scripts/edit.py` (CLI), `scripts/experimental_tasks/inference.py` (`make exp-test-directedit` / `exp-test-directedit-dry`), and `custom_nodes/comfyui-anima-directedit/` (ComfyUI nodes).

Use `make exp-test-directedit-dry` to verify ψ_tar == ψ_src reconstructs the source — gates whether the inversion is well-conditioned independent of edit semantics.

See `docs/experimental/directedit_editing_v3.md` (what's built) and `docs/experimental/anima_tagger.md` (tagger architecture).

## Modulation guidance

Text-conditioned AdaLN modulation via a learned `pooled_text_proj` MLP (Starodubcev et al., ICLR 2026). Distilled with `make distill-mod`: teacher uses real cross-attention, student uses zeroed cross-attention but receives pooled text through modulation. At inference, steers AdaLN coefficients toward quality-positive directions. See `docs/methods/mod-guidance.md`.

## IP-Adapter

Decoupled image cross-attention (Ye et al. 2023). DiT is frozen; trains only the Perceiver resampler and per-block parallel `to_k_ip`/`to_v_ip` projections (~150M params at default `K=16`, 28 blocks). Reference image → frozen vision tower (PE-Core-L14-336 by default) → resampler → K compact IP tokens → per-block KV → patched cross-attention adds `scale * SDPA(text_q, ip_k, ip_v)` to the existing text cross-attention. Reuses the LoRA pipeline's data layout — source images live under `post_image_dataset/resized/` and caches (latents, text-emb, PE features) live under `post_image_dataset/lora/`. Defaults to PRE-CACHED PE features (`{stem}_anima_pe.safetensors` from `make preprocess-pe`) so training never loads the vision encoder. CFG dropout (`image_drop_p`) zeros image conditioning so inference can do image-CFG independently of text-CFG. See `docs/experimental/ip-adapter.md`.

## EasyControl

Extended self-attention image conditioning. DiT is frozen; trains per-block cond LoRA on self-attn (q/k/v/o) + FFN (layer1/layer2) plus a per-block scalar logit-bias `b_cond` (init `-10`) that gates cond-position softmax mass. Reference is VAE-encoded and patch-embedded by the DiT's frozen `x_embedder` into condition tokens that flow through every block alongside the target stream; target self-attention attends to a key set extended with the cond stream's keys/values. Training uses a **two-stream block forward** (target + cond in one scope, no deferred-backward dance); inference prefills a per-block `(K_c, V_c)` cache once at setup and reuses it across every denoising step and every CFG branch (cond is deterministic — `cond_temb = t_embedder(0)`). Source images live in `easycontrol-dataset/`; caches go to `post_image_dataset/easycontrol/` via subset `cache_dir`. Reuses cached VAE latents — no new sidecar. See `docs/experimental/easycontrol.md`.

## Soft Tokens (SoftREPA parameterization)

Per-layer time-indexed soft text tokens (Lee et al., arXiv:2503.08250, NeurIPS 2025). DiT frozen; trains a `(n_layers, K, D)` token bank + `(n_t_buckets, n_layers, D)` t-offsets — ~1M params at default. For each of the first `n_layers` blocks, a `(layer, t-bucket)`-specific token slice is spliced into `crossattn_emb` via a per-block `Block.forward` monkey-patch (ReFT-pattern); end-of-sequence overwrite of zero-padding tail (or `front_of_padding` scatter) keeps `_run_blocks` torch.compile shape-static. Anima's cross-attention (not joint-stream MM-DiT) means each block independently sees a different `crossattn_emb` — no strip/re-prepend dance. Adopts only the parameterization from the SoftREPA paper; the contrastive InfoNCE objective is intentionally skipped (caused SD3 FID regression). Inference is wired both ways: standalone via `inference.py --soft_tokens_weight` (the denoising loop fires `network.append_postfix(..., timesteps=t)` per CFG branch before each forward — see `library/inference/generation.py`), and in ComfyUI via the `AnimaSoftTokensLoader` node (`custom_nodes/comfyui-hydralora/soft_tokens.py`), which reproduces the per-block splice with `forward_pre_hook`s and divides comfy's `sigma×1000` FLOW timesteps back to `[0,1]` for the t-bucket index.

## ChimeraHydra (dual-pool additive MoE)

Two HydraLoRAs glued at the residual: a **content** pool (K_c B-heads routed by the per-Linear rank-R lx-router from HydraLoRA, Tian et al. NeurIPS'24, arXiv:2404.19245) and a **frequency** pool (K_f B-heads routed by the network-level `FreqRouter` fed `concat(FEI(z_t), sinusoidal-σ)`, FeRA-shaped, arXiv:2511.17979). Two A's per adapted Linear — one shared within each pool — both Cayley-rotated off disjoint SVD subspaces of W, so the row spaces of A_c and A_f are orthogonal and every B in pool c is column-orthogonal to every B in pool f. T-LoRA's rank mask (TimeStep Master, arXiv:2503.07416) modulates the content branch only; the freq branch stays full-rank at every t. Pool outputs are added — no multiplicative gate, no σ-band overlap mask; specialization comes from router-input separation (content router can't see σ; freq router can't see pooled text). Single phase: both pools, both routers, and the shared A's train jointly under denoise + per-pool balance loss. See `docs/proposal/chimera_hydra.md` (with 2026-05-15 erratum) and `networks/lora_modules/chimera.py`.

## Postfix-tail per-image inversion

The postfix *training method* was archived (2026-05-20 — soft_tokens superseded it; see `_archive/postfix/`). What stays live is this **inversion probe**: optimize a per-image K-slot residual on an orthonormal postfix tail (`Q @ diag(s)`) so it reconstructs a target image against the frozen DiT — an observation tool, not a deployable adapter. It needs no postfix-trained checkpoint (it builds its own SVD-of-cached-TE / random basis). Code in `library/inference/postfix_inversion.py` (self-contained — the basis builders `_build_svd_te_basis` / `_make_orthonormal_basis` were inlined here when the method was archived) + `scripts/inversion/invert_postfix_tail.py`; wired in `tasks.py` (`exp-invert-directedit`). See `docs/proposal/postfix_residual_per_image_inversion.md`.

## Preprocessing

Data preparation scripts in `preprocess/`:
- `resize_images.py` — VAE-compatible image resizing (used by `make preprocess-resize`). Reads `image_dataset/`, writes resized PNGs to `post_image_dataset/resized/`. Drops images below `--min_pixels` (default 0.5MP). `--no_copy_captions` skips the `.txt` copy so captions stay only in `image_dataset/`.
- `cache_latents.py` — Cache VAE latents (used by `make preprocess-vae`). Reads `post_image_dataset/resized/`, writes `{stem}_{WxH}_anima.npz` into `post_image_dataset/lora/` via `--cache_dir`.
- `cache_text_embeddings.py` — Cache text encoder outputs (used by `make preprocess-te`). Reads `image_dataset/` (where `.txt` lives) and writes `{stem}_anima_te.safetensors` into `post_image_dataset/lora/` via `--cache_dir`. Mirrors `resize_images.py`'s `--min_pixels` filter so caches don't accumulate for images that would be dropped at resize.
- `cache_pe_encoder.py` — Cache PE-Core-L14-336 vision encoder features (`{stem}_anima_pe.safetensors`). Wrapped by `make preprocess-pe` (reads `post_image_dataset/resized/`, writes `post_image_dataset/lora/`). IP-Adapter and the DCW v4 fusion head consume the same sidecars from `post_image_dataset/lora/`. Pass `--centroid` (after cache pass) or `--centroid_only` (pool existing caches) to emit the dataset-mean PE-feature centroid sidecar consumed by IP-Adapter (`ip_centroid_path`) and DCW v4 (`cos(c_pool, μ_centroid)` channel).
- `make preprocess-pooled` — Cache pooled text-embedding sidecars (consumed by `make distill-mod`). CPU-only; reads existing `_anima_te.safetensors` and writes pooled companions next to them.
- `generate_masks.py` — SAM3-based text bubble mask generation
- `generate_masks_mit.py` — MIT/ComicTextDetector mask generation (manga-specific)
- `merge_masks.py` — Combine SAM3 + MIT masks into final mask set

## Scripts

Utility scripts in `scripts/`:
- `distill_mod/` — Modulation guidance distillation package: `prep.py` (Phase 1 + 2 staging, `make distill-prep`), `distill.py` (pooled_text_proj trainer, `make distill-mod`), shared `uncond.py` / `synth.py` / `teacher_cache.py` / `validation.py`
- `comfy_batch.py` — Run ComfyUI batch workflow from `workflows/` directory
- `merge_to_dit.py` — Bake a LoRA adapter into the base DiT (used by `make merge`)
- `export_logs_json.py` — Export TensorBoard run scalars to JSON/JSONL (used by `make export-logs`)
- `anima_tagger/cli.py` — Train the Anima Tagger checkpoint used by DirectEdit (invoke as `python -m scripts.anima_tagger.cli`). See `docs/experimental/anima_tagger.md`.
- `edit.py` — Standalone DirectEdit CLI entry (the `make exp-test-directedit` wrapper around it lives in `scripts/experimental_tasks/inference.py`).
- `scripts/dcw/` — DCW v4 calibration pipeline: `measure_bias.py` (per-aspect trajectory dump), `train_fusion_head.py` (fusion-head trainer), `trajectory.py`, `haar.py`, etc. Driven by `make dcw` / `make dcw-train` (`scripts/tasks/dcw.py`).

## Custom nodes

Spectrum KSampler and mod guidance ComfyUI nodes live in a separate repo: https://github.com/sorryhyun/ComfyUI-Spectrum-KSampler (the Spectrum node also ships DCW: scalar default `+0.01` plus `auto` mode that runs the v4 fusion head in-node — see `reference_spectrum_node_dcw_defaults` memory).

In-tree under `custom_nodes/`:
- `comfyui-hydralora/` — three loader nodes: **Anima Adapter Loader** (LoRA / Hydra / ReFT), **Anima FeRA Loader**, and **Anima Soft Tokens Loader** (SoftREPA per-block crossattn splice). (The Anima Postfix Loader was retired when the postfix method was archived.) See `custom_nodes/comfyui-hydralora/CLAUDE.md` for code-level details and the `forward_hook`-not-`forward`-override invariant; `README.md` for user-facing docs and changelog.
- `comfyui-anima-directedit/` — `AnimaDirectEdit` node (invert + edit on a frozen Anima checkpoint, using stock MODEL/CLIP/VAE sockets). Consumes the `ANIMA_TAGGER` socket from `comfyui-anima-tagger`, or skips the tagger via the `prompt_src_override` STRING input. See its own `README.md`.
- `comfyui-anima-tagger/` — `AnimaTaggerLoader` (→ `ANIMA_TAGGER` socket) and `AnimaTaggerCaption` (`ANIMA_TAGGER` + `IMAGE` → `STRING`). Standalone captioner usable outside DirectEdit (LoRA caption pre-fill, prompt scaffolding, etc.).
- `comfyui-anima-trainer/` — single **Anima LoRA Trainer** node: one image + caption → trains a T-LoRA + OrthoLoRA via the local daemon, then loads the chosen base checkpoint and returns it patched (a drop-in for the Anima Adapter Loader's MODEL). No MODEL input (avoids holding the DiT resident during training → OOM); errors if the daemon isn't already running.
- `comfyui-anima-blockcompile/` — single **Anima Block Compile** (`AnimaBlockCompile`) node that `torch.compile`s the DiT one transformer block at a time (faster compile, fewer graph breaks) — the same `--compile_mode blocks` strategy the training/inference pipeline uses by default.

The hydralora, anima-tagger, anima-directedit, and anima-trainer nodes each carry a `_vendor/` subset of the live tree so they keep working when installed outside the anima_lora repo. The vendor trees are regenerated by `scripts/sync_vendor.py` (`make vendor-sync`); for hydralora the canonical kernels are `library/inference/router_compute.py` + `library/runtime/fei.py` + `networks/lora_modules/router_state.py`; for anima-trainer it's the stdlib daemon *client* (`scripts/daemon/{config,client}.py` verbatim + a psutil-free `proc.py` trimmed to `read_pidfile`). Re-run before every node publish — see [[feedback_vendor_sync]].

## External tools

ComfyUI, SAM3, and manga-image-translator live in the parent directory (`../comfy/`, `../sam3/`, etc.).

## Contributing

PRs are reviewed against a tier system spelled out in `CONTRIBUTING.md` (Tier 1 = bug fixes / typos; Tier 1.5 = efficiency or numerics revisions to existing methods — bench script + invariant test required; Tier 2 = new adapter method — paper citation + `bench/<method>/` subdir + docs entry + `make <name>` / `make test-<name>` targets; Tier 3 = new base-model support, currently not accepted). The `bench/<method>/` convention (README + runnable script + timestamped `results/`) is how method PRs prove their claims. Bench scripts share `bench/_common.py` (`make_run_dir` + `write_result`) — every run drops a standard `result.json` envelope (git SHA, env, args, metrics, artifacts) into `bench/<method>/results/<YYYYMMDD-HHMM>[-<label>]/` so cross-run indexing and reproduction are uniform.
