# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

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

Both `make` (Unix) and `python tasks.py` (cross-platform/Windows) work — the `Makefile` is a thin dispatcher forwarding every target to `python tasks.py <target> $(ARGS)`. **`tasks.py` is the source of truth**; command bodies live in `scripts/tasks/{training,inference,preprocess,masking,webui,downloads,utilities,tagger,dcw}.py` and `scripts/experimental_tasks/` (for `exp-*`). Don't grep the Makefile for a recipe — look there.

All training runs `train.py --method <name> --preset <name>`. By default it's invoked **directly** (single-GPU fast path — skips the ~5s accelerate launcher bootstrap; `train.py` builds its own single-process `Accelerator()` and reads `mixed_precision` from the config chain). Set `ANIMA_ACCELERATE_LAUNCH=1` to wrap it in `accelerate launch` for multi-GPU / distributed runs (see `build_launch_cmd` in `scripts/tasks/_common.py`). Override any config value from CLI (`--network_dim 32 --max_train_epochs 64`) or the preset via `PRESET=low_vram make lora`. `exp-*` targets are experimental — may break or be removed.

```bash
# Training (run from anima_lora/) — method + hardware preset; method wins on overlap
make lora                                # methods/lora.toml + presets.toml[default]
make lora PRESET=low_vram|fast_16gb|half # override preset (half → sample_ratio=0.5)
make lora-gui GUI_PRESETS=tlora          # clean per-variant configs/gui-methods/ (no toggle blocks)
                                         #   `ls configs/gui-methods/` for the live variant list
make exp-ip-adapter | exp-easycontrol | exp-soft-tokens | exp-chimera | exp-turbo

# Inference (latest output) — SPECTRUM=1 / MOD=1 / NOLORA=1 compose into every test-* target
make test [MOD=1] [NOLORA=1] [SPECTRUM=1]
make test-hydra            # HydraLoRA / FeRA router-live checkpoints
make test-merge            # merged/baked DiT (no adapter)
make test-dcw | test-dcw-v4 | test-smc-cfg     # DCW scalar / v4 calibrator / SMC-CFG
make exp-test-soft | exp-test-turbo | exp-test-ip REF_IMAGE=... | exp-test-easycontrol REF_IMAGE=...
make exp-test-directedit PROMPT='...' | exp-test-directedit-dry

# Modulation guidance distillation
make distill-prep          # stage uncond sidecar + teacher-synthetic clean-latents pool
make distill-mod           # train pooled_text_proj MLP (add --synth_data_dir for paper-faithful fit)

# DCW v4 calibration (one-shot per LoRA checkpoint)
make dcw                   # sample 5 aspect buckets + train fusion head (~3-5h on a 5060 Ti)
make dcw-train             # train-only on existing pool (~30s)

python -m webui            # WebUI (FastAPI + Vue 3 — config editing, dataset browsing, training)
make mask | mask-clean     # SAM3 + MIT → post_image_dataset/masks/ (for masked loss)
make merge ADAPTER_DIR=output/ckpt [MULTIPLIER=0.8]   # bake LoRA into DiT (LoRA/Ortho/T-LoRA only)
make comfy-batch           # run ComfyUI batch workflow
make print-config METHOD=lora PRESET=default          # dump merged config chain
make test-unit             # pytest tests/ (smoke, config, loss/network registries)
make export-logs RUN=...   # export TensorBoard run to JSON
make update                # update from a GitHub release (--dry-run / --version / --no-sync)
ruff check . --fix && ruff format .
```

Gotchas: `merge` refuses ReFT / Hydra moe / postfix (not foldable) unless `--allow-partial`. `turbo` output is a normal LoRA — infer with `--infer_steps 4 --cfg 1.0`.

## Key entry points

| File | Purpose |
|------|---------|
| `anima_lora/__init__.py` | **Programmatic front door** — lazy (PEP 562) re-export of the curated embedder entry points (`generate`, `get_generation_settings`, `GenerationRequest`, `load_method_preset`, `load_dit_model`, `load_vae`, …) + `ROOT` (repo root). `import anima_lora` instead of reverse-engineering `main()`s. |
| `examples/` | Runnable API scripts (`01`–`04` high-level flows, `05`–`06` raw primitives). `examples/README.md` is the embedder guide. |
| `train.py` | `AnimaTrainer` — main training loop via HF Accelerate |
| `inference.py` | Standalone image generation (`--help` for all flags) |
| `networks/spectrum.py` | Spectrum inference acceleration |
| `webui/` | FastAPI + Vue 3 WebUI (config editing, dataset browsing, training, system management) |
| `tasks.py` | Cross-platform task runner — source of truth for every `make` target |
| `scripts/tasks/` + `scripts/experimental_tasks/` | Where command bodies actually live (`_common.py` = shared helpers) |

Docs: shipped method deep-dives in `docs/methods/`, experimental in `docs/experimental/`, active proposals in `docs/proposal/`, retired material under `_archive/`.

## Programmatic API (embedders)

`uv sync` installs the repo editable, so `anima_lora` is importable anywhere. It's a thin façade — canonical homes are unchanged (`library.inference` / `library.config.io` / `library.anima.weights` / `library.models.qwen_vae` / `library.runtime.device`). Inference is **request-driven**: build a typed `GenerationRequest`, call `.to_args()` (which routes through `inference.parse_args` so every `getattr()`-read knob is populated; long-tail method flags ride `extra_argv`). Adapter family lives **in the checkpoint metadata**, not the call — the DiT loader merges-or-keeps-live accordingly. Prompt encoding installs two process-global strategy singletons lazily (`ensure_text_strategies`). Repo-relative model/config paths resolve against the **repo home** (`library.env.anima_home()` / `resolve_under_home()`), not the CWD — so `import anima_lora` works from any directory; set `ANIMA_HOME` for a relocated checkout, or override individual model paths with `ANIMA_DIT` / `ANIMA_VAE` / `ANIMA_TEXT_ENCODER`. The anchor is wired at the config-loader chokepoint (`library/config/io.py`) and the model-loader leaves (`load_anima_model` / `load_vae` / `load_qwen3_text_encoder`); new code opening a repo-relative path should call `resolve_under_home()` rather than assuming CWD.

## Config flow

Config-driven via a three-layer merge chain: `base.toml → presets.toml[<preset>] → methods/<method>.toml → CLI args`. **Method settings win over preset settings on overlap**, so a method can force its own hardware requirements (e.g. a frozen-DiT method forcing `blocks_to_swap=0`).

- `configs/base.toml` — shared infra (model paths, optimizer, compile) AND the default LoRA dataset blueprint (`[general]` + `[[datasets]]` + `[[datasets.subsets]]`, consumed by `BlueprintGenerator`, skipped by the flat method+preset merge — see `_DATASET_CONFIG_SECTIONS`). Use `--dataset_config` for a different blueprint, or drop a `[general]`/`[[datasets]]` block into the method TOML to shallow-override top-level scalars (`_apply_dataset_overrides` in `library/config/io.py`; subset-level overrides not supported this way).
- `configs/presets.toml` — hardware profiles as sections: `[default]`, `[fast_16gb]`, `[low_vram]` (also Windows 8GB), `[half]`. Holds `blocks_to_swap`, gradient/offload checkpointing, etc.
- `configs/methods/` — one flat file per family read by `train.py` (`lora`, `chimera`, `ip_adapter`, `easycontrol`, `soft_tokens`), each holding rank + routing knobs + opinionated LR/epochs/output_name. `turbo.toml` is the **odd one out**: a bespoke sectioned schema read only by `scripts/distill_turbo/` — don't `print-config METHOD=turbo`. Variants inside `lora.toml` are comment-toggle blocks; default stacks LoRA + OrthoLoRA + T-LoRA + shared_A FEI-routed Hydra. **Pre-three-axis checkpoints (`ss_use_hydra`/`ss_use_fei_router` metadata) no longer load** — legacy fallback removed.
- `configs/gui-methods/` — clean per-**variant** parallel tree, no toggle blocks (what you see is what runs). Selected via `--methods_subdir gui-methods` (wrapped by `make lora-gui`). `ls` for the live list.

Subsets accept `cache_dir` — redirects all VAE/TE/PE caches to that dir with stem-mirrored names (IP-Adapter & EasyControl use this to keep source dirs user-facing while caches live under `post_image_dataset/`). `library.config.io.load_method_preset(method, preset, methods_subdir=...)` is the reusable merge helper (not re-exported via `train_util`). All config paths are relative to `anima_lora/`. Outputs split by kind: checkpoints (+ `.snapshot.toml` + `_moe` siblings) in `output/ckpt/`, inference images in `output/tests/`.

## Architecture

- **Modular `library/`**: `train_util.py` is a re-exporting facade; code lives in domain subpackages — `anima/` (DiT model, training helpers, weights, strategy), `datasets/` (incl. `cache.py` = general cached-pair train reader `CachedDataset`, re-exported by `distill.py` for back-compat), `training/` (optimizer/scheduler/checkpoint + loss/sampler/metric registries), `inference/` (engine: generation, sampling, models, text, adapters, sampler_context, `request.py` = typed `GenerationRequest`; plug-ins split into `corrections/` — DCW / SMC-CFG / mod-guidance — and `editing/` — DirectEdit + postfix inversion), `preprocess/` (dataset-caching **orchestration**: `images`/`latents`/`text`/`pe` — the scan→group-by-shape→batched-encode→idempotent-write loops), `models/` (VAE, metadata spec), `captioning/` (Anima Tagger), `vision/` (vision tower/resampler), `config/` (schema + loader), `io/` (`cache.py` = cache-path resolution + suffixes + discovery, `safetensors.py`), `runtime/` (device/offloading/noise + `cli.py` shared argparse surface + `harness.py` `build_anima` model-build harness), `env.py`, `log.py`.
- **Tooling layering contract**: **primitives** (`library/*` — load a model, encode a batch, resolve a cache path) → **façade** (`anima_lora/` — embedder entry points) → **orchestration** (`library/preprocess/`, `library/runtime/harness.py` — drive primitives over a whole dataset/run) → **entry points** (`scripts/preprocess/*.py`, `bench/**/run_bench.py`, `scripts/**`, `tasks.py` — thin argparse wrappers). `scripts/preprocess/*.py` are now thin CLI shells over `library/preprocess/`. `bench/`, `scripts/` are **not** installed packages (only `anima_lora`/`library`/`networks` are) — they keep a `sys.path` bootstrap to import siblings.
- **Strategy pattern** for tokenization/encoding (`library/anima/strategy.py`, `library/strategy_base.py`).
- **Pluggable adapters** under `networks/` — selected via `network_module` + (for LoRA family) the three-axis routing cfg. LoRA modules in `networks/lora_modules/` coordinated by `networks/lora_anima/`; IP-Adapter/EasyControl in `networks/methods/`; attention dispatcher `networks/attention_dispatch.py`; Spectrum `networks/spectrum.py`; SPD `networks/spd.py`. **See `networks/CLAUDE.md`** for the per-module map, three-axis surface, and dispatch invariants.

## Critical invariants

### Text encoder padding
The pretrained model expects max-padded text encoder outputs — zero-padded positions act as attention sinks in cross-attention softmax. Trimming to actual text length produces **black images**. Both training and inference must pad to `max_length` and must NOT mask out padding via `crossattn_seqlens`. Regenerate disk-cached `.npz` after any tokenizer/padding change.

### Constant-token bucketing — native shapes are the only mode
`CONSTANT_TOKEN_BUCKETS` (`library/datasets/buckets.py`) is **two token-count families — 4032 and 4200** — each entry *exactly* filling its count (zero intra-bucket padding by construction), tuples in `(W, H)` order. Each forward runs at its real token count. `compile_blocks()` is the single switch: when `torch_compile` is on it sets `_native_flatten` so the forward flattens each bucket's patch grid to a fake-5D `(B, 1, seq_len, 1, D)` shape, keying the block graph on **token count alone (2 graphs)** instead of per-resolution (24). No padding → no flash static-pad leak; bit-exact to the eager 5D path (eager forwards skip the flatten). The legacy pad-to-static path (`set_static_token_count(pad=True)` / `compile_core` / `--compile_mode full` / `static_token_count` / `static_pad`) was **removed 2026-05-24** — it leaked padding into flash self-attn and couldn't run this table (4200 > 4096). Note this diverges from `DCW_ASPECT_BUCKETS` (the 4056 HD pair is no longer a training bucket). Regenerate disk caches after changing the table.

### Lazy model loading
DiT loads AFTER text-encoder/VAE caching and unloading, to avoid OOM: text encoder → cache → free → VAE → cache → free → load DiT → attach adapter → train.

### compile-after-apply (`build_anima`)
`torch.compile` traces the adapter's monkey-patched forward, so `compile_blocks()` MUST run **after** `network.apply_to` + `load_weights`. `library/runtime/harness.py::build_anima` is the shared harness encoding this ordering (promoted from `bench/_anima.py`); use it from `bench`/`scripts`/`preprocess` rather than open-coding load→apply→compile.

## Methods

Each method has a deep-dive doc; the prose below is one-line orientation plus the load-bearing gotcha. Read the doc before working on one.

| Method | What it is | Gotcha / pointer |
|---|---|---|
| **Spectrum** | Training-free speedup via Chebyshev feature forecasting (`--spectrum`). Cached steps skip all blocks; `final_layer` pre-hook captures outputs. | `docs/methods/spectrum.md` |
| **SPD** | Training-free multi-resolution inference (`--spd`): early steps at low res, spectral noise-expansion handoff to full res. Sampler-level runner in `networks/spd.py`, registered like Spectrum. | v0 = Euler-only, no DCW/SMC/Spectrum compose, single-late `0.5→1.0 @ σ0.7` default. `docs/experimental/spd.md`; `bench/spd/plan.md` Phase 3, `docs/proposal/spd_finetune_lora.md` (Case B). |
| **DCW** | Training-free SNR-t bias correction at the sampler boundary; composes with everything. Scalar (`--dcw`) or v4 learnable (`--dcw_v4 auto`). | **Bias direction is (CFG × aspect)-dependent** — shipped scalar `−0.015` is CFG=1-only and wrong-sign on CFG=4 non-square. `docs/methods/dcw.md` |
| **SMC-CFG** | Training-free α-adaptive sliding-mode CFG correction in velocity space (λ=5, α=0.2). | Paper's fixed k was ~14× off; ships `sign()` only (tanh ε removed). `docs/methods/smc_cfg.md` |
| **Mod guidance** | Text-conditioned AdaLN via learned `pooled_text_proj` MLP, distilled with `make distill-mod`. | `docs/methods/mod-guidance.md` |
| **DirectEdit + Anima Tagger** | Inversion + edit-conditioning swap; Tagger (`library/captioning/`) maps image → Anima-format tags for ψ_src. | Edit leverage collapses if ψ_src is off-manifold — verify with `exp-test-directedit-dry`. `docs/experimental/directedit_editing_v3.md`, `anima_tagger.md` |
| **IP-Adapter** | Decoupled image cross-attention; frozen DiT, trains resampler + per-block `to_k_ip`/`to_v_ip`. Defaults to pre-cached PE features. | `docs/experimental/ip-adapter.md` |
| **EasyControl** | Extended self-attn image conditioning; frozen DiT, per-block cond LoRA + scalar `b_cond` gate. Source `easycontrol-dataset/`. | `docs/experimental/easycontrol.md` |
| **Soft Tokens** | SoftREPA per-layer × per-t soft text tokens (~1M params); frozen DiT, per-block `Block.forward` splice into `crossattn_emb`. | InfoNCE objective intentionally skipped. `configs/methods/soft_tokens.toml` |
| **ChimeraHydra** | Dual-pool additive MoE: content pool (network ContentRouter on pooled `crossattn_emb`) + freq pool (network FreqRouter on FEI+σ), two A's per Linear off disjoint SVD subspaces. Both pools always centered-gate; the per-Linear `lx_c` content router + non-centered path were removed. | T-LoRA mask hits content branch only. `docs/experimental/chimera-hydra.md`, `networks/lora_modules/chimera.py` |
| **Turbo** | Decoupled-Hybrid DMD2 distillation; output is a normal LoRA. | Bespoke schema read by `scripts/distill_turbo/` — don't `print-config`. `docs/experimental/dmd2-decoupled.md` (ops) + `docs/structure/dmd2-decoupled.md` (math). |
| **Postfix-tail inversion** | Per-image inversion *probe* (training method archived 2026-05-20). | Observation tool, not a deployable adapter. `library/inference/postfix_inversion.py` |

## Preprocessing & scripts

Data-prep scripts in `scripts/preprocess/` (resize → VAE latents → text embeddings → PE features → masks); see file headers for flags and `make preprocess-{resize,vae,te,pe,pooled}` / `make mask`. **The caching logic moved to `library/preprocess/` — these scripts are now thin argparse wrappers**; edit the orchestration in the library, the flags in the script. Other utility scripts in `scripts/` — notably `distill_mod/` (mod-guidance distillation), `merge_to_dit.py`, `dcw/` (DCW v4 calibration pipeline), `anima_tagger/cli.py`, `edit.py`, `export_logs_json.py`.

Caches live under `post_image_dataset/lora/`: `{stem}_{WxH}_anima.npz` (VAE), `{stem}_anima_te.safetensors` (text), `{stem}_anima_pe.safetensors` (PE). TE caching reads `.txt` from `image_dataset/` (the caption master); training reads only cached embeddings.

## Custom nodes

Spectrum KSampler + mod-guidance nodes live in a separate repo (https://github.com/sorryhyun/ComfyUI-Spectrum-KSampler; ships DCW scalar default `+0.01` + `auto` mode). In-tree under `custom_nodes/`: `comfyui-hydralora/` (Adapter / FeRA / Soft Tokens loaders — see its `CLAUDE.md` for the `forward_hook`-not-override invariant), `comfyui-anima-directedit/`, `comfyui-anima-tagger/`, `comfyui-anima-trainer/`, `comfyui-anima-blockcompile/`.

Several nodes carry a `_vendor/` subset of the live tree. **Regenerate vendor trees with `make vendor-sync` (`scripts/sync_vendor.py`), never `cp` by hand** — re-run before every node publish. See [[feedback_vendor_sync]]. Note `../comfy/custom_nodes/` is symlinked into this repo — edit the source here, not the symlink.

## External tools

ComfyUI, SAM3, and manga-image-translator live in the parent directory (`../comfy/`, `../sam3/`, etc.).

## Contributing

PRs follow a tier system in `CONTRIBUTING.md` (Tier 1 = bugfixes/typos; Tier 1.5 = numerics/efficiency revisions — bench script + invariant test required; Tier 2 = new adapter method — paper citation + `bench/<method>/` + docs + `make` targets; Tier 3 = new base-model support, not accepted). Bench scripts share `bench/_common.py` and drop a standard `result.json` envelope into `bench/<method>/results/<YYYYMMDD-HHMM>[-label]/`.
