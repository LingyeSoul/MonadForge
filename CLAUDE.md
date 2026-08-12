# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

## Project Overview

Anima — LoRA/T-LoRA training and inference pipeline for the Anima diffusion model (DiT-based, flow-matching). Supports several adapter families (LoRA / OrthoLoRA / T-LoRA / HydraLoRA / FeRA / ChimeraHydra / EasyControl) selectable via method config + hardware preset. The LoRA family is routed via a three-axis surface — `use_moe_style` / `route_per_layer` / `router_source` — see `configs/methods/lora.toml`.

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

`make help` lists every target; the canonical bodies are in `tasks.py`. Non-obvious knobs and gotchas worth knowing up front:

# Inference (latest output) — SPECTRUM=1 / MOD=1 / NOLORA=1 compose into every test-* target
make test [MOD=1] [NOLORA=1] [SPECTRUM=1]
make test-hydra            # HydraLoRA / FeRA router-live checkpoints
make test-merge            # merged/baked DiT (no adapter)
make test-dcw | test-dcw-v4 | test-smc-cfg     # DCW scalar / v4 calibrator / SMC-CFG
make exp-test-soft | test-turbo | exp-test-ip REF_IMAGE=... | exp-test-easycontrol REF_IMAGE=...
make exp-test-directedit PROMPT='...' | exp-test-directedit-dry

# Modulation guidance distillation
make distill-prep          # stage uncond sidecar + teacher-synthetic clean-latents pool
make distill-mod           # train pooled_text_proj MLP (add --synth_data_dir for paper-faithful fit)

# DCW v4 calibration (one-shot per LoRA checkpoint)
make dcw                   # sample 5 aspect buckets + train fusion head (~3-5h on a 5060 Ti)
make dcw-train             # train-only on existing pool (~30s)

python -m webui            # WebUI (FastAPI + Vue 3 — config editing, dataset browsing, training)
make mask | mask-clean     # SAM3 + MIT → post_image_dataset/masks/ (for masked loss)
make merge ADAPTER_DIR=output/ckpt [MULTIPLIER=0.8]   # bake LoRA into DiT (LoRA/Ortho/T-LoRA/LoKR/GLoKr)
make comfy-batch           # run ComfyUI batch workflow
make print-config METHOD=lora PRESET=default          # dump merged config chain
make test-unit             # pytest tests/ (smoke, config, loss/network registries)
make export-logs RUN=...   # export TensorBoard run to JSON
make update                # update from a GitHub release (--dry-run / --version / --no-sync)
ruff check . --fix && ruff format .
```

Gotchas: `merge` refuses ReFT / Hydra moe / postfix (not foldable) unless `--allow-partial`. `turbo` output is a normal LoRA — infer with `--infer_steps 2 --cfg 1.0` (matched to the DP-DMD `student_steps=2` rollout).
## Key entry points

| File | Purpose |
|------|---------|
| `anima_lora/__init__.py` | **Programmatic front door** — lazy (PEP 562) re-export of the curated embedder entry points (`generate`, `get_generation_settings`, `GenerationRequest`, `load_method_preset`, `load_dit_model`, `load_vae`, …) + `ROOT` (repo root). `import anima_lora` instead of reverse-engineering `main()`s. |
| `examples/` | Runnable API scripts (`01`–`04` high-level flows, `05`–`06` raw primitives). `examples/README.md` is the embedder guide. |
| `train.py` | `AnimaTrainer` — main training loop via HF Accelerate |
| `inference.py` | Standalone image generation (`--help` for all flags) |
| `networks/spectrum.py` | Spectrum inference acceleration |
| `webui/` | FastAPI + Vue 3 WebUI (config editing, dataset browsing, training, system management). Its `services/task_service.py` submits every task to the daemon (below) as a *command* job and tails the daemon-managed stdout + progress.jsonl. |
| `scripts/daemon/` | **Training daemon** — localhost serial job queue (`127.0.0.1:8765`, no auth): one job at a time (GPU guard), state persists to `output/daemon/` (survives WebUI restart), `chain_train` (preprocess→train), REST + Python client (`scripts/daemon/client.py`) + stdio MCP bridge (`mcp.py`) so AI agents can submit/watch/stop jobs. CLI: `python tasks.py daemon[-status\|-attach\|-kill\|-terminate]`. `--queue` on any train target enqueues here. **Also hosts the WebUI as a supervised sidecar** (`webui_sidecar.py`) — the daemon spawns the uvicorn server on boot, respawns it on crash, and tree-kills it on shutdown; the sidecar is NOT a job (excluded from the serial queue + `active_job`). Disable with `ANIMA_DAEMON_HOST_WEBUI=0` (CLI/ComfyUI-only setups). |
| `scripts/tray/` | **Windows system-tray app** (pystray) — status indicator (idle/running/error/down) + controller (open WebUI, pause/resume queue, stop job, restart daemon, language switcher 中/英). Run `pythonw -m scripts.tray` or the `monadforge-tray` gui-script. Icons are procedural (Pillow) in `icons.py`; strings localized in `i18n.py` (own choice persisted to `output/daemon/tray-prefs.json`, default 中文 — independent of the WebUI's language). |
| `tasks.py` | Cross-platform task runner — source of truth for every `make` target |
| `scripts/tasks/` + `scripts/experimental_tasks/` | Where command bodies actually live (`_common.py` = shared helpers, incl. `_queue_submit`/`queue_command` for the daemon `--queue` path) |

Docs: shipped method deep-dives in `docs/methods/`, experimental in `docs/experimental/`, active proposals in `docs/proposal/`, retired material under `_archive/`.

## Programmatic API (embedders)

`uv sync` installs the repo editable, so `anima_lora` is importable anywhere. It's a thin façade — canonical homes are unchanged (`library.inference` / `library.config.io` / `library.anima.weights` / `library.models.qwen_vae` / `library.runtime.device`). Inference is **request-driven**: build a typed `GenerationRequest`, call `.to_args()` (which routes through `inference.parse_args` so every `getattr()`-read knob is populated; long-tail method flags ride `extra_argv`). Adapter family lives **in the checkpoint metadata**, not the call — the DiT loader merges-or-keeps-live accordingly. Prompt encoding installs two process-global strategy singletons lazily (`ensure_text_strategies`). Repo-relative model/config paths resolve against the **repo home** (`library.env.anima_home()` / `resolve_under_home()`), not the CWD — so `import anima_lora` works from any directory; set `ANIMA_HOME` for a relocated checkout, or override individual model paths with `ANIMA_DIT` / `ANIMA_VAE` / `ANIMA_TEXT_ENCODER`. The anchor is wired at the config-loader chokepoint (`library/config/io.py`) and the model-loader leaves (`load_anima_model` / `load_vae` / `load_qwen3_text_encoder`); new code opening a repo-relative path should call `resolve_under_home()` rather than assuming CWD.

## Config flow

Config-driven via a layered merge chain: `model.toml → base.toml (legacy model keys only) → custom/model.toml → presets.toml[<preset>] → methods/<method>.toml → CLI args`; non-model settings enter at the normal `base.toml` layer. **Method settings win over preset settings on overlap**, so a method can force its own hardware requirements (e.g. a frozen-DiT method forcing `blocks_to_swap=0`).

- `configs/model.toml` — shipped repo-relative model defaults. Machine-local paths belong in gitignored `configs/custom/model.toml`; the training, preprocess, programmatic API, and WebUI model settings all read the same effective layer.
- `configs/base.toml` — shared infra (optimizer, compile) AND the default LoRA dataset blueprint (`[general]` + `[[datasets]]` + `[[datasets.subsets]]`, consumed by `BlueprintGenerator`, skipped by the flat method+preset merge — see `_DATASET_CONFIG_SECTIONS`). Three ways to override the blueprint: `--dataset_config` for a separate file; a scalar-only `[general]`/`[[datasets]]` block in the method TOML to **shallow-override** top-level scalars (`_apply_dataset_overrides`; subset-level overrides not supported this way); or a method TOML carrying a **full** `[[datasets]]` with `subsets` to **fully replace** base's blueprint inline (the self-contained per-method layout, see next bullet). Full-vs-shallow is decided by `load_dataset_config_from_base` on whether the method's `[[datasets]]` has a subset.
- `configs/preprocess.toml` — shipped template for preprocess knobs split out of base.toml (`source_image_dir`, `drop_lowres_images`, `min_pixels`, **`target_res`**, **`multires_per_image`**); the live WebUI-owned values are persisted under gitignored `configs/custom/preprocess.toml`. The preprocess pipeline reads them via `load_path_overrides`, layered **`custom/preprocess.toml → base.toml → preset → method`**. Training ignores the source/filter knobs, but `load_method_preset` seeds `target_res` and `multires_per_image` at lowest priority: in normal nearest-tier mode the cache files remain self-describing, while multi-resolution mode uses the selected tier list to require and expand one cache per image/tier. Dataset paths (`resized_image_dir`, `lora_cache_dir`) remain in base.toml because the blueprint interpolates them; model paths use the separate model configuration described above.
- `configs/presets.toml` — hardware profiles as sections: `[default]`, `[fast_16gb]`, `[low_vram]` (also Windows 8GB), `[half]`. Holds `blocks_to_swap`, gradient/offload checkpointing, etc.
- `configs/methods/` — one flat file per family read by `train.py` (`lora`, `chimera`, `soft_tokens`, `byg`, `spd`), each holding rank + routing knobs + opinionated LR/epochs/output_name. `turbo.toml` is the **odd one out**: a bespoke sectioned schema read only by `scripts/distill_turbo/` — don't `print-config METHOD=turbo`. Variants inside `lora.toml` are comment-toggle blocks; default stacks LoRA + OrthoLoRA + T-LoRA + shared_A FEI-routed Hydra. **Pre-three-axis checkpoints (`ss_use_hydra`/`ss_use_fei_router` metadata) no longer load** — legacy fallback removed.
- **Self-contained per-method dir** (`configs/<method>/<method>.toml`) — the consolidated layout: method config **+** full inline dataset blueprint in one file, no `dataset_config` cross-reference. `_resolve_method_path` (`library/config/io.py`) **prefers** `configs/<method>/<method>.toml` over the flat `configs/methods/<method>.toml` when present (default `methods` subdir only — `gui-methods` stays flat), so `--method <m>` auto-discovers it with no new flags. **EasyControl is the pilot**: `configs/easycontrol/easycontrol.toml` (alongside the miner-generated descriptor blueprints `near_twins.toml` / `colorize.toml` in the same dir). NB `configs/gui-methods/easycontrol.toml` still points at the standalone `configs/datasets/easycontrol.toml` — keep the inline subset in sync until gui-methods is migrated.
- `configs/gui-methods/` — clean per-**variant** parallel tree, no toggle blocks (what you see is what runs). Selected via `--methods_subdir gui-methods` (wrapped by `make lora-gui`). `ls` for the live list.

Subsets accept `cache_dir` — redirects all VAE/TE/PE caches to that dir with stem-mirrored names (EasyControl uses this to keep source dirs user-facing while caches live under `post_image_dataset/`). `library.config.io.load_method_preset(method, preset, methods_subdir=...)` is the reusable merge helper (not re-exported via `train_util`). All config paths are relative to `anima_lora/`. Outputs split by kind: checkpoints (+ `.snapshot.toml` + `_moe` siblings) in `output/ckpt/`, inference images in `output/tests/`. **Daemon per-job outputs** live under `output/daemon/jobs/<job_id>/`: `job.json` (persisted `Job` record), `stdout.log`, `progress.jsonl`, and `sample/` (training preview gallery, injected as `--sample_dir` so a new task's gallery never replays the previous task's). The daemon returns `sample_dir` on submit and the WebUI reads it back (via `task.sample_dir` in-session, or `GET /jobs/<id>` after a restart) to locate the gallery.

## Architecture

- **Modular `library/`** (`train_util.py` is a re-exporting facade): domain subpackages `anima/` (DiT model, weights, strategy), `datasets/` (`cache.py` = `CachedDataset`), `training/` (optimizer/scheduler/checkpoint + loss/sampler/metric registries), `inference/` (engine + `request.py` typed `GenerationRequest`; plug-ins split `corrections/` — DCW / SMC-CFG / mod-guidance — vs `editing/` — DirectEdit + postfix inversion), `preprocess/` (caching orchestration), `models/`, `captioning/`, `vision/`, `config/`, `io/` (cache-path resolution), `runtime/` (device/offloading + `cli.py` argparse + `harness.py` `build_anima`). Full per-subpackage map in `docs/structure/`.
- **Tooling layering contract**: **primitives** (`library/*` — load a model, encode a batch, resolve a cache path) → **façade** (`anima_lora/` — embedder entry points) → **orchestration** (`library/preprocess/`, `library/runtime/harness.py` — drive primitives over a whole dataset/run) → **entry points** (`scripts/preprocess/*.py`, `bench/**/run_bench.py`, `scripts/**`, `tasks.py` — thin argparse wrappers). `scripts/preprocess/*.py` are now thin CLI shells over `library/preprocess/`. `bench/`, `scripts/` are **not** installed packages (only `anima_lora`/`library`/`networks` are) — they keep a `sys.path` bootstrap to import siblings.
- **Strategy pattern** for tokenization/encoding (`library/anima/strategy.py`, `library/anima/text_strategies.py`).
- **Pluggable adapters** under `networks/` — selected via `network_module` + (for LoRA family) the three-axis routing cfg. LoRA modules in `networks/lora_modules/` coordinated by `networks/lora_anima/`; EasyControl in `networks/methods/`; attention dispatcher `networks/attention_dispatch.py`; Spectrum `networks/spectrum.py`; SPD `networks/spd.py`. **See `networks/CLAUDE.md`** for the per-module map, three-axis surface, and dispatch invariants.

## Critical invariants

### Text encoder padding
The pretrained model expects max-padded text encoder outputs — zero-padded positions act as attention sinks in cross-attention softmax. Trimming to actual text length produces **black images**. Both training and inference must pad to `max_length` and must NOT mask out padding via `crossattn_seqlens`. Regenerate disk-cached `.npz` after any tokenizer/padding change.

### Free-fit native-shape bucketing — the only resize mode
Free-fit is the sole resize mode (the discrete **constant-token bucket pool** — `CONSTANT_TOKEN_BUCKETS` and the per-tier tables — was **removed 2026-06-19**; the migration kept only each tier's numeric token band in `EDGE_TOKEN_BANDS`). Free-fit keeps each image's **native aspect ratio** and lands its patch-grid token count *anywhere* inside its tier's band (`freefit_bucket` / `freefit_band_for_edge` in `buckets.py`; bands defined in `EDGE_TOKEN_BANDS`), driving crop loss to ~zero (sub-patch <16px residual). There is no `freefit` flag any more — it's implicit. Each forward runs at its real token count; `compile_blocks()` sets `_native_flatten` (flattens each patch grid to a fake-5D `(B, 1, seq_len, 1, D)` shape, keying the block graph on **token count alone**), bit-exact to the eager 5D path. The legacy pad-to-static path was removed 2026-05-24 (`static_token_count`/`static_pad` etc.).

**Compile coupling**: free-fit populates many distinct `(W,H)` inside a tier's band, which would explode the static N-graph cascade, so it **requires `compile_dynamic_seq`** — auto-enabled by `train.py` whenever `torch_compile` is on (and unconditionally forced in the bespoke distill loops via `ensure_dynamic_seq_for_freefit`). `dynamic_seq` marks only the seq axis dynamic and bounds it to the tier's `seq_range`, collapsing the whole band to **one graph per tier**. `make_buckets()` uses the actual on-disk cached `(W,H)` as the bucket set (caches are literally the source of truth), so nothing AR-snaps at load. **Snap-era caches still train fine** (a snap pool is just a free-fit pool that landed only on the old discrete counts); re-preprocess only to gain the reduced-crop benefit.

**Multi-scale tiers**: `EDGE_TOKEN_BANDS` defines per-tier bands for edges **512 768 896 1024 1280 1536** (768→2160 / 1280→6300 / 1536→8640 tok = one family each; 512→{1008,1024}, 896→{3000,3024}, 1024→{4032,4200} = two families each). Preprocess `--target_res <subset>` selects which tiers are active. By default each image goes only to the tier that **resizes it the least** (`choose_edge` — an area-based `|log(nominal_tokens/native_tokens)|` minimum). With `multires_per_image=true` (or `--multires_per_image`), preprocessing additionally writes every source image under `post_image_dataset/multires/<edge>/`, the VAE pass creates one `{stem}_{WxH}_anima.npz` per selected tier, and `DreamBoothDataset` expands those caches into distinct samples so every image/tier pair is reachable in the same epoch. This mode requires at least two tiers and fails before training if any selected cache/key is missing. The 1024 tier's band stays **frozen at (4032, 4200)** (`FREEFIT_FROZEN_EDGES`) because DCW calibration keys off it. In nearest-tier mode training remains fully self-describing and does not need `target_res`; in multi-resolution mode it uses `target_res` only to validate/filter the expanded cache set. Compile budgets always derive from the buckets actually populated, plus sample-prompt resolutions. All tiers stay within the rope cap (≤256 patches/axis).

### Lazy model loading
DiT loads AFTER text-encoder/VAE caching and unloading, to avoid OOM: text encoder → cache → free → VAE → cache → free → load DiT → attach adapter → train.

### compile-after-apply (`build_anima`)
`torch.compile` traces the adapter's monkey-patched forward, so `compile_blocks()` MUST run **after** `network.apply_to` + `load_weights`. `library/runtime/harness.py::build_anima` is the shared harness encoding this ordering (promoted from `bench/_anima.py`); use it from `bench`/`scripts`/`preprocess` rather than open-coding load→apply→compile.

### The DiT operates on 5D latents `(B, C, T=1, H, W)` — the singleton is **dim 2**
The DiT forward (and `PatchEmbed`, which `assert x.dim() == 5`) takes a **5D** latent with a singleton temporal/frame axis at **dim 2** (`T=1` for images — Anima reuses a video-shaped layout). Everything *around* the DiT is 4D `(B, C, H, W)`: VAE `encode_pixels_to_latents` returns 4D, cached `.npz` latents are 4D, the training inner loop works in 4D, FFT/spectral helpers (Spectrum, CNS γ, Log-Gabor) want 3D/4D `(C,H,W)`/`(B,C,H,W)`, and the vision tower (PE-Core `encode_pe_from_imageminus1to1`) wants 4D `(B,3,H,W)`. So the boundary dance is **always `unsqueeze(2)` going into the DiT and `squeeze(2)` coming out** — target **dim 2 explicitly**, never `squeeze()`/`squeeze(0)` (which silently hits batch when B=1 and corrupts the layout). Two recurring bite points: **`vae.decode_to_pixels` returns 5D `(B,3,1,H,W)` when fed a 5D latent** (squeeze dim 2 before handing RGB to a vision tower / `F.interpolate`), and **sampler-boundary plug-ins (DCW/SMC/CNS/SGMI/etc.) receive 5D** while any reference latent they blend against is often 4D (match ndim first — see the archived FreeText `_match_latent_ndim`). Mishandling dim 2 was a repeated source of subtle freetext bugs.

## Methods

Adapter families (training methods) below — one-line orientation plus the load-bearing gotcha; read the linked deep-dive before working on one.

**Training-free inference stacks** (Spectrum, SPD, DCW, SMC-CFG, CNS, mod-guidance, embedding inversion, DAVE) are documented separately under [`docs/inference/`](docs/inference/README.md) — read the relevant doc when you touch one rather than carrying their details here. Most ride on the sampler boundary and compose with any checkpoint (DAVE is the exception — a block-forward hook for same-prompt diversity). Channel scaling (per-channel LoRA gradient rebalance, on by default) is a training-time feature — see [`docs/optimizations/channel_scaling.md`](docs/optimizations/channel_scaling.md); note it's exactly inert on frozen-basis ortho variants.

| Method | What it is | Gotcha / pointer |
|---|---|---|
| **DirectEdit + Anima Tagger** | Inversion + edit-conditioning swap; Tagger (`library/captioning/`) maps image → Anima-format tags for ψ_src. | Edit leverage collapses if ψ_src is off-manifold — verify with `exp-test-directedit-dry`. `docs/experimental/directedit_editing_v3.md`, `anima_tagger.md` |
| **EasyControl** | Extended self-attn image conditioning; frozen DiT, per-block cond LoRA + scalar `b_cond` gate. Source `easycontrol-dataset/`. | `docs/experimental/easycontrol.md` |
| **Soft Tokens** | SoftREPA per-layer × per-t soft text tokens (~1M params); frozen DiT, per-block `Block.forward` splice into `crossattn_emb`. | InfoNCE objective intentionally skipped. `configs/methods/soft_tokens.toml` |
| **ChimeraHydra** | Dual-pool additive MoE: content pool (network ContentRouter on pooled `crossattn_emb`) + freq pool (network FreqRouter on FEI+σ), two A's per Linear off disjoint SVD subspaces. Both pools always centered-gate; the per-Linear `lx_c` content router + non-centered path were removed. | T-LoRA mask hits content branch only. `docs/experimental/chimera-hydra.md`, `networks/lora_modules/chimera.py` |
| **Turbo** | DP-DMD (diversity-preserved DMD) distillation; output is a normal LoRA. | Bespoke schema read by `scripts/distill_turbo/` — don't `print-config`. Bespoke two-optimizer loop (student + fake/critic) kept out of `train.py`; converges only the leaves — honors `--queue` (daemon command-job) + writes a canonical `output/ckpt/<name>.snapshot.toml`. Shipped (promoted from `exp-turbo` → `make turbo` / `make test-turbo`); a published 4-step student lives at `huggingface.co/sorryhyun/anima-turbo-4step`. `docs/methods/turbo.md` (ops), `docs/structure/turbo.md` (structure); CA-era history in `_archive/proposals/dmd2_decoupled_improvements.md`. |
| **Postfix-tail inversion** | Per-image inversion *probe* (training method archived 2026-05-20). | Observation tool, not a deployable adapter. `library/inference/editing/postfix_inversion.py` |

## Preprocessing & scripts

Data-prep scripts in `scripts/preprocess/` are thin argparse wrappers (resize → VAE latents → text embeddings → PE features → masks); **the caching logic lives in `library/preprocess/`** — edit orchestration there, flags in the script. `make preprocess-{resize,vae,te,pe,pooled}` / `make mask`. Resize is **idempotent + size-aware** (skips images already at the correct bucket; `--overwrite` forces all). Multi-resolution staging is likewise idempotent; VAE caching walks each selected staging tier into the common LoRA cache, while TE/PE/masks continue to use the normal nearest-tier image and are shared by the virtual samples. After a `target_res` tier change in nearest-tier mode, run `make preprocess-reconcile` (dry-run; `ARGS="--delete"` to act) to drop orphaned latent npz / stale resized PNG / PE sidecar / mask for every image whose bucket moved — TE caches are text-only and never touched. Old multi-resolution caches outside the current selected tier set are ignored by training. Other utility scripts: `distill_mod/`, `merge_to_dit.py`, `dcw/`, `anima_tagger/cli.py`, `edit.py`, `export_logs_json.py`.

Caches live under `post_image_dataset/lora/`: `{stem}_{WxH}_anima.npz` (VAE), `{stem}_anima_te.safetensors` (text), `{stem}_anima_pe.safetensors` (PE). TE caching reads `.txt` from `image_dataset/` (the caption master); training reads only cached embeddings.

## Custom nodes

Spectrum KSampler + mod-guidance nodes live in a separate repo (https://github.com/sorryhyun/ComfyUI-Spectrum-KSampler; ships DCW scalar default `+0.01` + `auto` mode). In-tree under `custom_nodes/`: `comfyui-hydralora/` (Adapter / FeRA / Soft Tokens loaders — see its `CLAUDE.md` for the `forward_hook`-not-override invariant), `comfyui-anima-directedit/`, `comfyui-anima-tagger/`, `comfyui-anima-trainer/`, `comfyui-anima-blockcompile/`.

Several nodes carry a `_vendor/` subset of the live tree. **Regenerate vendor trees with `make vendor-sync` (`scripts/release/sync_vendor.py`), never `cp` by hand** — re-run before every node publish. See [[feedback_vendor_sync]]. Note `../comfy/custom_nodes/` is symlinked into this repo — edit the source here, not the symlink.

## External tools

ComfyUI, SAM3, and manga-image-translator live in the parent directory (`../comfy/`, `../sam3/`, etc.).

## Contributing

PRs follow a tier system — see `CONTRIBUTING.md`. Key constraint for code work: numerics/efficiency changes (Tier 1.5) and new methods (Tier 2) **require a bench script + invariant test**. Bench scripts share `bench/_common.py` and drop a `result.json` envelope into `bench/<method>/results/<YYYYMMDD-HHMM>[-label]/`.
