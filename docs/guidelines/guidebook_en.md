# Anima LoRA Guidebook

This document is a comprehensive English guide for the complete **Anima LoRA** training and inference pipeline from scratch. It covers everything from CUDA driver installation to dataset preparation, training, inference, and ComfyUI deployment. This guide is aimed at Windows beginners — for WSL, Linux, and (the core goal of this project) training optimization, please refer to other documentation.

---

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [CUDA 13.0.2 Installation](#2-cuda-1302-installation)
3. [Python Environment and Repository Setup](#3-python-environment-and-repository-setup)
4. [Hugging Face Authentication and Model Download](#4-hugging-face-authentication-and-model-download)
5. [Dataset Preparation](#5-dataset-preparation)
6. [Preprocessing: Resize, Latent Caching, Text Embedding Caching](#6-preprocessing-resize-latent-caching-text-embedding-caching)
7. [WebUI Usage](#7-webui-usage)
8. [Training Execution](#8-training-execution)
9. [LoRA / Adapter Variant Selection Guide](#9-lora--adapter-variant-selection-guide)
10. [Inference](#10-inference)
11. [Deploying to ComfyUI](#11-deploying-to-comfyui)
12. [Updating](#12-updating)

---

## 1. System Requirements

| Item | Minimum | Recommended |
|---|---|---|
| GPU | **RTX 3060 or newer; 2xxx series and older are not supported** | 16 GB VRAM or more |
| System Memory | 16 GB | 32 GB or more |
| Disk | 60 GB free space | 200 GB or more (for cache + accumulated outputs) |
| OS | Windows 11 / Ubuntu 22.04+ | Ubuntu 24.04 (stable FA2/CUDA 13 builds) |
| Python | **Must be 3.13** | - |

---

## 2. CUDA 13.0.2 Installation

The latest CUDA is required for stable operation of PyTorch 2.x + Flash Attention 2. Download and install version 13.0.2 from the official NVIDIA archive.

Download page: <https://developer.nvidia.com/cuda-13-0-2-download-archive>

### 2.1 Windows Installation

1. On the page above, select **Operating System: Windows -> Architecture: x86_64 -> Version: 11/10 -> Installer Type: exe (local)**.
2. Run the downloaded `cuda_13.0.2_windows.exe` -> select "Express (Recommended)" installation.
3. After installation, verify in PowerShell:

   ```powershell
   nvidia-smi
   nvcc --version
   ```

4. If `nvcc` is not recognized, add the following paths to the system environment variable `Path`:

   ```
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\libnvvp
   ```

5. Restart your computer and run `nvcc --version` again to verify.

> **Driver Note**: CUDA 13.x requires NVIDIA driver version 580 or above. If your driver version is older, please update it first via GeForce Experience or the NVIDIA Download Center.

### 2.2 (Optional) Switching to CUDA 13.2 + torch 2.12 nightly

The default installation is CUDA 13.0 + torch 2.11 stable. If you want approximately **10% training speed improvement** on GPUs like the RTX 50 series, you can switch to CUDA 13.2 + torch 2.12 nightly (benchmark reference: [docs/optimizations/cuda132.md](../optimizations/cuda132.md)). No compilation tools are needed — `pyproject.toml` URLs already include pre-compiled trimmed FA2 wheel packages, and `uv sync` can download and install them directly.

Steps:

1. **Download and install CUDA 13.2.**
   On the <https://developer.nvidia.com/cuda-downloads?target_os=Windows> page, select **Windows -> x86_64 -> 11/10 -> exe (local)**. You can install it directly on top of 13.0, and both versions can coexist (they create separate `v13.2` and `v13.0` directories).

2. **Modify the comment toggles in `pyproject.toml`.** Two changes are needed:

   **(a) torch / torchvision** — In `dependencies`, comment out the "Windows: stable" two lines and uncomment the "Windows: cuda132 opt-in" two lines:

   ```toml
   # Windows: stable (default).
   # "torch>=2.11.0,<2.12 ; sys_platform == 'win32'",
   # "torchvision>=0.26.0,<0.27 ; sys_platform == 'win32'",
   # Windows: cuda132 opt-in. ...
   "torch>=2.12.0.dev0,<2.13 ; sys_platform == 'win32'",
   "torchvision>=0.27.0.dev0,<0.28 ; sys_platform == 'win32'",
   ```

   **(b) flash-attn** — In the same `dependencies`, comment out the "Windows: stable (default) -- built against torch 2.11 + CUDA 13.0" line and uncomment the "Windows: cuda132 opt-in -- trimmed FA2" line below it:

   ```toml
   # Windows: stable (default) — built against torch 2.11 + CUDA 13.0.
   # "flash-attn @ https://github.com/mjun0812/.../flash_attn-2.8.3+cu130torch2.11-cp313-cp313-win_amd64.whl ; sys_platform == 'win32'",
   # Windows: cuda132 opt-in — trimmed FA2 ...
   "flash-attn @ https://github.com/sorryhyun/flash-attention-sm120-fix/releases/download/fa2cuda132/flash_attn-2.8.4-cp313-cp313-win_amd64.whl ; sys_platform == 'win32'",
   ```

3. **Re-sync**: Run `uv sync`. It will download torch 2.12 nightly from the cu132 index and download/install the trimmed FA2 wheel from the release.

> To revert: restore the original comments in `pyproject.toml` and re-run `uv sync`.
>
> If you need to compile manually (e.g., different GPU from RTX 5060 Ti, different Python version, or managing wheels in your own fork): please refer to [docs/optimizations/cuda132.md](../optimizations/cuda132.md).

---

## 3. Python Environment and Repository Setup

This project uses [`uv`](https://github.com/astral-sh/uv) for dependency management. It uses Python 3.13.

### 3.1 Installing `uv`

  ```powershell
  irm https://astral.sh/uv/install.ps1 | iex
  ```

After installation, open a new terminal window and confirm `uv --version` outputs correctly.

### 3.2 Cloning the Repository

```bash
git clone https://github.com/sorryhyun/anima_lora.git
cd anima_lora
```

> All paths in this guide are relative to the `anima_lora/` directory. All commands are executed within this folder.

### 3.3 Installing Dependencies

```bash
winget install ezwinports.make
uv sync
```

`uv sync` creates a virtual environment based on `pyproject.toml`/`uv.lock` and installs all dependencies. After installation, activate the environment using either of the following methods:

- Manual activation each session (recommended): `.venv\Scripts\activate`
- VSCode can automatically activate the environment for a smoother experience.

---

## 4. Hugging Face Authentication and Model Download

### 4.1 Creating a Token and Logging In

1. Create a **read** permission token at <https://huggingface.co/settings/tokens>.
2. Log in from the terminal:

   ```bash
   hf auth login
   ```

   Paste the token and press Enter.

### 4.2 Downloading Models

```bash
make download-models
```

This command automatically downloads the following three models and organizes them into the `models/` directory.

| File | Path |
|---|---|
| Anima DiT (main diffusion model) | `models/diffusion_models/anima-base-v1.0.safetensors` |
| Qwen3 0.6B text encoder | `models/text_encoders/qwen_3_06b_base.safetensors` |
| QwenImage VAE | `models/vae/qwen_image_vae.safetensors` |

If you also need masking functionality, download the SAM3 and MIT models as well (already included in the above command).

> **If download is interrupted**: You can use individual commands such as `make download-anima`, `make download-sam3`, `make download-mit` to download in separate batches.

---

## 5. Dataset Preparation

Anima LoRA uses a structure of *images + same-name `.txt` caption sidecar files*. Here is an example of the `image_dataset/` folder.

```
image_dataset/
├─ 00001.png
├─ 00001.txt
├─ 00002.jpg
├─ 00002.txt
├─ subfolder/
│  ├─ 00010.webp
│  └─ 00010.txt
└─ ...
```

### 5.1 Caption Writing Tips

- According to the official Anima specification, tag order is always [meta] [character] [series] [artist] [general]. For example:

```
absurdres, safe, 1girl, chitanda eru, hyouka, @channel (caststation), full body, serafuku, She is saying hi.
```

- Based on personal experimentation, quality tags such as absurdres, highres, and masterpiece are best omitted or used sparingly. Alternatively, they can be fully omitted once the officially released mod guidance is available.
- Place original images in `image_dataset/` (naming is flexible; use this path).

### 5.2 What is `num_repeats`, and When Should You Adjust It? (Summary: **Don't touch it**)

In `configs/base.toml` under `[[datasets.subsets]]`, you will see `num_repeats = 1`. This is a kohya-ss style option that specifies **how many times each image is reused within a single epoch**, and is frequently encountered when following other LoRA trainer guides.

- **Keep it at `1` for the standard workflow described in this guide.** In the common use case of training with all images in a single `image_dataset/` folder, increasing `num_repeats` only *extends the length of a single epoch*, which is equivalent to increasing `max_train_epochs`. It is always more intuitive to adjust training volume by changing the number of epochs, and all presets and method configurations in this project are tuned assuming `num_repeats = 1`.
- **When does increasing it make sense?** When training with *multiple different subsets (folders)* where the number of images per folder varies significantly, `num_repeats` can serve as a *balancing tool* to equalize the exposure frequency of smaller folders to match larger ones (e.g., Character A: 1000 images + Character B: 50 images, then set `num_repeats = 20` only for the B subset). This does not apply to single-folder training.
- **Where to modify it?** `num_repeats` belongs to the *dataset configuration*, not the method configuration, so it is not exposed in `configs/methods/`, `configs/gui-methods/`, or the WebUI Training view. If you need to modify it, directly edit `[[datasets.subsets]]` in `configs/base.toml` (or in the TOML file specified via `--dataset_config <path>`). *If you simply want to train the same image more times*, the correct approach is to increase `max_train_epochs`, not to modify `num_repeats`.

---

## 6. Preprocessing: Resize, Latent Caching, Text Embedding Caching

To optimize training speed and VRAM usage, three steps must be executed in advance: **resize -> VAE latent caching -> text embedding caching**.

```bash
make preprocess              # Run all three steps (for LoRA / general training)
# Or run step by step
make preprocess-resize       # 1) image_dataset/ -> post_image_dataset/resized/
make preprocess-vae          # 2) VAE latent caching -> post_image_dataset/lora/
make preprocess-te           # 3) Text encoder output caching -> post_image_dataset/lora/
make preprocess-pe           # (Optional) PE-Core vision encoder feature caching — for IP-Adapter / REPA only
```

> **Caches are reused and not automatically deleted.**
> `make preprocess` (and the *Preprocess* button in the GUI) **reuses existing caches**. The `.npz` / `_te.safetensors` / `_pe.safetensors` files in `post_image_dataset/lora/` *are not overwritten or deleted*; only missing items are processed. Therefore, re-running is very fast, and interrupting midway is safe.
>
> In other words, running `make preprocess` again with existing caches will not lose any existing data — rest assured. Conversely, if you **modify captions, tokenizer, or resize options and want to regenerate caches from scratch**, you need to manually delete the cache folder (`post_image_dataset/lora/` or `post_image_dataset/easycontrol/`) and re-run.

### 6.1 What Resize Does

- Resizes images according to the VAE's required pixel alignment
- Automatically sorts images into *fixed-token-resolution buckets* satisfying (H/16) x (W/16) ~ 4096 patches
- Automatically excludes images that are too small (default: below 0.5 MP) and generates a report
- Saves results as PNG files to `post_image_dataset/resized/`

### 6.2 Latent Caching

- Runs the VAE once on all resized images and saves results to disk
- The VAE does not need to be loaded on the GPU during training, significantly saving VRAM
- Cache location: `post_image_dataset/lora/{stem}_{WxH}_anima.npz`
- Script: `preprocess/cache_latents.py`

### 6.3 Text Embedding Caching

- Pre-computes the outputs of Qwen3 0.6B + LLM adapter
- When `use_shuffled_caption_variants = true`, it also caches comma-shuffled caption variants (randomly selected during training)
- Cache location: `post_image_dataset/lora/{stem}_anima_te.safetensors`
- Captions are always read from the original `.txt` files in `image_dataset/` (not copied to the resize folder)
- Script: `preprocess/cache_text_embeddings.py`

### 6.4 PE Vision Feature Caching (Optional)

- Only needed when REPA auxiliary loss is enabled (`use_repa = true`)
- Pre-computes PE-Core-L14-336 vision encoder outputs so that the vision encoder does not need to be loaded during training
- Cache location: `post_image_dataset/lora/{stem}_anima_pe.safetensors`

> **When do you need to regenerate caches?**
> - New images added -> Simply re-run `make preprocess` (existing caches are preserved; only new items are added).
> - **Captions modified** or **tokenizer/padding options changed** -> Follow the instructions above to manually delete the cache folder (`post_image_dataset/lora/`) and re-run. Simply re-running will *reuse existing caches* and will not reflect the changes.

---

## 7. WebUI Usage

Using the WebUI-based browser interface, you can complete configuration editing, dataset browsing, preprocessing, training execution/monitoring, and LoRA merging all in one interface.

```bash
python -m webui              # Start WebUI server (http://127.0.0.1:8000)
```

Main WebUI views:

- **Training Config**: Select the LoRA family variant from the dropdown (recommended: `tlora` — Ortho + T-LoRA / others include `lora`, `tlora-8gb`, `tlora_ortho_reft`, `hydralora`, `reft`, etc.), and directly modify `presets.toml` presets (default / low_vram, etc.) and all training parameters, then start training
- **Preprocess**: One-click resize + VAE + text embedding caching
- **Dataset**: Preview images/captions and edit captions directly
- **Merge**: Bake the trained LoRA into the base DiT and save as a standalone ComfyUI checkpoint (supports only basic LoRA / OrthoLoRA / T-LoRA)

WebUI training internally calls `train.py`, so identical parameters can be fully reproduced in the CLI. The WebUI reads from `configs/gui-methods/<variant>.toml` (single-file variants without toggle blocks), so the variant list exposed in the WebUI is consistent with the CLI's `make lora-gui GUI_PRESETS=<variant>`. The current state of the variant list can be viewed via `ls configs/gui-methods/`.

### 7.1 Form Editing and Save Behavior

The training/preprocessing subprocess re-reads the variant TOML file from disk, so if you only modify the form without saving, those changes will not be reflected in training. The WebUI handles this in two ways:

- **Change detection**: When any field in the form (or the `+ Extra args` text box) is edited, the `Save` button turns orange and displays a `Save *` marker. This indicates *the variant file on disk is inconsistent with what is shown on screen*. Click `Save` or re-select the variant to reload from disk to clear the marker.
- **Auto-save**: Even if you forget to save and click `Train` / `Preprocess`, the current form values are automatically written to the variant file before the subprocess is executed. In other words, the values displayed on screen are the ones actually used for training. (`Test` infers from the last training result's checkpoint, so it does not trigger auto-save.)

> If you only want to *try out* changes without saving them, after editing the form, do not click `Train` — instead, switch to another variant and switch back. It will reload from disk, and the edits will be discarded.

### 7.2 Auto-Resume (checkpointing_epochs)

Even if training is interrupted midway, clicking `Train` again will **automatically resume training from the last saved checkpoint**. This is one of the most useful features for handling power outages, OOM, accidentally closed windows, etc., and is enabled by default.

How to use it in the GUI:

- In the Training Config view, the **Training** group contains a `checkpointing_epochs` field (gui-methods variants default to `2`, `methods/lora.toml` defaults to `4`). It saves the resume state every N epochs, overwriting the same file, without growing disk usage.
- After training is interrupted, click `Train` again with the same variant, and the log window will display `auto-resuming from checkpoint at step N`, continuing training from that point. No manual flag adjustments are needed.
- After training completes normally, the resume temporary files are automatically deleted, and the final result is saved as `output/ckpt/<output_name>.safetensors`.
- **If you change the dataset or core configuration (rank, LR, epoch count, etc.)** and want to restart training, manually delete the `output/ckpt/<output_name>-checkpoint-state/` folder before clicking `Train`. Otherwise, training will continue from the old state.

For detailed behavior, see [Section 8.6 Auto-Resume](#86-auto-resume-checkpointing_epochs) — which also explains the difference from `save_every_n_epochs`.

---

## 8. Training Execution

All training is executed via TOML configuration files and HuggingFace Accelerate. The configuration merge order is `configs/base.toml -> configs/presets.toml[<preset>] -> configs/methods/<method>.toml -> CLI arguments`, with method configuration overriding preset configuration.

### 8.1 Quick Start

**The most recommended starting point is OrthoLoRA + T-LoRA (i.e., the `tlora` variant)**. This combination offers the best balance between stability, detail, and style preservation, and can be used directly for regular character/style LoRA training.

```bash
# Recommended: Ortho + T-LoRA (gui-methods/tlora.toml)
make lora-gui GUI_PRESETS=tlora                  # General environment
PRESET=low_vram make lora-gui GUI_PRESETS=tlora-8gb   # 8~12 GB VRAM

# Other variants (configs/gui-methods/<variant>.toml — single-file, no toggle blocks)
make lora-gui GUI_PRESETS=lora                   # Basic LoRA only
make lora-gui GUI_PRESETS=tlora_ortho_reft       # Ortho + T-LoRA + ReFT combination
make lora-gui GUI_PRESETS=hydralora              # MoE multi-head routing
make lora-gui GUI_PRESETS=reft                   # Standalone ReFT

# Toggle block method (select variant directly in configs/methods/lora.toml)
make lora                          # presets.toml[default]
PRESET=low_vram make lora          # presets.toml[low_vram] — 8~12 GB VRAM
PRESET=half make lora              # Use half the dataset for quick experiments
```

> **Overriding parameters directly in the CLI**: You can pass extra arguments like `make lora -- --network_dim 32 --max_train_epochs 24` (same for `tasks.py`).

### 8.2 Masked Loss (Excluding Text Bubbles)

In manga/comic-style data, excluding *text bubbles or text regions* from the training loss can produce cleaner results.

```bash
make mask          # SAM3 + MIT (runs in temp directory) -> post_image_dataset/masks/
make mask-clean    # Delete post_image_dataset/masks/
```

The resulting PNGs are black-and-white images: **white (255) = training target**, **black (0) = excluded region**. Dataset subsets automatically prefer `post_image_dataset/masks/` (if present), otherwise falling back to legacy `masks/{merged,sam,mit}/` (users with the old layout will also work). If no mask files exist, they are simply ignored — creating them is optional.

### 8.3 Common Configuration Parameters (LoRA Baseline Defaults)

| Parameter | Default | Description |
|---|---|---|
| `network_dim` | `32` | LoRA rank. Higher values increase expressiveness and parameter count |
| `network_alpha` | `32` | LoRA scaling coefficient (usually same as `network_dim`) |
| `learning_rate` | `2e-5` | Learning rate. Hydra variants can use even lower values |
| `max_train_epochs` | `4` | Smaller datasets should use more epochs |
| `save_every_n_epochs` | `2` (gui-methods) / `4` (methods) | Adapter weight accumulation save interval |
| `checkpointing_epochs` | `2` (gui-methods) / `4` (methods) | Resume state save interval (overwrites single file) |
| `caption_dropout_rate` | `0.1` | Replaces some captions with empty strings (helps with CFG) |
| `use_shuffled_caption_variants` | `true` | Uses comma-shuffled caption variants |

Variant toggle switches (`use_ortho`, `use_timestep_mask`, `add_reft`, `use_moe_style`, `router_source`, etc.) can be activated by uncommenting blocks in `configs/methods/lora.toml`, or by using the variant-specific files in `configs/gui-methods/<variant>.toml`. **The recommended starting point `tlora` variant is a pre-configured OrthoLoRA + T-LoRA combination with `use_ortho = true` + `use_timestep_mask = true`.**

### 8.4 What Happens During Training

1. Load text encoder -> create/verify cache -> unload
2. Load VAE -> create/verify cache -> unload
3. *Lazily load* DiT to avoid VRAM conflicts during the caching phase
4. Inject adapter network into DiT's attention / FFN modules (injection targets vary by variant)
5. Noise sampling -> DiT forward pass -> flow-matching loss -> backward pass -> optimizer update
6. (Optional) Measure validation loss and generate sample images via `validation_split`

### 8.5 Output Artifacts

- Trained weights: `output/ckpt/<output_name>.safetensors` (automatically branched by variant to `anima`, `anima_tlora_ortho`, `anima_tlora_reft`, `anima_hydra`, `anima_postfix`, etc.)
- Checkpoints: Saved to `output/ckpt/` at `save_every_n_epochs` intervals (with `.snapshot.toml` sidecar files; Hydra also has `_moe` companion files)
- Validation samples: `output/ckpt/sample/`
- Inference result images: `output/tests/`

### 8.6 Auto-Resume (checkpointing_epochs)

This feature allows training to **automatically resume from the last saved checkpoint** even if interrupted. It is extremely useful for power outages, OOM, accidentally closed windows, or when you need to shut down temporarily. It is enabled by default in method files and requires no additional configuration.

```toml
checkpointing_epochs = 2     # Save resume state every 2 epochs (overwrites)
```

This serves a different purpose from `save_every_n_epochs`.

| Config Key | What Is Saved | Cumulative? | Purpose |
|---|---|---|---|
| `save_every_n_epochs` | Adapter weights (e.g., `anima_lora-000004.safetensors`) | **Cumulative** (or limited by `save_last_n_epochs`) | Run inference with intermediate results or compare overfitting timepoints |
| `checkpointing_epochs` | Full training resume state (optimizer / scheduler / RNG / adapter weights) | **Overwrites single file** | Auto-resume after training interruption |

How it works:

- **Auto-save**: At `checkpointing_epochs` intervals, saves `output/ckpt/<output_name>-checkpoint-state/` (state directory) + `<output_name>-checkpoint.safetensors` (weights), overwriting the previous content. Does not grow disk usage.
- **Auto-resume**: When re-running the same command (e.g., `make lora`), if a saved checkpoint exists and `max_train_steps` has not been reached, training **automatically resumes**. No need to manually add `--resume` or similar flags. The log message `auto-resuming from checkpoint at step N` indicates successful resume.
- **Auto-cleanup**: After training completes normally, the above two files are automatically deleted — the resume state is a temporary file, and the final artifact is `output/ckpt/<output_name>.safetensors` (see Section 8.5).
- **Manual resume**: To return to a state from a different point, you can manually specify `--resume <state_dir>`.

> **When to disable?** For short experimental training runs or when disk space is limited, comment out `checkpointing_epochs`. For formal training with larger datasets, it is recommended to keep it enabled at all times.
>
> **Note**: If you change the dataset, captions, or training configuration (rank, LR, epoch count, etc.), resuming from an old checkpoint is meaningless and potentially risky. In this case, manually delete the `output/ckpt/<output_name>-checkpoint-state/` folder and start fresh.

---

## 9. LoRA / Adapter Variant Selection Guide

> **Recommended**: If you are a first-time user or need to create a general-purpose character/style LoRA, start with **`tlora` (OrthoLoRA + T-LoRA)**. It offers the best balance of detail/style preservation and training stability.

| Variant | How to Run | Use Case |
|---|---|---|
| **OrthoLoRA + T-LoRA** | `make lora-gui GUI_PRESETS=tlora` | **Recommended**. SVD-based orthogonal rotation (OrthoLoRA) + per-timestep rank masking (T-LoRA) combination. Outputs `anima_tlora_ortho.safetensors` |
| **OrthoLoRA + T-LoRA (8GB)** | `make lora-gui GUI_PRESETS=tlora-8gb` or `PRESET=low_vram make lora-gui GUI_PRESETS=tlora` | Use the recommended combination in 8~12 GB VRAM environments |
| **Basic LoRA** | `make lora-gui GUI_PRESETS=lora` or `make lora` | Simplest baseline, for comparison experiments |
| **Basic LoRA (8GB)** | `make lora-gui GUI_PRESETS=lora-8gb` or `PRESET=low_vram make lora` | 8~12 GB VRAM |
| **T-LoRA + Ortho + ReFT** | `make lora-gui GUI_PRESETS=tlora_ortho_reft` | Adds expressive editing (ReFT) to the recommended combination, with minimal extra parameters |
| **HydraLoRA** | `make lora-gui GUI_PRESETS=hydralora` (8GB version: `hydralora-8gb`) | MoE multi-head routing, integrating multiple concepts into a single adapter |
| **Standalone ReFT** | `make lora-gui GUI_PRESETS=reft` or set `add_reft = true` in `methods/lora.toml` | Representation Fine-Tuning (ReFT), minimal parameter count |
| **Postfix Tuning** (*Experimental*) | `make exp-postfix` or `make lora-gui GUI_PRESETS=postfix_ortho_cond` | Appends trainable N vectors at the end of cross-attention (caption-conditional + orthogonal variants) |
| **ChimeraHydra** (*Experimental*) | `make exp-chimera` or `make lora-gui GUI_PRESETS=chimera_hydra` | Content/frequency dual-pool MoE — for research only |

For detailed options per variant, see [`docs/guidelines/training.md`](training.md) and the individual documents in `docs/methods/`.

> **Compatibility Notes**
> - Adapter variants such as HydraLoRA / ReFT require `cache_llm_adapter_outputs = true` to be enabled by default for proper operation.
> - The OrthoLoRA + T-LoRA portion in `tlora` and `tlora_ortho_reft` can be baked into the base DiT via `make merge` to create standalone ComfyUI checkpoints (the ReFT portion cannot be baked — requires `--allow-partial`).

---

## 10. Inference

### 10.1 Fastest Way to Test

If you want to immediately generate samples with the adapter you just trained, use the corresponding variant's `make test-*`. All commands automatically select the most recently saved adapter of the appropriate type from `output/ckpt/`.

```bash
make test                        # Standard LoRA / OrthoLoRA / T-LoRA / ReFT
make test SPECTRUM=1             # Spectrum-accelerated inference
make test MOD=1                  # Modulation guidance (pooled_text_proj) — composable with SPECTRUM=1
make test NOLORA=1               # Bare DiT inference (skips --lora_weight); compose with MOD=1 for mod-only path
make test-hydra                  # HydraLoRA (router-live, anima_hydra*_moe.safetensors)
make test-merge                  # Inference with baked standalone DiT (`*_merged.safetensors`)
make test-dcw                    # LoRA + DCW scalar correction (sampler-level SNR-t correction)
make test-dcw-v4                 # LoRA + DCW v4 learnable calibrator
# Experimental inference
make exp-test-postfix            # Postfix tuning (standard)
make exp-test-postfix-exp        # postfix_exp variant
make exp-test-postfix-func       # postfix_func variant
```

### 10.2 General Inference (Manual)

```bash
python inference.py \
    --dit models/diffusion_models/anima-base-v1.0.safetensors \
    --text_encoder models/text_encoders/qwen_3_06b_base.safetensors \
    --vae models/vae/qwen_image_vae.safetensors \
    --lora_weight output/ckpt/anima_lora.safetensors \
    --lora_multiplier 1.0 \
    --prompt "masterpiece, best quality, an anime girl in a sunlit forest" \
    --negative_prompt "worst quality, low quality, blurry" \
    --image_size 1024 1024 \
    --infer_steps 30 \
    --guidance_scale 4.0 \
    --sampler er_sde \
    --flow_shift 1.0 \
    --seed 42 \
    --save_path output/tests
```

Common parameters:

| Parameter | Description |
|---|---|
| `--lora_weight` | Path to the trained adapter. Multiple can be specified |
| `--lora_multiplier` | Adapter strength (0.0~1.5) |
| `--image_size H W` | Output resolution (e.g., `1024 1024`, `1024 1536`) |
| `--infer_steps` | Number of denoising steps (typically 20~50) |
| `--guidance_scale` | CFG strength (recommended 3.0~5.0) |
| `--sampler` | `er_sde`, `euler`, `dpm++`, etc. |
| `--seed` | Random seed for reproducibility |
| `--spectrum` | Enable Spectrum acceleration |
| `--pgraft` | P-GRAFT (late-stage denoising LoRA truncation), letting the base model handle late-stage details |

For the complete list of inference options and P-GRAFT inference method, see [`docs/guidelines/inference.md`](inference.md).

---

## 11. Deploying to ComfyUI

ComfyUI core natively supports the Anima base DiT (loaded via `UNETLoader` / `CLIPLoader` directly). Deployment method varies depending on the adapter type.

### 11.1 Classic LoRA / OrthoLoRA / T-LoRA

Copy the `.safetensors` files generated in `output/ckpt/` directly to `ComfyUI/models/loras/`, and they can be used through ComfyUI's default LoraLoader node. To create a cleaner standalone checkpoint:

```bash
make merge ADAPTER_DIR=output/ckpt                 # Bake latest weights into base DiT
make merge ADAPTER_DIR=output/ckpt MULTIPLIER=0.8  # Adjust strength
```

The baked `*_merged.safetensors` can be loaded directly as a standalone model in ComfyUI's `UNETLoader`.

### 11.2 HydraLoRA / ReFT / Postfix

These variants cannot be loaded through ComfyUI's default LoraLoader (because they involve more than just weight deltas — routing and token insertion are involved) and require dedicated nodes:

- **Anima Adapter Loader** (`custom_nodes/comfyui-hydralora/`) — unified handling of LoRA / Hydra / ReFT / postfix. For detailed usage, see the `README.md` in that folder.
- **Spectrum KSampler / Mod Guidance / DCW nodes** — separate repository: <https://github.com/sorryhyun/ComfyUI-Spectrum-KSampler>

---

## 12. Updating

```bash
make update              # Download latest version from GitHub release, apply, and auto-run uv sync
make update -- --dry-run # Preview which files would be changed
```

`update` does not touch `image_dataset/`, `post_image_dataset/`, `output/`, or `models/` directories. For user-modified configuration files, it will prompt for confirmation on conflicts.

---

## Further Reading

- [`docs/guidelines/training.md`](training.md) — Adapter variants, caption shuffling, masked loss, dataset configuration details
- [`docs/guidelines/inference.md`](inference.md) — Inference parameters, DCW, Spectrum, prompt file format
- [`docs/guidelines/difference_between_comfy.md`](difference_between_comfy.md) — Differences between anima_lora and ComfyUI core implementation
- [`docs/methods/timestep_mask.md`](../methods/timestep_mask.md) — T-LoRA timestep masking
- [`docs/methods/psoft-integrated-ortholora.md`](../methods/psoft-integrated-ortholora.md) — OrthoLoRA details (orthogonal rotation part of the recommended `tlora` variant)
- [`docs/methods/spectrum.md`](../methods/spectrum.md) — Spectrum acceleration principles and options
- [`docs/methods/dcw.md`](../methods/dcw.md) — DCW (scalar + v4 learnable calibrator)
- [`docs/methods/mod-guidance.md`](../methods/mod-guidance.md) — Modulation guidance
- [`docs/methods/hydra-lora.md`](../methods/hydra-lora.md) — HydraLoRA multi-head routing
- [`docs/methods/reft.md`](../methods/reft.md) — ReFT expression editing
- [`docs/experimental/postfix.md`](../experimental/postfix.md) — Postfix (cond+ortho)
- [`docs/optimizations/cuda132.md`](../optimizations/cuda132.md) — How to upgrade to CUDA 13.2
- [`docs/optimizations/full_model_cudagraph.md`](../optimizations/full_model_cudagraph.md) — `compile_mode=full` + CUDAGraph invariants and debugging

If you have questions or bug reports, feel free to submit them in Chinese on GitHub Issues. Happy training!
