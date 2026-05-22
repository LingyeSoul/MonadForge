# Anima LoRA Guidebook

This document is a comprehensive English guide for using the **Anima LoRA** training/inference pipeline from start to finish. It covers everything from CUDA driver installation to dataset preparation, training, inference, and ComfyUI deployment. This guide is written for Windows beginners; for WSL, Linux, and training optimization topics, please refer to other documents.

---

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Installing CUDA 13.0.2](#2-installing-cuda-1302)
3. [Python Environment and Repository Setup](#3-python-environment-and-repository-setup)
4. [Hugging Face Authentication and Model Download](#4-hugging-face-authentication-and-model-download)
5. [Dataset Preparation](#5-dataset-preparation)
6. [Preprocessing: Resize · Latents · Text Embedding Cache](#6-preprocessing-resize--latents--text-embedding-cache)
7. [Using the GUI](#7-using-the-gui)
8. [Running Training](#8-running-training)
9. [LoRA / Adapter Variant Selection Guide](#9-lora--adapter-variant-selection-guide)
10. [Inference](#10-inference)
11. [Deploying to ComfyUI](#11-deploying-to-comfyui)
12. [Updating](#12-updating)

---

## 1. System Requirements

| Item | Minimum | Recommended |
|---|---|---|
| GPU | At least **RTX 3060 or higher — 2xxx series and below are not supported** | VRAM 16 GB or more |
| System RAM | 16 GB | 32 GB or more |
| Disk | 60 GB free | 200 GB or more (for cache + accumulated outputs) |
| OS | Windows 11 / Ubuntu 22.04+ | Ubuntu 24.04 (stable FA2/CUDA 13 builds) |
| Python | **Must be 3.13** | - |

---

## 2. Installing CUDA 13.0.2

You need the latest CUDA for stable operation with PyTorch 2.x + Flash Attention 2. Download 13.0.2 from the NVIDIA official archive.

Download page: <https://developer.nvidia.com/cuda-13-0-2-download-archive>

### 2.1 Windows Installation

1. On the page above, select **Operating System: Windows → Architecture: x86_64 → Version: 11/10 → Installer Type: exe (local)**.
2. Run the downloaded `cuda_13.0.2_windows.exe` → choose "Express (Recommended)" install.
3. After installation, verify in PowerShell:

   ```powershell
   nvidia-smi
   nvcc --version
   ```

4. If `nvcc` is not recognized, add the following to your system `Path` environment variable:

   ```
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\libnvvp
   ```

5. Reboot and verify `nvcc --version` again.

> **Driver note**: CUDA 13.x requires NVIDIA driver 580 or higher. If you have an older driver, update it first via GeForce Experience or the NVIDIA Download Center.

### 2.2 (Optional) Upgrading to CUDA 13.2 + torch 2.12 nightly

The default install is CUDA 13.0 + torch 2.11 stable. For approximately **10% faster training** on RTX 50-series GPUs and similar, you can switch to CUDA 13.2 + torch 2.12 nightly (see [docs/optimizations/cuda132.md](../optimizations/cuda132.md) for benchmarks). No build tools are needed — a pre-built trimmed FA2 wheel is already referenced in `pyproject.toml` and `uv sync` will download it automatically.

Steps:

1. **Download and install CUDA 13.2.**
   At <https://developer.nvidia.com/cuda-downloads?target_os=Windows>, select **Windows → x86_64 → 11/10 → exe (local)**. You can install it on top of 13.0 or let both versions coexist (they get separate `v13.2` and `v13.0` directories).
2. **Toggle comments in `pyproject.toml`.** Make two edits:

   **(a) torch / torchvision** — comment out the "Windows: stable" two lines and uncomment the "Windows: cuda132 opt-in" two lines inside `dependencies`:

   ```toml
   # Windows: stable (default).
   # "torch>=2.11.0,<2.12 ; sys_platform == 'win32'",
   # "torchvision>=0.26.0,<0.27 ; sys_platform == 'win32'",
   # Windows: cuda132 opt-in. ...
   "torch>=2.12.0.dev0,<2.13 ; sys_platform == 'win32'",
   "torchvision>=0.27.0.dev0,<0.28 ; sys_platform == 'win32'",
   ```

   **(b) flash-attn** — comment out the "Windows: stable (default) — built against torch 2.11 + CUDA 13.0" line in the same `dependencies` block, then uncomment the "Windows: cuda132 opt-in — trimmed FA2" line below it:

   ```toml
   # Windows: stable (default) — built against torch 2.11 + CUDA 13.0.
   # "flash-attn @ https://github.com/mjun0812/.../flash_attn-2.8.3+cu130torch2.11-cp313-cp313-win_amd64.whl ; sys_platform == 'win32'",
   # Windows: cuda132 opt-in — trimmed FA2 ...
   "flash-attn @ https://github.com/sorryhyun/flash-attention-sm120-fix/releases/download/fa2cuda132/flash_attn-2.8.4-cp313-cp313-win_amd64.whl ; sys_platform == 'win32'",
   ```

3. **Re-sync**: run `uv sync`. This downloads torch 2.12 nightly from the cu132 index and installs the trimmed FA2 wheel from the release.

> To revert: toggle the comments back to their original state and run `uv sync` again.
>
> If you need to build from source (different GPU, different Python version, or managing wheels from your own fork): see [docs/optimizations/cuda132.md](../optimizations/cuda132.md).

---

## 3. Python Environment and Repository Setup

This project uses [`uv`](https://github.com/astral-sh/uv) for dependency management with Python 3.13.

### 3.0 One-line install (easiest, no git required) ✅

Recommended for beginners. Paste this single line into PowerShell — it installs `uv` if missing, downloads the latest release, runs `uv sync`, and creates an **"Anima LoRA GUI" desktop shortcut**:

```powershell
irm https://raw.githubusercontent.com/sorryhyun/anima_lora/main/install.ps1 | iex
```

Installs into `.\anima_lora\` (override with `$env:ANIMA_DIR`; pin a version with `$env:ANIMA_VERSION='v1.4.0'`). When it finishes:

1. `cd anima_lora`
2. Create a Hugging Face token and log in with `hf auth login` (see **§4**)
3. Download models: `python tasks.py download-models`
4. Double-click the **"Anima LoRA GUI"** desktop shortcut to launch (or `python tasks.py gui`)

> This path assumes GUI-centric use and does **not** install `make`. Most tasks are GUI buttons, so that's fine — but if you want to run `make ...` from the CLI, either run `winget install ezwinports.make` (§3.3) or use `python tasks.py` instead of `make`.
>
> Update later from inside the folder with `python tasks.py update` (release-tarball merge, no git needed).

Sections **§3.1–§3.3** below are the manual install for those who prefer `git clone` or want to understand each step.

### 3.1 Install `uv`

  ```powershell
  irm https://astral.sh/uv/install.ps1 | iex
  ```

After installation, open a new shell and confirm `uv --version` prints output.

### 3.2 Clone the Repository

```bash
git clone https://github.com/sorryhyun/anima_lora.git
cd anima_lora
```

> This guide uses `anima_lora/` as the base for all paths. Run all commands from inside this directory.

### 3.3 Install Dependencies

```bash
winget install ezwinports.make
uv sync
```

`uv sync` reads `pyproject.toml`/`uv.lock`, then **creates a virtual environment (a self-contained Python install) in a `.venv/` folder inside `anima_lora/`** and installs every dependency into it. It does *not* touch your system Python, so nothing here pollutes other projects. This is normal — you do not need to `pip install` anything yourself.

> **This is the step most newcomers get stuck on.** After `uv sync` finishes, the packages live inside `.venv/`, not in your global Python. If you just open a terminal and run `python tasks.py ...` or `make lora`, you'll likely hit `ModuleNotFoundError` because that shell is still using the *system* Python, which doesn't have the dependencies. You have to point the commands at the `.venv/` interpreter. There are two ways to do that:

**Option A — `uv run` (no activation, works everywhere).** Prefix any command with `uv run` and it transparently uses `.venv/`:

```bash
uv run make lora              # or: uv run python tasks.py lora
uv run hf auth login
uv run make download-models
```

This is the most foolproof option — it never depends on your shell state, so it can't "forget" the environment.

**Option B — activate the venv once per terminal.** Activating puts `.venv/`'s `python` first on your `PATH` for that shell session, so plain `make ...` / `python tasks.py ...` work without the `uv run` prefix:

```powershell
.venv\Scripts\activate        # Windows (PowerShell / cmd)
```
```bash
source .venv/bin/activate     # Linux / macOS / WSL
```

You'll know it worked when your prompt shows a `(anima_lora)` (or `(.venv)`) prefix. **You must re-activate in every new terminal window** — activation does not persist. To leave it, run `deactivate`. VSCode users can select the `.venv` interpreter once (Command Palette → *Python: Select Interpreter*) and its integrated terminal will auto-activate.

> Throughout the rest of this guide, commands are written as plain `make ...` / `python tasks.py ...`, which assume you've **either** activated the venv (Option B) **or** are prefixing with `uv run` (Option A). If a command fails with `ModuleNotFoundError` or `command not found: make`, this is almost always the cause — activate the venv or add `uv run`.

---

> ## 🖥️ Prefer not to use the command line? Use the GUI.
>
> With dependencies installed, you can do almost everything from here on in the GUI instead of typing commands — **model download, preprocessing, training, dataset/caption browsing, and merging are all buttons in one window.** Most newcomers will be happiest this way:
>
> ```bash
> make gui          # (or: uv run make gui)
> ```
>
> You'll still want to skim **§4** to create a Hugging Face token and log in (`hf auth login`) before the GUI's download button can fetch models, and **§5** to lay out your dataset. After that, the GUI covers the rest. Sections §6–§11 document the equivalent CLI commands — read them if you want to understand or script what the GUI does, but you don't have to run them by hand. Full GUI walkthrough: **[§7 Using the GUI](#7-using-the-gui)**.

---

## 4. Hugging Face Authentication and Model Download

### 4.1 Get a Token and Log In

1. Create a token with **read** permissions at <https://huggingface.co/settings/tokens>.
2. Log in from the terminal:

   ```bash
   hf auth login
   ```

   Paste the token and press Enter.

### 4.2 Download Models

```bash
make download-models
```

This command automatically downloads the following three items and organizes them under `models/`:

| File | Path |
|---|---|
| Anima DiT (diffusion model) | `models/diffusion_models/anima-base-v1.0.safetensors` |
| Qwen3 0.6B text encoder | `models/text_encoders/qwen_3_06b_base.safetensors` |
| QwenImage VAE | `models/vae/qwen_image_vae.safetensors` |

The same command also pulls **SAM3** and **MIT**, which are only used by the optional masked-loss step (see §8.2). Masking itself is opt-in, and even within it either segmenter can be toggled off — feel free to ignore these checkpoints if you're not training with masked loss.

> **If the download is interrupted**: you can re-download individual components with targets like `make download-anima`, `make download-sam3`, or `make download-mit`.

---

## 5. Dataset Preparation

Anima LoRA uses an *image + same-name `.txt` caption sidecar* structure. Example layout of `image_dataset/`:

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

- Following Anima's official guidelines, tag order is always [meta] [character] [series] [artist] [general]. For example:

```
absurdres, safe, 1girl, chitanda eru, hyouka, @channel (caststation), full body, serafuku, She is saying hi.
```

- Based on personal experimentation, quality tags such as absurdres, highres, and masterpiece are best omitted or kept to a minimum. Alternatively, once the officially released mod guidance is available, you can skip quality tags entirely.
- Place original images in `image_dataset/` (filenames are free — keep them in this location).

### 5.2 What is `num_repeats` and when should I touch it? (Summary: **leave it alone**)

Inside `configs/base.toml`'s `[[datasets.subsets]]` you'll see `num_repeats = 1`. This specifies **how many times each image is used per epoch** — a kohya-ss style option that appears frequently in other LoRA trainer guides.

- **In this guide's standard workflow, leave it at `1`.** When training with all images in a single `image_dataset/` folder, increasing `num_repeats` only *lengthens each epoch* — the effect is identical to increasing `max_train_epochs`. Adjusting training volume through epoch count is more intuitive, and all presets and method configs in this project are tuned assuming `num_repeats = 1`.
- **When does increasing it make sense?** Only as a *balancing tool* when a single run has *multiple subsets (folders)* with very different image counts — to boost the exposure frequency of smaller folders (e.g., Character A with 1000 images + Character B with 50 images: set only the B subset to `num_repeats = 20`). It does not apply to single-folder training.
- **Where do I change it?** `num_repeats` is a dataset setting, not a method setting, so it is not exposed in `configs/methods/`, `configs/gui-methods/`, or the GUI training tab. If you truly need to change it, edit `[[datasets.subsets]]` in `configs/base.toml` directly (or in a separate TOML specified via `--dataset_config <path>`). *If you simply want more training on the same images*, increase `max_train_epochs` rather than `num_repeats`.

---

## 6. Preprocessing: Resize · Latents · Text Embedding Cache

To optimize training speed and VRAM usage, **resize → VAE latent caching → text embedding caching** are done in advance.

```bash
make preprocess              # Run all three steps (for LoRA / standard training)
# Or step by step
make preprocess-resize       # 1) image_dataset/ → post_image_dataset/resized/
make preprocess-vae          # 2) VAE latent caching → post_image_dataset/lora/
make preprocess-te           # 3) Text encoder output caching → post_image_dataset/lora/
make preprocess-pe           # (Optional) PE-Core vision encoder feature caching — IP-Adapter only
```

> **⚠️ Caches are reused — they are never automatically deleted.**
> `make preprocess` (and the GUI's *Preprocess* button) **reuses existing caches as-is**. The `.npz` / `_te.safetensors` / `_pe.safetensors` files inside `post_image_dataset/lora/` are *never overwritten or deleted* — only missing entries are processed. This makes re-running very fast and safe to interrupt.
>
> In other words, running `make preprocess` again with existing caches won't lose any data. Conversely, if you **change captions, the tokenizer, or resize options and need to regenerate from scratch**, you must manually delete the cache directory (`post_image_dataset/lora/` or `post_image_dataset/easycontrol/`) and re-run.

### 6.1 What the Resize Step Does

- Resizes images to the pixel alignment required by the VAE
- Automatically sorts images into fixed token buckets where *(H/16) × (W/16) ≈ 4096 patches*
- Automatically excludes images that are too small (default: below 0.5 MP) and reports them
- Saves results as PNGs in `post_image_dataset/resized/`

### 6.2 Latent Caching

- Runs the VAE once on all resized images and saves the results to disk
- The VAE is not loaded onto the GPU during training, saving significant VRAM
- Cache location: `post_image_dataset/lora/{stem}_{WxH}_anima.npz`
- Script: `preprocess/cache_latents.py`

### 6.3 Text Embedding Caching

- Pre-computes Qwen3 0.6B + LLM adapter outputs
- If `use_shuffled_caption_variants = true`, also caches comma-shuffled caption variants (randomly selected during training)
- Cache location: `post_image_dataset/lora/{stem}_anima_te.safetensors`
- Captions are always read from the original `.txt` files in `image_dataset/` (not copied to the resized folder)
- Script: `preprocess/cache_text_embeddings.py`

### 6.4 PE Vision Feature Caching (Optional)

- Only needed for IP-Adapter training (and the DCW v4 fusion head)
- Pre-computes PE-Core-L14-336 vision encoder outputs so the vision encoder doesn't need to be loaded during training
- Cache location: `post_image_dataset/lora/{stem}_anima_pe.safetensors`

> **When do I need to regenerate caches?**
> - New images were added → just run `make preprocess` again (existing caches are kept, only new items are processed).
> - **Captions were modified** or **tokenizer/padding options were changed** → follow the ⚠️ instructions above and manually delete the cache directory (`post_image_dataset/lora/`) before re-running. A simple re-run *reuses existing caches*, so changes won't take effect.

---

## 7. Using the GUI

The PySide6-based GUI lets you edit configs, browse datasets, run preprocessing, start/monitor training, and merge LoRA — all in one window.

```bash
make gui
make gui-shortcut   # (Optional) Create a no-console-window .lnk shortcut on the Windows desktop
```

Main GUI tabs:

- **Training Config**: Select a LoRA family variant from the dropdown (recommended: `tlora` — Ortho + T-LoRA / others include `lora`, `tlora-8gb`, `tlora_ortho_reft`, `hydralora`, `reft`, etc.), edit the `presets.toml` preset (default / low_vram / etc.) and all training keys, then start training.
- **Preprocess**: Run resize + VAE + text embedding caching in one shot.
- **Dataset**: Preview images/captions and edit captions directly.
- **Merge**: Bake a trained LoRA into the base DiT to produce a standalone ComfyUI checkpoint (supports base LoRA / OrthoLoRA / T-LoRA only).

GUI training internally calls `train.py`, so the same parameters can be reproduced identically on the CLI. The GUI reads `configs/gui-methods/<variant>.toml` (one clean file per variant, no toggle blocks), so the variant list in the GUI matches `make lora-gui GUI_PRESETS=<variant>` on the CLI. Check the current variant list with `ls configs/gui-methods/`.

### 7.1 Form Editing and Save Behavior

Training/preprocessing subprocesses re-read the variant TOML file from disk, so edits made in the form but not saved will not be reflected in training. The GUI handles this in two ways:

- **Change detection**: When any field (or the `+ Extra args` text box) is edited, the `Save` button turns orange and shows `Save *` — meaning *the disk file and the screen differ*. This clears when you press `Save` or re-select the variant to reload from disk.
- **Auto-save**: If you forget to save and click `Train` / `Preprocess` anyway, the current form values are automatically written to the variant file before the subprocess starts. What you see on screen is what gets trained. (`Test` runs inference on the last trained checkpoint and is not auto-saved.)

> To try a change without committing it, edit the form and then switch to a different variant and back — the form reloads from disk and your edits are discarded.

### 7.2 Auto-Resume (checkpointing_epochs)

If training is interrupted, clicking `Train` again **automatically resumes from the last saved checkpoint**. This is one of the most useful features for handling power cuts, OOM errors, and accidentally closed windows — it's enabled by default.

How to use it in the GUI:

- The **Training** group in the Training Config tab has a `checkpointing_epochs` field (default `2` in gui-methods variants, `4` in `methods/lora.toml`). The state is saved every N epochs, overwriting a single file, so disk usage doesn't grow.
- After an interruption, click `Train` again with the same variant — `auto-resuming from checkpoint at step N` in the log means it resumed successfully. No manual flags needed.
- When training finishes normally, the resume files are automatically deleted. The final output is `output/ckpt/<output_name>.safetensors`.
- **After changing the dataset or core settings (rank/LR/epoch count etc.)** and wanting a fresh start, manually delete `output/ckpt/<output_name>-checkpoint-state/` before clicking `Train`, otherwise training continues on top of the old state.

Detailed behavior is covered in [§8.6 Auto-Resume](#86-auto-resume-checkpointing_epochs), including the difference from `save_every_n_epochs`.

### 7.3 Stopping Training and Closing the GUI

Training does not run *inside* the GUI window — when you click `Train`, the job is handed to a small background **training daemon** that runs `train.py` as a detached process. This has two practical consequences:

- **The `Stop` button aborts the current training job.** The daemon keeps running and advances to the next queued job (if any), so stopping one run never tears down the queue. This is the same as `make daemon-kill` on the CLI. (`Stop` also cancels an in-progress `Test` or `Preprocess`, which *do* run inside the GUI.)
- **Closing the GUI does NOT stop training.** Because the job runs in the detached daemon, training keeps going after you close the window — handy for long runs you want to leave overnight. When you reopen the GUI (`make gui`), it automatically reconnects to the still-running job and you'll see `Re-attached to running job …` in the log, with the progress bar and output picking up live again. (This also surfaces jobs started from the CLI or the ComfyUI trainer node.)

> To fully shut training down — kill the active job *and* stop the daemon to free the GPU — use `make daemon-terminate` on the CLI. `Stop` alone leaves the daemon up.
>
> `Test` and `Preprocess` are the exception: they run as in-window subprocesses, so closing the GUI cancels them.

---

## 8. Running Training

All training runs through TOML config files and HuggingFace Accelerate. The config merge order is `configs/base.toml → configs/presets.toml[<preset>] → configs/methods/<method>.toml → CLI args`, with method settings winning over preset settings on overlap.

### 8.1 Quick Start

**The recommended starting point is OrthoLoRA + T-LoRA (the `tlora` variant).** This is the most balanced combination for stability, detail, and style preservation, and can be used as-is for typical character/style LoRA training.

```bash
# Recommended: Ortho + T-LoRA (gui-methods/tlora.toml)
make lora-gui GUI_PRESETS=tlora                  # Standard environment
PRESET=low_vram make lora-gui GUI_PRESETS=tlora-8gb   # VRAM 8~12 GB

# Other variants (configs/gui-methods/<variant>.toml — clean single files)
make lora-gui GUI_PRESETS=lora                   # Plain baseline LoRA
make lora-gui GUI_PRESETS=tlora_ortho_reft       # Ortho + T-LoRA + ReFT stack
make lora-gui GUI_PRESETS=hydralora              # MoE multi-head routing
make lora-gui GUI_PRESETS=reft                   # ReFT only

# Toggle-block style (select variant inside configs/methods/lora.toml directly)
make lora                          # presets.toml[default]
PRESET=low_vram make lora          # presets.toml[low_vram] — VRAM 8~12 GB
PRESET=half make lora              # Use half the dataset for quick experiments
```

> **Overriding keys from the CLI**: pass extra args like `make lora -- --network_dim 32 --max_train_epochs 24` (`tasks.py` works the same way).

### 8.2 Masked Loss (Excluding Text Bubbles)

On manga/comic-style data, excluding *speech bubbles and text regions* from the training loss produces much cleaner results.

```bash
make mask          # SAM3 + MIT (runs via temp dir) → post_image_dataset/masks/
make mask-clean    # Delete post_image_dataset/masks/
```

Output PNGs are black-and-white: **white (255) = train**, **black (0) = exclude**. Dataset subsets use `post_image_dataset/masks/` automatically if present, falling back in order to the legacy `masks/{merged,sam,mit}/` directories (so old-layout users still work). Missing mask files are simply ignored, so not generating them is perfectly fine.

### 8.3 Commonly Adjusted Settings (LoRA defaults)

| Parameter | Default | Description |
|---|---|---|
| `network_dim` | `32` | LoRA rank. Higher = more expressive, more parameters |
| `network_alpha` | `32` | LoRA scale (usually the same as `network_dim`) |
| `learning_rate` | `2e-5` | Learning rate. Can be lowered for Hydra |
| `max_train_epochs` | `4` | Increase as dataset size decreases |
| `save_every_n_epochs` | `2` (gui-methods) / `4` (methods) | Accumulative adapter weight save interval |
| `checkpointing_epochs` | `2` (gui-methods) / `4` (methods) | Resume-state save interval (single file, overwritten) |
| `caption_dropout_rate` | `0.1` | Replaces some captions with empty strings (helps CFG) |
| `use_shuffled_caption_variants` | `true` | Use comma-shuffled caption variants |

Variant toggles (`use_ortho`, `use_timestep_mask`, `add_reft`, `use_moe_style`, `router_source`, etc.) are activated by uncommenting the relevant block in `configs/methods/lora.toml`, or by using the per-variant single-file from `configs/gui-methods/<variant>.toml`. **The recommended `tlora` variant has `use_ortho = true` + `use_timestep_mask = true` pre-enabled, giving you the OrthoLoRA + T-LoRA combination out of the box.**

### 8.4 What Happens During Training

1. Load text encoder → generate/verify cache → unload
2. Load VAE → generate/verify cache → unload
3. *Lazily* load DiT to avoid VRAM conflicts with the caching stages
4. Patch adapter network into the DiT's attention / FFN modules (targets differ per variant)
5. Sample noise → DiT forward pass → flow-matching loss → backward → optimizer step
6. (Optional) Compute validation loss and generate sample images using `validation_split`

### 8.5 Outputs

- Trained weights: `output/ckpt/<output_name>.safetensors` (auto-differentiated per variant as `anima`, `anima_tlora_ortho`, `anima_tlora_reft`, `anima_hydra`, `anima_postfix`, etc.)
- Checkpoints: saved to `output/ckpt/` every `save_every_n_epochs` (`.snapshot.toml` sidecar + `_moe` companion for Hydra)
- Validation samples: `output/ckpt/sample/`
- Inference output images: `output/tests/`

### 8.6 Auto-Resume (checkpointing_epochs)

If training is interrupted, **it can automatically resume from the last saved point**. This is extremely useful for power cuts, OOM errors, accidentally closing the window, or pausing to shut down for the night. It is already enabled in the default method files; no additional configuration needed.

```toml
checkpointing_epochs = 2     # Save resume state every 2 epochs (overwrite)
```

Difference from `save_every_n_epochs`:

| Key | Saves | Accumulates? | Purpose |
|---|---|---|---|
| `save_every_n_epochs` | Adapter weights (e.g., `anima_lora-000004.safetensors`) | **Yes** (or limited with `save_last_n_epochs`) | Compare intermediate results or find the overfitting point |
| `checkpointing_epochs` | Full resume state (optimizer / scheduler / RNG / adapter weights) | **No — overwrites a single file** | Automatic resume after interruption |

How it works:

- **Auto-save**: Every `checkpointing_epochs` epochs, `output/ckpt/<output_name>-checkpoint-state/` (state directory) and `<output_name>-checkpoint.safetensors` (weights) are saved together, overwriting the previous version. Disk usage stays flat.
- **Auto-resume**: Running the same command again (`make lora` etc.) automatically continues training if a checkpoint exists and `max_train_steps` hasn't been reached. No `--resume` flag needed. `auto-resuming from checkpoint at step N` in the log confirms it worked.
- **Auto-cleanup**: When training finishes normally, both files are automatically deleted — the resume state is temporary, and the final artifact is `output/ckpt/<output_name>.safetensors` (see 8.5).
- **Manual resume**: To restore to a specific earlier state, you can pass `--resume <state_dir>` explicitly.

> **When to disable**: for short experimental runs or when disk space is tight, comment out `checkpointing_epochs`. For serious training on larger datasets, it's almost always worth keeping enabled.
>
> **Warning**: if you change the dataset, captions, or core settings (rank/LR/epoch count etc.), resuming from an old checkpoint is meaningless or harmful. Delete `output/ckpt/<output_name>-checkpoint-state/` manually and start fresh.

---

## 9. LoRA / Adapter Variant Selection Guide

> **🌟 Recommended**: if you're new or making a typical character/style LoRA, start with **`tlora` (OrthoLoRA + T-LoRA)**. It offers the best balance of detail, style preservation, and training stability.

| Variant | How to Run | When to Use |
|---|---|---|
| **OrthoLoRA + T-LoRA** ⭐ | `make lora-gui GUI_PRESETS=tlora` | **Recommended.** SVD-based orthogonal rotation (OrthoLoRA) + timestep rank masking (T-LoRA) stacked. Produces `anima_tlora_ortho.safetensors` |
| **OrthoLoRA + T-LoRA (8 GB)** | `make lora-gui GUI_PRESETS=tlora-8gb` or `PRESET=low_vram make lora-gui GUI_PRESETS=tlora` | Same recommended combination on VRAM 8~12 GB |
| **Plain LoRA** | `make lora-gui GUI_PRESETS=lora` or `make lora` | Simplest baseline, for comparison experiments |
| **Plain LoRA (8 GB)** | `make lora-gui GUI_PRESETS=lora-8gb` or `PRESET=low_vram make lora` | VRAM 8~12 GB |
| **T-LoRA + Ortho + ReFT** | `make lora-gui GUI_PRESETS=tlora_ortho_reft` | Adds representation editing (ReFT) to the recommended combination for fine-grained control with fewer extra parameters |
| **HydraLoRA** | `make lora-gui GUI_PRESETS=hydralora` (8 GB: `hydralora-8gb`) | MoE multi-head routing; fit multiple concepts in one adapter |
| **ReFT only** | `make lora-gui GUI_PRESETS=reft` or set `add_reft = true` in `methods/lora.toml` | Representation Fine-Tuning — very few parameters |
| **Postfix Tuning** *(experimental)* | `make exp-postfix` or `make lora-gui GUI_PRESETS=postfix_ortho_cond` | Learnable N-vector appendix on cross-attention (caption-conditional + orthogonal) |
| **ChimeraHydra** *(experimental)* | `make exp-chimera` or `make lora-gui GUI_PRESETS=chimera_hydra` | Content/frequency dual-pool MoE — for research |

For detailed options per variant, see [`docs/guidelines/training.md`](training.md) and the individual docs under `docs/methods/`.

> **Compatibility notes**
> - HydraLoRA / ReFT and similar adapter variants require `cache_llm_adapter_outputs = true` (enabled by default) to work correctly.
> - The OrthoLoRA + T-LoRA portions of `tlora` and `tlora_ortho_reft` can be baked into the base DiT with `make merge` to produce a standalone ComfyUI checkpoint (the ReFT portion cannot be baked — use `--allow-partial`).

---

## 10. Inference

### 10.1 Quickest Test

To immediately generate samples with the adapter you just trained, use the matching `make test-*` command. All of them automatically pick the most recently saved adapter from `output/ckpt/`.

```bash
make test                        # Plain LoRA / OrthoLoRA / T-LoRA / ReFT
make test SPECTRUM=1             # Spectrum-accelerated inference
make test MOD=1                  # Modulation guidance (pooled_text_proj) — composable with SPECTRUM=1
make test NOLORA=1               # Base DiT only (omits --lora_weight); combine with MOD=1 for mod-only path
make test-hydra                  # HydraLoRA (router-live, anima_hydra*_moe.safetensors)
make test-merge                  # Inference with baked standalone DiT (*_merged.safetensors)
make test-dcw                    # LoRA + DCW scalar correction (sampler-level SNR-t bias correction)
make test-dcw-v4                 # LoRA + DCW v4 learnable calibrator
# Experimental inference
make exp-test-postfix            # Postfix tuning (vanilla)
make exp-test-postfix-exp        # postfix_exp variant
make exp-test-postfix-func       # postfix_func variant
```

### 10.2 Manual Inference

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

Commonly used flags:

| Flag | Description |
|---|---|
| `--lora_weight` | Path to the trained adapter. Multiple adapters can be provided |
| `--lora_multiplier` | Adapter strength (0.0–1.5) |
| `--image_size H W` | Output resolution (e.g., `1024 1024`, `1024 1536`) |
| `--infer_steps` | Number of denoising steps (typically 20–50) |
| `--guidance_scale` | CFG strength (3.0–5.0 recommended) |
| `--sampler` | `er_sde`, `euler`, `dpm++`, etc. |
| `--seed` | Seed for reproducibility |
| `--spectrum` | Enable Spectrum acceleration |
| `--pgraft` | P-GRAFT (LoRA cutoff in late denoising — base model handles late detail) |

For the full list of inference options and P-GRAFT usage, see [`docs/guidelines/inference.md`](inference.md).

---

## 11. Deploying to ComfyUI

ComfyUI core natively supports the Anima base DiT (`UNETLoader` / `CLIPLoader` work out of the box). Deployment differs by adapter type.

### 11.1 Classic LoRA / OrthoLoRA / T-LoRA

Copy the `.safetensors` file from `output/ckpt/` directly to `ComfyUI/models/loras/` — the standard ComfyUI LoraLoader node will work immediately. For a cleaner standalone checkpoint:

```bash
make merge ADAPTER_DIR=output/ckpt                 # Bake the latest weights into the base DiT
make merge ADAPTER_DIR=output/ckpt MULTIPLIER=0.8  # Adjust strength
```

The baked `*_merged.safetensors` can be loaded as a standalone model with ComfyUI's `UNETLoader`.

### 11.2 HydraLoRA / ReFT / Postfix

These variants cannot be loaded with ComfyUI's default LoraLoader (they involve routing and token insertion, not simple weight deltas) and require dedicated nodes:

- **Anima Adapter Loader** (`custom_nodes/comfyui-hydralora/`) — unified handling for LoRA / Hydra / ReFT / postfix. See the `README.md` in that folder for usage details.
- **Spectrum KSampler / Mod Guidance / DCW nodes** — separate repository at <https://github.com/sorryhyun/ComfyUI-Spectrum-KSampler>

---

## 12. Updating

```bash
make update              # Fetch the latest release from GitHub and apply it + auto-run uv sync
make update -- --dry-run # Preview which files would change
```

`update` does not touch `image_dataset/`, `post_image_dataset/`, `output/`, or `models/`, and will prompt you on config file conflicts.

---

## Further Reading

- [`docs/guidelines/training.md`](training.md) — Adapter variants, caption shuffling, masked loss, dataset config details
- [`docs/guidelines/inference.md`](inference.md) — Inference flags, DCW, Spectrum, prompt file format
- [`docs/guidelines/difference_between_comfy.md`](difference_between_comfy.md) — Implementation differences between anima_lora and ComfyUI core
- [`docs/methods/timestep_mask.md`](../methods/timestep_mask.md) — T-LoRA timestep mask
- [`docs/methods/psoft-integrated-ortholora.md`](../methods/psoft-integrated-ortholora.md) — OrthoLoRA details (the orthogonal rotation part of the recommended `tlora` variant)
- [`docs/methods/spectrum.md`](../methods/spectrum.md) — Spectrum acceleration: how it works and options
- [`docs/methods/dcw.md`](../methods/dcw.md) — DCW (scalar + v4 learnable calibrator)
- [`docs/methods/mod-guidance.md`](../methods/mod-guidance.md) — Modulation guidance
- [`docs/methods/hydra-lora.md`](../methods/hydra-lora.md) — HydraLoRA multi-head routing
- [`docs/methods/reft.md`](../methods/reft.md) — ReFT representation editing
- [`docs/experimental/postfix.md`](../experimental/postfix.md) — Postfix (cond+ortho)
- [`docs/optimizations/cuda132.md`](../optimizations/cuda132.md) — How to upgrade to CUDA 13.2
- [`docs/optimizations/full_model_cudagraph.md`](../optimizations/full_model_cudagraph.md) — `compile_mode=full` + CUDAGraph invariants and debugging

Questions and bug reports are welcome on GitHub Issues. Happy training!
