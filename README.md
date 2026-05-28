# anima_lora

[한국어](README.ko.md) · 📖 [가이드북 (Windows 초보자용 한국어 종합 가이드)](docs/guidelines/가이드북.md)

LoRA / T-LoRA training and inference engine for the [Anima](https://huggingface.co/circlestone-labs/Anima) diffusion model (DiT-based, flow-matching).

Four things this repo aims to do well:

1. **Fast LoRA training** on consumer GPUs — per-block `torch.compile` over a tiny fixed shape set (one block graph per token-count family), end to end.
2. **Solid conventional implementations** — LoRA, OrthoLoRA, and T-LoRA stack together and bake losslessly into a standalone DiT checkpoint.
3. **Recent methods, engineered for Anima** — Spectrum inference, DCW & SMC-CFG samplers, OrthoHydraLoRA, and modulation guidance, each implemented end-to-end against Anima's compile contract rather than dropped in as a toy port.
4. **A broad experimental surface** — SPD, ChimeraHydra, Soft Tokens, Turbo distillation, ReFT, IP-Adapter, EasyControl, DirectEdit, embedding inversion.

> **At-a-glance diagrams** for every method (DiT internals, LoRA, OrthoLoRA, T-LoRA, HydraLoRA, ReFT, Spectrum, modulation, compile optimizations) live in [`docs/structure_images/`](docs/structure_images/) — paired with prose walkthroughs in [`docs/structure/`](docs/structure/).

---

## How to start

One line — installs [uv](https://astral.sh/uv) if missing, fetches the latest release, and runs `uv sync` (no git required). The installer is published as a signed-by-checksum release asset:

```bash
# Linux / macOS
curl -LsSf https://github.com/sorryhyun/anima_lora/releases/latest/download/install.sh | sh
```
```powershell
# Windows (PowerShell)
irm https://github.com/sorryhyun/anima_lora/releases/latest/download/install.ps1 | iex
```

Installs into `./anima_lora/` (override with `ANIMA_DIR`). On Windows it also drops an **"Anima LoRA GUI"** shortcut on your desktop.

<details>
<summary><b>Safer install</b> — inspect &amp; verify the script before running</summary>

Every release ships a `checksums.txt` (SHA-256 of the installers + source archives). Download, verify, then run:

```bash
# Linux / macOS
curl -fLO https://github.com/sorryhyun/anima_lora/releases/latest/download/install.sh
curl -fLO https://github.com/sorryhyun/anima_lora/releases/latest/download/checksums.txt
grep install.sh checksums.txt | sha256sum -c -    # must print "install.sh: OK"
less install.sh                                    # read it
sh install.sh
```
```powershell
# Windows (PowerShell)
iwr https://github.com/sorryhyun/anima_lora/releases/latest/download/install.ps1 -OutFile install.ps1
iwr https://github.com/sorryhyun/anima_lora/releases/latest/download/checksums.txt -OutFile checksums.txt
(Get-FileHash install.ps1 -Algorithm SHA256).Hash.ToLower()   # compare against checksums.txt
notepad install.ps1                                            # read it
powershell -ExecutionPolicy Bypass -File .\install.ps1
```
</details>

**Reproducible / pinned install** — set `ANIMA_VERSION` to install a specific tag instead of latest (the recommended path when you need a known-good environment):

```bash
ANIMA_VERSION=v1.4.0 sh install.sh       # or: $env:ANIMA_VERSION='v1.4.0'; irm ... | iex
```

Then authenticate and pull models:

```bash
cd anima_lora
hf auth login
make download-models      # DiT + Qwen3 TE + QwenImage VAE (+ SAM3 / MIT / PE for masking & image conditioning) into models/
make gui                  # recommended — config editor + dataset browser + training monitor
```

Update later in place with `make update` (release-tarball merge, no git needed). Prefer cloning the repo? See [Setup → Manual](#manual-from-a-clone).

---

## 1. Fast training

**13.4 GB peak VRAM · 1.1 s/step** on a single RTX 5060 Ti while **rank=32 1MP resolution lora training** — achieved by co-designing the data pipeline, attention, and compiler stack so Dynamo sees a tiny fixed set of shapes (one block graph per token-count family) for the whole run.

| Lever | Summary |
|---|---|
| Constant-token bucketing | Buckets fall into two token-count families — 4032 and 4200 patches — each resolution *exactly* filling its count, so there is zero intra-bucket padding. Forwards run at native token counts, so `torch.compile` traces one block graph per distinct count (2). The legacy pad-to-static path was removed (it leaked padding into flash self-attn and couldn't run this table — 4200 > 4096). |
| Max-padded text encoder | Text outputs padded to 512 and zero-filled — the pretrained DiT uses zero keys as cross-attn sinks, so trimming breaks it. Also gives the compiler another fixed dim. |
| Per-block `torch.compile` | Each DiT block compiled independently with Inductor (`compile_blocks()`). Combined with native-token bucketing this pins the trace to 2 block graphs and eliminates guard recompilation. |
| Compile-friendly hot path | Audited every forward for patterns dynamo can't trace cleanly — `einops.rearrange` replaced with explicit `.unflatten()/.permute()` chains, `torch.autocast` context managers replaced with direct `.to(dtype)` casts, dict `.items()` loops hoisted out of compiled regions, FA4 wrapped in `@torch.compiler.disable` for clean graph breaks. |
| Flash Attention 2 | `flash_attn` 2.x with SDPA fallback. FA4 evaluated and removed — see [fa4.md](docs/optimizations/fa4.md). |

Compile pipeline details in [docs/optimizations/for_compile.md](docs/optimizations/for_compile.md).

---

## 2. Solid conventional implementations

The default training config stacks **LoRA + OrthoLoRA + T-LoRA** together. All three fold losslessly into a standalone DiT checkpoint via thin-SVD export at save time, so you can ship ComfyUI-compatible `*_merged.safetensors` with no adapter loader dependency.

| Variant | Pitch | Details |
|---|---|---|
| **LoRA** | Classic low-rank, rank 16–32. | — |
| **OrthoLoRA** | SVD-parameterized with orthogonality regularization; exports as plain LoRA. | [psoft-integrated-ortholora.md](docs/methods/psoft-integrated-ortholora.md) |
| **T-LoRA** | Timestep-dependent rank masking — low rank at high noise, full rank at low noise. Training-only mask, so merge is bit-equivalent. | [timestep_mask.md](docs/methods/timestep_mask.md) |

**Side-by-side** — same prompt, `er_sde` 30 steps, `cfg=4.0`, 1024². Each LoRA trained at rank 16 for 2 epochs on a 20% subset with training seed 42; inference seeds `{41, 42, 43}`. Reproduce with `python _archive/bench_methods.py`.

|  | **LoRA** | **OrthoLoRA + T-LoRA** |
|:---:|:---:|:---:|
| seed 41 | <img src="docs/side_by_side/lora/20260423-154854-014_41_.png" width="320"> | <img src="docs/side_by_side/ortho_tlora/20260423-155545-258_41_.png" width="320"> |
| seed 42 | <img src="docs/side_by_side/lora/20260423-154938-584_42_.png" width="320"> | <img src="docs/side_by_side/ortho_tlora/20260423-155631-762_42_.png" width="320"> |
| seed 43 | <img src="docs/side_by_side/lora/20260423-155024-080_43_.png" width="320"> | <img src="docs/side_by_side/ortho_tlora/20260423-155718-280_43_.png" width="320"> |

<details>
<summary>Base model and individual variants (plain, OrthoLoRA, T-LoRA)</summary>

|  | **plain (base)** | **OrthoLoRA** | **T-LoRA** |
|:---:|:---:|:---:|:---:|
| seed 41 | <img src="docs/side_by_side/plain/20260423-160513-382_41_.png" width="240"> | <img src="docs/side_by_side/ortholora/20260423-155109-338_41_.png" width="240"> | <img src="docs/side_by_side/tlora/20260423-155327-834_41_.png" width="240"> |
| seed 42 | <img src="docs/side_by_side/plain/20260423-160556-697_42_.png" width="240"> | <img src="docs/side_by_side/ortholora/20260423-155155-526_42_.png" width="240"> | <img src="docs/side_by_side/tlora/20260423-155413-304_42_.png" width="240"> |
| seed 43 | <img src="docs/side_by_side/plain/20260423-160640-759_43_.png" width="240"> | <img src="docs/side_by_side/ortholora/20260423-155241-905_43_.png" width="240"> | <img src="docs/side_by_side/tlora/20260423-155458-996_43_.png" width="240"> |

</details>

**Merging**:

```bash
make merge                                  # bake latest LoRA at multiplier 1.0
make merge ADAPTER_DIR=output/ckpt MULTIPLIER=0.8
```

Refuses non-linear-delta variants (ReFT / HydraLoRA `_moe`) by default; `--allow-partial` drops those and bakes only the LoRA portion.

---

## 3. Recent methods, engineered for Anima

Five recent papers picked up, implemented against Anima end-to-end, and shipped with the engineering they need to be actually usable — not toy reimplementations.

| Method | What it is | Engineering notes | Doc |
|---|---|---|---|
| **Spectrum inference** | Training-free speedup via Chebyshev polynomial feature forecasting (Han et al., CVPR 2026) — ≈1.75× at default settings, up to ~5× on more aggressive schedules (quality tradeoff). On cached steps every transformer block is skipped — only `t_embedder` + `final_layer` + `unpatchify` run. | `register_forward_pre_hook` on `final_layer` captures block outputs without monkey-patching the model; adaptive window schedule concentrates real forwards on early high-noise steps. Stable ComfyUI node in a separate repo: [ComfyUI-Spectrum-KSampler](https://github.com/sorryhyun/ComfyUI-Spectrum-KSampler). | [spectrum.md](docs/methods/spectrum.md) |
| **DCW calibrator** | Sampler-level SNR-t bias correction (Yu et al., CVPR 2026) — mixes each Euler step's `prev_sample` toward the model's `x0_pred` along the LL Haar band. Two modes: scalar `λ` (offline-tuned) and **v4 learnable** per-prompt calibrator with online observation. | v4 head conditions on `(aspect, prompt, observed prefix gap)` and fires after `k=7` warmup steps. Bias direction characterized as **(CFG × aspect)-dependent** on Anima — paper-direction at CFG=4 non-square, paper-opposite at CFG=1 / 1024². Trained per-checkpoint via `make dcw`. | [dcw.md](docs/methods/dcw.md) |
| **SMC-CFG** | Training-free sliding-mode CFG correction in velocity space (Wang et al., CFG-Ctrl) — treats the cond/uncond combine as a control problem applied to the residual `e = v_cond − v_uncond`. No extra DiT forwards. | Ships the **α-adaptive variant**: the paper's fixed gain `k` (≈14× off on Anima at CFG=4, visibly chattering) is replaced with `k_t = α·mean(\|e_t\|)` per step. `make test-smc-cfg` (λ=5, α=0.2); composes with Spectrum and mod-guidance. | [smc_cfg.md](docs/methods/smc_cfg.md) |
| **OrthoHydraLoRA** | MoE-style multi-head LoRA with orthogonalized experts and layer-local routing — shared `lora_down`, per-expert `lora_up_i`, learned per-sample router. Targets multi-style training without the cross-style bleed a single low-rank subspace produces. Original paper: [arXiv:2605.03252](https://arxiv.org/abs/2605.03252). | Saves two side-by-side files: `anima_hydra.safetensors` (baked-down LoRA, ComfyUI drop-in) and `anima_hydra_moe.safetensors` (full multi-head). Live routing in ComfyUI via the bundled **Anima Adapter Loader** node (`custom_nodes/comfyui-hydralora/`), which installs per-Linear forward hooks reproducing `HydraLoRAModule.forward`. | [hydra-lora.md](docs/methods/hydra-lora.md) |
| **Modulation guidance** | Distill a `pooled_text_proj` MLP that steers AdaLN modulation coefficients toward quality-positive directions (Starodubcev et al., ICLR 2026). Teacher sees real cross-attention; student sees zeroed cross-attention but receives pooled text through modulation. | Trained with `make distill-mod` against the frozen DiT. Inference applies the projection at AdaLN time so it composes with any LoRA variant; `make test MOD=1` runs a sample with it enabled (composes with `SPECTRUM=1`). | [mod-guidance.md](docs/methods/mod-guidance.md) |

---

## 4. Experimental surface

Each ships with a doc — see the link for usage, flags, and caveats.

| Feature | What it is | Doc |
|---|---|---|
| **SPD** | Spectral Progressive Diffusion (Xiao et al., 2026) — training-free multi-resolution inference (`--spd`): run early noise-dominated steps at low resolution, then inject high-frequency detail via spectral noise expansion. Optional trajectory-adapter fine-tune (`make exp-spd`). | [spd.md](docs/experimental/spd.md) |
| **ChimeraHydra** | Dual-pool additive MoE: a content pool (layer-local router) plus a frequency pool (network router on FEI + σ features), each an asymmetric HydraLoRA off a disjoint SVD subspace. Fuses HydraLoRA + TimeStep Master + FeRA. `make exp-chimera`. | [chimera-hydra.md](docs/experimental/chimera-hydra.md) |
| **Soft Tokens** | SoftREPA (Lee et al., NeurIPS 2025) — per-layer × per-t learnable text tokens (~1M params) spliced into `crossattn_emb`; DiT frozen. `make exp-soft-tokens`. | [soft_tokens.md](docs/experimental/soft_tokens.md) |
| **Turbo** | Decoupled DMD distillation (Liu et al., 2025) of the 28-step teacher into a 4–8-step generator. Output is a normal LoRA — infer with `--infer_steps 4 --cfg 1.0`. `make exp-turbo`. | [dmd2-decoupled.md](docs/experimental/dmd2-decoupled.md) |
| **DirectEdit** | Flow-inversion image editing (Yang & Ye, 2026) — invert to noise, swap edit conditioning, re-denoise with V-injection. Source captions come from the **Anima Tagger** (image → Anima-format tags). `make exp-test-directedit`. | [directedit_editing_v3.md](docs/experimental/directedit_editing_v3.md) |
| **ReFT** | Block-level residual-stream intervention (LoReFT, NeurIPS 2024). Composes with any LoRA variant. | [reft.md](docs/methods/reft.md) |
| **IP-Adapter** | Decoupled image cross-attention (Ye et al. 2023). DiT frozen; trains Perceiver resampler + per-block `to_k_ip`/`to_v_ip`. | [ip-adapter.md](docs/experimental/ip-adapter.md) |
| **EasyControl** | Extended self-attention image conditioning. DiT frozen; trains per-block cond LoRA on self-attn + FFN + scalar `b_cond` gate. | [easycontrol.md](docs/experimental/easycontrol.md) |
| **Embedding inversion** | Optimize a text embedding to match a target image through the frozen DiT. | [invert.md](docs/methods/invert.md) |

> **Want to contribute?** Two areas where outside help would have outsized impact: **IP-Adapter productionization** (tests, public reference checkpoint, lighter vision encoder) and **EasyControl adapters** (canny / depth / pose / … — each control type is one self-contained PR). See [CONTRIBUTING.md → Priority areas](CONTRIBUTING.md#priority-areas).

---

## Setup

> Quick one-line install is up top in [How to start](#how-to-start). The manual clone path is below.

### Manual (from a clone)

```bash
uv sync                   # Python 3.13 with pre-built flash attention 2
hf auth login
make download-models      # DiT + Qwen3 TE + QwenImage VAE (+ SAM3 / MIT / PE for masking & image conditioning) into models/
# place training images in image_dataset/ with .txt caption sidecars
make gui                  # recommended — config editor + dataset browser + training monitor
```

`uv sync` resolves to **torch 2.12 + CUDA 13.2** .

> **Anima ships as a uv-locked application environment, not a generic pip package.** `pyproject.toml` pins `python ==3.13.*`, specific torch / flash-attn wheel URLs, and `index-strategy = "unsafe-best-match"` — these are maintainer-chosen, known-good builds. Install with `uv sync` against the committed `uv.lock`; don't `pip install` from `pyproject.toml` (pip won't honor uv's index strategy or the prebuilt flash-attn wheels).

CLI path:

```bash
make preprocess           # VAE-compatible resize & validation
make lora                 # or: PRESET=fast_16gb make lora / PRESET=low_vram make lora / make exp-chimera
make test                 # sample generation with the latest trained LoRA
```

Config chain: `configs/base.toml → configs/presets.toml[<preset>] → configs/methods/<method>.toml → CLI args`. Override with `PRESET=low_vram make lora` or `--network_dim 32 --max_train_epochs 64`. Full flag reference in [docs/guidelines/training.md](docs/guidelines/training.md) and [docs/guidelines/inference.md](docs/guidelines/inference.md).

---

## Documentation

| Doc | Contents |
|-----|----------|
| [guidelines/training.md](docs/guidelines/training.md) | Training flags, LoRA variants, caption shuffle, masked loss, dataset config |
| [guidelines/inference.md](docs/guidelines/inference.md) | Inference flags, P-GRAFT, prompt files, LoRA format conversion |
| [optimizations/](docs/optimizations/) | Compile pipeline, FA4 post-mortem, CUDA 13.2 |
| [methods/](docs/methods/) | One doc per method — HydraLoRA, ReFT, Spectrum, inversion, mod guidance, T-LoRA, OrthoLoRA |

---

## License

Toolkit code: [MIT](LICENSE).

Anima / CircleStone **base model weights** ship under the **CircleStone Labs Non-Commercial License v1.0** and are not relicensed by this repo. Any LoRA, fine-tune, or merged checkpoint trained from those weights is a Derivative and inherits the non-commercial terms. See [NOTICE](NOTICE).
