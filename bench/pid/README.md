# PiD × Anima — integration bench (SUCCESS)

NVIDIA **PiD** (Pixel Diffusion Decoder, [`nv-tlabs/PiD`](https://github.com/nv-tlabs/PiD),
checkpoints released 2026-06-02) decodes Anima's Qwen-Image latents **directly**
into 4× super-resolved pixels — a drop-in replacement for VAE decode. Verified
end-to-end and shipped as a self-contained ComfyUI node.

> **TL;DR** — Anima's latents are byte-compatible with PiD's `qwenimage` checkpoint.
> A self-contained, gemma-free, hydra-free node ships from its own repo
> ([`sorryhyun/ComfyUI-Anima-PiD`](https://github.com/sorryhyun/ComfyUI-Anima-PiD))
> and produces clean 4× upscales. Single-pass up
> to ~3072px on 16 GB; tiling for 4K. Weights are NSCLv1 (non-commercial).

## Why it works (and why it's not L2P)

PiD operates **at the VAE-decode position** — it never touches the DiT. So the
frozen-core failure that killed L2P (RGB-token transplant upstream of generation)
doesn't apply. PiD just consumes the latent and emits pixels.

Compatibility is exact, not approximate:
- Anima's `qwen_vae.latents_mean/std` == PiD's `QwenImage_VAE_2d` constants
  (the AutoencoderKLQwenImage / Wan2.1 defaults) — byte-for-byte.
- Anima's `encode_pixels_to_latents` produces `(mu-mean)/std`, which **is** PiD's
  `LQ_latent` convention. ComfyUI stores raw VAE latents, so the node applies
  `(samples-mean)/std` (= Wan21 `process_in`, `scale_factor=1.0`).
- In the `LQ_latent` path PiD emits RGB directly, so the **Qwen VAE is not needed**
  and the **gemma text encoder is skipped** (zero caption embeddings: the 4-step
  distill path uses no CFG and the y-embedder is RMSNorm-fronted → zeros are safe).
  No multi-GB text-encoder download.

## Validation results

| Stage | What | Result |
|---|---|---|
| 1. VAE round-trip | PiD's own Qwen VAE decodes an Anima cached latent | **0 missing / 0 unexpected** keys, **PSNR 30.4 dB** vs source → latent space identical |
| 2. PiD SR (hydra path) | full nv-tlabs pipeline, gemma stubbed | 64→2048px, 4.18 s, 12.7 GB (fp32 params) |
| 3. Node core (self-contained) | vendored net, bf16, no hydra | 64→2048px, **3.80 s, 6.62 GB**, matches stage 2 (4.65/255) |
| 4. Node full path | loader+decode+raw-latent conversion+IMAGE | output `(1,2048,2048,3)`, matches bench (4.65/255) |
| 5. `torch.compile` | per-resolution graph, precomputed RoPE caches | warm **2.14 s** vs eager 3.81 s (**~1.8×**), +37 s one-time, matches eager (1.18/255) |

## Memory ceilings (RTX 5070 Ti, 16 GB, bf16, 4-step)

| Output | Latent grid | No-tile peak | Fits 16 GB? | Time |
|---|---|---|---|---|
| 2048² | 64 | 6.6 GB | ✅ | 3.8 s |
| 3072² | 96 | 11.3 GB | ✅ (seam-free) | 11.0 s |
| 4096² (full 4K) | 126×128 | OOM | ❌ → use tiling | — |

**Takeaway:** single-pass (no tiling, no seams) is viable up to **~3072px** on 16 GB.
Full-frame **4K requires tiling** (`tile_latent=64` → 2048px tiles, feather-blended).

## The ComfyUI node

Ships from the standalone [`sorryhyun/ComfyUI-Anima-PiD`](https://github.com/sorryhyun/ComfyUI-Anima-PiD) repo (symlinked into `comfy/custom_nodes/comfyui-anima-pid`):
- **Anima PiD Loader** — `models/pid/*.pth` → `ANIMA_PID`.
- **Anima PiD Decode (4x SR)** — `ANIMA_PID` + `LATENT` → `IMAGE`; `steps`, `sigma`,
  `tile_latent`/`tile_overlap`, `compile`.

Drop it where `VAEDecode` was: `KSampler → LATENT → Anima PiD Decode → Save Image`.
Self-contained (vendored PiD net under `pid_net/`, no hydra/imaginaire); needs only
torch/numpy/pillow. Checkpoint: `models/pid/pid_qwenimage_2kto4k_4step.pth`.

## Caveats

- **License**: PiD weights are NVIDIA **NSCLv1 — non-commercial**. Don't ship them
  in a paid product. (Vendored net *code* is Apache-2.0; wrapper is MIT.)
- **Tiling seams**: multi-tile feather blending is implemented but **not yet
  eyeballed on a real >1-tile 4K decode** — verify before relying on it. Single-tile
  / single-pass (≤3072px) is fully verified and clean.
- **gemma-skip**: caption is null → no text-guided detail (LQ latent drives the SR).
  Faithful caption-conditioned mode would need `Efficient-Large-Model/gemma-2-2b-it`
  (~5 GB, ungated mirror).
- **`sigma`** is passed without actually noising the latent, so above 0 it's an
  off-label detail/sharpening dial; `sigma=0` is exactly faithful.

## Artifacts

- `stage1_vae_roundtrip.py` — latent-space identity check (only needs the 0.5 GB VAE).
- `results/stage1_pidvae_recon.png` — VAE round-trip recon.
- `results/stage2_pid_sr.png` — first PiD 4× decode (hydra path).
- `results/node_core_validate.png`, `results/node_full_validate.png` — node validation.
- `results/notile_3072.png` — single-pass 3072px (seam-free).
- `results/<ts>-qwen-feasibility/result.json` — structured envelope.
- Self-contained core: `pid_core.py` in the [`ComfyUI-Anima-PiD`](https://github.com/sorryhyun/ComfyUI-Anima-PiD) repo.
