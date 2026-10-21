---
name: custom-nodes
description: The ComfyUI node map — which node lives in which standalone repo vs in-tree under custom_nodes/, where each is symlinked, and the vendor-sync rule for the _vendor/ subsets. Load before editing, publishing, or hunting for a ComfyUI node, or before touching anything under custom_nodes/.
---

# ComfyUI custom nodes

Most nodes live in standalone repos, **symlinked** into `../comfy/custom_nodes/`. Edit
the source repo, never the symlink.

## Out of tree

| Node | Repo / path | Notes |
|---|---|---|
| Spectrum KSampler + mod-guidance | https://github.com/sorryhyun/ComfyUI-Spectrum-KSampler | ships DCW scalar default `+0.01` + `auto` mode |
| PiD decode | https://github.com/sorryhyun/ComfyUI-Anima-PiD | symlinked as `comfyui-anima-pid` |
| EasyControl KSampler | `~/ComfyUI-EasyControl-KSamplerCompat` | |
| Block Compile | https://github.com/sorryhyun/ComfyUI-Anima-BlockCompile | standalone at `~/ComfyUI-Anima-BlockCompile`, symlinked as `comfyui-anima-blockcompile` |
| DAVE | https://github.com/sorryhyun/ComfyUI-Anima-DAVE | standalone at `~/ComfyUI-Anima-DAVE`, symlinked as `comfyui-anima-dave`. One node for Anima and the `moss-image` transformer; `mask=auto` picks the bundled pool per model. Its npz masks are copies of `networks/calibration/dave_alpha.npz` and `library/moss/calibration/{dave_alpha,massive_channels}.npz`, not vendor-synced |
| MOSS Toolkit | https://github.com/sorryhyun/ComfyUI-MOSS-Toolkit (renamed from ComfyUI-MOSS-i2i; registry id `comfyui-moss-toolkit`, the old `comfyui-moss-i2i` id is retired), standalone at `~/ComfyUI-MOSS-Toolkit`, symlinked as `comfyui-moss-toolkit` | i2i: LoRA loader + page-through-vision-tower text encode + VAE-reference MODEL patch for the `moss-image` node. The ref forward (`reference.py`) mirrors `library/moss/model.py` and runs as an `APPLY_MODEL` wrapper: the OpenMOSS forward never calls `DIFFUSION_MODEL` wrappers. Text: `AnimaCaptionToMossLayout` (Anima caption → layout JSON, moved out of the Tagger pack with its id kept), registered only when `anime_tools` imports |
| Anima Adapter Loader | https://github.com/sorryhyun/ComfyUI-Anima_lora-Adapter | Adapter / FeRA / Soft Tokens loaders; standalone at `~/ComfyUI-Anima_lora-Adapter`. **Read its `CLAUDE.md` for the `forward_hook`-not-override invariant** |
| Anima Tagger | inside the `anime_tools` repo (`comfyui/anima_tagger/` there) | symlinked as `comfyui-anima-tagger`; imports `anime_tools.tagger` and vendors nothing. The standalone `ComfyUI-Anima-Tagger` repo is **retired but still on disk** — don't edit it |

## In tree

Under `custom_nodes/`: `comfyui-anima-directedit/`, `comfyui-anima-register/`,
`comfyui-anima-trainer/` (daemon-backed one-shot trainer).

## Vendor trees

Several nodes carry a `_vendor/` subset of the live tree. **Regenerate with `make
vendor-sync` (`scripts/release/sync_vendor.py`), never `cp` by hand** — and re-run it
before every node publish.
