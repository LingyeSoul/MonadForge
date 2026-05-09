# Anima Tagger (ComfyUI)

Multi-label image tagger trained on the Anima caption distribution. Drop in an image, get back a comma-separated tag string in exactly Anima's training-time T5 format — `rating, count, characters, copyrights, @artists, generals`, underscores replaced by spaces.

Two nodes in the `anima` category:

| Node | Inputs | Outputs | Use |
|------|--------|---------|-----|
| **Anima Tagger Loader** | `tagger_dir` (STRING) | `tagger` (ANIMA_TAGGER) | Load the checkpoint once; ComfyUI memoizes the output so the tagger persists across graph runs. |
| **Anima Tagger Caption** | `tagger` (ANIMA_TAGGER), `image` (IMAGE) | `caption` (STRING) | Tag an image. Drop the STRING into any text input. |

## What it's for

- **DirectEdit ψ_src.** The `ANIMA_TAGGER` socket plugs straight into [`comfyui-anima-directedit`](https://github.com/sorryhyun/anima_lora/tree/main/custom_nodes/comfyui-anima-directedit). DirectEdit's edit leverage collapses when ψ_src is structurally far from Anima's training-time embedding manifold — Anima Tagger fixes that vs. a generic WD-tagger.
- **Caption pre-fill for LoRA training.** Tag your dataset, paste into `.txt` sidecars.
- **Prompt scaffolding.** Wire the caption STRING into `CLIPTextEncode` to seed a generation from an existing image's tag set.

## Install

Drop `custom_nodes/comfyui-anima-tagger/` into your ComfyUI `custom_nodes/`, restart ComfyUI. The nodes appear under the `anima` category.

The node works in two install shapes:

1. **Inside the anima_lora repo** (dev / monorepo). It imports the live `library.captioning.anima_tagger`, so edits in the parent repo are picked up immediately.
2. **Standalone** (just this directory dropped into a vanilla ComfyUI `custom_nodes/`). It falls back to a bundled inference subset under `_vendor/` — no need to clone the parent repo or run `uv sync`. Pip deps are listed in `pyproject.toml` (ComfyUI ships everything except possibly `einops` / `timm` / `pyyaml`, all small).

The PE-Core-L14-336 vision encoder checkpoint (~1 GB) is auto-fetched from `facebook/PE-Core-L14-336` on first use into the `pe_ckpt` path on the loader node.

### For maintainers — keeping the vendor copy fresh

The `_vendor/` tree is generated from the live anima_lora source. Regenerate it before bumping the node version:

```bash
python scripts/sync_vendor.py     # from the anima_lora repo root (refreshes both tagger + directedit vendor trees)
```

## Checkpoint layout

`tagger_dir` must contain (produced by `python -m scripts.anima_tagger.cli` in the parent repo):

```
<tagger_dir>/
  config.json              # model config + training metadata
  model.safetensors        # AnimaTaggerHead state dict
  pe_lora.safetensors      # PE-LoRA delta on PE-Core trailing blocks (optional)
  thresholds.safetensors   # per-tag F1-optimal thresholds
  vocab.json               # tag list with category + median_pos info
  rules.yaml               # caption-normalization rules snapshot
```

Default `tagger_dir` is `models/captioners/anima-tagger-v1` (relative to the `anima_lora/` repo root). Absolute paths used as-is.

## Usage

### Caption an image

```
[Load Image] ──┐
               ├─► [Anima Tagger Caption] ──► [Save Text File]
[Anima Tagger Loader] ──┘
       tagger_dir: models/captioners/anima-tagger-v1
```

### Drive a normal text-to-image generation from an existing image's tags

```
[Load Image] ──┐
               ├─► [Anima Tagger Caption] ──► caption ──► [CLIPTextEncode] ──► [KSampler] ──► …
[Anima Tagger Loader] ──┘
```

### Plug into DirectEdit (cross-package)

```
[Anima Tagger Loader] ──► tagger ──┐
                                    │
                                    ▼
[Load Image] ─────────────────► [Anima DirectEdit] ──► edited image
                                    ▲
                  edit_text: "double peace"
```

DirectEdit owns its own ψ_tar logic and only needs the `ANIMA_TAGGER` socket — see [`comfyui-anima-directedit`](https://github.com/sorryhyun/anima_lora/tree/main/custom_nodes/comfyui-anima-directedit).

## Files

| File | Role |
|------|------|
| `nodes.py` | `AnimaTaggerLoader` + `AnimaTaggerCaption`. |
| `__init__.py` | Re-exports `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS`. |
| `pyproject.toml` | ComfyUI Registry metadata. |

## References

- **AnimaTagger architecture.** `docs/experimental/anima_tagger.md` in the parent repo.
- **DirectEdit integration.** `docs/experimental/directedit_editing_v3.md` (why ψ_src manifold-fit matters).
- **Trainer.** `python -m scripts.anima_tagger.cli` in the parent repo (`scripts/anima_tagger/cli.py`).
