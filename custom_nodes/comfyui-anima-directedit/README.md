# Anima DirectEdit (ComfyUI)

A ComfyUI node that edits an image by adding tag(s) to its caption. Drop in an image, type `glasses` (or `double peace`, or `school_uniform`, …), and out comes the same image with that change applied — backgrounds, composition, and unchanged subject details preserved.

Built on **DirectEdit** (Yang & Ye, [arXiv:2605.02417](https://arxiv.org/abs/2605.02417v1)) — a training-free flow-inversion editor. Reference implementation: [Tr1stesse/DirectEdit](https://github.com/Tr1stesse/DirectEdit).

This node ports that idea to the Anima (DiT, flow-matching) model. ψ_src can come from an **AnimaTagger** socket (sibling [`comfyui-anima-tagger`](https://github.com/sorryhyun/anima_lora/tree/main/custom_nodes/comfyui-anima-tagger) package) or as a plain string via `prompt_src_override`.

## What the node does

```
IMAGE ─► AnimaTagger ─► ψ_src ──┐
       (or prompt_src_override) │
                                 ├─► ψ_tar = ψ_src + ", " + edit_text
              edit_text ─────────┘                │
                                                  ▼
                       ┌─────────────────────────────────────┐
                       │ DirectEdit                          │
                       │   1. invert(image, ψ_src)           │
                       │      ─► z_inv, Δz                   │
                       │   2. edit_forward(z_inv[0], Δz, ψ_tar) │
                       │      ─► z_edit                      │
                       └─────────────────────────────────────┘
                                                  │
                                                  ▼
                                          edited IMAGE
```

Two passes through the DiT:

1. **Inversion** (clean → noise): step backward through the same Euler ODE the generator runs forward, querying v_θ at each step's input. Record per-step residuals `Δz_i = z_inv[i+1] − z_inv[i]` — these are the "anchor" the paper uses to make reconstruction bit-exact.
2. **Editing** (noise → clean): standard generation loop, but every model call is queried at `z[i] + Δz[i]` instead of `z[i]`. The cross-attn prompt is the edit target ψ_tar; the residual Δz pins the trajectory to the source.

Result: regions the prompt *doesn't change* stay locked to the source; regions it *does* change get re-rendered under the target prompt.

## Install

Drop `custom_nodes/comfyui-anima-directedit/` into your ComfyUI `custom_nodes/`. For image-driven ψ_src, also install the sibling [`comfyui-anima-tagger`](https://github.com/sorryhyun/anima_lora/tree/main/custom_nodes/comfyui-anima-tagger) package (which provides `AnimaTaggerLoader` → the `ANIMA_TAGGER` socket this node consumes). Restart ComfyUI; the node appears as **Anima DirectEdit** in the `anima` category.

The package imports from the parent `anima_lora/` repo (it lives at `anima_lora/custom_nodes/comfyui-anima-directedit/`), so keep the directory inside the repo or symlink so that `Path(__file__).resolve().parents[2]` lands on the `anima_lora/` root.

## Inputs

| Input | Type | Notes |
|-------|------|-------|
| `model` | MODEL | Anima DiT (`UNETLoader` / `CheckpointLoaderSimple`). `LoraLoader` / `comfyui-hydralora` adapter loaders compose naturally upstream. |
| `clip` | CLIP | Anima text encoder (Qwen3 06B + T5xxl tokenizer) via `CLIPLoader`. |
| `vae` | VAE | Qwen Image VAE via `VAELoader`. |
| `image` | IMAGE | Source image. Auto-snapped to the closest `CONSTANT_TOKEN_BUCKETS` aspect ratio. |
| `edit_text` | STRING (multiline) | Tag(s) to add. `ψ_tar = ψ_src + ", " + edit_text`. Empty → reconstruction sanity check. |
| `negative_prompt` | STRING | CFG negative for the edit pass. Default `"worst quality"`. |
| `infer_steps` | INT | Both inversion and edit step count. Default 28. |
| `flow_shift` | FLOAT | Sigma-shift schedule. Default 1.0 (Anima preview3 standard). |
| `guidance_scale` | FLOAT | CFG for the edit (target) pass. Default 4.0. |
| `invert_guidance` | FLOAT | CFG during inversion. Default 1.0 (no CFG). |
| `seed` | INT | Reserved for future stochastic hooks; current pipeline is deterministic. |
| `tagger` | ANIMA_TAGGER (optional) | From `AnimaTaggerLoader` in `comfyui-anima-tagger`. Required unless `prompt_src_override` is set. |
| `prompt_src_override` | STRING (optional) | Replace the tagger's caption with your own ψ_src. Useful when the source is an Anima-generated image and you already know the original prompt. If set, the tagger socket is ignored. |

## Outputs

| Output | Type | Notes |
|--------|------|-------|
| `image` | IMAGE | Edited image. |
| `prompt_src` | STRING | What the tagger derived (or `prompt_src_override` if set). Useful for debugging — if the edit fails, check whether ψ_src actually describes the source. |
| `prompt_tar` | STRING | The full target caption fed to the edit pass. |

## Usage

```
[UNETLoader] ──► model ──┐
[CLIPLoader] ──► clip ──┤
[VAELoader] ──► vae ────┤
[Load Image] ──► image ─┤
                         ├─► [Anima DirectEdit] ──► [Save Image]
[AnimaTaggerLoader] ──► tagger ──┘
                                  edit_text: "double peace"
```

CLI equivalent (for reference):

```bash
make exp-test-directedit PROMPT='double peace'
```

## When to use `prompt_src_override`

The tagger is great for *external* images (web rips, screenshots) where you have no recorded prompt. For images you generated with Anima yourself, the original prompt is already a much better ψ_src than anything a tagger can recover. Paste it into `prompt_src_override` and the node skips the tagger entirely — and you don't need to install `comfyui-anima-tagger` at all.

## Caveats (v0)

- **No V-injection / no mask blending.** v1 of the underlying DirectEdit primitive (`library/inference/directedit.py`) is the paper's pure ΔZ-anchored edit at `t_inj=0, mask=None`. V-injection and background-lock are deferred to v2.
- **Inversion runs at `invert_guidance=1.0`** by default (no CFG). Raise only if you need the inverted noise to match a high-CFG generation seed.
- **Single-frame only.** Anima's qwen-image VAE is a video VAE that this pipeline drives at `T=1`. If you wire in a video-shaped IMAGE, only the first frame is processed.

## Files

| File | Role |
|------|------|
| `nodes.py` | The `AnimaDirectEdit` node — encode prompts → invert → edit_forward → decode. |
| `__init__.py` | Re-exports `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS`. |
| `pyproject.toml` | ComfyUI Registry metadata. |

## References

- **DirectEdit paper.** Yang & Ye, "Direct flow-inversion image editing for rectified flow models." [arXiv:2605.02417](https://arxiv.org/abs/2605.02417v1).
- **Reference implementation.** [Tr1stesse/DirectEdit](https://github.com/Tr1stesse/DirectEdit) — original PyTorch reference, source for the inversion/edit-forward step rules ported here.
- **AnimaTagger.** Sibling package [`comfyui-anima-tagger`](https://github.com/sorryhyun/anima_lora/tree/main/custom_nodes/comfyui-anima-tagger). Architecture: `docs/experimental/anima_tagger.md`. Integration rationale: `docs/experimental/directedit_editing_v3.md`.
- **Anima editing pipeline.** `scripts/edit.py` (CLI) and `library/inference/directedit.py` (primitives).
