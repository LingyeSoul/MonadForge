# DirectEdit + img2emb — image editing proposal

Source paper: arXiv 2605.02417v1 (Yang & Ye, *DirectEdit: Step-Level Accurate
Inversion for Flow-Based Image Editing*, 2026). Sister proposal: see
[`img2emb_plan.md`](./img2emb_plan.md) for the inversion model this document
relies on.

## Goal

Add a training-free **image editing** capability to Anima: take an existing
image + a desired change, produce an edited image that preserves non-edited
regions. Works on three input scenarios — Anima-generated images
(with/without recorded prompt) and external images (uploads, photos, art
from elsewhere).

Out of scope: re-composition tasks ("same character, different scene"). That
remains IP-Adapter's job. Editing is structurally tied to the source image's
spatial layout by design.

## Why now

Two pieces fell into place:

1. **DirectEdit** kills the inversion-fidelity problem that has blocked
   training-free editing on flow models. Step-level reconstruction MSE
   drops 5 orders of magnitude (231 → 6e-4) at zero extra NFEs vs prior
   flow-inversion methods (FireFlow, FTEdit, DNAEdit, RFEdit). The single
   technical idea is: don't rectify the inversion path, just align the
   reconstruction path to it bit-exactly via a recorded step residual
   `ΔZ_t = Z_{t+1}^inv − Z_t^inv`.

2. **img2emb revival** ([`img2emb_plan.md`](./img2emb_plan.md)) produces
   exactly the missing component DirectEdit needs but doesn't propose: a
   real-time map from image to a Anima-natural post-T5 embedding. Without
   img2emb the only options for external images are captioning (lossy,
   distributionally off) or null-prompt (mathematically reconstructs but
   collapses edit leverage).

Each is useful on its own; together they cover all three input scenarios
end-to-end.

## Three scenarios

| # | Input | What we have | Bottleneck | Proposed path |
|---|---|---|---|---|
| 1 | External image | Pixels only | No prompt, no noise | img2emb → ψ_src; DirectEdit invert; edit |
| 2 | Generated image | Pixels + (usually) ComfyUI metadata | Maybe no recorded prompt | If prompt present: same as (3). Else: same as (1). |
| 3 | Generated image + prompt | Pixels + ψ_src | None | Skip inversion entirely — sidecar replay (see below) |

Case (3) is by far the cheapest and is the design target for
**ComfyUI-integrated** editing of self-generated images. Case (1) is the
most general and is the design target for the full editing pipeline.

## Component 1 — DirectEdit primitive

Mechanics, in the order they fire during a single edit pass:

1. **Inversion** with source prompt ψ_src:
   ```
   Z_T = VAE_encode(I_src)
   for t = T-1 .. 0:
       Z_t^inv  = Z_{t+1}^inv − (σ_{t+1} − σ_t) · v_θ(Z_{t+1}^inv, ψ_src)
       ΔZ_t     = Z_{t+1}^inv − Z_t^inv     # the only "non-standard" line
   ```
2. **Editing forward**, two branches in lockstep:
   ```
   Z_0^src = Z_0^tar = Z_0^inv
   for t = 0 .. T-1:
       Ẑ_t^src = Z_t^src + ΔZ_t      # anchor to inversion path
       Ẑ_t^tar = Z_t^tar + ΔZ_t      # same anchor, different prompt
       v̂_t^tar = v_θ(Ẑ_t^tar, ψ_tar) with self-attn V from src injected for t < t_inj
       Z_{t+1}^src = Z_t^src + (σ_{t+1} − σ_t) · v_θ(Ẑ_t^src, ψ_src)
       Z_{t+1}^tar = Z_t^tar + (σ_{t+1} − σ_t) · v̂_t^tar
       Z_{t+1}^tar = Z_{t+1}^tar ⊙ (1 − M) + Z_{t+1}^src ⊙ M   # background lock
   ```

The four orthogonal levers, each addressing a different failure mode:

| Lever | Job | Failure mode it prevents |
|---|---|---|
| `ΔZ_t` residual | Trajectory anchor | Drift / over-edit / under-edit from accumulated Euler error |
| Cross-attn to ψ_tar | The actual edit | n/a — this is what produces the change |
| Self-attn V-injection (first `t_inj` steps) | Source detail preservation | Texture/identity loss inside the edit region |
| Mask blending | Background lock | Unintended changes outside the edit region |

V-injection scope and `t_inj` schedule follow the paper (Section 3.3,
Fig. 7): more steps → output looks more like source; default `t_inj = 3`
out of 30 steps balances fidelity vs editability across most edit types.

**Anima-specific note.** Anima uses cross-attention (not joint-stream
MM-DiT like SD3.5/FLUX). The paper reports injection placements per
backbone (SD3.5: every block but the last; FLUX: single blocks only).
Anima's block layout matches FLUX's single-stream stage more than SD3.5's
joint stream — single-block injection is the conservative starting choice;
needs a sweep against PIE-Bench to confirm.

## Component 2 — img2emb as the inversion-prompt provider

DirectEdit requires a ψ_src. Where it comes from determines edit quality:

| Source | Reconstruction | Edit quality | Cost |
|---|---|---|---|
| Recorded ψ_src (case 3) | Exact | Best | 0 |
| ComfyUI PNG metadata (case 2) | Exact | Best | 0 |
| Captioner → T5 (case 1, fallback) | Exact (residual trick) | Degrades on un-captioned features | Captioner forward |
| Null prompt | Exact | Near-useless — residual dominates, prompt has no leverage | 0 |
| **img2emb prediction** | **Exact** | **Anima-natural — features stay encoded in ψ_src even when un-verbalized** | **Resampler forward** |

The last row is what unlocks general external-image editing. img2emb's
output lives inside Anima's training-time embedding distribution by
construction (FM-loss supervision through the frozen DiT — see
[`img2emb_plan.md`](./img2emb_plan.md) §"What changes vs the archived
design"), so cross-attention is operating in-domain rather than on
captioner-shaped distribution-shifted text.

The `img2emb_plan.md` revival is independently motivated; this proposal
just rides on it for the editing use case. If img2emb is delayed or
abandoned, DirectEdit still works with a captioner-based fallback at
reduced edit quality on un-captioned features.

## Component 3 — embedding-space editing UI

With img2emb in place, the editing interface no longer requires the user
to verbalize *what's in the image*. Only the **delta** needs words:

```python
ψ_src   = img2emb(I)                      # what the image "is" in Anima's terms
ψ_delta = T5("a boy") − T5("a girl")      # natural language for the change only
ψ_tar   = ψ_src + α · ψ_delta             # arithmetic in post-T5 space
# DirectEdit(ψ_src, ψ_tar) → edited image
```

This sidesteps the captioner-fidelity problem entirely. The image's full
content stays encoded in ψ_src whether or not any human ever wrote it
down; the cross-attn delta only needs to specify the change.

Design questions for the embedding-arithmetic UI:

- **Delta scale α.** Unit norm of `ψ_delta` varies wildly between concept
  pairs. Normalize? Per-token? Use cosine-only direction with magnitude
  from a learned scale?
- **Direction validity.** Not every direction in post-T5 space is a valid
  editing direction. Need a probe/regularizer to keep `ψ_tar` on the same
  manifold ψ_src lives on (otherwise cross-attn produces noise).
- **Compositional edits.** `ψ_delta_1 + ψ_delta_2` for "change boy + add
  glasses" — does this stack linearly or interfere?

These are exploratory; v1 of the editing tool can ship with full-prompt
ψ_tar (user types the target description) and add delta-arithmetic later.

## ComfyUI integration shape

Two distinct nodes / tabs, matching the two cost regimes:

### Fast Edit (case 3: self-generated images)

The user-generates-then-edits flow doesn't need DirectEdit at all — the
noise is already known. The simpler P2P-style sidecar pattern covers it:

```
Generate node (opt-in flag "save edit sidecar"):
  outputs: image.png, image.editstate.{safetensors|zarr}
  sidecar contains:
    - Z_0 (initial noise)                           ~1 MB
    - per-step Z_t                                  ~30 MB  [optional]
    - per-block self-attn V tensors, first ~half    ~1–4 GB [optional]
    - model_state_hash (DiT + LoRA stack + scheduler)

Edit node:
  inputs: (image, sidecar, ψ_tar, optional mask, t_inj)
  validates: model_state_hash matches current state — refuse otherwise
  pass: 1 forward, source-anchored via cached V injection + mask blend
  cost: T NFE, same as a generation
```

Three sidecar tiers (cheap / medium / full) let the user trade disk for
edit quality — `Z_0` alone is too thin for proper anchoring, V-cache is
the actual lever.

### Import Edit (case 1: external images)

```
Import Edit node:
  inputs: (image, ψ_tar | embedding_delta, optional mask, t_inj)
  pipeline:
    1. img2emb(I) → ψ_src
    2. DirectEdit invert → {Z_0^inv, ΔZ_t, optional V_t}
    3. DirectEdit edit branch with (ψ_src, ψ_tar) → I_tar
  cost: T NFE invert + T NFE edit ≈ 2× generation
```

The ψ_tar input accepts either a natural-language target prompt (passes
through T5 normally) or a delta on top of ψ_src (the embedding-arithmetic
UI from Component 3, when shipped).

## Existing code that already covers most of this

| What we need | What already exists |
|---|---|
| Flow inversion primitive | `archive/inversion/invert_embedding.py` (per-image gradient descent) and `archive/inversion/invert_reference.py` — same primitive, less efficient form |
| Image → embedding | `archive/img2emb/{preprocess,pretrain,finetune,infer}.py` (legacy direct-alignment design); revival plan in [`img2emb_plan.md`](./img2emb_plan.md) |
| Vision encoders + resampler + buckets | `library/vision/{encoder,resampler,buckets,data}.py` — already extracted from the archive for IP-Adapter; reuse directly |
| DiT cross-attention hooks | `networks/methods/ip_adapter.py`, `networks/methods/easycontrol.py` — both monkey-patch attention; same pattern works for V-injection |
| Mask blending | `library/datasets/` mask handling + the masking pipeline (`make mask*`) for source-image masks; SAM3 already integrated |
| Flow sampler / scheduler | `library/inference/`, `library/runtime/noise.py` |
| Cached `(I, ψ_T5)` pairs for img2emb training | `post_image_dataset/lora/{stem}_anima_te.safetensors` already on disk |

What's actually new code:

- DirectEdit invert+edit loop wrapper around the existing sampler
  (`scripts/edit.py` + `library/inference/edit.py` or similar; ~few
  hundred LoC).
- Sidecar save/load (additive on top of `inference.py`'s generation loop —
  another forward hook on each block to capture self-attn V).
- ComfyUI nodes (separate repo or `custom_nodes/` if we go that route).
- img2emb revival itself — see [`img2emb_plan.md`](./img2emb_plan.md).

## Open design questions

1. **V-injection scope on Anima.** SD3.5 vs FLUX have different
   recommendations. Anima's block layout needs its own sweep — paper
   doesn't cover us. Bench against a small PIE-Bench-style mini-set.
2. **Mask source.** SAM3 (already integrated via `make mask`) can produce
   the edit-region mask. Do we wire this in automatically (with a target
   description → SAM segmentation prompt), or require user-provided
   masks? MLLM-based mask generation (paper's approach) needs a Qwen2-VL
   integration we don't have yet.
3. **Sidecar V-cache size.** At 1024² × 30 steps × 28 blocks, full self-attn
   V at fp16 is ~1–4 GB. That's brutal as a per-image sidecar. First-half
   only? Quantized? Or only save when user explicitly opts in per generate?
4. **Captioner fallback.** If img2emb isn't ready in v1, what's the
   captioner? Qwen2-VL is the natural choice (already in the broader
   ecosystem) but we don't ship it. BLIP-2 is smaller but less aligned to
   anime/manga content.
5. **Spectrum compatibility.** Spectrum (Chebyshev step caching) skips
   most blocks on cached steps. The V-injection needs source-pass V at
   every step for editing — likely incompatible with Spectrum on the
   source pass. Edit pass might still benefit. Quick code review needed.
6. **Eval signal.** PIE-Bench is the paper's benchmark; PSNR/CLIP-Sim are
   uninformative on Anima content (cf. `project_fm_val_loss_uninformative`
   memory). Need an Anima-specific edit-quality bench — likely pairwise
   human or MLLM-judge on a curated 50-image set.

## Phasing

### Phase 0 — Read and verify (no code)

- [ ] Read `archive/img2emb/` and confirm `img2emb_plan.md`'s assessment
      of what was tried and what to drop.
- [ ] Skim Yang & Ye's appendix for the per-architecture injection
      placement details (Anima vs SD3.5 vs FLUX layer mapping).
- [ ] Check Spectrum's hook surface vs our V-injection needs — note any
      incompatibility upfront.

### Phase 1 — DirectEdit primitive with caption fallback

- [ ] `library/inference/edit.py` — invert+edit loop, V-injection hook,
      mask blending. Pure DirectEdit, no img2emb yet.
- [ ] `scripts/edit.py` — CLI wrapper: `--image`, `--prompt_src`,
      `--prompt_tar`, optional `--mask`, `--t_inj`.
- [ ] `make exp-edit` and `make exp-test-edit` targets.
- [ ] Mini-bench: 20 self-generated images with known prompts, swap one
      attribute via ψ_tar, eyeball quality. Should match paper's reconstruction
      claims (step-MSE near VAE floor).

This is shippable on its own for case (3) editing — no img2emb needed.

### Phase 2 — img2emb revival

Defer to [`img2emb_plan.md`](./img2emb_plan.md). Independent project.

### Phase 3 — full pipeline

- [ ] Wire img2emb output as DirectEdit's ψ_src for case (1).
- [ ] Embedding-space delta UI (`scripts/edit.py --embedding_delta_src
      ... --embedding_delta_tar ...` or similar).
- [ ] Compare against captioner-based fallback on a small external-image
      set; if img2emb wins, deprecate captioner path. If close, keep both
      as user-selectable.

### Phase 4 — ComfyUI integration

- [ ] Sidecar save in our existing generation flow (additive flag on
      `inference.py`).
- [ ] Two ComfyUI nodes: Fast Edit (sidecar) and Import Edit (full
      pipeline). Probably in `custom_nodes/comfyui-anima-edit/` (new
      package) or extending `custom_nodes/comfyui-hydralora/`.
- [ ] Model-state hash validation on sidecar load.

## Decision points before starting

1. **Phase 1 ship-on-its-own?** If yes, captioner-fallback DirectEdit
   becomes a real feature even if img2emb stalls. Recommended: yes.
2. **Sidecar default opt-in vs opt-out?** Disk cost is significant for
   full V-cache. Recommend: opt-in flag, with a "lite" default that
   saves only Z_0 + per-step Z_t (~30 MB total).
3. **ComfyUI node placement.** New `comfyui-anima-edit` repo, or extend
   `comfyui-hydralora`? The hydralora node already centralizes adapter
   loading; editing is a different concern and probably wants its own
   node. Recommend: new repo.

## Why this matters

- Closes a feature gap with closed-source tools that already do
  invert-and-edit (Adobe Firefly, Midjourney's editor, etc.).
- Validates the "Anima as a flexible image-manipulation backbone" framing
  beyond pure generation.
- Reuses ~80% existing infrastructure (vision encoders, resampler, mask
  pipeline, attention monkey-patching, embedding inversion). Net new code
  is the DirectEdit loop wrapper + sidecar plumbing + ComfyUI nodes.
- Composes cleanly with the embedding-arithmetic direction we explored in
  `archive/inversion/` — gives that work a real downstream consumer.

## Non-goals

- Replacing IP-Adapter. Different task (re-composition vs editing).
- Replacing inpainting via outpainting models. Mask-blend keeps us inside
  the source distribution; out-of-distribution inpainting still needs a
  dedicated model.
- General-purpose flow inversion as a research artifact. We're consuming
  DirectEdit, not extending it.
