# bench/mod_guidance — mod-guidance channel attribution

Image-space probe of how a prompt edit reaches the image through the mod-guidance
pooled-text path. Replaces the geometry-only analysis behind
`docs/findings/mod_guidance_quality_tag_axis.md` (which was demoted — see its
STATUS box: the "quality axis" was a content-magnitude axis, and its
"degrades quality" claim was never imaged).

## The premise

A tag edit is not one intervention. It enters the DiT through **two separable
inputs**:

- **pooled / mod channel** — `crossattn_emb.max(1) → pooled_text_proj → AdaLN`
- **cross-attn channel** — the full `crossattn_emb` sequence → cross-attention

They are separable at the forward via `pooled_text_override` (models.py:1643), so
we can run cross-attn from prompt A while feeding the pooled vector of prompt B.
That decoupling is what every experiment here exploits.

## Experiments (`--experiment`)

| name | question | how |
|---|---|---|
| `swap` | how much of a tag's *image* effect flows through pooled vs cross-attn, and do the channels reinforce or cancel? | render the 2×2 of {cross=base\|tag}×{pool=base\|tag}; split into Δ_cross, Δ_pool; report `pool_share`, `cos(cross,pool)`, additivity residual |
| `order` | does tag *order* matter, through cross-attn alone? | permute the comma-tags in cross-attn while **pinning** `pooled=pool(canon)` (pooling is post-encoder, so reorder is *not* pooled-invariant — pinning makes it a true isolator); compare to the same-prompt seed floor |
| `intensity` | does a *hard push* on the mod channel drift the image worse (DC-blowout)? | sweep the steering weight `w`; measure off-`w0` PE movement + pixel-std collapse / tone shift |

Metrics live in **latent space** (the model's own output; spatially sensitive)
and **PE-Core space** (the repo's CMMD features; note PE-pooled is somewhat
pose-blind — a guide, not the verdict). **Read the saved grids.**

## Run

```bash
# full run (real dense captions, both quality + content tags, compiled)
uv run python bench/mod_guidance/channel_attribution.py \
    --pooled_text_proj output/ckpt/pooled_text_proj-0530.safetensors \
    --experiment all --dataset_samples 6 \
    --tags "score_9,masterpiece,holding a sword" --seeds 0,1 --compile --label full

# smoke
uv run python bench/mod_guidance/channel_attribution.py \
    --pooled_text_proj output/ckpt/pooled_text_proj-0530.safetensors \
    --experiment swap --prompts "1girl, solo, outdoors" --tags "score_9" \
    --infer_steps 12 --label smoke
```

Key flags: `--dataset_samples N` (sample real captions from `image_dataset/` —
the dense regime where the geometry numbers collapse), `--prompts` (explicit,
wins), `--tags`, `--seeds`, `--compile` (amortises the sweep), `--grid_thumb`
(grid resolution), `--w_points` / `--steer_pos` (intensity sweep).

Output: `results/<ts>[-label]/` with `result.json` (standard bench envelope),
`rows.csv` (per-pair metrics), and one labelled grid PNG per render group.

## Reading the numbers

- `pool_share ≈ 0` → the mod channel barely carries the edit; the topic is
  low-value as a steering lever. `pool_share` shrinking on dense vs neutral
  prompts is the max-pool-saturation collapse, now in image space.
- `cos(cross,pool) > 0` → channels reinforce (the geometry doc's "double-drive");
  `< 0` → conflict/cancel; `≈ 0` → orthogonal.
- high `additivity_resid` → the channels are *not* a linear superposition; the
  clean double-drive story is too simple.
- `order_dist / seed_floor ≪ 1` → cross-attn is only weakly order-sensitive
  (reordering moves the image less than a seed change).
- large `pixel_std` drop at high `w` → DC-blowout is real in pixels.
