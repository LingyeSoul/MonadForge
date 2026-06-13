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

## `text_jacobian.py` — does the mod head have a sensitivity GAD could repair?

A cheap, generation-free probe (no text encoder, reuses the distill forwards) for
the one question that decides whether GAD-for-mod-guidance is worth pursuing:
**does the distilled `pooled_text_proj` reproduce the teacher's *local response*
to a text change, not just its pointwise output?** That first-order response is
exactly what inference steering (`emb + w·delta`) rides on.

For held-out (latent, σ, noise) it perturbs text A→B by a factor `h` and compares
output deltas across the two pathways:

- teacher: `Δv` from perturbing the cross-attn input (`skip_pooled_text_proj=True`)
- student: `Δv` from perturbing the pooled input (`pooled_text_override`, crossattn pinned at uncond)

```bash
uv run python -m bench.mod_guidance.text_jacobian \
    --pooled_text_proj output/ckpt/pooled_text_proj.safetensors \
    --n_pairs 96 --sigmas 0.1 0.4 0.7 0.9 --h 1.0
# GAD-faithful local Jacobian: --h 0.1
```

Reading: `cos(ΔS, ΔT)` per σ is the verdict. **≈1** → the MSE-only head already
matches the teacher's text geometry, GAD has nothing to fix (confirms the
`docs/experimental/gad.md` skepticism). **Well below 1**, especially at high σ
where modulation dominates → that's the deficiency GAD targets; train a GAD head
and re-run to score the gain (Δcos). `ratio = ‖ΔS‖/‖ΔT‖` is cross-pathway so
won't hit 1 exactly — read it for collapse (`≈0` = text-blind) / blowout. Set
`--validation_seed` to the distill run's so the holdout is genuine.
