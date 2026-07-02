# qwen_vae_2d — 2D-folded VAE for single-image encode/decode

Ports kohya-ss sd-scripts' `--qwen_image_vae_2d` idea to Anima's vendored
`AutoencoderKLQwenImage`.

## Idea

The VAE is built from `QwenImageCausalConv3d` (3D causal Conv3d). For a single
image (`T=1`) the causal temporal padding is **zero-pad** with the real frame in
the last temporal slot, so only the **last** temporal tap sees data. The exact
2D equivalent of each conv is therefore `weight[:, :, -1, :, :]` run as a plain
`nn.Conv2d` on the squeezed `(B,C,H,W)` frame (bias unchanged). The temporal
feature cache is a no-op for single frames and is disabled.

`AutoencoderKLQwenImage.convert_to_2d()` (or `load_vae(..., vae_2d=True)`)
rewrites all 61 `QwenImageCausalConv3d` to `Folded2DConv` in place and disables
the temporal feature cache; the 5D `(B,C,1,H,W)` plumbing around them is
preserved (squeeze→Conv2d→unsqueeze). After folding the VAE is **image-only**
(asserts `T=1`).

**Shipped:** ON by default in latent caching
(`scripts/preprocess/cache_latents.py` → `make preprocess-vae`); opt out with
`ARGS="--no_vae_2d"`. Invariants pinned in `tests/test_qwen_vae_2d.py`.

## Equivalence

- **Per-layer, fp64: bit-exact** (`max|Δ| = 0.00e+00` for the real `(3,3,3)`
  conv — see the inline check in the conversation / `run_bench.py`).
- **Whole network, fp32 @256px:** encode `max|Δ| ≈ 7e-4` (rel `3e-4`), decode
  `≈ 1.4e-2`. This is fp32 rounding compounding over 61 layers + cuDNN picking
  different Conv3d vs Conv2d algorithms — **not** an approximation in the fold.
- **bf16 (production/caching dtype):** encode `max|Δ| ≈ 7.8e-3`, i.e. below
  bf16 latent quantization. Latents are cached in bf16, so this is the number
  that matters and it is negligible.

## Results (RTX 5070 Ti 16GB, bf16, batch 1, real 1024 image)

3D baseline uses the production `spatial_chunk_size=64`; the 2D fold runs
unchunked.

| res   | encode 3D→2D       | speed | enc peak 3D→2D     | mem  | decode 3D→2D       | speed | dec peak 3D→2D     | mem  |
|-------|--------------------|-------|--------------------|------|--------------------|-------|--------------------|------|
| 512   | 37.2 → 15.5 ms     | 2.40× | 2966 → 1920 MB     | 0.65×| 63.3 → 26.6 ms     | 2.38× | 3017 → 2021 MB     | 0.67×|
| 768   | 94.2 → 40.7 ms     | 2.32× | 6268 → 4196 MB     | 0.67×| 154.8 → 66.8 ms    | 2.32× | 6382 → 4423 MB     | 0.69×|
| 1024  | 164.3 → 78.1 ms    | 2.10× | 10885 → 7370 MB    | 0.68×| 271.5 → 124.9 ms   | 2.17× | 11087 → 7773 MB    | 0.70×|

**Takeaway:** ~2.1–2.4× faster encode *and* decode, ~30–35% lower peak memory,
numerically equivalent within bf16 noise. The win lands on the latent-caching
preprocess pass (`scripts/preprocess/cache_latents.py` → `library/preprocess/`)
and on inference/preview decode — **not** on the training step (VAE is unloaded
before DiT training).

Memory note: peak is ~0.68× here because the 3D baseline is *already* chunked
(chunk=64). vs an unchunked 3D path the reduction is far larger — unchunked 3D
OOMs at batch 4 / 1024 on 16GB where the folded 2D path uses less. Per kohya,
with the 2D path peak shifts to full-res activations / mid-block attention, so
`spatial_chunk_size` helps the 2D path less.

## Run

```bash
uv run python bench/qwen_vae_2d/run_bench.py --res 1024 --batch 1 \
    --image post_image_dataset/resized/<artist>/<id>.png
```
