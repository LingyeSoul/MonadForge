# MOSS LoRA comparison grids

Reference grids for judging changes to the LoRA recipe, all on the inference setup we
keep using: ER-SDE + Spectrum, 50 steps (flex_window 0.5), Base DiT, CFG 3, uncond skip
22, 1024², prompts through `--prompt_format layout_json`. Write-up: `docs/experimental/moss_lora.md`.

Each grid is 2 columns (LoRA ×0, ×1) × 2 rows (seeds 1, 2), 512 px tiles. Prompts
(`project/moss_lora/src/lora_ab.py` `PROMPTS`): p0 Miku tag bag (in the set), p1 Akita
Neru tag bag (not in the set), p2 a natural-language sentence with the artist tag.

| File | LoRA |
|---|---|
| `caststation_layout_ersde_spec50_p{0,1,2}.webp` | `caststation_layout_r16`: channel (caststation), 30 images, layout-JSON captions, 20 epochs, rank 16, lr 1e-4 |

Full-resolution images and latents (not in git): `output/moss/lora/ab_caststation_layout_r16_50st/`.
A new recipe gets its own row here, rendered with `lora_ab.py` at its defaults.
