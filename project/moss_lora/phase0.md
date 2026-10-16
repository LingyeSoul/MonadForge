# Phase 0 — port and reproduce (2026-10-16)

Code: `library/moss/{model,loader,generate}.py`, sidecar `scripts/moss/generate.py`,
CPU invariants `tests/test_moss_model.py`, gate script `project/moss_lora/src/parity.py`
(`make daemon-run ARGS="project/moss_lora/src/parity.py"`; writes
`output/moss/parity/parity.json` + the final latents).

## Gate: vs the reference `MOSSImagePipeline`

Base, 1024², 50 steps, CFG 3, uncond skip 22, seed 42 (fp32 CPU-generator noise), the
same prompt embeds and initial latents fed to both. Reference = the shipped
`pipeline_moss_image.py` with its own `MOSSImageTransformer`, `output_type='latent'`.

| check | prompt 1 (55 tok) | prompt 2 (49 tok) |
|---|---|---|
| TE: `loader.encode_prompt` vs `MOSSImagePipeline.encode_prompt` | bit-exact | bit-exact (also `""`, 10 tok) |
| `denoise(batched=False)` final latent vs reference | **bit-exact** (`torch.equal`) | **bit-exact** |
| `denoise(batched=True)` final latent, max / mean \|Δ\| | 5.19 / 0.058 | 5.12 / 0.256 |
| s/image: reference / unbatched / batched | 72.3 / 71.4 / 89.6 | 71.7 / 72.5 / 91.0 |
| denoise peak, unbatched / batched | 10.06 / 10.61 GiB | same |

**Batched cond+uncond is not the default, and should not be.** It is ~25 % *slower*:
cond and uncond have different text lengths (55 vs 10 tokens), so the joint batch needs a
key mask, and a boolean `attn_mask` pushes SDPA off the flash kernel onto a slower one.
The two separate unmasked forwards stay on flash. It is also not "within bf16 noise" at
the end of the trajectory: the per-forward difference is kernel rounding (the CPU tiny-model
test agrees to 1e-5), but over 50 steps that rounding grows to max |Δ| ≈ 5 on the final latent,
mean 0.06 to 0.26. Same distribution (latent |mean| 0.715 vs 0.716), different sample. `--batched`
stays an opt-in flag. A fused path would need a variable-length (unpadded) flash call to
win anything.

## Space reproductions (the two the day-one run lost to the decode OOM)

Space `OpenMOSS-Team/MOSS-Image`, gallery `assets/images/{004,017}/parameters.json`
(Base, Euler, shift 5, uncond skip 22; the Space seeds a CPU generator, as ours does).
The Space's `pipeline_moss_image.py` differs from the model repo's only by a CPU prompt
cache.

| entry | size | prompt tokens | CFG | denoise | peak (denoise) | decode peak (DiT off) |
|---|---|---|---|---|---|---|
| 004 Blue Archive ensemble | 1776×1184 | 1937 | 5 | RESULT_004 | | |
| 017 beach panorama | 2880×720 | 2065 | 3 | RESULT_017 | | |

Outputs: `/media/sorryhyun/data/moss_image/samples/space_repro/`.

## Notes

- With the TE layer-paging budget now sized from free VRAM (`loader._offload_by_layer`,
  25 of 32 decoder layers resident), encoding the two ~2k-token JSON prompts plus two
  short negatives took one TE load.
- `generate.py` caches TE outputs under `<out_dir>/_te_cache/` keyed by text, so a rerun
  of the same prompts skips the 17 GB text encoder.
