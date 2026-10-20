# moss_lora — MOSS-Image on a 16 GB card

**Not Anima.** MOSS-Image (OpenMOSS, Apache-2.0, released 2026-10-11) is a 5B single-stream
DiT trained from scratch on a frozen Qwen3.5-9B text encoder and the frozen FLUX.2 VAE. The
line stacks into this repo the way `library/qwen21/` does: its own `library/moss/`,
`scripts/moss/`, with only the runtime plumbing shared. Repos:
`OpenMOSS-Team/MOSS-Image-Base`, `OpenMOSS-Team/MOSS-Image-SFT` (SFT = fine-tune of Base,
same architecture and sampler defaults). Technical report: `MOSS_Image_Technical_Report.pdf`
(35 pp., in the HF Space `assets/`).

Goal: the two things Anima already has, ported: **Spectrum inference** and a **light LoRA
with activation checkpointing**, both on the 5070 Ti (16 GB).

## Why (first look, 2026-10-16)

One seed, five 1024² prompts, Base and SFT, 50 steps, CFG 3, uncond skip 22:

- Quoted CJK text in the main subject came out correct in both models: "待ってくれ！"
  written vertically in a manga bubble, three handwritten Chinese lines, a "오늘도 수고했어" neon
  sign. Small background text (signage behind glass) is still pseudo-glyphs.
- A raw Danbooru tag bag (`1girl, solo, akita neru, …`) rendered on-prompt, even though the report names no booru source
  and says the captions are gemma4 natural language. Whether the model knows the character
  is untested: every trait was also spelled out in the prompt.
- SFT renders more finished (lighting, backgrounds, saturation) and pulls toward a glossy
  gacha look; Base is rawer and more varied (flat cel shading, scan-like manga). Base looks
  like the better LoRA substrate, SFT the better out-of-the-box sampler. That is an
  impression from five prompts, not a measurement.

Samples: `/media/sorryhyun/data/moss_image/samples/` (`{prompt}_{Base|SFT}_s42.png`).

## Measured so far

| | value |
|---|---|
| Weights on disk | Base 27.2 GB (TE 16.8 + DiT 10.2 + VAE 0.25); SFT adds only its DiT (10.2), TE/VAE/tokenizer are byte-identical (same LFS oids) and symlinked |
| Location | `/media/sorryhyun/data/moss_image/MOSS-Image-{Base,SFT}` (nvme0n1p4) |
| DiT resident, bf16 | 9.7 GiB |
| 1024², 50 steps, CFG 3 (2 forwards/step) | 71–73 s/image, peak 12.6 GiB, both models |
| TE encode, Qwen3.5-9B layer-offloaded | 86 s for 9 prompts (cold, includes load); 10–2065 tokens |
| 1776×1184 (8214 img tokens) + 1937-token JSON prompt | denoise fits; **fp32 VAE decode OOMs** with the DiT still resident |

Prototype script (outside the repo, session scratch):
`run_moss.py` — TE phase → free → per-model DiT, `--example <Space parameters.json>` to
replay a gallery entry. It is the reference the Phase 0 port must reproduce.

## Architecture facts the code has to respect

From `pipeline_moss_image.py` (shipped in the repo, 440 lines) and report App. A:

- **Latents are 4D `(B, 128, H/16, W/16)`**: FLUX.2 VAE 32 ch at /8, packed 2×2 into
  128 ch, 16 px stride. Sides must be multiples of 16. No 5D, no frame axis.
- **Latent normalisation lives in the VAE's BatchNorm buffers**: decode is
  `z·sqrt(var+eps) + mean` then un-patch `(b, c/4, 2, 2, h, w)` → `(b, c/4, 2h, 2w)`.
  Encoding for the cache is the exact inverse; derive it from the decode, don't guess.
- **Text is variable-length and masked, never padded.** The DiT loops per sample and
  slices each context to its true length (the GDN layers in the TE also misbehave on
  padding, so the TE runs one prompt per forward). This is the opposite of Anima's
  max-pad rule and the same as qwen21's.
- **TE input** = Qwen chat template, `add_generation_prompt=True, enable_thinking=True`,
  `last_hidden_state` (4096), up to 4096 tokens. The model was trained on exactly this
  format; changing it means re-caching.
- **Single-stream**: text (after a linear 4096→3072 and a 2-block text refiner) is
  concatenated after the image tokens, and both go through all 42 blocks. Attention cost
  grows as `(N_img + N_txt)²`, and every text token also pays the full FFN/QKV. At
  1776×1184 a 1.9k-token prompt adds ≈ +24 % linear and +53 % attention FLOPs. Caption
  length is a direct step-time lever in training.
- **4D RoPE `(r, t, w, h)`**: `r = 0` (reserved for references), so the leading 32/128
  head dims are unrotated; spatial coords centred; image sequence position = text length.
  Frequencies must stay fp32 (`_RoPE` rebuilds them on every `.to()`).
- Blocks: shared modulation (one `Linear(3072, 6·3072)` for all blocks), bias-free
  `q k v o ffn_up_gate ffn_down`, SiTU-GLU FFN, RMSNorm with q/k norm.
- **Sampler**: Euler, `shift = 5` snapped to the nearest training-grid σ
  (`MOSSImageScheduler`), CFG 3, **uncond branch skips block 22** (still 2 forwards/step),
  optional extra SLG forward. Timestep input is `1000·σ`. Take every default from the
  pipeline file, not from Anima habit.
- **Training objective (report App. A.2)**: flow matching, target `ε − z0`, per-token
  mean, no σ/SNR weighting, text tokens excluded. Final pretraining interval: logit-normal
  μ = −1.09, s = 0.95, cutoff 0.95, 1 % uniform mix, shift 1. Muon–AdamW, EMA 0.9999.

## Layout (mirrors `library/qwen21/`)

| Path | Role |
|---|---|
| `library/moss/CLAUDE.md` | "This is not Anima" + the table of root invariants that do not carry over |
| `library/moss/requests.py` | torch-free `CacheRequest` / `GenerateRequest` / `TrainRequest`, `to_argv()` / `from_argv()` |
| `library/moss/model.py` | vendored `MOSSImageTransformer` / scheduler (Apache-2.0, attribution kept), our own sampler loop. No `trust_remote_code` at run time |
| `library/moss/loader.py` | phase-split load: TE → encode → free → DiT; DiT off the card before an fp32 decode |
| `library/moss/cache.py` | TE cache (natural length + mask) and latent cache (BN-normalised, packed) |
| `library/moss/generate.py` | `run_generate(req)`, Spectrum hook-in |
| `library/moss/lora.py` | adapters held outside `layers` (the qwen21 rule: the swapper moves every `.weight` under a block) |
| `library/moss/train.py` | `run_train(req)` |
| `scripts/moss/{cache,generate,train}.py` | thin argparse sidecars, submitted through the daemon |
| `library/downloads.py` | catalog rows: `moss-base` pack (DiT + TE + VAE + tokenizer), `moss-sft` = DiT only, reusing Base's TE/VAE |

Shared with Anima, and nothing else: `library.runtime.offloading`, `library.runtime.dynamo`,
`library.env`, the GUI job infra, the daemon, and the pure Spectrum core
(`networks/spectrum_forecast.py`, `networks/spectrum_sea.py`, already free of Anima imports
for the ComfyUI node). Nothing Anima-side imports `library.moss`.

## Phases

### Phase 0 — port and reproduce

- Catalog rows; vendored model + scheduler; `loader` / `cache` (TE side) / `generate`
  with the reference CFG + skip-22 loop.
- TE encode: offload **whole decoder layers**. Leaf-level group offload breaks on
  Qwen3.5's GDN path, which reads `conv1d.weight` functionally.
- Batch cond + uncond into one forward with a key mask instead of the reference
  per-sample loop, if it is numerically equivalent.
- **Gate**: same seed, same prompt → matches the reference `MOSSImagePipeline` output
  (bit-exact for the unbatched path; batched path within bf16 noise, checked by
  `max|Δ|` on the final latent). Also finish the two Space reproductions the
  2026-10-16 run lost to the decode OOM (`004` Blue Archive ensemble, CFG 5, 1776×1184;
  `017` 2880×720 beach panorama).

### Phase 1 — Spectrum

- Forecast the image-token hidden state after block 41 (before `final_norm_video`). Fast
  path = `time_embedding` → `video_head_modulation` → `final_norm_video` →
  `final_linear_video`. Cheap: no unpatchify.
- Cond and uncond need separate forecasters: the uncond forward skips block 22, so its
  feature trajectory is a different function.
- Text tokens are not forecast (they are dropped before the head anyway).
- Step-count sweep in the same bench (50 / 30 / 25 / 20, both models). The SFT card
  gives no fewer-step recommendation, so this is an open question, not an assumption.
- **Gate**: wall-clock speed-up vs plain 50-step at matched seeds, plus side-by-side
  grids reviewed by eye. The numeric target is set after the baseline is in.

### Phase 2 — light LoRA with activation checkpointing

- Caches: TE (`last_hidden_state` + length, never padded) and latents (native aspect,
  multiples of 16, ≤ ~1 MP to match Stage 3), both precached so training never loads
  the TE or VAE.
- LoRA on the six linears per block (`q k v o ffn_up_gate ffn_down`, 42 × 6), held
  outside the swapped blocks; fp32 master, bf16 GEMMs, saved bf16.
- Memory: 9.7 GiB resident DiT + checkpointed activations at ~4k image + caption tokens.
  Checkpointing first, then block swap sized from `mem_get_info` free (the qwen21
  measurement rule), compile last as a speed lever.
- Train on **Base**; test whether a Base LoRA transfers onto SFT.
- **Gate**: a small character or style set trains to a recognisable result in under an
  hour on the 5070 Ti, peak below 15 GiB.

### Later / out of scope for now

GUI tab, ComfyUI node, Anima-side inference stacks beyond Spectrum (SMC-CFG, CNS, …),
fp8 DiT weights.

**Soup (2026-10-20, `soup.md`).** A 3-seed soup at fixed lr 5e-5 on caststation keeps
the style even though the ingredients' full-sum ΔW cos is only ~0.17. The earlier
LR-diverse sincos soup (2e-5 / 5e-5 / 1e-4) washed out because one ingredient carried
~82 % of the energy. Before building, check the energy shares with
`src/dw_cos.py`: they should be roughly even. cos alone is not a gate. Ingredients of
one soup share one `svd_slice`. Slices are for merging different artists.

## Open questions

1. Caption format for LoRA data: tag bag vs natural language vs layout JSON. Attention
   cost is quadratic in caption length, and the report credits structured JSON with its
   best human ratings.
2. Character knowledge: does a name-only prompt (`akita neru`) work without the trait
   list?
3. Does a Base-trained LoRA carry onto SFT, or is SFT far enough to need its own?
4. Report inconsistency: the text says three pretraining stages and ships "the Stage 3
   checkpoint at step 310,000", but Table 1 lists a Stage 4 (up to ~2×1024², 520K
   steps). Which checkpoint is Base? It decides the native resolution ceiling.
5. `torch.compile` with variable text length: the joint sequence length changes with every
   caption. Needs dynamic shapes or bucketing of the text length.

## Gotchas hit on day one

- `hf download --include/--exclude a b c` takes **one** pattern per flag; the extras
  become positional filenames. Repeat the flag or use one pattern.
- Forward hooks that return `module.to(...)` replace the layer's input/output (the return
  value is the module). Return `None`.
- The fp32 VAE decode at ~2 MP needs ~1.5 GiB more than is free with the DiT resident.
  Finish all denoising, move the DiT off, then decode.
