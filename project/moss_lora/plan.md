# moss_lora — MOSS-Image on a 16 GB card

**Not Anima.** MOSS-Image (OpenMOSS, Apache-2.0, released 2026-10-11) is a 5B single-stream
DiT trained from scratch on a frozen Qwen3.5-9B text encoder and the frozen FLUX.2 VAE
(`OpenMOSS-Team/MOSS-Image-Base`, `-SFT`; SFT is a fine-tune of Base). The line ported
Spectrum and a light LoRA onto the 5070 Ti and grew from there. Code map:
`library/moss/CLAUDE.md`; user doc: `docs/moss/README.md`.

## Status

| Sub-line | State | Where |
|---|---|---|
| Phase 0 — port + reference parity | done (`75d9d01a`): unbatched bit-exact, batched opt-in | `bench/moss_parity/` |
| Phase 1 — Spectrum | done: 2.6× at 50 steps, compiled blocks on top | `bench/moss_spectrum/`, `docs/moss/README.md` |
| Phase 2 — light LoRA (block swap + activation budget) | done | `docs/moss/README.md`; phase logs removed 2026-10-18, in git history |
| Channel scaling + `dc_isolate` | shipped (α 0.5, `dc_isolate` on, lr 5e-5); control arm open | `docs/moss/channel_scaling.md`, log `channel_scaling.md` |
| DAVE | shipped for Base as `dave=auto` (blocks 19–30) | `docs/moss/dave.md`, log `dave.md` |
| ER-SDE sampler | shipped as the default | `sampler.md` |
| CNS | shipped γ files (bench + ComfyUI `er_sde_cns`), no `generate.py` flag | `docs/inference/cns.md`, `sampler.md` |
| GUI tab + ComfyUI | shipped: `make gui-moss`, kohya-key LoRAs, Spectrum KSampler for MOSS, DAVE node 0.2.0 | `docs/guidelines/moss-guidebook.md` |
| Caption dropout | A/B done: keep 0 | `caption_dropout.md` |
| Seed soup | done: a 3-seed fixed-lr soup keeps the style; check energy shares first (`scripts/soup/dw_cos.py`) | `soup.md` |
| Relational REPA | parked: seed variance swamps any effect | `repa.md` |
| Vision-tower reference | finding | `docs/moss/findings/vision_tower_reference.md`, `bench/moss_vision_reference/` |
| Anima negatives revisit (sinks, 2:4 sparsity) | Tier 1 measured, Tier 2 unmeasured | `docs/proposal/moss_anima_negatives_revisit.md` |
| i2i / colorize (VAE reference on RoPE `r`) | **active** | `docs/proposal/moss_i2i.md`, `src/colorize_{cache,eval,wild,scale}.py` |
| Technical-report digest | reference | `report_notes.md` |

`src/`: the colorize probes, `lora_ab.py` (LoRA A/B grids), `vision_te.py` (TE call with
an image in the user turn) and `_render.py` (their shared render loop). `space/`: the two
Space prompts every A/B renders.

## Open questions

Per sub-line open items live in each log's `## Open`. Line-wide:

1. Character knowledge: does a name-only prompt (`akita neru`) work without the trait
   list?
2. Resolution ceiling: the report's text caps Base at ~1 MP, the Space presets run
   ~2.1 MP (`report_notes.md` § (a)). Keep LoRA buckets ≤ ~1 MP until Base is compared
   at 1 MP vs 2 MP.

Closed: caption format (`layout_json` is the default since `643f957f`); which checkpoint
is Base (Stage 3 EMA at 310k, `report_notes.md` § (a)); whether a Base LoRA carries onto
SFT (it does, `docs/moss/README.md`); `torch.compile` with variable text length (blocks
compiled since `0d852ebb`).

## First look (2026-10-16)

One seed, five 1024² prompts, Base and SFT, 50 steps, CFG 3, uncond skip 22:

- Quoted CJK text in the main subject came out correct in both models: "待ってくれ！"
  written vertically in a manga bubble, three handwritten Chinese lines, a "오늘도 수고했어" neon
  sign. Small background text (signage behind glass) is still pseudo-glyphs.
- A raw Danbooru tag bag (`1girl, solo, akita neru, …`) rendered on-prompt, even though the
  report names no booru source and says the captions are gemma4 natural language.
- SFT renders more finished and pulls toward a glossy gacha look; Base is rawer and more
  varied. Base looked like the better LoRA substrate, SFT the better out-of-the-box
  sampler.

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
