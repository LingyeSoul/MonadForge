# MOSS-Image i2i — a VAE reference on RoPE's `r` axis

Status: PROPOSED 2026-10-19. Two probes measured (colorize, and colorize with manga pages,
below). Code on branch
`i2i_probe`.

## Premise

MOSS-Image has no image conditioning. Its 4D RoPE `(r, f, h, w)` keeps an `r` axis that
training always sets to 0 (`project/moss_lora/plan.md`). Qwen3.5's vision tower conditions
MOSS zero-shot, but at 32 px per token it brings back content, not line placement
(`docs/moss-findings/vision_tower_reference.md`). This proposal puts a clean VAE latent of
the condition image into the single-stream sequence on `r = 1`, and trains a plain LoRA to
read it.

## Mechanism (as built on `i2i_probe`)

- **Sequence** `[image | ref | text]`. The ref tokens go through the same `video_embedder`
  as the image. They get no noise and share the step's timestep modulation.
- **RoPE**: ref tokens are at `r = 1`, the image's `f` (= text length). Their `h, w`
  coordinates are stretched onto the target grid, `(i + 0.5)·H/h_ref − 0.5`, centred like
  the image's. So a half-size ref lines up with the target position for position.
- **Head and loss** read the image tokens only. Text RoPE and the key mask shift by the
  ref token count.
- **Inference**: the ref goes into every CFG branch.
- **Off path**: `ref=None` is bit-exact with the old forward, both single and batched
  (checked on a tiny CPU model). With a ref, forward and `forward_batched` agree to 2.4e-7.
- **Files**:
  - `library/moss/model.py`: `_RoPE.forward(ref_hw=)`, `embed_sample` / `forward` /
    `forward_batched(ref=)`.
  - `generate.py`: `denoise(ref=)`.
  - `cache.py` / `scan.py` (`*.moss_ref.safetensors`), `train.py`, `tune.py`.
  - Probe scripts: `project/moss_lora/src/colorize_{cache,eval}.py`.

## Probe: EasyControl colorize, 1-hour budget

- **Data**: 1000 pairs from EasyControl colorize (synthetic mangafied page → colour
  original) and 8 held out. The 3 probe images go first in the holdout, then random pages.
- **Both runs share**:
  - The text encoder input: the page through the vision tower plus the colour caption
    (`filter_to_colors_and_protected`).
  - The LoRA: rank 32 on q/k/v/o/ffn, lr 1e-4, 1600 steps, `--plain_caption_rate 0.3`
    (the page with no caption).
- **Difference**: the ref run adds the page as a VAE latent at half size per side
  (1/4 area, 720–1080 tokens).
- **Cost**: v1 ran 58.6 min at 2.16 s/step; ref ran 71.3 min at 2.67 s/step (swap 36,
  budget 0.2).
- **Eval**: Base + LoRA, ER-SDE + Spectrum, 30 steps, CFG 3, seed 1. The ref goes to the
  LoRA arms only.

| Arm | 64 px luma PSNR | luma r (128 px) | mean chroma | chroma error vs target |
|---|---|---|---|---|
| target | — | 1 | 16.1 | 0 |
| B&W page, copied as is | — | 0.994 | 0.0 | 16.1 |
| base, page + colour caption | 9.96 | 0.25 | — | — |
| v1 (vision tower only) + caption | 11.28 | 0.42 | 15.3 | 16.4 |
| v1 page only | 11.30 | 0.40 | — | — |
| **ref + caption** | **25.62** | **0.97** | 17.2 | **13.6** |
| **ref, page only** | **24.77** | **0.96** | 17.2 | 18.9 |

Chroma is the mean CbCr distance from grey, at 1/8 size. The chroma error is the mean
CbCr distance to the target.

- **Alignment comes from the ref.**
  - Every held-out page scores 22.6–28.3 on the 64 px PSNR; v1 ranged 8.6–14.1.
  - By eye: line work, faces, hands, lettering and background match the page.
  - v1 kept the content but redrew the layout.
- **It colours; it does not copy.** Chroma is at the target's level. With the caption, the
  chroma error beats grey (13.6 vs 16.1); v1 did not. The colour tags steer the hues.
- **Without a caption** the colours are plausible but are not the target's (18.9). That is
  expected: a B&W page does not fix hue.
- **Limits**: one seed, 8 pages. Base gets no ref, so the fair comparison is v1 vs ref.

## Probe 2: colour manga pages, lr 5e-5

- **Data**: the same 1000 pairs plus 500 pages from EasyControl's `manga_color` slice:
  hitocrawl's full-colour works, JA / EN / KO editions, 7117 pairs in all
  (`colorize_cache.py --manga 500`). The holdout is the same 8 plus 4 manga works; no page
  of a held-out work trains, in any edition.
- **Run**: ref at half size as before, lr 5e-5 (v1: 1e-4), 2 epochs = 3000 steps, 137 min
  at 2.75 s/step (swap 28, budget 0.15 from `tune_swap`).
- **Loss**: 5e-5 starts slower. It meets ref_v1 at about 0.5 epoch, and from there the two
  overlap on an epochs axis. Epoch means: 0.309 / 0.278 (ref_v1: 0.302 / 0.276). The
  datasets differ, so only the shapes compare. ref_v1's epoch 1 also ran size-sorted
  (since fixed, see `--fit_report`).
- **Eval**: as above, 12 held out, seed 1, compared by eye against ref_v1 rendered on
  the same holdout.

| | manga run vs ref_v1 |
|---|---|
| Lettering on the manga pages | Better. `889354_en`: "YOU ARE SO CUTE, NANOHA." and "OOH YOU SQUIRTED." read right; ref_v1 gives "YO U ARE", "DON YOU SQUIRTES". Small balloons garble in both. Red balloon lettering comes out black in both |
| Hair colour on the manga pages | Closer on 2 of 3 works (orange `889354`, blonde `403801`); ref_v1 is closer on `623751` |
| The 8 illustrations, with caption | Even. One clear miss: `eight_tohyama` gets two green-haired characters where the target has one purple, one green |
| Page only | More conservative. Where the hue is ambiguous it stays near the page's grey, and saturation is lower; ref_v1 picks a vivid hue, often a wrong one |

- **Limits**: one seed, 12 pages, by eye; no metric pass yet.
- **Lettering**: the half-size ref is the likely limit on small text; untested. It adds a
  lettering read to the ref-size arms (open question 2).

## Probe 3: real B&W pages, no target

- **Data**: 8 pages from `comfy/input/to_colorize{,2}`, never through Mangafy: 4 Korean
  manga pages (screentone, panels, lettering) and 4 grey-shaded illustrations. Page
  only, no caption; ref_v1 and the manga run, seed 1 (`colorize_wild.py`).
- **Read**: luma r against the page; chroma as above (the colorize targets average 16).

| | luma r, manga / illus. | chroma, manga / illus. |
|---|---|---|
| ref_v1 | 0.90 / 0.94 | 3.1 / 12.4 |
| manga run | 0.91 / 0.97 | 3.0 / 7.7 |

- **Alignment holds off-distribution.** Panels, black fills, screentone gradients and
  balloon placement stay. Screentone is not copied as noise. On the Korean pages the
  manga run is clearly ahead: its lettering reads closer to the page, it keeps the faces
  (ref_v1 redraws some), and it adds no background wash.
- **The manga pages barely colour.** Both LoRAs leave them mostly white and grey, with a
  few accents (a hoodie, jackets, hair, eyes). Real pages are sparser than Mangafy's
  output, and with no caption nothing fixes hue. A colour caption is the untested lever.
- **On the illustrations**, ref_v1 colours vividly but invents: cat ears, a teal tail, a
  starred background, bokeh dots on one manga page. The manga run stays faithful but
  desaturated.
- **No degradation augmentation** is needed for alignment.

## Settled

- **The vision tower stays** in the TE input; no text-only-TE arm.
- **Ref size** is capped at 1/2 per side, the current setting; no size arms.

## Probe 4: the reference at t = 0

- **Why**: the ref tokens share the image's block modulation, so a clean latent is
  modulated as if noised; with the median training σ near 0.75 that is most steps.
- **Change**: `--ref_timestep 0` gives the ref tokens the t = 0 modulation (shift,
  scale and both gates) in every block; the image and text keep the step's. A negative
  value is the old behaviour, and `ref=None` stays bit-exact. The value goes into the LoRA
  metadata, and the eval scripts set it from there.
- **Arm**: `ref_v1`'s request as it ran (`colorize_ref`, 1000 pairs, 1600 steps, lr
  1e-4, rank 32, ref 1/2), with only `--ref_timestep 0`.
- **Baseline**: `moss_colorize_ref_v1`; it is not retrained. Its epoch 1 ran
  size-sorted (since fixed), which the new arm does not.
- **Read**: the 12-page holdout of probe 2, metrics as in probe 1, seed 1.
- **Cost**: 71.9 min at 2.70 s/step (ref_v1: 71.3 min, 2.67 s/step).
- **Loss**: epoch means 0.290 / 0.270 (ref_v1: 0.302 / 0.276). Epoch 1 also differs in
  order, so read epoch 2.

| Arm | 64 px luma PSNR, 8 illus. / 4 manga | luma r | chroma error, 8 illus. |
|---|---|---|---|
| ref_v1 + caption | 25.62 / 23.73 | 0.966 | 13.6 |
| **t0 + caption** | **27.27 / 26.10** | **0.975** | **13.3** |
| ref_v1, page only | 24.77 / 23.34 | 0.964 | 18.9 |
| **t0, page only** | **26.58 / 25.96** | **0.974** | **16.7** |
| manga run + caption (more data, other lr) | 27.49 / 26.74 | 0.975 | 13.0 |

- **t = 0 wins.** With the caption it is ahead on 11 of 12 pages.
- **One page carries much of the illustration mean.** `yomu` gains 8 dB because t0
  gets its pink background where ref_v1 painted it blue. Without that page the gain on
  the illustrations is +0.7 dB; on the 4 manga pages it is +2.4.
- **By eye**:
  - `eight_tohyama` keeps one purple-haired and one green-haired character, as the
    target and ref_v1 have them; the manga run made both green.
  - `ru_zhai`'s background turns warm, as in the target, where ref_v1 made it night
    blue.
  - `ama_mitsuki`'s hoodie turns blue where the target's is beige.
- **On 1000 pairs and 1600 steps**, t0 comes within 0.2 dB of the manga run on the
  illustrations.
- **Limits**: one seed, 12 pages.
- **Default**: `--ref_timestep` is now 0 by default; `-1` opts out. LoRAs from before
  the flag carry no stamp and load with the old, shared timestep.
- **Then**: masking the ref to attend only to itself. With t = 0, that makes the ref's
  per-block K/V one computation per image, reused across steps and CFG branches.

## Open questions, in order

1. **Ref guidance.**
   - The ref is in both CFG branches now.
   - Arms: ref dropped from uncond (image CFG), and a separate ref guidance scale
     (InstructPix2Pix-style).
   - Training needs ref dropout for this.
2. **Other tasks on the same path**: lineart / sketch → colour, edits (needs pairs), and
   2× SR. A half-size ref *is* an SR condition. Check against the finished ResShift line
   (`project/finished/`).
3. **Eval protocol**:
   - More seeds and a larger holdout.
   - A safe-only holdout for anything published.
   - Real B&W pages (`colorize_wild.py`): the training conditions all come from
     Mangafy.
4. **Ship surface** (after 1):
   - `GenerateRequest(ref_image=, ref_scale=)` with VAE-encode in the request path.
   - A ref input on the ComfyUI MOSS node.
   - `reference_latents` / ref scale stamped in the LoRA metadata (train.py already
     writes `reference_latents`) so inference can refuse a mismatch.
   - A cache stage for refs outside the probe script.

Gate before the ship surface: alignment holds on a second
task (lineart or SR).
