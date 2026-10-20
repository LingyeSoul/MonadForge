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

## Open questions, in order

1. **Is the vision tower still needed?**
   - Arm: ref + text-only colour caption, through the stock TE and its cache.
   - If it matches, i2i is a plain LoRA with plain caching: no 1000-row TE, a cheaper
     joint sequence, and Qwen's tower drops out of the loop.
2. **Ref size.**
   - Arms: 1/4 per side (~250 tokens), 1/2 (this run), full (≈ image tokens).
   - Read alignment (luma r / PSNR) and small-balloon lettering against s/step.
3. **Ref guidance.**
   - The ref is in both CFG branches now.
   - Arms: ref dropped from uncond (image CFG), and a separate ref guidance scale
     (InstructPix2Pix-style).
   - Training needs ref dropout for this.
4. **Other tasks on the same path**: lineart / sketch → colour, edits (needs pairs), and
   2× SR. A half-size ref *is* an SR condition. Check against the finished ResShift line
   (`project/finished/`).
5. **Eval protocol**:
   - More seeds and a larger holdout.
   - A safe-only holdout for anything published.
6. **Ship surface** (after 1–3):
   - `GenerateRequest(ref_image=, ref_scale=)` with VAE-encode in the request path.
   - A ref input on the ComfyUI MOSS node.
   - `reference_latents` / ref scale stamped in the LoRA metadata (train.py already
     writes `reference_latents`) so inference can refuse a mismatch.
   - A cache stage for refs outside the probe script.

Gate before the ship surface: open questions 1 and 2 are answered, and alignment holds on
a second task (lineart or SR).
