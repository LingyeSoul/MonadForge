# Colorize with an OCR caption

Status: arm C trained and read 2026-10-21. Sub-line of the i2i / colorize probe
(`docs/proposal/moss_i2i.md`). Code: `src/colorize_cache.py`, `src/colorize_eval.py`,
`src/colorize_run.py` (both chained).

## Current recipe

`moss_colorize_3k_ocr` (2026-10-22) is the kept run: 3.3k pairs (half manga), fp8 TE,
lr 5e-5, 3 epochs, page alone on 30 % of steps, of the rest the layout caption 70 %.
Its checkpoint, eval (`output/tests/colorize_moss_colorize_3k_ocr`) and cache stay;
every earlier colorize checkpoint, eval arm and cache (`colorize_te`, `_ref`,
`_manga`, `_ocr`, `moss_colorize_3k` and its fp8 / GPTQ holdouts) was deleted, so the
runs below are record only.

A cache from `colorize_cache.py` holds two captions after the page, plus the page
alone, and the eval renders one arm per kind:

| TE key | Caption | Eval arm |
|---|---|---|
| `embeds` | OCR: layout JSON, the colour caption + one text item per OCR line | `lora_ocr` (and `base`, no LoRA) |
| `embeds_v{1..3}` | sparse: tag-dropped colour caption, no OCR | `lora_sparse` (`v1`) |
| `embeds_plain` | the page alone | `lora_page` |

`moss_colorize_3k_ocr`'s own cache predates this: its `embeds_v*` carry the text +
bbox OCR clause after the sparse caption. A `colorize_run.py` run draws its own pairs
and holdout, so its numbers do not line up page for page with that run.

## Why

Lettering in colorize renders reads off: glyphs drift and small balloons garble. The
half-size ref fixes where the text goes, not which glyphs it shows. Give the glyphs to
the TE as text: OCR the B&W page and add what it says, with its boxes, to the caption.

## Caption

- The OCR reads the **condition** (the B&W page), never the target. At inference only
  the page exists.
- Reader: anime_tools (AnimeText detector + the manga VL reader), ASCII lines kept
  (`skip_en` off), `usable_lines` floors. Speech and SFX alike, in reading order; an
  exact repeat is dropped and its first box kept.
- Boxes are MOSS's `abs_centralized` `yxyx` in the canvas's pixels (`centered_box`).
  The condition is read at the canvas size, so no rescale is needed.
- Each TE file holds two captioned encodes and the page alone:
  - `embeds`: layout JSON, the colour caption through `caption_to_layout` plus one
    `{"label":"text","caption":"text","text":…,"bbox":…}` item per line;
  - `embeds_v1`: `{colour caption}. OCR result as follows "…" [y0, x0, y1, x1], …`;
  - `embeds_plain`: the page alone.
- A page with no usable line keeps the layout of its colour caption and the colour
  caption itself. 224 of the 1012 pages carry lines; 5 of the 12 held out do (the 4
  manga pages, `ama_mitsuki`).
- All encodes use the fp8 TE (the catalog's GPTQ v2 file), as does the cache's
  `uncond.safetensors`.

## Arms

| Arm | LoRA | Eval caption |
|---|---|---|
| A (baseline) | `moss_colorize_ref_t0`, bf16 TE | colour caption |
| C | t0's request on `colorize_ocr`, `--caption_variant_rate 0.5` | layout (`lora_cap`) / text + bbox (`lora_sparse`) |

- C copies `moss_colorize_ref_t0.json`'s request: the same 1000 pairs as
  `colorize_ref`, 1600 steps, lr 1e-4, rank 32, ref 1/2, `ref_timestep 0`, seed 0. Steps
  split 35 % layout / 35 % text + bbox / 30 % page only.
- C changes three things against A: the OCR clause, the caption format (layout or
  text), and the TE (fp8 instead of bf16). Arm B (t0 with the OCR caption, no
  retraining) was skipped.
- The holdout is `colorize_manga`'s 12 pages. Its `holdout/` had lost 6 of the 12
  entries; `colorize_ocr.py` takes the missing refs from `colorize_ref/holdout` (the
  same encode) and re-encoded `manga__403801_en_0009`'s.

## Results

Cost: OCR 94 s, TE cache 22 min (3 fp8 encodes a page, torch GDN path), training
72.8 min (2.73 s/step, peak 12.6 GB; epoch 2 loss 0.271, t0 0.270). Eval seed 1,
ER-SDE + Spectrum, 30 steps, CFG 3.

**Lettering: better wherever the OCR read the line.** On the 4 manga pages:

- `623751_ja`: "それじゃ ドクター俺Pが 舌を使って 健康診断するぞ" and "あはっ
  プロデューサー くすぐったいよ" come out exact under layout; A garbles both
  ("ぐれじゃ おデーセーか…").
- `2550520_ja`: "あら? エミちゃん まだ帰って いないの?" exact under layout; A gives
  "エミぃかん… よぁっての?".
- `889354_en`: "NANOHA, YOU'RE FINALLY GONNA HAVE SEX WITH ME." exact under both
  captions; A gives "NANOL/ YOU'RE GONNA GONNA HAIR SEX".
- `403801_en`: "AND SO, BECAME ONE OF HIS GIRL-FRIENDS." and "ON OUR DATE, HE ENTERED
  DRESSING ROOM AND STARTED FUCKING ME." under layout (one word dropped each); A gives
  "AND SO ECAME GRB GO.T SO HENSS".
- Lines the OCR missed (most EN balloons on the two EN pages: "YOU ARE SO CUTE",
  "H-HEY, WAIT! NOT HERE!", "THE STAFF WOULD SEE US…") stay as garbled as A, or worse.
- Layout reads closer than text + bbox on every page.

**Colorization: even by eye; the luma metric drops.** 64 px luma PSNR / luma r
(128 px) / chroma error against the target:

| Arm | 8 illus. | 4 manga | 5 with OCR | 7 without |
|---|---|---|---|---|
| A, caption | 27.27 / 0.969 / 13.3 | 26.10 / 0.944 / 14.0 | 26.44 / 0.950 / 12.9 | 27.20 / 0.968 / 14.0 |
| A, page only | 26.58 / 0.968 / 16.7 | 25.96 / 0.942 / 14.6 | 26.14 / 0.947 / 13.9 | 26.54 / 0.968 / 17.5 |
| C, layout | 25.52 / 0.947 / 15.1 | 25.54 / 0.935 / 13.4 | 25.82 / 0.941 / 12.1 | 25.32 / 0.945 / 16.3 |
| C, text + bbox | 25.47 / 0.947 / 15.1 | 25.45 / 0.932 / 13.7 | 25.67 / 0.937 / 12.1 | 25.32 / 0.946 / 16.5 |
| C, page only | 25.48 / 0.949 / 17.5 | 25.74 / 0.937 / 14.0 | 25.91 / 0.942 / 12.8 | 25.32 / 0.947 / 18.8 |

- The luma drop is uneven: 8 of 12 pages lose, most of the mean is `yomu` (31.3 →
  24.7, warmer skin, more saturated bedding) and `ru_zhai` (25.8 → 22.6); `moursho`
  gains 1.9. It also holds on the 7 pages without OCR lines and on C's page-only arm,
  so the OCR clause does not cause it.
- Hue (chroma error with caption, per page) splits 6 / 5 / 1: C is closer on `ru_zhai`
  (brown sofa as the target, A blue), `oldsickkim` (grey background, A magenta),
  `ama_mitsuki` (grey hoodie, A blue), and three more; A is closer on `eight_tohyama`
  (C 25.2 vs 16.8, over-saturated green hair), `yomu`, `ningen_mame` and two more. Mean
  luma does not shift either way.
- Side by side (target | A | C, all 12 pages) neither arm reads as the better
  colorizer. The metric gap is within what a different training run can move; one run
  cannot tell the TE switch, the caption-format mix and run variance apart.
- An earlier read (`moss_colorize_3k`, a bf16-trained LoRA) moved only 34.6 dB PSNR
  between a bf16 and an RTN fp8 TE at eval, so an fp8 TE at eval alone is small. This
  run also trained on fp8.

## Open

- Whether the luma gap is real: a second bf16 seed of t0 (run variance), or a t0 rerun
  on an fp8 cache without OCR (TE only).
- OCR recall on EN pages: the reader missed most English balloons. Lines it reads are
  rendered; lines it misses are not.

## Limits

- One seed, 12 pages, one training run per arm.
- Only 224 of the 1000 training pairs carry lines, so C sees the clause on a
  minority of steps.
