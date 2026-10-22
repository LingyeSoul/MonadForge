# Colorize with an OCR caption

Sub-line of the i2i / colorize probe (`docs/proposal/moss_i2i.md`). Code:
`scripts/moss/colorize_cache.py` (`library/moss/colorize.py`), `src/colorize_eval.py`,
chained by `src/colorize_run.py`. The recipe below ships as `configs/moss/i2i/colorize/`.

## Current recipe

`moss_colorize_3k_ocr` (2026-10-22) is the kept run: 3.3k pairs (half manga), fp8 TE,
lr 5e-5, 3 epochs, page alone on 30 % of steps, of the rest the layout caption 70 %. Its
checkpoint, eval (`output/tests/colorize_moss_colorize_3k_ocr`) and cache stay; every
earlier colorize checkpoint, eval and cache was deleted, so arm C below is record only.

A `colorize_cache.py` cache holds two captions after the page, plus the page alone; the
eval renders one arm per kind:

| TE key | Caption | Eval arm |
|---|---|---|
| `embeds` | OCR: layout JSON, the colour caption + one text item per OCR line | `lora_ocr` (and `base`, no LoRA) |
| `embeds_v{1..3}` | sparse: tag-dropped colour caption, no OCR | `lora_sparse` (`v1`) |
| `embeds_plain` | the page alone | `lora_page` |

`moss_colorize_3k_ocr`'s own cache predates this: its `embeds_v*` carry the text + bbox
OCR clause after the sparse caption. A `colorize_run.py` run draws its own pairs and
holdout, so its numbers do not line up page for page with that run.

## OCR caption

Goal: colorize renders garble lettering; the half-size ref fixes where text goes, not
which glyphs. So the B&W page is OCR'd and its lines, with boxes, go into the caption.

- The OCR reads the **condition** (the B&W page), never the target: at inference only
  the page exists.
- Reader: anime_tools (AnimeText detector + the manga VL reader), ASCII lines kept
  (`skip_en` off), `usable_lines` floors. Speech and SFX alike, in reading order; an
  exact repeat is dropped and its first box kept.
- Boxes are MOSS's `abs_centralized` `yxyx` in canvas pixels (`centered_box`); the
  condition is read at canvas size, so no rescale.
- Layout item per line: `{"label":"text","caption":"text","text":…,"bbox":…}`. Arm C's
  text form (`embeds_v1`): `{colour caption}. OCR result as follows "…" [y0, x0, y1, x1], …`.
- A page with no usable line keeps its colour caption and its layout. 224 of the 1012
  pages carry lines; 5 of the 12 held out do (the 4 manga pages, `ama_mitsuki`).
- All encodes, including the cache's `uncond.safetensors`, use the fp8 TE (the catalog's
  GPTQ v2 file).

## Arm C vs A (2026-10-21)

| Arm | LoRA | Eval caption |
|---|---|---|
| A (baseline) | `moss_colorize_ref_t0`, bf16 TE | colour caption |
| C | t0's request on `colorize_ocr`, `--caption_variant_rate 0.5` | layout (`lora_cap`) / text + bbox (`lora_sparse`) |

C copies `moss_colorize_ref_t0.json`'s request (the 1000 pairs of `colorize_ref`, 1600
steps, lr 1e-4, rank 32, ref 1/2, `ref_timestep 0`, seed 0); steps split 35 % layout /
35 % text + bbox / 30 % page only. C differs from A in three ways at once: the OCR
clause, the caption format, the TE (fp8 vs bf16). Holdout: `colorize_manga`'s 12 pages
(6 missing refs taken from `colorize_ref/holdout`, `manga__403801_en_0009` re-encoded).

Cost: OCR 94 s, TE cache 22 min (3 fp8 encodes a page, torch GDN path), training 72.8 min
(2.73 s/step, peak 12.6 GB; epoch 2 loss 0.271, t0 0.270). Eval seed 1, ER-SDE +
Spectrum, 30 steps, CFG 3.

**Lettering: better wherever the OCR read the line** (4 manga pages):

- `623751_ja`: "それじゃ ドクター俺Pが 舌を使って 健康診断するぞ" and "あはっ
  プロデューサー くすぐったいよ" exact under layout; A garbles both ("ぐれじゃ おデーセーか…").
- `2550520_ja`: "あら? エミちゃん まだ帰って いないの?" exact under layout; A gives
  "エミぃかん… よぁっての?".
- `889354_en`: "NANOHA, YOU'RE FINALLY GONNA HAVE SEX WITH ME." exact under both
  captions; A gives "NANOL/ YOU'RE GONNA GONNA HAIR SEX".
- `403801_en`: "AND SO, BECAME ONE OF HIS GIRL-FRIENDS." and "ON OUR DATE, HE ENTERED
  DRESSING ROOM AND STARTED FUCKING ME." under layout (one word dropped each); A gives
  "AND SO ECAME GRB GO.T SO HENSS".
- Lines the OCR missed (most EN balloons on the two EN pages) stay as garbled as A, or
  worse. Layout reads closer than text + bbox on every page.

**Colorization: even by eye; the luma metric drops.** 64 px luma PSNR / luma r (128 px)
/ chroma error against the target:

| Arm | 8 illus. | 4 manga | 5 with OCR | 7 without |
|---|---|---|---|---|
| A, caption | 27.27 / 0.969 / 13.3 | 26.10 / 0.944 / 14.0 | 26.44 / 0.950 / 12.9 | 27.20 / 0.968 / 14.0 |
| A, page only | 26.58 / 0.968 / 16.7 | 25.96 / 0.942 / 14.6 | 26.14 / 0.947 / 13.9 | 26.54 / 0.968 / 17.5 |
| C, layout | 25.52 / 0.947 / 15.1 | 25.54 / 0.935 / 13.4 | 25.82 / 0.941 / 12.1 | 25.32 / 0.945 / 16.3 |
| C, text + bbox | 25.47 / 0.947 / 15.1 | 25.45 / 0.932 / 13.7 | 25.67 / 0.937 / 12.1 | 25.32 / 0.946 / 16.5 |
| C, page only | 25.48 / 0.949 / 17.5 | 25.74 / 0.937 / 14.0 | 25.91 / 0.942 / 12.8 | 25.32 / 0.947 / 18.8 |

- The luma drop is uneven: 8 of 12 pages lose, most of the mean is `yomu` (31.3 → 24.7)
  and `ru_zhai` (25.8 → 22.6); `moursho` gains 1.9. It holds on the 7 pages without OCR
  and on C's page-only arm, so the OCR clause does not cause it.
- Hue (chroma error with caption, per page) splits 6 / 5 / 1: C closer on `ru_zhai`,
  `oldsickkim`, `ama_mitsuki` and three more; A closer on `eight_tohyama` (C 25.2 vs
  16.8, over-saturated green hair), `yomu`, `ningen_mame` and two more.
- Side by side, neither reads as the better colorizer; one run per arm cannot separate
  the TE switch, the caption mix and run variance.
- `moss_colorize_3k` (bf16-trained) moved only 34.6 dB PSNR between a bf16 and an RTN
  fp8 TE at eval, so the fp8 TE at eval alone is small.

## Open

- Whether the luma gap is real: a second bf16 seed of t0 (run variance), or t0 rerun on
  an fp8 cache without OCR (TE only).
- OCR recall on EN pages: the reader missed most English balloons, and missed lines are
  not rendered.
- Limits: one seed, 12 pages, one run per arm; only 224 of the 1000 training pairs carry
  lines.
