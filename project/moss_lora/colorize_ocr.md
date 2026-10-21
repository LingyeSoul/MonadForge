# Colorize with an OCR caption

Status: arm C trained and read 2026-10-21. Sub-line of the i2i / colorize probe
(`docs/proposal/moss_i2i.md`). Code: `src/colorize_ocr.py` (cache), `src/colorize_eval.py`.

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

**Colorization: down, page-only arm included.** 64 px luma PSNR / luma r (128 px) /
chroma error against the target:

| Arm | 8 illus. | 4 manga | 5 with OCR | 7 without |
|---|---|---|---|---|
| A, caption | 27.27 / 0.969 / 13.3 | 26.10 / 0.944 / 14.0 | 26.44 / 0.950 / 12.9 | 27.20 / 0.968 / 14.0 |
| A, page only | 26.58 / 0.968 / 16.7 | 25.96 / 0.942 / 14.6 | 26.14 / 0.947 / 13.9 | 26.54 / 0.968 / 17.5 |
| C, layout | 25.52 / 0.947 / 15.1 | 25.54 / 0.935 / 13.4 | 25.82 / 0.941 / 12.1 | 25.32 / 0.945 / 16.3 |
| C, text + bbox | 25.47 / 0.947 / 15.1 | 25.45 / 0.932 / 13.7 | 25.67 / 0.937 / 12.1 | 25.32 / 0.946 / 16.5 |
| C, page only | 25.48 / 0.949 / 17.5 | 25.74 / 0.937 / 14.0 | 25.91 / 0.942 / 12.8 | 25.32 / 0.947 / 18.8 |

- The drop sits on the 7 pages without OCR lines and on C's page-only arm, where no
  OCR text is in the prompt. So the OCR clause does not cause it; the TE switch, the
  caption-format mix or run-to-run variance does. One training run cannot tell them
  apart.
- By eye it is a tone shift: C renders brighter and warmer (`yomu` 31.3 → 24.7, warmer
  skin and more saturated bedding; `ru_zhai` 25.8 → 22.6, a lighter, lower-contrast
  room, though its brown sofa matches the target better than A's blue one).
- Chroma error on the 5 OCR pages is lower for C (12.1 vs 12.9).
- An earlier read (`moss_colorize_3k`, a bf16-trained LoRA) moved only 34.6 dB PSNR
  between a bf16 and an RTN fp8 TE at eval, so an fp8 TE at eval alone is small. This
  run also trained on fp8.

## Open

- What dropped the colour: a t0 rerun on an fp8 cache without OCR (TE only), or a
  second bf16 seed of t0 (run variance).
- OCR recall on EN pages: the reader missed most English balloons. Lines it reads are
  rendered; lines it misses are not.

## Limits

- One seed, 12 pages, one training run per arm.
- Only 224 of the 1000 training pairs carry lines, so C sees the clause on a
  minority of steps.
