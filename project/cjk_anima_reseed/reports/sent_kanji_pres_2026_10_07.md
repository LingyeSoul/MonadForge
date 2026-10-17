# sent_kanji_pres — L_pres in a full run (2026-10-07)

`configs/sent_kanji_pres.toml`: `sent_kanji_f0`'s rows, data, lr, row_lr
and free_residual, plus `pres = { lam = 10, band = [0.8, 0.9], every = 2 }`
— λ 10 · L_pres at σ ~ U(0.8, 0.9) on every 2nd scene step, the band
`probe_pres_train` h32 p10c09 settled (`reports/probe_pres_2026_10_06.md`
§ h32, band capped at 0.9). 1 348 rows, 53 920 steps × batch 4, 650 min at
1.38 it/s (f0 2.31 it/s: the pres pass on half the steps costs × 1.67).

## 1. Training

Last 200 log rows (f0 / pres): loss 0.089 / 0.090, in-box 0.108 / 0.111,
out-box 0.075 / 0.075, warm cos 0.953 / 0.949, rel 1.187 / 1.186. The pres
run gives back a little in-box fit (+0.002, the probe's +0.0026 order) and
moves the rows about as far from the warm start. L_pres (mean over the
logged steps) 1.15e-3 in the first 5 k steps, 0.86e-3 at 5–15 k, 0.71e-3
at 25–35 k and 0.70e-3 in the last 5 k — flat from mid-run; no in-run
reading without the term to set it against.

## 2. Ruler

`ruler.py run --pack punct --arms seed_fixed_1005_stick080@punct,sent_kanji_f0,sent_kanji_pres`
→ `results/20261007-2048-ruler-sensitive-sent_kanji_pres/` (preview51 and f0
from cache). All 96 strings:

| arm | g_f1 | g_p | g_r | g_r_kanji | g_r_kana | drawn | exact | le2 | dup | cer | text_area | iou_en | en_match | en_tok_out |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| preview51 | 0.241 | 0.170 | 0.635 | 0.193 | 0.722 | 45.6 | 9 | 14 | 32 | 0.681 | 0.091 | 0.248 | 0.436 | 0.724 |
| sent_kanji_f0 | 0.271 | 0.202 | 0.670 | 0.336 | 0.738 | 37.3 | 3 | 13 | 28 | 0.689 | 0.083 | 0.244 | 0.436 | 0.718 |
| **sent_kanji_pres** | 0.270 | 0.224 | 0.611 | 0.294 | 0.672 | 35.2 | 4 | 13 | 20 | 0.676 | **0.055** | **0.318** | **0.533** | **0.759** |

Paired, pres − f0 (mean Δ, up / down strings, p):

| | all 96 | unseen 42 |
|---|---|---|
| `en_tok_out` | +0.041, 77 / 19, **2e-9** | +0.037, 32 / 10, 9e-4 |
| `en_match` | +0.097, 71 / 20, **7e-8** | +0.076, 25 / 13, 0.07 |
| `iou_en` | +0.075, 70 / 22, **5e-7** | |
| `text_area` | −0.028, 20 / 76, **7e-9** | −0.016, 12 / 30, 0.008 |
| `cer` (lower better) | −0.013, 27 / 35, 0.37 | −0.028, 0.31 |
| `g_f1` | −0.001, 49 / 43, 0.6 | −0.010, 0.42 |
| `g_r_kana` | −0.066, 26 / 41, 0.086 | |
| `g_r_kanji` | −0.042, n.s. | |
| `le2` / `exact` | 13 → 13 (5 / 5) / 3 → 4 | |

Against preview51: `g_f1` +0.029 (61 / 32, p 0.004), `g_r_kanji` +0.102
(p 0.002), `drawn` −10.4 (p 2e-5), `en_match` +0.097 (68 / 18, p 5e-8),
cer −0.005 n.s. — f0's text gains kept, and the page gain preview51 bought
over the old floor (+0.076) doubled.

By bin (preview51 / f0 / pres):

| bin | g_f1 | text_area | en_match | en_tok_out |
|---|---|---|---|---|
| short | 0.249 / 0.222 / 0.188 | 0.060 / 0.059 / 0.055 | 0.471 / 0.500 / 0.547 | 0.725 / 0.729 / 0.749 |
| mid | 0.212 / 0.259 / 0.299 | 0.074 / 0.075 / 0.052 | 0.487 / 0.461 / 0.578 | 0.748 / 0.735 / 0.782 |
| long | 0.261 / 0.331 / 0.323 | 0.139 / 0.113 / 0.058 | 0.350 / 0.346 / 0.473 | 0.698 / 0.688 / 0.745 |

## 3. The shrunk text is not the whole page gain

The text area falls a third (long strings halve: 0.113 → 0.058), and a
smaller text box alone lifts the out-of-box page reads. On the 30 strings
whose text area held (|Δ| < 0.01), pres − f0: `en_tok_out` +0.030
(p 0.002), `en_match` +0.057 (p 0.017), `iou_en` +0.034 (p 0.004).
Spearman of Δtext_area against Δ: `iou_en` −0.45 (p 5e-6), `en_match`
−0.29 (p 0.004), `en_tok_out` −0.16 (n.s.) — about half the `iou_en` gain is
the shrink, `en_tok_out`'s mostly is not.

## Verdict

- **The probe carried over.** p10c09's pattern holds at full scale: the
  text at f0's (cer, le2, g_f1 n.s.), the page up on every page read, and
  the gain held where the text did not shrink.
- **What it costs.** The text box shrinks (−33 %, long strings −49 %) and
  kana recall leans down (−0.066, p 0.09; short `g_f1` 0.222 → 0.188). The
  kanji recall f0 bought over preview51 stays (+0.10).
- Fewer doubled glyphs (`dup` 28 → 20, n.s.).

## Open

- The shrink by eye: `sheets/` — whether long dialogue reads as a smaller
  bubble or as text squeezed out of it.
- λ below 10, to buy back text area against the page.
