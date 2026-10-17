# stick_scene: kana_up's stick re-fit on the scene tiers alone (2026-10-04)

From the user (10-04, after `stick_fit_2026_10_04.md`): the grids are flat
canvases too (`recipes.grid`: a white or tinted canvas, half of them with a
bare bubble), so `stick_nolone` still trained the stick on scene-less
pictures. Two stick runs on the four bubble tiers only: `stick_nolonegrid` at
their `kana_up` bands, `stick_nlg_high` with every item at σ 0.75–0.95 (the
layout is set above 0.85, `probe_split` § 3).

**Verdict.**
- **Without any flat canvas the white canvas stays: the look is not the
  stick's.** `stick_nolonegrid` reads as `stick_full` (words ≤ 1 edit 51 vs
  51, paired 18 / 18; singles official 32 vs 35, p 0.68) and its scene is no
  better: singles p01 cos out 0.740 vs 0.779, flat white 0.529 vs 0.470,
  words flat white 0.126 vs 0.069. On the sheets it draws kana_up's
  pictures; ひ p01 goes to a full white canvas (0.569). `stick_fit`'s "if the
  lone tiers make the look, it is in the spikes" now holds with the grid
  confound out.
- **At σ ≥ 0.75 alone the stick takes the data's layout: the first stick run
  that changes the picture.** `stick_nlg_high` writes the words in a white
  speech-bubble box over the blackboard in thin, small glyphs — the bubble
  tiers' picture. The stick moves 122 (|m| 141 → 187, cos 0.758 to
  kana_up's); the low-σ items that held the other re-fits to render-neutral
  moves are gone.
- **That layout is not the word fit; the read falls.** Against `stick_full`:
  words ≤ 1 edit 28 vs 51 (15 / 38, p 0.002), ≤ 2 edits 45 vs 85 (8 / 48,
  p 5e-8), singles contained 74 vs 98 (3 / 27, p 8e-6). The move is
  orthogonal to retrain_kana − kana_up (cos −0.005).
- **The scene outside the text comes back on singles, the boxes whiten the
  words.** Singles cos out 0.929 (`stick_full` 0.908), p01 0.860 (0.779;
  kana_mix 0.811, anchor 0.852): く う と p01 draw the bed and the girl again,
  with the glyph small or wrong (う → ら, く garble). Words flat white
  0.157, 27 / 52 renders ≥ 0.1 whiter than their EN ref (11 of them p02):
  the white boxes.

## 1. The runs

`stick_full`'s recipe (`stick_fit` § 1: warm from `kana_up/trained.pt`, the
166 rows' mean trained alone, 1 992 steps, lr 1e-3 cosine) on `kana_up`'s
data less `lone_*` and `grid_*`: 11 206 of 16 601 items.

| tier | items | `kana_up` band | `stick_nlg_high` |
|---|---|---|---|
| `bubble1_52` | 2 241 | 0.55–0.9 | 0.75–0.95 |
| `bubble1_32` | 3 362 | 0.35–0.7 | 0.75–0.95 |
| `bubbleN_34` | 3 362 | 0.45–0.8 | 0.75–0.95 |
| `bubbleN_18` | 2 241 | 0.2–0.6 | 0.75–0.95 |

`stick_nlg_high`'s band is the run config's `band` (a stick run's data keeps
its stamped bands; `cjk_scale.train(band=…)` replaces them at train, so the
table's 0.9 cap does not apply).

The stick's path (`train_log.json`, cos to kana_up's stick):

| step | 300 | 600 | 900 | 1 200 | 1 500 | 1 975 |
|---|---|---|---|---|---|---|
| `stick_full` | 0.959 | 0.940 | 0.941 | 0.941 | 0.953 | 0.956 |
| `stick_nolonegrid` | 0.939 | 0.915 | 0.911 | 0.915 | 0.919 | 0.921 |
| `stick_nlg_high` | 0.893 | 0.814 | 0.787 | 0.770 | 0.760 | 0.758 |
| `stick_nlg_high` \|m\| | 157.5 | 170.2 | 182.0 | 184.5 | 186.8 | 187.0 |

`stick_nolonegrid` is flat from step 600 at lr 8e-4; `stick_nlg_high` walks
one way until 1 350 and stops as the lr decays. More steps would not move
either; a lower lr would stop `stick_nlg_high` part way along its path.

Train jobs `20261004-123437-fd9bce` (17 min), `…-123653-c81d7c` (14 min);
read jobs `20261004-130614-04991f`, `…-9fb578` (12 min each);
`results/20261004-1318-stick_nolonegrid/`, `20261004-1330-stick_nlg_high/`.

## 2. The read

| | words official / 104 | contained | ≤ 1 | ≤ 2 | ≤ 1c | dup | singles official / 112 | contained | repeats |
|---|---|---|---|---|---|---|---|---|---|
| kana_up | 14 | 34 | 55 | 88 | 88 | 73 | 39 | 93 | 21 |
| `stick_full` | 16 | 32 | 51 | 85 | 87 | 71 | 35 | 98 | 23 |
| `stick_nolone` | 15 | 35 | 54 | 81 | 82 | 71 | 40 | 96 | 25 |
| `stick_nolonegrid` | 11 | 29 | 51 | 79 | 82 | 73 | 32 | 93 | 19 |
| `stick_nlg_high` | 11 | 13 | 28 | 45 | 48 | 65 | 30 | 74 | 19 |
| retrain_kana | 33 | 51 | 73 | 94 | 95 | 52 | 47 | 91 | 7 |

Paired against `stick_full` (gained / lost, p):

| | official | contained | ≤ 1 edit | ≤ 2 edits | dup / repeats |
|---|---|---|---|---|---|
| `stick_nolonegrid`, words | 5 / 10, 0.3 | 7 / 10, 0.63 | 18 / 18, 1.0 | 10 / 16, 0.33 | dup 17 / 15, 0.86 |
| `stick_nolonegrid`, singles | 10 / 13, 0.68 | 4 / 9, 0.27 | — | — | rep 4 / 8, 0.39 |
| `stick_nlg_high`, words | 8 / 13, 0.38 | 7 / 26, 0.001 | 15 / 38, 0.002 | 8 / 48, 5e-8 | dup 14 / 20, 0.39 |
| `stick_nlg_high`, singles | 12 / 17, 0.46 | 3 / 27, 8e-6 | — | — | rep 13 / 17, 0.58 |

Per key (words ≤ 1 edit / singles official, of 8; `stick_full` /
`stick_nolonegrid` / `stick_nlg_high`): `stick_nolonegrid` stays within
± 2 of `stick_full` on every key but り (4 → 0) and ア (6 → 3).
`stick_nlg_high` loses most where the words are long: たすけて 4 / 4 / 0,
なにしてる 5 / 4 / 0, たいせつ 6 / 4 / 1, パソコン 2 / 2 / 0, かんがえ
4 / 4 / 1; it gains on かなしい 2 / 2 / 5 and ことば 5 / 4 / 7.

## 3. The scene

`probe_split`'s numbers on the same renders at seed 0 (`stick_fit.py --legs
scene`, `results/20261004-1331-stick_nlg/`):

| | cos out words / singles / p01 | flat white words / singles / p01 | fw+ words / singles |
|---|---|---|---|
| kana_mix | 0.937 / 0.920 / 0.811 | 0.103 / 0.154 / 0.421 | 11 / 12 |
| kana_up | 0.935 / 0.905 / 0.769 | 0.103 / 0.196 / 0.501 | 11 / 17 |
| `stick_full` | 0.938 / 0.908 / 0.779 | 0.069 / 0.177 / 0.470 | 7 / 14 |
| `stick_nolone` | 0.939 / 0.907 / 0.774 | 0.080 / 0.182 / 0.475 | 7 / 15 |
| `stick_nolonegrid` | 0.933 / 0.898 / 0.740 | 0.126 / 0.202 / 0.529 | 13 / 17 |
| `stick_nlg_high` | 0.934 / 0.929 / 0.860 | 0.157 / 0.189 / 0.430 | 27 / 17 |
| retrain_kana | 0.937 / 0.900 / 0.745 | 0.114 / 0.236 / 0.672 | 11 / 20 |

`fw+`: renders ≥ 0.1 whiter than their EN ref, of 52 / 56. By prompt
(words and singles together) `stick_nlg_high` has p00 3, p01 26, p02 11,
p03 4; every other arm has 0–1 on p02 — the boxes.

Sheets `output/cjk_anima_reseed/stick_nlg/sheets/` (gitignored), overviews
`overview_{singles_p01,words_p00}_{a,b}.png` with both new runs beside
`stick_full` / `stick_nolone`:
- words p00: the four warm re-fits but `stick_nlg_high` keep kana_up's
  banner, colour and slot count; `stick_nlg_high` draws a white rounded box
  on the blackboard, black thin text, the girl smaller (`te い わっ`,
  `ここううええいん`, `ごてんにちちには`).
- singles p01: `stick_nolonegrid` keeps kana_up's white canvas (く も) and
  pasted column (ひ → the whole canvas white); と moves the girl to a
  corner. `stick_nlg_high` brings the bed back on く う と (と a small `とと`
  bubble over a pink bed), ひ keeps the pasted column.

## 4. The stick in row space (`stick_fit.py --legs geo`)

| move from kana_up's stick | \|Δ\| | cos with retrain_kana − kana_up | cos with kana_mix − kana_up |
|---|---|---|---|
| `stick_full` | 45.3 | 0.062 | 0.005 |
| `stick_nolone` | 45.6 | 0.024 | 0.027 |
| `stick_nolonegrid` | 59.7 | 0.293 | 0.022 |
| `stick_nlg_high` | 122.1 | −0.005 | −0.171 |

Moves between runs: `stick_full` · `stick_nolonegrid` 0.32, `stick_nolone` ·
`stick_nolonegrid` 0.35, `stick_nlg_high` to the others 0.04–0.16.
`stick_nolonegrid` moves furthest toward retrain_kana's stick of the four
and renders as `stick_full`; geometry again does not call the render.

## What it leaves open

- **The spikes on scene data**: kana_up's rows, the mean held in the
  optimizer, the spikes trained on the bubble tiers alone (`stick_fit`'s
  open item, with the grids now left out too).
- **Part way along `stick_nlg_high`'s move**: kana_up's spikes on sticks at
  t = 0.25 / 0.5 / 0.75 from kana_up's to `stick_nlg_high`'s (no training) —
  is there a point where the read holds and p01's scene comes back. What a
  lower lr would reach, if the path is straight (no mid-run snapshots).

## Code

- `../cjk_anima_scale/cjk_scale/train.py`: `band` (every kept item's σ band
  replaced at train; experiments only).
- `reseed/config.py`: `band` (stick runs); `_archive/configs/stick_nolonegrid.toml`,
  `_archive/configs/stick_nlg_high.toml`.
- `stick_fit.py`: both runs in `STICK_RUNS`, sheets under
  `output/cjk_anima_reseed/<label>/sheets/`, move cos per pair.
