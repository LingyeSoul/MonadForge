# probe_split: kana_up's +0.1 split by σ and by stick / spikes (2026-10-04)

From the user (10-04, after `kana_up_2026_10_04.md`): the +0.1 bought text and
cost the scene, dup and the "fill the blank / white canvas" look stay — what
causes each, and can it be separated without training? `probe_split.py`
swaps rows at render: `kana_up` and `kana_mix` differ only in their 166 kana
rows.

**Verdict.**
- **Neither split separates the text gain from the scene cost.** Rows
  switched at σ 0.75: each arm is the run above the switch on every count.
  At 0.85: kana_mix above keeps kana_mix's scene *and* its text (words ≤ 1
  edit 17 vs kana_up 29, paired 4 / 16, p 0.012); kana_up above takes all of
  kana_up's scene cost and half its text gain (22).
- **The gain is mostly layout.** Above 0.85 the run sets the text's colour,
  size, position and slot count. kana_mix's layout is a red top banner with
  more slots than the word; kana_up's rows below 0.85 write into it and
  stutter (`たさいいそつつ`). kana_up's own layout is fewer, larger slots,
  so the word fits (≤ 1 edit after collapsing repeats 34 → 46). The same
  move is the scene cost: on p01 a large black block across the duvet, the
  scene washed white.
- **Dup is slot count against glyph size.** x̂0 per σ: the text region is
  placed by σ 0.95, before anything reads (≤ 7 / 52 renders read at 0.9);
  glyph size and identity land at 0.85–0.8, and so do the repeats (kana_mix
  dup 4 → 15 → 28 across 0.9 / 0.85 / 0.8). 42–43 / 52 final reads are
  longer than the word in both runs; dup 33 in both.
- **The white canvas is the lone tier's picture, on singles and p01.**
  EN refs have no flat white on p01; every reseed arm makes 15–22 / 27 p01
  renders ≥ 0.1 whiter than their ref. `く` on p01 under kana_up rows above
  0.75: the whole canvas white, one centred glyph (cos out 0.52, flat white
  0.97) — a glyph alone on a blank page, plain MSE over the canvas, which
  kana_up trained up to 0.9 (`lone_44`). Singles' flat white rises with the
  high-σ lone training: kana_mix 0.154, kana_up 0.196, retrain_kana 0.236;
  words stay at 0.103. Not trained out, so not shown to be the tier.
- **Stick and spikes alone read nothing**, as `stick_2026_10_03.md`'s s = 0
  predicts. The stick alone (`up_mean`) keeps part of the white (singles
  0.128, a white sign on p00); the spikes alone (`up_nomean`) keep almost none
  (0.042) and the best scene (cos out 0.959). The large scene loss needs
  both: a glyph drawn large.
- **The stick's direction carries the word fit (§ 4).** kana_up's spikes on
  retrain_kana's stick (cos 0.93 to kana_up's, |m| 147 vs 141): words
  official 4 → 13 (10 / 1, p 0.012), dup 36 → 23 (4 / 17, p 0.007) —
  retrain_kana's 20 / 22 — and the scene cost comes with it (singles p01 cos
  out 0.746, retrain_kana 0.745). `stick_2026_10_03.md`'s "the deficit is in
  the spikes, not the stick" read the stick's length (± 10 %); its direction
  was not read.

## 1. Setup

The plain read's grid (`run.py read`: 13 words + 14 singles × p00–p03,
`PLAIN_CLAUSE`) at **seed 0 only** (user, 10-04: not all 216): 108 renders an
arm, paired against the runs' own cached renders. Rows split as in
`../cjk_anima_scale/experiments/sigma_split` (`context_alt` +
`tag_drop_sigma`, negative pass untouched). Plumbing: kana_up on both sides
through the split path vs its cached render, mean |Δpx| 0.000 (first job) /
2.20 / 255.

| arm | above the switch | below |
|---|---|---|
| `up_mix_s<σ>` | kana_up | kana_mix |
| `mix_up_s<σ>` | kana_mix | kana_up |
| `up_mean` | kana_up's 166 rows all = their mean (the stick), every σ | |
| `up_nomean` | kana_up less that mean (the spikes), every σ | |

The mean of kana_up − kana_mix is 1.6 % of the difference's energy, so the
stick / spike split is of kana_up's own rows (mean energy 0.31; kana_mix
0.30; the two sticks at cos 0.985, |m| 141 / 138, rows 0.70).

Jobs `20261004-094909-f449e8` (0.75 + stick / spikes + traj, 33 min) and
`20261004-104503-922c3f` (0.85, 13 min); `results/20261004-1013-s075/`,
`20261004-1022-s075-traj/`, `20261004-1057-s085/`; renders under
`output/cjk_anima_reseed/probe_split/`.

## 2. The read

Flat white: share of 16² patches with std < 6 and mean > 225
(`sigma_split.placement`); EN refs p00–p03 0.117 / 0 / 0 / 0.027. `fw+ p01`:
p01 renders ≥ 0.1 whiter than their EN ref, of 27.

| arm | words ≤ 1 / ≤ 2 | ≤ 1c | dup / 52 | singles official / 56 | repeats | cos out words / singles / p01 | flat white singles | fw+ p01 |
|---|---|---|---|---|---|---|---|---|
| kana_mix | 15 / 37 | 34 | 36 | 20 | 12 | 0.937 / 0.920 / 0.811 | 0.154 | 17 |
| kana_up | 29 / 42 | 46 | 36 | 23 | 8 | 0.935 / 0.905 / 0.769 | 0.196 | 20 |
| `up_mix_s0.75` | 28 / 41 | 43 | 34 | 24 | 7 | 0.936 / 0.905 / 0.766 | 0.195 | 19 |
| `mix_up_s0.75` | 15 / 37 | 35 | 40 | 18 | 10 | 0.935 / 0.920 / 0.811 | 0.155 | 17 |
| `up_mix_s0.85` | 22 / 40 | 40 | 34 | 22 | 6 | 0.936 / 0.905 / 0.766 | 0.196 | 19 |
| `mix_up_s0.85` | 17 / 31 | 36 | 36 | 19 | 7 | 0.936 / 0.919 / 0.811 | 0.155 | 17 |
| `up_mean` | 0 / 0 | 0 | — | 0 | 0 | 0.957 / 0.940 / 0.879 | 0.128 | 7 |
| `up_nomean` | 0 / 0 | 0 | — | 0 | 0 | 0.955 / 0.959 / 0.915 | 0.042 | 5 |
| kana_big | 15 / 27 | 31 | 37 | 18 | 11 | 0.938 / 0.923 / 0.817 | 0.164 | 16 |
| anchor | 19 / 38 | 38 | 41 | 15 | 14 | 0.929 / 0.934 / 0.852 | 0.155 | 19 |
| recap hp | 17 / 34 | 36 | 41 | 17 | 9 | 0.933 / 0.934 / 0.867 | 0.165 | 15 |
| retrain_kana | 39 / 50 | 49 | 22 | 24 | 2 | 0.937 / 0.900 / 0.745 | 0.236 | 22 |

`up_mean` / `up_nomean` dup counts doubled characters in reads of garble, not
of the word; left out.

Paired, words ≤ 1 edit (gained / lost, McNemar p):

| | vs kana_up | vs kana_mix |
|---|---|---|
| `up_mix_s0.75` | 1 / 2, 1.0 | 17 / 4, 0.007 |
| `mix_up_s0.75` | 2 / 16, 0.001 | 3 / 3, 1.0 |
| `up_mix_s0.85` | 0 / 7, 0.016 | 13 / 6, 0.17 |
| `mix_up_s0.85` | 4 / 16, 0.012 | 6 / 4, 0.75 |

## 3. x̂0 per σ (`--legs traj`)

kana_mix / kana_up rows at every σ, the 13 words × p00–p03 at seed 0, x̂0
decoded at the step nearest each σ and read (sfx, longest non-whole box).

| σ | 0.95 | 0.9 | 0.85 | 0.8 | 0.7 | 0 |
|---|---|---|---|---|---|---|
| kana_mix: any read / longer than the word / dup / ≤ 1 | 0 / 0 / 0 / 0 | 7 / 4 / 4 / 1 | 33 / 18 / 15 / 8 | 45 / 30 / 28 / 13 | 52 / 39 / 33 / 14 | 52 / 43 / 33 / 12 |
| kana_up | 1 / 0 / 0 / 0 | 2 / 1 / 1 / 1 | 34 / 20 / 11 / 16 | 47 / 31 / 22 / 25 | 52 / 42 / 30 / 25 | 52 / 42 / 33 / 27 |

On `traj_こんにちは`: the banner's width and place are in x̂0 at 0.95 for
both runs; kana_mix's glyphs come in smaller and the banner takes 6–7 of
them (`こんにこちちは`, `こんにに ち は`), kana_up's larger, 5–6.

## 4. The stick swapped (`up_rkstick`)

kana_up's rows less their mean plus retrain_kana's mean over the same 166
ext ids (in kana_up's row units), every σ. The sticks (delta units, 166 kana
rows):

| | \|stick\| | stick energy | cos vs kana_mix | cos vs retrain_kana |
|---|---|---|---|---|
| kana_up | 140.9 | 0.309 | 0.985 | 0.931 |
| kana_mix | 138.4 | 0.298 | — | 0.906 |
| kana_big | 142.1 | 0.302 | 0.989 | 0.886 |
| anchor | 137.5 | 0.291 | 0.977 | 0.889 |
| recap hp | 141.1 | 0.297 | 0.981 | 0.894 |
| retrain_kana | 147.4 | 0.337 | 0.906 | — |
| 0921 seed (162) | 143.6 | 0.282 | 0.802 | 0.725 |

The reseed sticks agree to 0.97–0.99; the +0.1 turned kana_up's toward
retrain_kana's (up − mix · retrain − mix cos 0.52; kana_big's move −0.15).

| arm | words official / contained / ≤ 1 / ≤ 1c | dup | singles official | cos out singles / p01 | flat white words / singles |
|---|---|---|---|---|---|
| kana_mix | 2 / 10 / 15 / 34 | 36 | 20 | 0.920 / 0.811 | 0.103 / 0.154 |
| kana_up | 4 / 17 / 29 / 46 | 36 | 23 | 0.905 / 0.769 | 0.103 / 0.196 |
| `up_rkstick` | 13 / 21 / 33 / 42 | 23 | 21 | 0.898 / 0.746 | 0.131 / 0.206 |
| retrain_kana | 20 / 28 / 39 / 49 | 22 | 24 | 0.900 / 0.745 | 0.114 / 0.236 |

Paired against kana_up, words: official 10 / 1 (p 0.012), dup 4 / 17
(p 0.007), ≤ 1 edit 9 / 5 (p 0.42); singles flat (official 3 / 5).

On the sheets: `こんにちは` exact on p00–p02 in retrain_kana's rounded pink
banner, its slot count the word's; the bed scene on p01 kept. `パソコン`
moves nothing (every arm weak; p01 a large word on a white duvet).

Job `20261004-110151-f1ef6f` (7 min, plumbing |Δpx| 3.6 / 255),
`results/20261004-1108-rkstick/`.

## What it leaves open

- Whether the white canvas is the lone tier: a micro arm with kana_up's
  `lone_*` bands back at kana_mix's, every other tier as kana_up.
- The stick at 0.8 (preview5's lever) on kana_up rows: no ruler read on the
  reseed rows yet (`up_s08`, no training); likewise retrain_kana's stick
  × 0.8, the stick half way between the two, and kana_mix's spikes on
  retrain_kana's stick (is the gain the stick's whatever the spikes).
- Seed 0 only: 52 word / 56 single renders an arm.
