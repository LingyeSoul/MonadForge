# grad_bands — the bands read off the gradient, trained (2026-10-03)

`../../../cjk_anima_scale/experiments/grid_lone` (`recap_h0`, `recap_hp`) on
`experiments/grad_identity`'s pass 2
(`grad_identity_2026_10_02.md`; the bubble tiers:
`../../../cjk_anima_scale/reports/grad_identity_2026_10_02.md` § 5). The
question, from the night of 10-02: the step-0 gradient gives, per tier, the σ
where a cold row's gradient stops depending on the glyph drawn. Do bands set
from that read train better than the law's per-px bands, on the same items?

**Verdict.**
- **No lower edge loses** (`recap_h0`: every tier at σ from 0 up to its
  identity half point). Under `recap` on every count — singles official
  1 vs 12 of 64, contained 26 vs 44; words ≤ 2 edits 20 vs 35 of 72.
- **With a lower edge it ties the law** (`recap_hp`: the same upper edges,
  the lower edge where the identity gradient is half its peak). Singles
  official 12 vs 12, contained 40 vs 44; words ≤ 1 edit 16 vs 14, ≤ 2 edits
  33 vs 35; no paired count under p 0.1.
- **The gradient read confirms the law's bands and does not improve on
  them.** Its bands are the law's widened downward by 0.05–0.1, with two
  tiers moved more (`bubble1_32`, `lone_28`: 13 % of the items). At
  60 steps / row on 72 words and 64 singles that is not a difference this
  read sees.
- **"There is no lower edge in identity" is not "the lower edge is free."**
  Below its peak the share f holds and the gradient collapses; a band
  without a lower edge spends ≈ 40 % of its draws there.
- Every arm stays far under `retrain_kana` (words ≤ 1 edit 16 vs 45,
  singles official 12 vs 34). The gap is not a band's width.

Sheets were not looked at; every number is a read tally.

## The two edges

From pass 2 (one cold row, the true caption, the glyph in its slot swapped;
medians over 40–60 items a tier, σ 0.2 … 0.95):

- **Upper edge — f's half point.** f = the share of the row's gradient that
  changes with the glyph drawn. Plateau = mean f over σ 0.2–0.4; the half
  point is where f crosses half of it (linear between the σ read). f's
  level is the form's (≈ 0.9 a glyph alone in its box, ≈ 0.6 → 0.4 a
  window, 0.3–0.5 a grid cell), so a fixed threshold does not carry across
  forms: f > 0.2 puts `bubble1_32` at 0.9 and gives `bubble1_52` no edge.
- **Lower edge — ‖I‖ at half its peak, low side.** ‖I‖ = ‖g‖·√f, the
  glyph-dependent gradient's size. f has no lower edge; ‖I‖ does.

| tier | px | f plateau | f half point | ‖I‖ peak (σ) | ‖I‖ half peak, low side | `recap_hp` | the law (`recap`) |
|---|---|---|---|---|---|---|---|
| `grid_16` | 15 | 0.31 | 0.62 | 34.6 (0.4) | < 0.2 (20.0 at 0.2) | 0.2–0.6 | 0.3–0.5 |
| `grid_29` | 28 | 0.37 | 0.70 | 48.5 (0.5) | 0.41 | 0.4–0.7 | 0.5–0.7 |
| `lone_16` | 15 | 0.56 | 0.56 | flat, 2–5 | — | 0.2–0.5 | 0.3–0.5 |
| `lone_28` | 28 | 0.64 | 0.66 | 8.1 (0.75) | 0.36 | 0.35–0.65 | 0.5–0.7 |
| `bubbleN_18` | 19 | 0.41 | 0.51 | 20.8 (0.4) | < 0.2 (11.3 at 0.2) | 0.2–0.5 | 0.3–0.5 |
| `bubbleN_34` | 36 | 0.59 | 0.68 | 26.0 (0.6) | 0.44 | 0.45–0.7 | 0.5–0.7 |
| `bubble1_32` | 32 | 0.88 | 0.61 | 153.8 (0.5) | 0.35 | 0.35–0.6 | 0.5–0.7 |
| `grid_44` | 44 | 0.50 | 0.75 | 46.9 (0.6) | 0.53 | (0.55–0.75) | 0.7–0.9 |
| `lone_44` | 45 | 0.67 | 0.79 | 12.8 (0.8) | 0.45 | (0.45–0.8) | 0.7–0.9 |
| `bubble1_52` | 50 | 0.93 | 0.79 | 123.4 (0.7) | 0.53 | (0.55–0.8) | 0.7–0.9 |

‖I‖ × 1e-3. Edges rounded to 0.05. The last three tiers are not in the
`grid_lone` data and were not trained here. `lone_16` has no ‖I‖ peak and
takes `grid_16`'s lower edge; its upper edge was set at 0.5 (the earlier
eyeball 0.53), under the 0.56 the rule above gives. For the 15–19 px tiers
the lower edge is the lowest σ read, not a measured half peak.

## Setup

Both arms: `reband` on `run1002_grid_lone/data_recap` — the same 8 200
items, plain captions, latents and TE cache, the records differing in `band`
alone (checked) — trained as `recap`: the 81 hiragana + `ー` cold at the
pack rows, 60 steps / row = 4 920, old seed as context, routed, batch 4,
lr 1e-3 cosine. σ is drawn sigmoid-normal inside the item's band
(`train.noisy_by_band`), so a band's draws sit around its middle.

| arm | bands | job | result |
|---|---|---|---|
| `recap` (10-02) | the law's, per px | `20261002-143355-beb03c` | `grid_lone/results/20261002-1519-recap` |
| `recap_h0` | 0 – f half point | `20261002-231259-a6db0c` (54.0 min, with both reads) | `grid_lone/results/20261002-2312-recap_h0` |
| `recap_hp` | ‖I‖ half peak – f half point | `20261003-001816-dbca3d` (51.9 min) | `grid_lone/results/20261003-0018-recap_hp` |

`recap_h0`'s bands: `grid_29` 0–0.7, `grid_16` 0–0.6, `lone_28` 0–0.65,
`lone_16` 0–0.5, `bubbleN_34` 0–0.7, `bubble1_32` 0–0.6, `bubbleN_18`
0–0.5. `20261003-001101-db9ff7` (`recap_hp` with `bubbleN_18` at the law's
0.3–0.5) was stopped 7 min in, unread.

Reads: `grid_lone`'s — 9 hiragana words + 8 singles, 4 prompts × 2 seeds;
the plain wording (`{p}. Text reads as "…".`, the recap items' own) paired
arm against arm on prompt × seed, and the clauses of record (`en` words,
`swap` singles).

## Reads

Plain wording. Pairs are left arm only / right arm only, McNemar p.

| words (72) | official | contained | ≤ 1 edit | ≤ 2 edits | dup | ≤ 1 edit, doubles collapsed |
|---|---|---|---|---|---|---|
| `recap` | 6 | 9 | 14 | 35 | 61 | 37 |
| `recap_hp` | 1 | 4 | 16 | 33 | 56 | 33 |
| `recap_h0` | 1 | 2 | 7 | 20 | 50 | 14 |
| `retrain_kana` | 18 | 28 | 45 | 63 | 44 | 67 |
| `recap_hp` vs `recap` | 1 / 6, 0.12 | 2 / 7, 0.18 | 9 / 7, 0.8 | 8 / 10, 0.81 | 10 / 15, 0.42 | 14 / 18, 0.6 |
| `recap_h0` vs `recap` | 1 / 6, 0.12 | 2 / 9, 0.065 | 4 / 11, 0.12 | 7 / 22, 0.008 | 7 / 18, 0.043 | 2 / 25, 6e-06 |
| `recap_hp` vs `recap_h0` | 1 / 1, 1.0 | 4 / 2, 0.69 | 15 / 6, 0.078 | 21 / 8, 0.024 | 16 / 10, 0.33 | 29 / 10, 0.003 |

| singles (64) | official | contained | kana | repeat |
|---|---|---|---|---|
| `recap` | 12 | 44 | 58 | 15 |
| `recap_hp` | 12 | 40 | 58 | 16 |
| `recap_h0` | 1 | 26 | 52 | 12 |
| `retrain_kana` | 34 | 56 | 61 | 5 |
| `recap_hp` vs `recap` | 8 / 8, 1.0 | 10 / 14, 0.54 | 4 / 4, 1.0 | 7 / 6, 1.0 |
| `recap_h0` vs `recap` | 0 / 11, 0.001 | 3 / 21, 0.0003 | 2 / 8, 0.11 | 7 / 10, 0.63 |
| `recap_hp` vs `recap_h0` | 11 / 0, 0.001 | 22 / 8, 0.016 | 9 / 3, 0.15 | 13 / 9, 0.52 |

The clauses of record:

| | `en` words: official / contained / ≤ 1 / ≤ 2 edits | `swap` singles: official / contained |
|---|---|---|
| `recap` | 0 / 4 / 5 / 13 | 7 / 27 |
| `recap_hp` | 0 / 2 / 4 / 19 | 7 / 26 |
| `recap_h0` | 0 / 2 / 1 / 10 | 0 / 12 |

- **`recap_h0` is under `recap` on every count**, singles most. No single
  improves: く and り go from contained 5 and 4 of 8 to 0, も 8 → 2. Among
  the words only ことば holds (≤ 1 edit 5 / 5); かなしい loses its 4
  official.
- **`recap_hp` ties `recap`.** Words official leans to `recap` (かなしい
  4 → 0 of 8) and ≤ 1 edit the other way (なにしてる 0 → 4); the singles
  move by one or two renders a glyph, both ways.
- **`recap_hp` is over `recap_h0` with the upper edges equal** (singles
  official 11 / 0): what `recap_h0` lost is the lower edge.

## Training trajectories

Window means of the 25-step log, first 500 → last 500 steps. The loss level
is not comparable between bands, only the move inside a run.

| | in-box | out-of-box | Δ norm mean, last 500 | rel |
|---|---|---|---|---|
| `recap` | 0.162 → 0.112 (−31 %) | 0.070 → 0.071 | 197 | 1.04 |
| `recap_hp` | 0.155 → 0.108 (−30 %) | 0.072 → 0.073 | 196 | 1.03 |
| `recap_h0` | 0.136 → 0.118 (−13 %) | 0.094 → 0.098 | 183 | 0.96 |

`recap_hp` follows `recap`. `recap_h0`'s rows moved almost as far while its
in-box loss fell under half as much — `b7593`'s shape
(`grid_small_lone_2026_10_02.md` § Training trajectories), from the low
side.

## Why the lower edge costs

A band (0, 0.7) has its median draw at σ 0.35 and ≈ 40 % of its draws under
0.3. There the identity gradient of a 28–36 px item is a fraction of its
peak:

| ‖I‖ × 1e-3 | σ 0.2 | 0.3 | 0.4 | peak |
|---|---|---|---|---|
| `grid_29` | 3.1 | 6.4 | 21.2 | 48.5 (0.5) |
| `bubbleN_34` | 1.6 | 3.2 | 7.4 | 26.0 (0.6) |
| `bubble1_32` | 13.7 | 33.7 | 115.1 | 153.8 (0.5) |
| `lone_28` | 1.1 | 2.9 | 4.8 | 8.1 (0.75) |

Those four tiers are 58 % of the items. The 15–19 px tiers peak at 0.4 and
lose little (`grid_16` 20.0 at σ 0.2, 34.6 at its peak). At 60 / row the
draws under the peak are draws the rows did not get — the gradient read's
"a band's lower edge is an exposure edge", paid in training.

## What it does not show

- **The upper edges alone.** `recap_h0` and `recap_hp` share them; against
  `recap` they move three tiers (`bubble1_32` 0.7 → 0.6, `lone_28` 0.7 →
  0.65, `grid_16` 0.5 → 0.6). No arm changes only those.
- **The two tiers that moved**, read apart: 1 107 of 8 200 items; the read
  is 72 words and 64 singles, none of them per tier.
- **The 44–50 px tiers.** The read puts `grid_44` and `lone_44` 0.1–0.15
  under the law's 0.7–0.9; untrained here.
- **A longer budget.** 60 / row; `recap` itself is far under `retrain_kana`.
- **Bands below σ 0.2**: the gradient was not read there.

## What it changes in `motivation2.md`

- § 6's open item — "the same 15–28 px items at two low bands, the law's
  against windows on the identity peak" — is read: a tie.
- Verdict 1 ("the seed's bands were right for its glyph sizes") holds on
  cold rows and small glyphs from the other side: moving the bands to
  where the gradient says identity is does not beat them.

## Open

- The upper edge alone on the tiers the read moves most: `grid_44` /
  `lone_44` at 0.55–0.75 against 0.7–0.9 on `grid_44`'s items
  (`run1002_grid_44`), where the law's band sits one step above the
  identity peak.
- A σ density rather than an edge: draws weighted to the ‖I‖ peak inside
  the law's band. Unbuilt.

Code: `../../../cjk_anima_scale/experiments/grid_lone/run_exp.py` (`--legs
reband train read read_plain --tag recap_h0 | recap_hp`; the bands are
`BANDS`).
