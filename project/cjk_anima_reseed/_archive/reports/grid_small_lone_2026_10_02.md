# grid_small / grid_lone — hiragana cold on small glyphs only (2026-10-02)

`../../../cjk_anima_scale/experiments/grid_small` (`r0`, `b7593`) and
`experiments/grid_lone` (`r0`, `recap`).
The question, from `motivation2.md` verdict 4: can the hiragana rows be
seeded cold with **no glyph above ≈ 40 px**, the kana run's large lone tier
(`b0709`, σ 0.7–0.9) replaced by grids of small glyphs.

**Verdict: not in this shape.** Every arm reads under `retrain_kana` on the
same prompts × seeds, on words and on singles.

- `grid_small r0` (per-px bands, 135 steps / row): words ≤ 1 edit 12 vs 33 of
  72, singles contained 31 vs 44 of 64.
- `grid_small b7593` (the same items, every one at σ 0.75–0.93): **0** on
  every count. The in-box loss barely moved in training.
- `grid_lone r0` (lone 1×1 small glyphs added, 60 steps / row): words ≤ 1
  edit 14 of 72, singles contained 23 of 64 — no better than `grid_small r0`.
  Budget and composition changed together, so the 1×1 tier is not isolated.
- `grid_lone recap` (the same items, plain grid captions): words fall
  further — ≤ 1 edit 5 of 72, ≤ 2 edits 13 against `r0`'s 33 — and singles
  hold (contained 27 vs 23). The word read asks with `Japanese text reads
  as`, the clause `r0`'s grid items carry and `recap`'s do not.
- **Asked in the plain wording (`{p}. Text reads as "…".`), `recap` and `r0`
  read the same** (words ≤ 1 edit 14 vs 19 of 72, p 0.27), and every arm
  reads better than under its clause of record — `retrain_kana` most:
  words ≤ 1 edit 33 → 45, singles official 17 → 34. The training caption
  did not matter; the asking clause did (§ The plain read).
- `grid_lone recap_h0` (10-03; `recap`'s items, every tier at σ from 0 up to
  its identity half point, no lower edge): **under `recap`** on the plain
  read — singles official 1 vs 12, contained 26 vs 44 of 64; words ≤ 2 edits
  20 vs 35 of 72 (`grad_bands_2026_10_03.md`).
- `grid_lone recap_hp` (10-03; the same upper edges, the lower edge where
  the identity gradient is half its peak): **ties `recap`** — singles
  official 12 vs 12, contained 40 vs 44; words ≤ 1 edit 16 vs 14, ≤ 2 edits
  33 vs 35 — and reads over `recap_h0`. The bands read off the gradient buy
  nothing over the law's; the lower edge is what `recap_h0` lacked.
- The scene is kept: EN-reference cos equals the kana run's on words and is
  higher on singles. What is lost is the glyph, not the picture.

Sheets were sent to the user and are not described here; the one render
note below (§ b7593) is from a 12-image peek.

## Setup

All arms: rows cold from the pack rows, old seed (`rows_step1_0921_merged`)
as context, routed, batch 4, lr 1e-3 cosine, warmup 0.1, ≈ 2.4 it/s. The
read is `kana_reband`'s: 9 hiragana words (`en` clause) + 8 singles (`swap`
clause), 4 prompts × 2 seeds, paired against `retrain_kana`'s reads of
record.

| arm | rows | steps | bands | job | result |
|---|---|---|---|---|---|
| `grid_small r0` | 81 hiragana | 135 / row = 10 935 | per px: 0.3–0.5, 0.5–0.7 | `20261002-112155-8e8225` (87.9 min) | `grid_small/results/20261002-1121-r0` |
| `grid_small b7593` | 81 hiragana | 10 935 | one band 0.75–0.93 | `20261002-121029-24622c` (84.9 min) | `grid_small/results/20261002-1249-b7593` |
| `grid_lone r0` | 81 hiragana + `ー` | 60 / row = 4 920 | per px | `20261002-143355-1c397b` (45.9 min) | `grid_lone/results/20261002-1433-r0` |
| `grid_lone recap` | as `r0` | 4 920 | per px | `20261002-143355-beb03c` (44.0 min) | `grid_lone/results/20261002-1519-recap` |
| `grid_lone recap_h0` | as `r0` | 4 920 | per tier: 0 – half point | `20261002-231259-a6db0c` (54.0 min, with both reads) | `grid_lone/results/20261002-2312-recap_h0` |
| `grid_lone recap_hp` | as `r0` | 4 920 | per tier: ‖I‖ half peak – half point | `20261003-001816-dbca3d` (51.9 min, with both reads) | `grid_lone/results/20261003-0018-recap_hp` |

### Data

The tables name the pools as the data dirs do (band group / recipe). The
tiers' names since the evening of 10-02, by form and median glyph px:
`g0507` → `grid_29`, `g0305` → `grid_16`, `l0507` → `lone_28`, `l0305` →
`lone_16`, `b0507` `scene_window` → `bubbleN_34`, `b0507`
`scene_single_small` → `bubble1_32`, `b0305` `scene_window` →
`bubbleN_18`; `retrain_kana`'s `b0709` → `bubble1_52` (`scene_single`),
`grid_82` (grids), `lone_190` (flat 1×1). Table:
`../../../cjk_anima_scale/README.md` § Item pools.

`grid_small` (8 100 items, `run1002_grid_small/data`; `b7593` is the same
records with every band replaced):

| group / tier | share | glyph px (min / median / max) | glyphs / item |
|---|---|---|---|
| `g0507` grid 2×2–3×3 | 30 % | 24 / 29 / 36.5 | 6.3 |
| `g0305` grid 2×2–3×3 | 30 % | 12 / 17 / 23 | 6.3 |
| `b0305` scene_window | 20 % | 12 / 18 / 24 | 4.5 |
| `b0507` scene_window | 14 % | 24 / 34 / 64 | 4.5 |
| `b0507` scene_single_small | 6 % | 24 / 32 / 40 | 1.0 |

No 1×1, no flat lone glyph, nothing of `b0709`. `retrain_kana` for
comparison (17 400 items): 33 % of its items are `b0709` at σ 0.7–0.9 —
`scene_single` 17 % (40 / 52 / 136 px), grids 13 % (40 / 82 / 179 px), flat
1×1 3 % (52 / 191 / 383 px).

`grid_lone` (8 200 items, `run1002_grid_lone/data`): `g0305` cut to 15 % and
the 15 % given to lone 1×1 items at the grids' font px, bare or one bubble,
glyph at the canvas centre ± 8 %.

| group / tier | σ | share | items |
|---|---|---|---|
| `g0507` grid | 0.5–0.7 | 30 % | 2 460 |
| `g0305` grid | 0.3–0.5 | 15 % | 1 230 |
| `l0507` 1×1, font 28–42 px | 0.5–0.7 | 7.5 % | 615 |
| `l0305` 1×1, font 14–26 px | 0.3–0.5 | 7.5 % | 615 |
| `b0507` scene | 0.5–0.7 | 20 % | 1 640 |
| `b0305` scene | 0.3–0.5 | 20 % | 1 640 |

Exposure in `grid_small`: 8 100 items over 10 935 steps × 4 ≈ 5.4 epochs; a
row sits in 536 items on average (377 `ぅ` – 918 `っ`), ≈ 2 900 item draws
over the run.

### The gate drops thin and small glyphs from a lone tier

The builder's gate measures an item by its ink box, √(box area / glyphs). A
lone glyph is measured by its own box, so a thin or small one reads a band
below its font size and is re-drawn away. Ink box ÷ font px, measured on the
grid cells (median glyph 0.83):

| row | ratio | first build: `l0507` / `l0305` items |
|---|---|---|
| `ー` | 0.38 | 0 / 0 |
| `っ` | 0.58 | 1 / 3 |
| `ぅ` | 0.60 | 0 / 5 |
| `ょ` | 0.61 | 1 / 2 |
| `ぃ` | 0.61 | 2 / 3 |
| `げ` | 0.91 | 10 / 10 |

A pass rate predicted from the ratio alone correlates 0.89 (`l0507`) and
0.75 (`l0305`) with the counts. A grid item averages its cells, so the same
glyphs survive there (`ー` still thinner: 147 items in `g0507` against a
median 190). Fix: the lone tiers take the group's band ungated
(`builder` tier param `gate = "group"`), glyphs drawn at the grid tiers'
font px. Rebuilt: all 82 rows in both tiers, 4–10 items each. Both
`grid_lone` arms trained on the rebuilt data.

## Reads

`off` = both readers exact, `cont` = contained, `≤1` / `≤2` = edit distance,
`dup` = a doubled glyph, `≤1c` = ≤ 1 edit after collapsing doubles.

### Words — `en`, 9 words × 8 = 72

| | off | cont | ≤1 | ≤2 | dup | ≤1c |
|---|---|---|---|---|---|---|
| `retrain_kana` | 10 | 22 | 33 | 53 | 47 | 53 |
| `grid_small r0` | 1 | 8 | 12 | 38 | 58 | 40 |
| `grid_small b7593` | 0 | 0 | 0 | 0 | 33 | 0 |
| `grid_lone r0` | 3 | 7 | 14 | 33 | 52 | 33 |
| `grid_lone recap` | 0 | 4 | 5 | 13 | 57 | 14 |

Paired against `retrain_kana` (arm only / kana only, McNemar p):

| | off | cont | ≤1 | ≤2 |
|---|---|---|---|---|
| `grid_small r0` | 1 / 10, 0.012 | 3 / 17, 0.0026 | 4 / 25, 0.0001 | 5 / 20, 0.0041 |
| `grid_lone r0` | 2 / 9, 0.065 | 3 / 18, 0.0015 | 6 / 25, 0.0009 | 7 / 27, 0.0008 |
| `grid_lone recap` | 0 / 10, 0.002 | 3 / 21, 0.0003 | 1 / 29, 6e-08 | 2 / 42, 1e-10 |

C2's eight words from the caches (of 64): `p1_mix` ≤ 1 edit 34, `p1_cold`
26. Both grid arms are under `p1_cold` (≤ 1 edit 12 and 14 on all nine
words).

### Singles — `swap`, 8 glyphs × 8 = 64

| | off | cont | read as kana |
|---|---|---|---|
| `retrain_kana` | 17 | 44 | 52 |
| `grid_small r0` | 9 | 31 | 45 |
| `grid_small b7593` | 0 | 0 | 20 |
| `grid_lone r0` | 8 | 23 | 40 |
| `grid_lone recap` | 7 | 27 | 43 |

Per glyph, off / cont of 8:

| | あ | う | が | く | と | ひ | も | り |
|---|---|---|---|---|---|---|---|---|
| `retrain_kana` | 5 / 7 | 4 / 8 | 3 / 3 | 0 / 5 | 3 / 8 | 0 / 5 | 2 / 5 | 0 / 3 |
| `grid_small r0` | 2 / 4 | 2 / 6 | 0 / 0 | 0 / 5 | 0 / 2 | 1 / 2 | 4 / 7 | 0 / 5 |
| `grid_lone r0` | 3 / 5 | 4 / 5 | 0 / 0 | 0 / 2 | 0 / 1 | 0 / 3 | 1 / 5 | 0 / 2 |
| `grid_lone recap` | 0 / 2 | 3 / 5 | 3 / 4 | 0 / 4 | 0 / 2 | 0 / 2 | 1 / 5 | 0 / 3 |

`が` and `と` are gone in `grid_small r0` and `grid_lone r0`; `recap` has
`が` back at the kana run's count (3 / 4) and loses `あ` (0 / 2). `も` is
the one glyph a grid arm reads better than the kana run (`grid_small r0`),
and it does not repeat in either `grid_lone` arm.

### EN reference — cos to the `hi` render of the same prompt × seed

Means over the same keys; `retrain_kana` from its `native_reads.json`.

| | words: en cos | cos out | box IoU | CER sfx | singles: en cos | cos out | box IoU | CER sfx |
|---|---|---|---|---|---|---|---|---|
| `retrain_kana` | 0.924 | 0.922 | 0.19 | 0.46 | 0.922 | 0.916 | 0.28 | 0.42 |
| `grid_small r0` | 0.927 | 0.924 | 0.27 | 0.65 | 0.947 | 0.941 | 0.38 | 0.70 |
| `grid_small b7593` | 0.873 | 0.877 | 0.07 | 0.99 | 0.957 | 0.952 | 0.46 | 1.00 |
| `grid_lone r0` | 0.923 | 0.923 | 0.27 | 0.70 | 0.955 | 0.952 | 0.40 | 0.75 |
| `grid_lone recap` | 0.936 | 0.934 | 0.31 | 0.83 | 0.962 | 0.959 | 0.44 | 0.78 |

The small-glyph arms sit as close to the EN reference as the kana run on
words and closer on singles, with the glyph box nearer the EN word's box.
`b7593` has the highest singles cos and reads nothing: the cos says the
scene is the EN render's, not that the glyph is right.

## The plain read

`grid_lone`'s `read_plain` leg (jobs `20261002-162623-8a8fee`,
`20261002-162953-43fb22`; results `20261002-1626-plain`,
`20261002-1641-plain_kana`): the same words, singles, prompts and seeds
asked with `{p}. Text reads as "…".` — no `japanese text` tag, no language
in the clause. Renders in each arm's `native_r4_plain/`; `retrain_kana` was
rendered for it (136 images). "Of record" below is `en` for words and `swap`
(an English clause) for singles, so the singles' move mixes dropping the tag
with dropping the English clause.

Words, 72:

| arm | clause | off | cont | ≤1 | ≤2 | dup | ≤1c |
|---|---|---|---|---|---|---|---|
| `retrain_kana` | `en` | 10 | 22 | 33 | 53 | 47 | 53 |
| `retrain_kana` | plain | 18 | 28 | 45 | 63 | 44 | 67 |
| `grid_lone r0` | `en` | 3 | 7 | 14 | 33 | 52 | 33 |
| `grid_lone r0` | plain | 3 | 10 | 19 | 38 | 58 | 35 |
| `grid_lone recap` | `en` | 0 | 4 | 5 | 13 | 57 | 14 |
| `grid_lone recap` | plain | 6 | 9 | 14 | 35 | 61 | 37 |

Singles, 64:

| arm | clause | off | cont | read as kana | rep |
|---|---|---|---|---|---|
| `retrain_kana` | `swap` | 17 | 44 | 52 | 2 |
| `retrain_kana` | plain | 34 | 56 | 61 | 5 |
| `grid_lone r0` | `swap` | 8 | 23 | 40 | 3 |
| `grid_lone r0` | plain | 16 | 43 | 59 | 11 |
| `grid_lone recap` | `swap` | 7 | 27 | 43 | 5 |
| `grid_lone recap` | plain | 12 | 44 | 58 | 15 |

Paired under the plain clause (first only / second only, McNemar p):

| | words ≤1 | words off | singles off | singles cont |
|---|---|---|---|---|
| `recap` vs `r0` | 4 / 9, 0.27 | 5 / 2, 0.45 | 5 / 9, 0.42 | 9 / 8, 1.0 |
| `r0` vs `retrain_kana` | 8 / 34, 7e-05 | 3 / 18, 0.0015 | 3 / 21, 0.0003 | 2 / 15, 0.0023 |
| `recap` vs `retrain_kana` | 3 / 34, 1e-07 | 6 / 18, 0.023 | 1 / 23, 3e-06 | 1 / 13, 0.0018 |

EN reference and the text's area (means; area = the detector boxes' summed
share of the canvas, computed here from `native_reads.json`):

| arm | clause | words: en cos | box IoU | CER sfx | text area | singles: en cos | box IoU | CER sfx | text area |
|---|---|---|---|---|---|---|---|---|---|
| `retrain_kana` | of record | 0.924 | 0.19 | 0.46 | 0.158 | 0.922 | 0.28 | 0.42 | 0.061 |
| `retrain_kana` | plain | 0.946 | 0.30 | 0.35 | 0.107 | 0.921 | 0.14 | 0.28 | 0.038 |
| `grid_lone r0` | of record | 0.923 | 0.27 | 0.70 | 0.113 | 0.955 | 0.40 | 0.75 | 0.080 |
| `grid_lone r0` | plain | 0.944 | 0.28 | 0.66 | 0.097 | 0.951 | 0.26 | 0.66 | 0.050 |
| `grid_lone recap` | of record | 0.936 | 0.31 | 0.83 | 0.118 | 0.962 | 0.44 | 0.78 | 0.069 |
| `grid_lone recap` | plain | 0.952 | 0.33 | 0.66 | 0.081 | 0.955 | 0.27 | 0.70 | 0.040 |

- The plain clause shrinks the text's area in every arm (words 0.158 →
  0.107 on the kana run) and moves the words' renders toward the EN
  reference (en cos +0.02 in all three). This is `findings.md` § 3 again
  ("the `japanese text` tag sets a text area beyond the quoted word"), now
  on nine 3–5 glyph words and three arms.
- The doubled glyph does not follow the area: `dup` 47 → 44, 52 → 58,
  57 → 61. What the kana run gains is outside the doubling (≤ 1 edit after
  collapsing doubles 53 → 67).
- `recap`'s loss on the `en` read was the clause, not the rows: under the
  plain clause it reads as `r0` does.

## Training trajectories

Window means of the 25-step log. The loss level is not comparable between
bands (`b7593` is measured at σ 0.75–0.93), only the move inside a run.

| | in-box, first 500 → last 500 | out-of-box | Δ norm mean, last 500 | rel |
|---|---|---|---|---|
| `grid_small r0` | 0.162 → 0.100 (−38 %) | 0.068 → 0.069 | 240 | 1.27 |
| `grid_small b7593` | 0.185 → 0.160 (−13 %) | 0.104 → 0.104 | 243 | 1.28 |
| `grid_lone r0` | 0.162 → 0.108 (−34 %) | 0.070 → 0.071 | 195 | 1.03 |
| `grid_lone recap` | 0.162 → 0.112 (−31 %) | 0.070 → 0.071 | 197 | 1.04 |
| `grid_lone recap_h0` | 0.136 → 0.118 (−13 %) | 0.094 → 0.098 | 183 | 0.96 |
| `grid_lone recap_hp` | 0.155 → 0.108 (−30 %) | 0.072 → 0.073 | 196 | 1.03 |

- `b7593`'s rows moved as far as `r0`'s (Δ norm 243 vs 240) while its in-box
  loss fell a third as much; it sat at ≈ 0.175 from step 1 000 to 2 500.
  This is `motivation2.md` verdict 2 once more, cold: items trained above
  where their glyphs resolve move the rows without teaching the glyph.
- `grid_lone r0` stops at a smaller Δ norm (195): 45 % of `r0`'s steps.
- `grid_lone recap` follows `r0`'s curve: in-box 0.001–0.004 above it in
  the windows compared (to step 2 725, and the last 500: 0.112 vs 0.108),
  the same out-of-box loss and Δ norm.

### `b7593`, the 12-image peek

Prompt p00, six words × two seeds, beside `r0`'s renders. `r0` writes a
large coloured banner close to the word with doubled glyphs (`こんにちはは`,
`かななししい`). `b7593` writes small text in bubbles, white panels and
columns — a manga layout — but the content is pseudo-kanji scribble, and no
cell shows the word.

## The plain caption (`recap`)

User, 10-02: the grid captions lose `manga`, the `japanese text` tag and the
clause's language, and the clause's wording varies per item.

- bubble: `simple background, no humans, multiple speech bubbles. On the top
  left, text reads as "ご". On the top right, …`
- bare: `white background, simple background, no humans, text focus. On the
  top left, text reads as "わ". … In the center, …`
- 1×1: no position header, one bubble — `simple background, no humans,
  speech bubble. Text reads as "ぉ".`
- wordings (`common.prompts.GRID_CLAUSES`, one per item, uniform):
  `text reads as "…"`, `the text says "…"`, `"…" is written`, `"…"`, and on
  bubble items `a speech bubble reads "…"`. The record keeps the draw
  (`clause`).

`recap` is a derive leg: the same images, latents and order, the TE cache
its own. `grid_lone recap` is the only arm trained on it; a
`grid_small` arm with the plain captions at 0.75–0.93 was started and
stopped 8 min in, unread (its control, `b7593`, read 0).

## The bands read off the gradient (`recap_h0`, `recap_hp`, 10-03)

Two `reband` arms on `recap`'s items, bands from `grad_identity`'s pass 2:
`grad_bands_2026_10_03.md` holds the bands, reads and pairs. `recap_h0`
(0 – the identity half point, no lower edge) reads under `recap`;
`recap_hp` (the lower edge at half the identity peak) ties it.

## Open

- The reads of record ask with `japanese text`. The kana run reads better
  without it (words ≤ 1 edit 33 → 45 of 72); whether that holds on the seed
  (`findings.md` § 3: no direction over seven captions) and on kanji is
  unread.
- `grid_lone` changed rows (+ `ー`), budget (135 → 60 / row) and composition
  at once. An arm at 135 / row would separate the 1×1 tier from the budget.
- No arm here has a glyph above 64 px or an item at σ 0.7–0.9 on a large
  glyph. `p1_cold` read ≤ 1 edit 26 / 64 on C2's words with in-word items
  alone; the grid arms, which add grids to those, read lower on the same
  words. What separates them (the grids' share, 81 rows against 36, the
  budget) is unread.
