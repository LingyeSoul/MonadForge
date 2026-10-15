# Item pools renamed by glyph px (2026-10-02)

Until today an item pool was named by its band group and its recipe
(`b0507` / `scene_window`): the name said which σ band the pool was bound
to. The user asked for the rename (10-02: the band-binding theory looks
stale — today's arms move the same pools between bands), so the name now
says what does not move: the form of the item and the glyph size it was
built at.

**The rename took the band out of the names and nowhere else.** The binding
of px to band is still what the code runs:

- a tier is drawn in a group that carries one band (`builder.Group.band`);
- the builder gates each item on `windows.window(kind, px, layout)` — the
  band law — and stamps the band on the record;
- the trainer draws each item's σ inside its own band
  (`train.noisy_by_band`).

A build with no override trains per px band, as before the rename. An arm
trains at another band only through a `reband` / `recap` leg's
`--band lo hi`, which rewrites every record's `band` after the build
(`grid_small b7593`, `grid_44 recap_b7593`). The user read the rename as the
per-band data mix being gone (10-02, evening); it is not. Whether the
binding leaves the builder and the trainer is open (§ Open).

Code: `../../../cjk_anima_scale/cjk_scale/{builder,recipes,conflict}.py`. The
live table is `../../../cjk_anima_scale/README.md` § Item pools.

## The name

`<form>_<px>`. The form is what is drawn; the px is the median ink px
(√(box area / glyphs)) of the pool's items as built on the kana run.

| form | what |
|---|---|
| `lone` | one glyph alone on its canvas (1×1, bare or one bubble) |
| `grid` | 2×2 – 3×3 cells, one glyph per cell |
| `bubble1` | one glyph in a bubble of a generated scene |
| `bubbleN` | a 2–6 glyph window in a scene bubble, routed per glyph |
| `piece_bubble`, `piece_grid`, `line_bubble` | the piece kind: one piece in a bubble, pieces in word cells, a corpus line in a bubble |

The number is a fixed label (`builder.TIER_PX`), not recomputed per build.
A kanji run draws the large pools larger (`retrain_kanji_b4`: `grid_82` →
100, `lone_190` → 235, `bubble1_52` → 55, `bubble1_32` → 35.5); each
build's own px is `build.json` `tiers.<tier>.px_kept`.

## Old → new

| until 10-02 (group / recipe) | now | px p10 / median / p90 | σ band its table stamps | px source |
|---|---|---|---|---|
| `b0709` / `grid_single`, 1×1 | `lone_190` | 119 / 191 / 268 | 0.7–0.9 | `retrain_kana/data` |
| `b0709` / `grid_single`, 2×2 – 3×3 | `grid_82` | 55 / 82 / 118 | 0.7–0.9 | 〃 |
| `b0709` / `scene_single` | `bubble1_52` | 42 / 52 / 75 | 0.7–0.9 | 〃 |
| `b0507` / `scene_window` | `bubbleN_34` | 29 / 34 / 45 | 0.5–0.7 | 〃 |
| `b0507` / `scene_single_small` | `bubble1_32` | 27 / 32 / 38 | 0.5–0.7 | 〃 |
| `b0305` / `scene_window` | `bubbleN_18` | 14 / 18 / 22 | 0.3–0.5 | 〃 |
| `g0507` / `grid_single` | `grid_29` | 25 / 29 / 33 | 0.5–0.7 | `run1002_grid_lone/data` |
| `l0507` / `grid_single` | `lone_28` | 22 / 28 / 35 | 0.5–0.7 | 〃 |
| `g0305` / `grid_single` | `grid_16` | 13 / 16 / 20 | 0.3–0.5 | 〃 |
| `l0305` / `grid_single` | `lone_16` | 12 / 16 / 21 | 0.3–0.5 | 〃 |
| `b0507` / `scene_piece` | `piece_bubble_38` | 30 / 38 / 53 | 0.5–0.7 | `run0925_300f` build log |
| `b0507` / `scene_short` | `line_bubble_32` | 29 / 32 / 39 | 0.5–0.7 | 〃 |
| `b0507` / `grid_string` | `piece_grid_29` | 26 / 29 / 32 | 0.5–0.7 | 〃 |
| `b0305` / `scene_piece` | `piece_bubble_19` | 15 / 19 / 23 | 0.3–0.5 | 〃 |
| `b0305` / `scene_sentence` | `line_bubble_19` | 16 / 19 / 23 | 0.3–0.5 | 〃 |
| `b0305` / `grid_string` | `piece_grid_17` | 13 / 17 / 21 | 0.3–0.5 | 〃 |
| — | `grid_44` | 41 / 44 / 49 | 0.7–0.9 | `run1002_grid_44/data` |
| — | `lone_44` | 34 / 44 / 52 | 0.7–0.9 | 〃 |

The σ column is the band of the group the tier is drawn in — the band law's
window for that px — not a read on the tier, and not part of the name. The
first six rows trained at it on `retrain_kana`. The four small `grid` / `lone` tiers got it as the
experiments' default (§ The bands the 10-02 arms trained at). No arm has
trained `grid_44` / `lone_44` at 0.7–0.9.

Recipes (the draw functions, `recipes.RECIPES`): `scene_single` +
`scene_single_small` → `bubble1` (one function; `fill = [lo, hi]` takes the
small path), `scene_window` → `bubbleN`, `grid_single` → `grid`. The piece
recipes keep their names.

## What changed in the code

- `builder.Tier` carries the name; it is the image prefix
  (`img/<tier>_<i>.png`), the record's `tier` key, the `build.json` `tiers`
  key and the sheet name. New records have no `group` key.
- `builder.Group` has no name: it is the draw unit — the tiers of one kind
  at one band, from one rng restart. `build.json` `groups` is a list.
- `builder.tier_of(record)` returns a record's tier for new and old records
  alike (`LEGACY_TIERS`; a record of an experiment's own table falls back to
  its old file prefix, `b0507_scene_spelled`). `builder.tiers(*names)`
  returns `TABLE`'s tiers by name.
- `conflict` groups items by their band (`0.7-0.9`, not `b0709`) and breaks
  the price table down by tier.
- 13 experiment scripts moved to the new API; the ones that read data of
  record ask `tier_of`. `grid_small` and `grid_lone` build under the new
  names.
- A run holding both kinds no longer collides on a group name (`b0507` was
  the single kind's and the piece kind's); the check is now that tier names
  are unique.

## What did not change

- Data dirs, reports, results and `experiments/README.md` entries of record
  keep the old names. Nothing on disk was rewritten.
- The draws. `lone_190` and `grid_82` are one tier whose deck deals 1×1
  beside the grids; splitting it would reorder rng consumption, so it stays
  one draw and each item is named by its layout (`Tier.flat`). Shares are
  still group share × tier weight.
- The band law (`windows.py`), the gate, the trainer: px still decides the
  band an item trains in (the top of this file).
- Tier names of the older experiments' own tables (`stage_b`, `polish_rows`,
  …) are their old file prefixes; their px was not measured.

## Checks

- `retrain_kana`'s table built with the code before and after the rename, 10
  workers: 17 400 records equal on every field compared (text, caption,
  boxes, px, band, shape, scene, …), 360 images pixel-equal.
- `grid_lone`'s table rebuilt with the new code against
  `run1002_grid_lone/data`: 8 200 records equal, 280 images pixel-equal.
- Line tests: 86 pass (`test_tier_of_reads_the_records_of_record` added).
  Every experiment imports and builds its table; no GPU leg was run for the
  rename.

Found on the way: `retrain_kana/data` (built 09-28) is no longer what a
rebuild gives, before or after the rename. The window pool is 187 218
against the record's 178 846, and the first tier already draws other items.
The cause was not traced past that.

## `grid_lone recap` in the new names

82 rows (81 hiragana + `ー`) cold, 60 steps / row = 4 920, 8 200 items.

| tier | items | px p10 / median / p90 | σ | glyphs / item | caption |
|---|---|---|---|---|---|
| `bubbleN_34` | 1 148 (14 %) | 29 / 34 / 45 | 0.5–0.7 | 4.5 | as built |
| `bubble1_32` | 492 (6 %) | 28 / 32 / 38 | 0.5–0.7 | 1 | as built |
| `grid_29` | 2 460 (30 %) | 25 / 29 / 33 | 0.5–0.7 | 6.3 | plain |
| `lone_28` | 615 (7.5 %) | 22 / 28 / 34 | 0.5–0.7 | 1 | plain |
| `bubbleN_18` | 1 640 (20 %) | 14 / 18 / 22 | 0.3–0.5 | 4.5 | as built |
| `grid_16` | 1 230 (15 %) | 13 / 16 / 20 | 0.3–0.5 | 6.3 | plain |
| `lone_16` | 615 (7.5 %) | 12 / 16 / 21 | 0.3–0.5 | 1 | plain |

The σ column is what the arm trained at: the per-px bands of the
experiment's table. No `grid_lone` arm trained at 0.75–0.93.

## The bands the 10-02 arms trained at

Job argv and each data dir's `train.jsonl`; "how" is from the session
record.

| arm | job | band | how it was set |
|---|---|---|---|
| `grid_small r0` | `20261002-112155-8e8225` | per px: 0.3–0.5, 0.5–0.7 | the experiment's table; the request named grid share and glyph size, no band. Asked mid-run whether the bands were one, the user kept it as the control |
| `grid_small b7593` | `20261002-121029-24622c` | every item 0.75–0.93 | user |
| `grid_small recap_b7593` | `20261002-134455-2aaa5a` | every item 0.75–0.93 | user, replacing a per-px job queued at the default; stopped 8 min in, unread |
| `grid_lone r0` | `20261002-143355-1c397b` | per px | the experiment's table; the request named the budget, the lone share and `g0305`'s, no band. The band was one column of the briefing and one of four items to confirm, and went in on an OK to the briefing |
| `grid_lone recap` | `20261002-143355-beb03c` | per px | as `r0` |
| `grid_44 recap_b7593` | `20261002-180924-362d76` | every item 0.75–0.93 | user: "as before, every item at 0.75–0.93" — the arm before it, `grid_lone recap`, was per px, and the mismatch was not raised |

Every per-px band above was the table's default, not a band the user chose.
The arms that read (`grid_small r0`, both `grid_lone` arms) are the per-px
ones; every arm at 0.75–0.93 read 0 (`kana_reband`, `grid_small b7593`,
`grid_44 recap_b7593`).

## `grid_44`

`../../../cjk_anima_scale/experiments/grid_44` (user, 10-02): `grid_lone` with
a 44 px pair. Half of `grid_29` goes to `grid_44`; the lone share is split
evenly over three sizes.

| tier | items | vs `grid_lone` | px p10 / median / p90 | σ as built |
|---|---|---|---|---|
| `grid_44` | 1 230 (15 %) | new | 41 / 44 / 49 | 0.7–0.9 |
| `grid_29` | 1 230 (15 %) | 30 → 15 % | 25 / 29 / 33 | 0.5–0.7 |
| `grid_16` | 1 230 (15 %) | same | 13 / 16 / 20 | 0.3–0.5 |
| `lone_44` | 410 (5 %) | new | 34 / 44 / 52 | 0.7–0.9 |
| `lone_28` | 410 (5 %) | 7.5 → 5 % | 22 / 28 / 34 | 0.5–0.7 |
| `lone_16` | 410 (5 %) | 7.5 → 5 % | 12 / 16 / 21 | 0.3–0.5 |
| `bubbleN_34` | 1 148 (14 %) | same | 29 / 34 / 45 | 0.5–0.7 |
| `bubble1_32` | 492 (6 %) | same | 28 / 32 / 38 | 0.5–0.7 |
| `bubbleN_18` | 1 640 (20 %) | same | 14 / 18 / 22 | 0.3–0.5 |

- `grid_44` is drawn at 44–62 font px (median 42 as drawn) through the
  ≥ 40 px gate, which rejects 38 % of the draws and leaves 44. `lone_44` is
  ungated, so it is drawn 2 px up (46–64) to land on 44.
- A row sits in 300 (`ぅ`) – 841 (`っ`) items, median 434; in `grid_44` 63
  (`ー`) – 114, in `lone_44` 2 (`ぅ`) – 8.
- As built: 1 640 items at 0.7–0.9, 3 280 at 0.5–0.7, 3 280 at 0.3–0.5.

The "σ as built" column is the table's per-px default; no arm has trained
at it. The one arm trained is `grid_44_cold_hira_recap_b7593` (job
`20261002-180924-362d76`, 56.2 min; result
`../../../cjk_anima_scale/experiments/grid_44/results/20261002-1809-recap_b7593`):
this build, the plain captions on the 4 920 grid and lone items, **every
item at σ 0.75–0.93**, 60 steps / row.

### `recap_b7593` reads 0

Words, 72 (`en` of record, and the plain clause):

| arm | clause | off | cont | ≤1 | ≤2 | dup | ≤1c | read as kana |
|---|---|---|---|---|---|---|---|---|
| `retrain_kana` | `en` | 10 | 22 | 33 | 53 | 47 | 53 | 72 |
| `grid_44 recap_b7593` | `en` | 0 | 0 | 0 | 0 | 27 | 0 | 72 |
| `retrain_kana` | plain | 18 | 28 | 45 | 63 | 44 | 67 | 72 |
| `grid_lone recap` | plain | 6 | 9 | 14 | 35 | 61 | 37 | 72 |
| `grid_44 recap_b7593` | plain | 0 | 0 | 0 | 0 | 62 | 0 | 41 |

Singles, 64 (`swap` of record, and the plain clause):

| arm | clause | off | cont | read as kana | rep |
|---|---|---|---|---|---|
| `retrain_kana` | `swap` | 17 | 44 | 52 | 2 |
| `grid_44 recap_b7593` | `swap` | 0 | 0 | 24 | 0 |
| `retrain_kana` | plain | 34 | 56 | 61 | 5 |
| `grid_lone recap` | plain | 12 | 44 | 58 | 15 |
| `grid_44 recap_b7593` | plain | 0 | 1 | 34 | 0 |

Training, first 500 → last 500 steps: in-box 0.182 → 0.165 (−9.5 %),
out-of-box 0.105 → 0.105, Δ norm 191, rel 1.01. `grid_lone recap`: in-box
0.162 → 0.112 (−31 %), Δ norm 197; `grid_small b7593`: 0.185 → 0.160
(−13 %). The rows moved as far as `grid_lone recap`'s and the in-box loss
did not follow — `grid_small b7593`'s trajectory again.

Two sheets looked at (`こんにちは` `en`, `あ` `swap`): monochrome manga
layouts with bubbles and vertical columns filled with pseudo-kanji scribble
on the word; on the single, a garbled Latin caption strip and a pseudo-glyph.

- This arm differs from `grid_lone recap` in two things, the 44 px pair and
  the band, so the paired plain read does not isolate the 44 px pair. What
  it repeats is `grid_small b7593`.
- The data is the build above: the nine tiers at their counts and px, every
  `band` 0.75–0.93 (`data_recap_b7593/train.jsonl`).

## Open

- The per-px arm of `grid_44` (`--label recap --legs data recap train read
  read_plain --tag recap`: the 44 px pair at 0.7–0.9, the rest as
  `grid_lone recap`) has not run. It is the arm that compares to
  `grid_lone recap`, and the first per-px train after the rename.
- Whether the px → band binding stays in the builder and the trainer. The
  rename only stopped the names from saying it.
