# grad_identity — the band law read on the training gradient, no training (2026-10-02)

`../../../cjk_anima_scale/experiments/grad_identity` (full report:
`../../../cjk_anima_scale/reports/grad_identity_2026_10_02.md`). The question,
from the evening of 10-02: the band law's **upper** edge is confirmed by
same-items two-band training pairs (`grid_small r0` vs `b7593`, `band_c2_kanji`,
the reband arms), but its **lower** edge and its peak were only ever read on
the EN ceiling (`cf_sense`, base model, no rows). Is there a training-free
read of what a JA row's gradient carries, by glyph px and σ?

**Verdict.**
- **The ceiling transfers to the JA training gradient.** Holding a cold row
  fixed and swapping the glyph drawn in its slot, the share of the row's
  gradient that depends on the glyph (f) falls with σ, earlier the smaller
  the glyph; its half point — 0.62 / 0.72 / 0.76 for 15 / 28 / 44 px grid
  cells — is the EN ceiling's upper edge for that px (0.6 / 0.7 / 0.8).
- **The identity gradient peaks at σ 0.4 / 0.5 / 0.6 for 15 / 28 / 44 px.**
  That is the ceiling's flat-letter peak, one step under its grid-cell peak.
- **There is no lower edge in identity.** Below the peak the share holds
  and the gradient itself collapses (`grid_44`: ‖g‖ 50× smaller at σ 0.2
  than at 0.6). A band's lower edge is an exposure edge.
- **0.75–0.93 is layout, measured.** Every 15–28 px tier at that band has
  f ≤ 0.16, mostly under 0.1: a gradient over 90 % the same whichever glyph
  is drawn. `hypothesis.md` H1's "an item above where its glyphs resolve
  teaches layout alone", at step 0; `grid_small b7593` reading 0 is its
  trained form.
- **grid_44's bands sit on the peak for 15 and 28 px and one step above it
  for 44 px** (0.7–0.9; the 0.8–0.9 half gives f ≤ 0.15).
- **A lone 1×1 gets 3–10× less identity per draw than a grid cell of the
  same px.** Its loss is the plain canvas mean (`src` font, no box share).
- **The bubble tiers follow the same px law** (pass 2 on scenes, the same
  evening). The identity peak is ordered by px whatever the form — σ 0.4
  (15–19 px), 0.5 (28–32), 0.6 (36–44), 0.7 (50) — and the recipe table's
  bands hold it in all four bubble tiers. The small scene tiers lose the
  share ≈ 0.1 σ earlier than a grid cell of the same px (half point 0.52 /
  0.68 for 19 / 36 px windows). `bubbleN_18` at 0.75–0.93 — `b0305_reband`'s
  items at its band — reads f 0.06–0.12.
- **A window pays every row for every slot.** With slot k's glyph swapped
  in the image, a row of the window whose own glyph did not change takes as
  large a glyph-dependent share as slot k's own row (f_cross ≈ f_own at
  every σ, the sizes equal too). The own row's edge is +0.02 at σ ≤ 0.5 on
  the 19 px windows and not separable on the 36 px ones. `hypothesis.md`
  H2, on the gradient: the draw says which glyph is in the bubble, not which
  row it belongs to.
- **Per draw, a lone glyph with the box share is the strongest identity
  teacher** (`bubble1`: ≈ 3× a grid cell, 10–20× a lone 1×1 under the plain
  loss, same px); a window row gets about half a grid cell's.
- **The brief's own design was null by construction.** Comparing the true
  row's gradient with other rows' in the same slot (pass 1) finds nothing:
  different rows' gradients are near-orthogonal at every σ (cos ≈ 0.15), so
  there is no identity / layout split *in row space* at a cold start. Pass 2
  (one row, the image swapped) is the read.

## Setup

Rows at the grid_44 train leg's initial point: the 81 hiragana + ー cold at
the pack rows, everything else at the old seed, routed, raw pack (sha
`7b9fce0b…`). Items from `output/cjk_anima_scale/run1002_grid_44/` (plain
captions, `data_recap_b7593`; the records' bands ignored — σ swept 0.2 … 0.95,
one ε per item × σ). Loss = the trained loss (box-share FM on grids, plain
MSE on a lone 1×1). u = a hiragana occurring once in the item; three wrong
glyphs v_j absent from the item and from u's dakuten / small-kana family.

| pass | contrast | items | job |
|---|---|---|---|
| 1 | same image, caption's slot glyph u → v_j; gradient of row u vs row v_j | 40 × 9 tiers (plain) + 16 × 9 (clause) | `20261002-195428-37dfb2`, 38.5 min |
| 2 | one row u, true caption; image re-drawn with the slot's glyph u (A) or v_j (B_j) | 40 × 6 grid / lone tiers | `20261002-203655-1e4c1d`, 18.5 min |
| 2, scenes | the same on the bubble tiers; a window's other rows read from the same backward | 40 × 2 `bubble1` + 60 × 2 `bubbleN` tiers | `20261002-223707-64728b`, 18.0 min |

Pass 2's renders are fresh twins of the records (`render_grid`, the build's
`bubble_fit` / `cell_jitter`, one seeded rng per item, one font for A and
every B_j), kept iff the pixel difference lies inside the slot's cell
(240 / 240). f = 1 − cos(g(u | B_j), g(u | B_k)); the bf16 floor is f ≈ 0.01.

The scene renders are `render_into_scene(ref_text=…)` on the record's scene,
fill and orientation — one fit, the sibling's glyphs in place, one font and
one rng for A and every B_j; a window is kept iff each difference spans one
glyph pitch at slot k (6 dropped). Twin px median 50 / 32 / 36 / 19. Loss =
the trained box share, on the union of the 1 + m ink boxes. `bubble1_52` is
not a grid_44 tier: its records are `retrain_kana`'s, read at the same
initial point.

## f, the glyph-dependent share of row u's gradient (pass 2, medians)

| tier | px | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.75 | 0.8 | 0.85 | 0.9 | 0.95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `grid_44` | 44 | .48 | .58 | .45 | .44 | .40 | .36 | .25 | .15 | .11 | .07 | .01 |
| `grid_29` | 28 | .42 | .34 | .36 | .32 | .33 | .19 | .12 | .10 | .08 | .03 | .01 |
| `grid_16` | 15 | .33 | .31 | .29 | .19 | .17 | .09 | .08 | .06 | .02 | .01 | .01 |
| `lone_44` | 45 | .55 | .71 | .76 | .85 | .71 | .66 | .42 | .30 | .17 | .07 | .01 |
| `lone_28` | 28 | .56 | .73 | .63 | .69 | .52 | .19 | .16 | .11 | .04 | .04 | .01 |
| `lone_16` | 15 | .64 | .60 | .44 | .35 | .23 | .16 | .10 | .11 | .04 | .02 | .01 |
| `bubble1_52` | 50 | .91 | .99 | .88 | .85 | .81 | .63 | .58 | .44 | .47 | .27 | .27 |
| `bubble1_32` | 32 | .89 | .94 | .83 | .78 | .45 | .38 | .24 | .28 | .29 | .23 | .19 |
| `bubbleN_34` | 36 | .58 | .60 | .58 | .46 | .39 | .27 | .25 | .15 | .13 | .13 | .06 |
| `bubbleN_18` | 19 | .49 | .38 | .35 | .22 | .09 | .08 | .10 | .09 | .06 | .12 | .05 |

‖I‖ = ‖g‖·√f, the identity gradient's size (× 1e-3): grids peak 46.9 at
σ 0.6 (`grid_44`), 48.5 at 0.5 (`grid_29`), 34.6 at 0.4 (`grid_16`); the
lone tiers stay under 13 everywhere. Scenes: 123.4 at 0.7 (`bubble1_52`),
153.8 at 0.5 (`bubble1_32`), 26.0 at 0.6 (`bubbleN_34`), 20.8 at 0.4
(`bubbleN_18`), per row.

| tier | identity peak σ | f half point | f < 0.1 from | EN ceiling (flat letter, peak · live) | grid_44 band | f across it | at 0.75–0.93 |
|---|---|---|---|---|---|---|---|
| `grid_16` | 0.4 | 0.62 | 0.7 | 16 px: 0.4 · 0.2–0.6 | 0.3–0.5 | .31 → .19 | .08 → .01 |
| `grid_29` | 0.5 | 0.72 | 0.8 | 24–32 px: 0.5 · 0.35–0.7 | 0.5–0.7 | .32 → .19 | .12 → .03 |
| `grid_44` | 0.6 | 0.76 | 0.9 | 48 px: 0.6 · 0.5–0.7 | 0.7–0.9 | .36 → .07 | .25 → .07 |
| `bubbleN_18` | 0.4 | 0.52 | 0.6 | — | 0.3–0.5 | .38 → .22 | .10 → .06 |
| `bubble1_32` | 0.5 | 0.60 | never (.19 at 0.95) | 24–32 px: as `grid_29` | 0.5–0.7 | .78 → .38 | .24 → .23 |
| `bubbleN_34` | 0.6 | 0.68 | 0.95 | — | 0.5–0.7 | .46 → .27 | .25 → .13 |
| `bubble1_52` | 0.7 | 0.79 | never (.27 at 0.95) | 48 px: as `grid_44` | 0.7–0.9 (`TABLE`'s) | .63 → .27 | .58 → .27 |

The drawn glyph being the row's own buys nothing at a cold start; at 28–45 px
it costs a little (the true render's gradient sits closer to the common part,
excess −0.03 … −0.06, p ≤ 2e-3) — the base already expects u a little where
it can see it. At 15 px the shift vanishes.

## A window's other rows (pass 2 on scenes)

Slot k's glyph swapped in the image; f on slot k's own row and the mean f
over the window's other rows (their own glyphs unchanged), 60 items a tier:

| tier | | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.75 | 0.8 | 0.85 | 0.9 | 0.95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bubbleN_34` | own | .58 | .60 | .58 | .46 | .39 | .27 | .25 | .15 | .13 | .13 | .06 |
| | cross | .52 | .54 | .56 | .44 | .34 | .25 | .26 | .15 | .14 | .11 | .07 |
| `bubbleN_18` | own | .49 | .38 | .35 | .22 | .09 | .08 | .10 | .09 | .06 | .12 | .05 |
| | cross | .42 | .39 | .32 | .20 | .10 | .08 | .10 | .10 | .07 | .10 | .05 |

- No (tier, σ) cell separates own from cross at p < 0.01; ‖I_cross‖ ≈
  ‖I_own‖ throughout (26.0 · 25.3 and 20.8 · 20.1 at the peaks).
- Per item over σ ≤ 0.5: `bubbleN_18` own − cross +0.021, 46 / 60 > 0
  (p 4e-5); `bubbleN_34` +0.009, 37 / 60 (p 0.09). Nothing above 0.5 on the
  19 px windows.
- Rows next to slot k take slightly more than rows further off at low σ
  (`bubbleN_18` σ 0.2–0.4: 0.44 / 0.40 / 0.33 against 0.37 / 0.31 / 0.27).
- A shared amount, not a shared vector: the own row's change between two
  renders against a cross row's over the same two has cos ≈ 0.09 — what two
  rows' gradients have anyway. Each row takes the swap through its own
  Jacobian.
- Fewer rows, more each: at the peak a 2–3 glyph window pays a row 42–45
  (19 px) / 45 (36 px) × 1e-3 against 10–14 / 24 for 4–6 glyphs.

## What it does not show

- **Training.** A step-0 gradient is necessary, not sufficient: Adam
  rescales per coordinate and the rows leave the pack rows within a few
  hundred steps. f prices what a draw *offers* the row, not what a run keeps.
  Where the best low band is for small text is still a training read.
- **Balanced window cells.** Length and orientation are the tiers' own mix
  (13–21 items in the small cells); lines live on the `sl1w` pool only.
- **Clause captions in pass 2**; any warm or trained rows.
- The renders are twins of the records' specs, not the data dir's pixels.

## What it changes in `motivation2.md`

- Verdict 2 ("the band decides what gradient a row sees") now has a
  measurement behind it rather than three warm arms and one confounded cold
  one: at 0.75–0.93 a 15–28 px item's gradient is ≥ 90 % glyph-independent.
- The band law's lower edges are exposure edges, so a band below the
  identity peak wastes draws rather than teaching the wrong thing; the peaks
  (0.4 / 0.5 / 0.6 for 15 / 28 / 44 px) give a candidate table
  0.3–0.5 / 0.4–0.6 / 0.5–0.7, one step under the law for 44 px.
- A lone 1×1 under the plain canvas loss is the weakest identity teacher per
  draw in the set; whether it takes the box share is a trainer question —
  now priced: the one-glyph form that does take it (`bubble1`) is paid
  10–20× as much at the same px.
- H2 has a gradient-side read: at a cold start one window draw does not
  tell its rows apart — the own row's edge is +0.02 at most, the rest is
  the bubble's glyphs, paid to all. Windows alone still buy identity
  (verdict 4, `p1_cold` 65 / 128), so it is bought across draws, not within
  one: a row's own glyph is the only thing constant over its windows. A
  reading, not a measurement — the trained-row read (Open) is the test.
- `b0305_reband`'s layout break is § Verdict's "0.75–0.93 is layout" on its
  own items: 19 px windows at that band, f 0.06–0.12.

## Open

- Pass 2 on trained rows (`grid_44_cold_hira_recap_b7593`, `retrain_kana`):
  once identity is bought, does the true render separate, and at which σ?
- The window read on trained rows: once the rows carry identity, does
  f_own pull away from f_cross, or does a window keep paying every row for
  every slot?
- ~~The two-low-band training pair for 15–28 px items~~ — read 10-03
  (`grad_bands_2026_10_03.md`): bands set from this read tie the law's; no
  lower edge loses.

Code: `../../../cjk_anima_scale/experiments/grad_identity/run_exp.py`
(`--dry_run`, `--items`, `--sigmas`, `--wrong`, `--captions`, `--render`,
`--render_tiers`, `--window_items`, `--analyze <dir>`); results under its
`results/` (`20261002-1954-r1`, `20261002-2036-render`,
`20261002-2237-scene`).
