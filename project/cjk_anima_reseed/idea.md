# idea — per-occurrence credit (2026-10-14)

Extracted from `_archive/idea.md` ① (the 10-05 review); `status.md` § 6
lists it as not tried. A glyph occurrence gets its own training credit: the
loss on its region flows back only through its own embedding lookup.

## The problem

- **The loss gives every row in a box the same credit.** Box-share FM
  (`reseed/loss.py`): `s · mean_in + (1 − s) · mean_out` over the item's
  box; a grid item's box is the union of its cells (`item_boxes`). No glyph
  region is tied to the token that names it.
- **Cold, the neighbours take the same gradient**
  (`_archive/reports/grad_identity_2026_10_02.md`, pass 2 on scenes, rows at
  the pack): with slot k's glyph swapped, the window's other rows move as
  much as slot k's own — `bubbleN_34` own / cross f .46 / .44 at σ 0.5; no
  (tier, σ) cell separates them at p < 0.01.
- **Trained, still** (`_archive/reports/probe_twin_2026_10_08.md`, rows at
  preview51): in the box the neighbours' rows turn 0.6 – 0.9 as much as the
  swapped glyph's own (kana own 0.39 – 0.44, cross 0.32 – 0.33 at
  σ 0.45 – 0.65). The review's first diagnostic — "if trained rows already
  show strong specificity, the idea weakens" — is this read, and they do
  not.

## The variant

- Per-glyph boxes. Per item, one occurrence: a local FM term over its box
  (dilated), whose gradient reaches the rows only through that
  occurrence's token position; the caption and every other row stay in the
  forward as they are.
- The ordinary whole-item FM stays in the batch: it teaches composition
  and count.
- A repeated glyph is gated per occurrence, not by its shared row.
- The local term only below the tier's identity half point
  (`grad_bands`): above it a glyph-sized box is mostly layout.
- The export is a plain static table.

## In this tree

- **Boxes.** Grid records carry `boxes`, one per cell in `units` order
  (`reseed/builder.py`). Scene windows carry one `box` (the union); a
  glyph's own box there needs the renderer — `probe_twin`'s
  `render_into_scene(ref_text=…)` already redraws one glyph in place.
- **The gate.** `ExtDelta` adds the row at `llm_adapter.embed`
  (`reseed/rows.py`); with `glyph_route` every glyph is its own ext id, so
  an occurrence is a token position. At the hook:
  `e = e.detach() + m · (e − e.detach())`, m = 1 on the occurrence's
  positions.
- **Cost.** The local term's gate differs from the whole-item term's, so
  it needs its own forward — a second pass as L_pres has (× 1.67 at every
  2nd step on `sent_kanji_pres`).
- **Fit with `motivation2.md` § 8.** The grid-heavy cold arms put half
  their items on canvases that already carry per-glyph boxes; the variant
  can start there, grid items only.

## First read

A micro pair (≈ 25 min each) on grid items: plain vs occurrence-local,
same data, start and steps. Read by glyph (singles, a few words; sheets
looked at), and `probe_twin`'s own / cross f on both arms' rows — the
variant should lift own over cross.

What it cannot do: omissions, repeats and count are sequence-level
(`_archive/idea.md` ④, a quote-scoped adapter); sharper rows need not fix
them.
