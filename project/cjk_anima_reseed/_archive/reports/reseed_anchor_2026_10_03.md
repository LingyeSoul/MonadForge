# reseed_anchor — the columns lettered as Japanese, the base's text beside the rows (2026-10-03)

`../../../cjk_anima_scale/experiments/reseed_anchor` on `reseed_recap`'s arm
(the 81 hiragana + 85 katakana rows cold on the old seed, 135 / row =
22 410 steps, 16 600 items, plain grid captions, the `hp` bands). Two
questions from the user (10-03):

1. Were `reseed_recap`'s windows drawn with the lettering `seed_synth`
   fixed? **No** — and the fix went further than `seed_synth`'s.
2. Does text the base writes from its own rows — a ！ / ？ after a window,
   an EN word in one grid cell — beside the cold rows raise what they
   learn? **No.** Words tie `reseed_recap hp`; singles read lower.

**Verdict.**
- **Words tie.** Plain read, paired against `reseed_recap hp`: official
  11 vs 15 of 104 (7 / 11, p 0.48), ≤ 1 edit 40 vs 42, ≤ 2 edits 73 vs 73,
  dup 75 vs 76.
- **Singles read lower.** Official 21 vs 30 of 112 (7 / 16, p 0.09;
  hiragana 4 / 13, p 0.05), contained 87 vs 90, repeats 32 vs 23 (p 0.15).
  The `en` / `swap` read agrees: words official 2 vs 4, ≤ 2 edits 41 vs 35;
  singles official 8 vs 14.
- **The gap to `retrain_kana` is unchanged.** Plain words official 33,
  ≤ 1 edit 73; singles official 47, repeats 7.
- The arm changes the lettering and adds the base's text at once, so the
  singles' drop is not attributed to either. The lettering-only control
  (`--variant fix`) is built and not trained.

## 1. The lettering

`reseed_recap`'s windows (`bubbleN_34` / `bubbleN_18`) were drawn with
`render_into_scene`'s defaults, as every data dir of record: of 3 970 column
windows, 489 carry a turned ー 〜 up to 0.2 em left of the column axis, 1 356
a small kana drawn as the horizontal glyph (at the bottom of its cell), and
507 of the 5 644 windows open on ー or a small kana (`ーシングラか`, `ゅぱ`,
`ォこれ`). `seed_synth`'s redraw (`tategaki=True`, 10-01) put the turned mark
on the axis and top-aligned its two columns; it never reached a builder
recipe, and it left the small kana horizontal.

The fix, opt-in (the data of record stays reproducible):

- `render_into_scene(vert_forms=True)`: a column's ー 〜 …, small kana, 、。
  and brackets are the font's own vertical alternates (OpenType `vert`,
  through libraqm). 14 of the 15 render fonts carry them; TanukiMagic does
  not, and keeps the turned bar, centred, its small kana nudged 0.1 em
  up-right. Checked on a sheet: the bar sits on the axis as the font draws
  it in a column, the small kana top-right (`シャクマ？`, `スムぅー！`).
- No window opens on `scene.NO_HEAD` (a small kana, ー): 15 408 of 187 218
  windows dropped, no glyph loses its windows (per glyph min 4, median
  1 610).

Cut mid-word windows (`タンドバ` from スタンドバイ) stay: a window may cross
a word boundary by design (C3).

## 2. The base's text

- **Marks.** 30 % of the window draws take a window that a ！ / ？ closes in
  the dialogue corpus — the run's last 2–6 glyphs before the mark, under
  the pool's rules — with the mark drawn in the bubble and written in the
  caption: 10 930 such windows (！ 7 041, ？ 3 889), 158 glyphs (ぴ ぷ ぺ ヂ
  ヅ ヶ have none and keep a plain window); 1 667 items, 29.5 % of the
  windows.
- **EN cell.** Half the multi-cell grids give one cell, at random, to an EN
  word from `sigma_split`'s 36-word pool (the words the base writes in a
  3 × 3 at recall 0.84, `findings.md` § 6), at the cells' font px, captioned
  plain (`On the bottom middle, text reads as "WAIT".`): 3 067 grids, even
  over 2×2 … 3×3. The cell is outside the item's kind and px; the deck
  deals one glyph fewer, so those tiers carry ≈ 8 % fewer kana cells.
- **Neither touches a row.** The pack's encode fold sends ！ ？ to `!` `?`,
  and the closing quote joins them into one stock T5 piece (`!"` / `?"`):
  the full-width and half-width captions encode to the same ids, and on
  all 1 667 marked items the ext rows are the bare window's. Without the
  fold ！ would be `ext4`. No EN cell carries an ext row.

## 3. The read

Job `20261003-074735-8ffb80` (train 153 min, 186 min with the reads),
`results/20261003-0747-anchor/`. `retrain_read`'s grid (4 prompts × 2 seeds)
on the kana run's 13 words and 14 singles; `read` = `en` / `swap` against
the reads of record, `read_plain` = the plain clause, paired against
`reseed_recap hp` and `retrain_kana` (both cached).

| plain | `reseed_anchor` | `reseed_recap hp` | `retrain_kana` |
|---|---|---|---|
| words official / 104 | 11 | 15 | 33 |
| words contained | 28 | 33 | 51 |
| words ≤ 1 / ≤ 2 edits | 40 / 73 | 42 / 73 | 73 / 94 |
| words dup | 75 | 76 | 52 |
| singles official / 112 | 21 | 30 | 47 |
| singles contained | 87 | 90 | 91 |
| singles repeats | 32 | 23 | 7 |

Paired, `reseed_anchor` vs `reseed_recap hp` (wins / losses, McNemar p):

| | official | contained | ≤ 1 edit | repeats / dup |
|---|---|---|---|---|
| words, hiragana (72) | 2 / 6, 0.29 | 8 / 7, 1.0 | 9 / 14, 0.40 | dup 15 / 8, 0.21 |
| words, katakana (32) | 5 / 5, 1.0 | 4 / 10, 0.18 | 7 / 4, 0.55 | dup 4 / 12, 0.08 |
| singles, hiragana (64) | 4 / 13, 0.05 | 5 / 8, 0.58 | — | rep 12 / 6, 0.24 |
| singles, katakana (48) | 3 / 3, 1.0 | 8 / 8, 1.0 | — | rep 8 / 5, 0.58 |

On a 16-key × 2-render sheet the two arms draw the same scenes and banners
for the same prompt and seed, both with Latin garble in places. A sheet
that size settles nothing on its own.

## What it does not show

- Which change costs the singles: the lettering, the marks, the EN cells or
  the ≈ 8 % fewer grid kana cells. `--variant fix` (the lettering alone)
  separates the first from the rest.
- Whether the base's text helps a read that asks for it — a word with ！
  (`findings.md` § 2: こんにちは！ 5 / 8 official on the seed). No ruler here
  asks for a mark or an EN cell.
