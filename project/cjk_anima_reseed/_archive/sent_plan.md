# sent_plan — the kana rows trained on dialogue lines, at the shipped stick (2026-10-05)

Run: `_archive/configs/sent_ball.toml`. Judged on the dialogue ruler (`criteria.md`).

**Read (10-05)**: `reports/sent_ball_2026_10_05.md` — worse than preview51
(short exact 8 → 2), mid / long unmoved; the spikes turned to cos 0.77 at
their length. At lr 2e-4 (`sent_ball_lr2`) the rows stay (cos 0.978)
and the same short strings fall.
Rows free (`sent_whole`) tie preview51; the stick alone (`sent_stick`)
ties it on short / mid and loses long (§ 7). No split buys mid / long.

## Why

On the sensitive ruler (`reports/ruler_2026_10_05.md` § 4) no arm moves past
a word: mid exact ≤ 1 / 32, long within 2 edits 0 / 32 for every table.
Every reseed data dir so far trains words as 2–6 glyph windows in one
column (`scene_window` `max_lines=1`); the target is a dialogue line over
2–3 columns of a bubble. The `sent` tiers (`sent_34` / `sent_22`, README)
draw that line; here they are the main data.

## The rows: a warm ball on preview51

- **Warm from the shipped rows**: `ball_on = "seed_fixed_1005_stick080"` —
  the rows preview51 was baked from: seed_fixed_1005 with each family's
  mean (kana 167 rows, kanji 1 185) scaled × 0.8, the six mark rows
  unscaled. The run starts on what ships, so the trained rows bake at
  scale 1.0 (no second × 0.8).
- **Trained**: the 163 kana rows (hiragana + ゔ, katakana + ヴ + ー). The kana
  family's other four — `゙`, `ヵ`, `ヶ`, `・` — stay frozen at preview51:
  under the punct pack `・` routes to a dot run and `゙` is never drawn, and a
  ball needs every trained row drawn.
- **Held**: the 163 rows' mean as preview51 has it, put back after every
  step (`warm = true`: `train.py` asserts the warm rows' mean is the file's).
  |m| 119.4 = 0.803 × the 1005 mean over the same rows (cos 0.99999; 0.5 %
  off an exact 0.8, the family mean carrying the four rows above); the rows
  less it start as 1005's, identical (max |Δ| 3e-5). Row norm mean 253.6 →
  237.7.
- **Frozen**: the kanji rows (0930's, × 0.8) and the marks, as shipped;
  the punct pack's routing at build, train and read.

## The data

163 rows × 100 items = 16 300 items. Kana per item measured on the 3 %
look build (`--frac 0.03`, earlier shares; the per-tier counts do not
depend on the shares).

| tier | form | σ band | px | items | share | kana / item | kana occurrences |
|---|---|---|---|---|---|---|---|
| bubble1_52 | one glyph in a bubble | 0.55–0.8 | ~48 | 326 | 2 % | 1.0 | 0.3 % |
| bubble1_32 | one glyph in a bubble | 0.35–0.6 | ~33 | 489 | 3 % | 1.0 | 0.5 % |
| bubbleN_34 | 2–6 glyph window | 0.45–0.7 | ~34 | 2 445 | 15 % | 3.8 | 9.0 % |
| bubbleN_18 | 2–6 glyph window | 0.2–0.5 | ~17 | 1 630 | 10 % | 4.5 | 7.1 % |
| sent_34 | dialogue line, 2–3 columns, 8–10 cells | 0.45–0.7 | ~35 | 6 520 | 40 % | 6.7 of 8.8 | 42.1 % |
| sent_22 | dialogue line, 2–3 columns, 8–14 cells | 0.3–0.6 | ~22 | 4 890 | 30 % | 8.7 of 11.1 | 41.0 % |

- By items singles 5 % / windows 25 % / dialogue 70 %; by kana occurrence
  0.8 % / 16 % / 83 %. The loss box's share grows by log glyph count to its
  cap at 8 glyphs (`BOX_SHARE*`), so the gradient split sits between the two.
- A line's kanji and marks (about a quarter of its cells) ride frozen.
- Lines: 19 133 of the 42 974 Manga109 dialogue lines pass (8–14 cells,
  ruler 5-grams and `read` trigrams held out); `sent_34`'s 8–10 cells hold
  ≈ 9.5 k of them for its 6 520 items.
- Singles average 5 per row: the four bubble1 tiers are what guarantee every
  row a draw.

## The budget

40 steps per row → 6 520 steps (ball_rk_bubble: 135 × 81 = 10 935), batch
4, ≈ 26 k draws, ≈ 1.6 passes over the items. lr 1e-3 cosine, warmup 0.1,
μ 0 (no anchor on the warm rows), the trainer of record otherwise.

## The read

The sensitive ruler, paired on prompt × seed against
`seed_fixed_1005_stick080` — the run differs from it only by this training.
That table has no ruler cache yet (the floor of record is retrain_kana /
seed_retrain_0930 / seed_fixed_1005@punct): it renders once, as the shipped
floor, and every later arm on these rows reads against it.

What to look at:

- **mid / long**: cer and ≤ 2 edits (exact counts sit at 0–1 there);
- **short**: the warm risk. Every warm pass of record lost identity — the
  rows barely move in aggregate, each turns a few percent, and that costs the
  strings (`_archive/motivation2.md` § 4; those reads were on the banner
  set, not this ruler). The ball holds the mean, not the turn; watch
  `warm_cos` in the train log and the short bin's exact;
- `dup`, and the page columns (en_tok_out, flat white over the EN ref) for
  paste / wipe / banner.

## Run

```bash
rm -rf output/cjk_anima_reseed/sent_ball   # the 3 % look build
make daemon-run ARGS="--stall-timeout 900 project/cjk_anima_reseed/run.py sent_ball data"
make daemon-run ARGS="project/cjk_anima_reseed/run.py sent_ball train"
# the floor (seed_fixed_1005_stick080) on the ruler, then the arm
```
