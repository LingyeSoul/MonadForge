# cjk_anima_reseed — motivation 2 (2026-10-02, revised 2026-10-14)

Takes over the morning draft of 10-02 (`motivation.md`, "the seed trained in
the wrong bands"; withdrawn in verdict 1, deleted 10-14). § 1 – 5 are the
record of 09-30 – 10-02 read for what it says about re-seeding the JA vocab
pack's rows cold; every number there is from a dated report under
`../finished/cjk_anima_scale/` (`reports/`, `findings.md`, `hypothesis.md`,
`proposal_seed_synthesis.md`). § 6 is what the reseed read after it, § 7 the
motivation as it stands on 10-14, § 8 the two cold arms it asks for.

**The seed:** `seed_retrain_0930` (`retrain_kana` 174 kana rows cold +
`retrain_kanji_b1`–`b4` on top, routed). On `sent` (184 renders): official 29,
≤ 1 edit 92, a doubled glyph in 100; the word comes out as a banner at
≈ 78 px (long side 401 px).

**The target** (user, 10-02): manga-size dialogue in speech bubbles —
multi-panel and single-panel manga, illustrations — not the `sent` banner.
The large glyphs in the seed's data were for identity.

## Verdict

1. **The seed's bands were right for its glyph sizes.** Every arm of record
   that produced identity trained its glyphs inside the ceiling window of
   their px; every arm that lost it moved the same px above that window
   (§ 3). The morning draft's § 1 is withdrawn.
2. **A row is one vector at every σ.** The band decides what gradient a row
   sees, not where it acts. An item trained above where its glyphs resolve
   teaches its layout alone, and that layout overrides every caption the row
   appears in (§ 2). Confirmed three times on warm arms and once cold, and
   measured on the gradient itself (`_archive/reports/grad_identity_2026_10_02.md`):
   at 0.75–0.93 a 15–28 px item's gradient on a cold row is ≥ 90 % the same
   whichever glyph is drawn. The glyph-dependent share falls with σ, earlier
   the smaller the glyph, and its half point is the EN ceiling's upper edge
   for that px; the identity gradient peaks at σ 0.4 / 0.5 / 0.6 for
   15 / 28 / 44 px, and below the peak it is exposure, not identity, that
   runs out.
3. **Where identity is written depends on rendered size.** Layout and slot
   count at σ ≥ 0.9; identity at 0.85–0.7 for large glyphs, below 0.8 for the
   seed's small JA; nothing moves below 0.5 (§ 1).
4. **Identity does not need large glyphs.** `p1_cold` — 36 hiragana rows cold,
   in-word items at 0.3–0.7 only, nothing above ≈ 40 px — reads ≤ 1 edit
   65 / 128. The large lone tier buys the lone single (official 29 → 82 / 144)
   and a larger rendered glyph (banner 75 → 87 → 131 px across
   `p1_cold` / `p1_mix` / `p1_lone`) (§ 3).
5. **Every warm arm on the seed lost strings** — polish, garble_replace (nine
   variants), span_reband, seed_synth, b0305_reband, Δ scaling — whatever it
   moved in layout (§ 4). The rows already carry what their items taught; a
   warm pass adds to it. None of the data questions has been read cold.
6. **A σ gate does not give manga-size text.** The base above 0.8 with the
   seed rows below keeps the base's small bubbles and writes the word's
   glyphs into them, with repeats: 0–1 / 16 official against the seed's 7
   (§ 1).

So the reseed is: cold rows, small glyphs at the band law's bands for their
px, a bubble-shaped target — and the open reads are whether small-only
training writes manga-size text when the caption gives it a bubble, and
where small-text identity sits in σ (§ 6).

## 1. Where the text is decided on the trajectory

`reports/sigma_split_2026_09_30.md`, `findings.md` (x̂0 per σ, σ-gated rows;
`sent` grid, 512², 28 steps, cfg 4, shift 3: σ 1 → 0.9 is steps 0–7,
0.9 → 0.69 steps 7–16, < 0.5 the last 7).

- **Layout at σ ≥ 0.9, identity 0.85–0.7, nothing below 0.5.** The base
  writing 12–18 px EN in a bubble reads 61 % of words at σ 0.85, 80 % at
  0.75, 83 % at the end; a 3 × 3 EN grid has its rows placed at σ 1.0, slots
  at 0.95, most words by 0.85. The seed's rows commit earlier than EN: the
  JA banner already reads at 0.9. Below 0.5 only stroke detail moves, and
  it follows x_t, not the caption — rows gated to act only below 0.5 read
  0 / 184; the caption itself has no leverage there.
- **The σ that sets a glyph follows its rendered size.** The seed's trained
  singles in a ≈ 100 px grid are set by 0.9 and survive the conditional
  dropped at 0.8 (64 / 72 → 61; what flips is dakuten / handakuten); the
  seed's own b0305 captions (12–24 px dialogue windows) lose every string
  when it is dropped at 0.8 (7 / 16 → 0, CER 0.29 → 0.80). For small text
  the seed's identity is written **below** 0.8.
- **Rows only above 0.8** keep the floor's layout to the third decimal and
  a third of its strings (official 29 → 10–13, the top banners); the rest
  are rewritten by the base in the same slots. Rows only below 0.8, with
  the base's garble layout above, recover 16 / 29 official and 61 / 92
  ≤ 1 edit — into the base's bubbles, split across them. Neither side alone
  reaches the floor.
- **The mirror on small text (10-02):** the base above 0.8 and the seed
  rows below, on the 16 b0305 renders — official 0–1 (CER 0.80 → 0.55–0.62):
  the rows write the word's glyphs into the base's small bubbles, into more
  slots than the word has (`んだだよよウチ`, `なアロロナ`). The seed arm's 7
  draws the same word large; the comparison is two sizes. n = 16, captions
  of 3–5 glyphs, one switch.
- **The repeats are the base's text region minus the word.** The region's
  span is set by σ 0.95 with or without rows (`japanese text` tag → the
  long banner; without it, a 2-glyph line); the slot count commits between
  0.95 and 0.9 (`inject_count`: a hand-made n-slot banner injected at 0.9
  turns 0 / 60 official into 29 / 60, dup 60 → 21). A caption that fills the
  slots reads better (こんにちはは 4 / 8 vs こんにちは 2 / 8; こんにちは！ 5 / 8
  with 7 / 8 free of repeats).

## 2. A row is one vector at every σ (`hypothesis.md` H1)

A row is added at encode (`ExtDelta` on `llm_adapter.embed`); it has no σ
input. The ceiling table (`band_experiment_results.md` § 2) is where a
teacher-forced loss carries a caption gradient for a glyph of that px —
training is teacher-forced, so it is the only thing a band can follow. The
trajectory (§ 1) is where the vector is consumed. Two quantities, not a
contradiction (`reports/band_size_2026_10_02.md`).

What follows, and what read it:

- **An item above where its glyphs resolve teaches layout only, and that
  layout lands in the high-σ steps that set every caption's layout.**
  `b0305_reband` (the seed's 12–24 px windows warm at 0.75–0.93, μ 0.02):
  the floor's banner turns into small columns and sentence-length lines in
  bubbles. `shared_dir` splits its update: the break is in the per-row
  residual, not the mean direction, and it acts above 0.8 — the seed rows
  above 0.8 restore the floor's layout on every placement measure, and the
  arm's rows below 0.8 alone cost strings (≤ 1 edit 45 → 25–32, seed 0).
- **What the band teaches is the item's glyph size relative to its region;
  the region stays the base's.** `span_reband` (≈ 34 px word windows at
  0.85–0.95, warm): the banner stays, the glyphs in it get smaller and
  more — official 29 → 16, dup 100 → 113, banner glyph 78 → 68 px; its
  whole effect is above 0.8. `seed_synth` (the seed's own renders with the
  word redrawn to fill the base's region, 208 canvases × 10 same-length
  swaps, at 0.85–0.95 warm): size did not transfer (px 43 items, banner
  78 → 78) but the count did not either — dup 100 → 116, extra glyphs on
  the long words (`たたすすけけて`). Three item sets at 0.75–0.95 each put
  more glyphs in the base's region: the slot count is not a σ-less row's
  to carry from items trained above 0.85.
- **Cold, the same:** `kana_reband` (the kana run's all-hiragana items,
  every one at 0.75–0.93, 81 rows cold, 135 / row): words 0 / 72 at every
  threshold against `retrain_kana`'s 10 / 33 / 53, singles 0 / 64 vs
  17 / 44; the renders are small columns in bubbles with no identity, and
  the scene style leaks (monochrome sketches). Confounded (the hiragana
  filter dropped 2 272 of 2 314 multi-cell grids; 81 rows vs 174; the lone
  glyphs moved too), but it is what the ceiling table says of 18–34 px
  words above σ 0.6.
- **`garble_replace` short50 at 0.8–0.95** (14 px lines): the rows learn
  the canvases' bubbles (white blobs at σ 1.0 / 0.95) and no string
  (≤ 1 edit 7 / 184; strokes only form at σ ≤ 0.5). At its own band
  (0.6–0.85) the same items keep the banner and read 18.
- **The gradient, no training** (`_archive/reports/grad_identity_2026_10_02.md`; the
  grid_44 items, the 81 hiragana rows cold at the pack rows): with the row
  and caption fixed and the glyph drawn in its slot swapped, the share of
  the row's gradient that depends on the glyph is 0.31 / 0.32 / 0.36 at the
  top of the law's band for 15 / 28 / 44 px grid cells, 0.09 / 0.19 / 0.36 at
  σ 0.7, and ≤ 0.16 everywhere in 0.75–0.93 for 15–28 px. Its half point
  (0.62 / 0.72 / 0.76) is the EN ceiling's upper edge (0.6 / 0.7 / 0.8); the
  identity gradient's size peaks at 0.4 / 0.5 / 0.6. Below the peak the
  share holds and the gradient collapses (50× at σ 0.2 for 44 px). A lone
  1×1 under the plain canvas loss gets 3–10× less identity per draw than a
  grid cell of the same px. Step 0 only — necessary, not sufficient.

## 3. The seed's data, measured against the law

`reports/band_size_2026_10_02.md` § 1, § 4–5.
Every run in the chain drew `builder.TABLE`'s three groups 1 : 1 : 1.
Drawn px (√(box area / glyphs)), p5 – p50 – p95, `retrain_kana` / `b4`:

| group, recipe | σ | text | `retrain_kana` | `b4` |
|---|---|---|---|---|
| `b0709` `grid_single` | 0.7–0.9 | one glyph per cell | 52 – 91 – 232 | 62 – 108 – 271 |
| `b0709` `scene_single` | 0.7–0.9 | one glyph | 41 – 52 – 82 | 42 – 55 – 92 |
| `b0507` `scene_window` | 0.5–0.7 | word | 28 – 34 – 51 | 30 – 35 – 53 |
| `b0507` `scene_single_small` | 0.5–0.7 | one glyph | 26 – 32 – 38 | 31 – 36 – 40 |
| `b0305` `scene_window` | 0.3–0.5 | word | 14 – 18 – 23 | 14 – 18 – 23 |

- Each size has one recipe and one band. A word is never above 64 px and
  never above σ 0.7; a glyph above 64 px is always a lone glyph. The seed's
  banner (≈ 78 px) is a size no word item was drawn at — that, not the band
  table, is the gap between the items and where a large banner's identity
  is committed. It matters only if the banner is the target; it is not.
- Where identity has been bought, by px and band (every arm inside the
  ceiling window for its px; every arm above it lost):

| arm | glyph px | band | identity |
|---|---|---|---|
| `band_s_0923` singles on scenes | 48 | 0.7–0.9 | native 84 / 192 (0.5–0.7: 49) |
| `micro_cf_0922` pieces | 35 | 0.5–0.7 | exact 7 / 32 (0.7–0.9: 1) |
| `p1_cold` / `p1_mix` / `retrain_kana` windows | 34, 18 | 0.5–0.7, 0.3–0.5 | 65, 80 / 128; 29 / 64 ≤ 1 edit |
| singles | 48 | 0.8–0.95 | dead (`windows.py`, `band_c2_kanji`) |
| `garble_replace` cold, warm | 14 | 0.6–0.85 | 0, 5 / 184 official |
| `garble_replace` short50hb | 14 | 0.8–0.95 | 1 / 184 |
| `b0305_reband` warm | 18 | 0.75–0.93 | the banner → small columns |
| `span_reband` warm | 34 | 0.85–0.95 | official 29 → 16 |
| `seed_synth` warm | the redrawn words | 0.85–0.95 | official 29 → 19 |
| `kana_reband` cold | 18 – 52 | 0.75–0.93 | 0 / 72, 0 / 64 |

- The P1 arms re-read for size (36 hiragana donors cold, C2's eight words,
  `en`): in-word identity comes from the small in-word items (`p1_cold`
  65 / 128 with nothing above ≈ 40 px); the lone tier alone composes nothing
  (`p1_lone` 9 / 128) but makes a lone glyph come out once (singles official
  29 → 74–82 / 144) and scales the rendered banner glyph 75 → 87 → 131 px.
  All three still draw a banner under the `reads as` caption; `p1_cold`
  under a bubble caption is unread.

## 4. What the arms on the finished seed closed

All warm from the seed unless noted, read on `sent` 184 against the floor
(official 29 / ≤ 1 edit 92 / dup 100).

| arm | what it changed | official / ≤ 1 edit / dup | where |
|---|---|---|---|
| `polish_seed` μ 0.1, 4 steps / row, 1 362 singles | the seed's own recipe + `scene_line` tiers | 11 / 60 / more doubling | `polish_seed_2026_09_30.md` |
| `polish_seed --color` | coloured scenes only (973 / 1 897) | 10 / 46 | same — "the canvases were not the cause" (warm; open cold) |
| `garble_replace` warm μ 0.1 | the base's garble canvases, string replaced at the garble's px (14), 0.6–0.85 | 5 / 29 / 137 — banner kept, hiragana words → 0 | `garble_replace_2026_09_30.md` |
| `garble_replace` cold | same items from the pack rows | 0 / 0 / 114 — bubbles with long pseudo-JA lines, no banner | same |
| + quoted captions, grid20 / grid50, no humans, μ 0.02, short lines, inverse frequency, 0.8–0.95 | one variable each | 1–3 / ≤ 24; inv_freq ≤ 2 edit 51 → 28 | same, follow-ups |
| `delta_scale` Δ 0.9 / 0.75 (no training) | the seed's Δ shrunk | 10 / 48 / 124; 2 / 5 / 107 — order information restored, identity gone | `delta_scale_2026_10_01.md` |
| `b0305_reband` μ 0.02 | 12–24 px windows at 0.75–0.93 | layout broken (read stopped at 158 / 184) | `hypothesis.md` |
| `span_reband` μ 0.02 | ≈ 34 px windows at 0.85–0.95 × 4 | 16 / 70 / 113 | `proposal_seed_synthesis.md` |
| `seed_synth` swap μ 0.02 | own renders, word redrawn to the region, 0.85–0.95 × 6 | 19 / 71 / 116 | same |

Closed by them: a Δ cap or row scale; garble items at the base's px at any
band; order-paired losses on routed singles; the span lever (rows at
0.85–0.95 refill the region, they do not shorten it); a σ gate at inference
for 0.85–0.95 arms (their whole effect is above 0.8); `cf_sense`-style
teacher-forced count reads; warm polish of converged rows in any shape
(every warm pass of record lost identity; every arm that held it trained
cold from the pack).

Polish's own diagnosis holds for all of them: the rows barely move in
aggregate (warm_cos 0.96–0.996), each rotates a few percent on its own, and
that alone costs the strings. There is no warm move left that is not a
cold question.

## 5. Data properties still unread as causes

From the morning draft's § 2–3, measured on the data dirs of record; none
has a cold read behind it.

- **Half the scenes are monochrome or line art**: 49.7 % of scene items
  sit on a scene with coloured share < 0.08, 58 % of all items with the
  flat / grid canvases. The warm `--color` polish found no effect;
  `kana_reband` cold leaks monochrome sketch style into the renders.
- **Lettering**: every data dir was drawn with `tategaki=False` (a turned
  ー 〜 … up to 0.2 em off the column axis: 860 / 7 004 vertical windows in
  `retrain_kana`, 606 / 22 931 in `b4`); `scene_window` draws `max_lines=1`,
  so no item has a second column or line while the base breaks JA into
  columns by itself (placed by σ 0.69); small kana in a column are the
  horizontal glyph (2 278 / 7 004, 2 027 / 22 931); 774 / 9 860 and
  929 / 32 141 windows start with ー, a small kana or a closing mark.
- **A grid cell's bubble is not a speech bubble**: a 130–256 px cell filled
  by a bubble around a 16 px glyph. Two renderer opt-ins came out of
  `grid_small`'s builds (user, 10-02): `bubble_fit` (the bubble's inscribed
  rectangle 1.15–1.7 × the ink) and `cell_jitter` (the glyph at its cell's
  centre ± 8 %), both off by default in `src/data/grid.py`.

## 6. What the reseed read after this (10-02 – 10-09)

Reports under `_archive/reports/` and `reports/`; `status.md` § 2 – 4 holds
the tables.

- **Small-only cold loses.** `grid_small` / `grid_lone` (81 hiragana cold,
  grids ≥ 45 % of the items, no glyph above 64 px, 135 / 60 a row): words
  ≤ 1 edit 12 / 14 of 72 against `retrain_kana`'s 33, singles contained
  31 / 23 of 64 against 44 (`grid_small_lone_2026_10_02.md`).
- **The bands off the gradient tie the law's** (`grad_bands_2026_10_03.md`):
  the lower edge at half the identity peak, the upper at its half point —
  `reseed/table.py`'s `hp` bands.
- **The reseed table stays under `retrain_kana`.** `reseed_anchor`,
  `kana_mix`, `kana_big` (81 hiragana + 85 katakana cold, 135 / row): plain
  words official 9 – 15 of 104 against 33, ≤ 1 edit 34 – 42 against 73.
  What `kana_mix` lacks is a large glyph trained high: by glyph instance,
  ≥ 64 px 0.8 % against 19.7 %, ≥ 40 px with a band reaching 0.85 0.0 %
  against 28.3 %. A 65 px grid at its own gradient band (`kana_big`, 12.5 %
  of the items, σ 0.65–0.8) does not close it (`grid_64_2026_10_03.md`).
- **Upper edges + 0.1 is the one data move that read.** `kana_up` (every
  tier's upper edge + 0.1, capped at 0.9): words ≤ 1 edit 55 vs 34 / 104
  (p 0.002), the gain in hiragana words (26 / 7); exact words flat
  (14 vs 11); the scene degrades as `retrain_kana`'s does
  (`kana_up_2026_10_04.md`).
- **On dialogue no cold kana arm beats `retrain_kana`** (`ruler_2026_10_05.md`
  § 3 – 4): `kana_up` ties leaning down (exact 1 / 6, p 0.12);
  `seed_retrain_0930` is the best table.
- **Warm has a mode now.** Verdict 5 holds for the 10-02 arms (μ-pulled,
  on the finished seed); whole-warm with the norm pull off
  (`free_residual = 0`) ties or gains (`sent_whole`, `sent_kanji_f0`).
  **L_pres** (λ 10 · the out-of-box match to the frozen DiT under an EN
  caption, σ 0.8 – 0.9, every 2nd scene step) lifts every page read —
  `en_match` +0.097 (p 7e-8), text area −33 % — with the text at f0's
  (g_f1, cer n.s.) and kana recall −0.066 (p 0.09)
  (`sent_kanji_pres_2026_10_07.md`).
- **Identity has no row direction of its own** (`probe_twin_2026_10_08.md`,
  rows at preview51): swapping one glyph turns the neighbours' rows 0.6 – 0.9
  as much as its own row. The loss gives every row in a box the same
  credit; `idea.md`.

## 7. The motivation (10-14)

1. **Glyph size is bound to the items that taught identity.**
   `retrain_kana`'s identity came with its σ 0.7 – 0.9 large items
   (`grid_82`, `lone_190`, `bubble1_52`), and so did its rendered size: the
   lone tier scales the banner glyph 75 → 87 → 131 px (§ 3), the seed draws
   a ≈ 78 px banner. The reseed took those items out and lost the identity
   with them (§ 6).
2. **The bands are older observations.** The band law (the EN ceiling
   table) and the gradient read (step 0, cold rows at the pack) were both
   read on banners, not on the dialogue target; the one recipe past them
   (`kana_up`) moved the read.
3. **The page is the warm pass's job now.** L_pres fixes the page warm
   without costing the text (§ 6), and it stays warm-only (user 10-14). So
   the cold pass can be judged on identity alone: rows whose identity does
   not ride on one glyph size, every row at even exposure. Asked with it:
   does that cold pass need fewer steps than `retrain_kana`'s 135 / row?

**Exposure, measured** (10-14, `output/cjk_anima_scale/retrain_kana/data/`,
23 490 steps × batch 4 over 17 400 items = 5.4 epochs). A hiragana row is
drawn 2 465 times on average (median 1 987, 788 – 7 209): 75 % in windows
(`bubbleN_18` 1 086, `bubbleN_34` 760 a row), 18 % in the large grids
(444), the rest single glyphs (bubble1 156, flat 1×1 20). Windows follow
the dialogue's frequency; a grid or lone deck deals every row alike.
`retrain_kana`'s items: windows 56.7 %, grids 13.3 %, bubble1 26.7 %,
flat 1×1 3.4 %; glyphs an item 4.4 (windows), 6.19 (grids), 1.

## 8. The two cold arms

Both: the 81 hiragana rows (`retrain_kana`'s 80 + ゔ) cold from the pack,
every other row frozen at `seed = "0930"` (so the pair against
`retrain_kana` differs in those 81 rows only, as `ball_rk_bubble` /
`gs_rkstick` did), `reseed/table.py`'s tiers. The mix (user 10-14) — grid
50 / lone 10 / bubbleN 20 / bubble1 20 — split inside each form by
`TABLE`'s own ratios unless set otherwise:

| tier | share | A: `hp` band | B: upper + 0.1 (cap 0.9) |
|---|---|---|---|
| `grid_44` | 10 | 0.55–0.75 | 0.55–0.85 |
| `grid_29` | 20 | 0.4–0.7 | 0.4–0.8 |
| `grid_16` | 20 | 0.2–0.6 | 0.2–0.7 |
| `lone_44` / `lone_28` / `lone_16` | 3.3 / 3.3 / 3.4 | 0.45–0.8 / 0.35–0.65 / 0.2–0.5 | 0.45–0.9 / 0.35–0.75 / 0.2–0.6 |
| `bubbleN_34` / `bubbleN_18` | 8.2 / 11.8 | 0.45–0.7 / 0.2–0.5 | 0.45–0.8 / 0.2–0.6 |
| `bubble1_52` / `bubble1_32` | 11.1 / 8.9 | 0.55–0.8 / 0.35–0.6 | 0.55–0.9 / 0.35–0.7 |

- **Steps.** Glyphs an item ≈ 4.2 – 4.3 (grid 6.25 over 2×2 / 3×3 / 2×3 /
  3×2, windows 3.6 – 4.5, singles 1). `retrain_kana`'s mean exposure is
  81 × 2 465 / (4 × 4.25) ≈ 11 750 steps (145 / row); the arms take
  `steps_per_row = 150` (12 150 steps, ≈ 95 – 100 min each). A row is then
  drawn ≈ 1 850 times in grids (4× `retrain_kana`'s large grids), ≈ 60 lone,
  ≈ 120 in bubble1 (single glyphs 30 % of the items, as `retrain_kana`),
  ≈ 470 in windows (¼ of `retrain_kana`'s).
- **B needs its knob back.** `upper_shift` (kana_up's) was pruned from
  `reseed/config.py`; its config is in `_archive/configs/kana_up.toml`
  (freeze point `b3dee68e`). `UPPER_MAX = 0.9` already caps it.
- **The package is ahead on `cjk-ko`** (cjk_ko's tiers and knobs on
  `reseed/`): build and train from that tree, or merge it here first.

**Reads.** The dialogue ruler (`criteria.md`, `ruler.py run`) against the
cached `retrain_kana` / `seed_retrain_0930` renders: g_f1, exact, cer,
short kana exact. The steps question on the same ruler from checkpoints
(`trainer.SAVE_EVERY`, 5 000 now, lowered to read ≈ 50 / 100 / 150 a row).
Singles and the kana run's 13 words as diagnosis, not as the verdict.
Size binding has no read yet: the rendered glyph px against the bubble the
prompt asks for is the candidate.

**Risks.**
- `grid_small` was grid-heavy too and lost; it had no glyph above 40 px and
  nothing over σ 0.7. A has `grid_44` at 0.75; B takes it to 0.85.
- In-word exposure falls to ¼ of `retrain_kana`'s, and composition came
  from the in-word items (`p1_lone` 9 / 128, C2). Words and `dup` decide,
  not singles.
- `kana_up`'s gain was near misses, not exact words, and it cost the scene.

**After.** The arm that holds against `retrain_kana` goes warm on
`sent_kanji_pres`'s recipe (whole-warm, `free_residual = 0`, L_pres).
