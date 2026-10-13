# plan — the KO dialogue warm on jamo_j1024_ko_cf, micro arms A / B / B′ (2026-10-13)

User 10-13: warm the KO sentence run from `jamo_j1024_ko_cf`, and read it as
the 10-11 micro arms were read (`reports/micro_2026_10_11.md`), against those
arms. Nothing below is built or run yet.

## 1. The start

`output/cjk_ko/jamo_j1024_ko_cf/trained.pt`: J1024_ko's rows and 51 compat
jamo + the counterfactual forms (25 %, flat and bubble) + `faces.legible`,
60 200 steps on a RunPod 5090 (236 min). No residual, so every syllable row is
its jamo composition (the 626 cf hosts back at their own, checked 626 / 626).

Its read, seed 0, one render each (`results/20261013-1452-micro-read/`, the
three arms scored in one batch):

| group | floor (j256_ko) | jamo_j1024_ko | jamo_j1024_ko_cf |
|---|---|---|---|
| trained (64) | 21 | **29** · glyph F1 0.34 | 21 · 0.29 |
| next (64) | 9 | 16 · 0.26 | **22** · 0.32 |
| held strings (27) | 1 | 0 · 0.11 | 0 · 0.08 |
| H (32) | 6 | 9 · 0.26 | **16** · 0.45 |

The cf start leads where the composition carries the syllable (H, next) and
trails on trained rows and strings. Strings are what this run is for.

## 2. The arms

The 10-11 micro design with only the start moved: the 192 most frequent
chatbot syllables (`assets/micro192.json`), the lines made of them
(`ko_chat_192`, 3 153), 20 steps a row (3 840 steps, batch 4), lr 2e-4
cosine, free_residual 0, no L_pres, punct pack, seed_1008's JA rows frozen.

| arm | config | rows |
|---|---|---|
| A_cf | `ko_warm_cf_a` | the 192 free, `rows_from = "cjk_ko/jamo_j1024_ko_cf"` |
| B_cf | `ko_warm_cf_b` | the 192 composed from the factors only, `factor = "jamo"`, `factor_from = "cjk_ko/jamo_j1024_ko_cf"` |
| B′_cf | `ko_warm_cf_bp` | B_cf + a per-syllable residual, `factor_residual = 0.0` |

As on 10-11, B's step-1 loss should equal A's (the factors compose the start's
rows exactly). Check it before reading anything else.

Of the 192, 166 are J1024 rows; the other 26 are the H syllables (26 of H's
32 sit in the 192). See § 4.

## 3. The mix — to choose before the build

The 10-11 mix (bubble1 5 / bubbleN 25 / sentko 70) taught every arm to draw
a line for a lone syllable: trained exact 18 → 1–2 / 64 and drawn 3 → 6.
Three options:

1. **The 10-11 mix, unchanged.** Reuses `ko_micro_a`'s build
   (`data_from = "ko_micro_a"`, 19 200 items, still on disk, 14 GB), so it
   needs no build. Only the start differs from 10-11, so the arm-by-arm
   comparison against ko_micro_a / _b / _bp is clean. The line habit is
   expected again.
2. **The 10-11 report's next mix**: bubble1 20 / bubbleN 40 / sentko 40
   (grid / lone perhaps on). This needs a new build (~5 min CPU). It reads
   whether a mix that keeps lone glyphs holds the start's identity while the
   strings rise. It moves the start and the mix together against 10-11.
3. **2 + the counterfactual tier kept (~10–15 %)**, so B / B′ do not trade
   away the composition gain the start carries. Before the build this needs
   a fix: with `factor_residual` set (B′), `Jamo.merge_all` skips every row
   that has a residual (`reseed/jamo.py:308`), so the cf hosts would be
   saved as their forms.

## 4. The read

`probes/micro_read.py render` on the three new arms only, then one `score`
over every arm, because the reader moves between batches (±2–3 a group on
identical renders, 10-13). The arms:

- floor (j256_ko), ko_micro_a / _b / _bp: the 10-11 arms, renders cached
- jamo_j1024_ko_cf: the start, renders cached
- A_cf / B_cf / B′_cf: new, added to `ARMS`

Groups: trained (64), next (64), held (27), H, jamo. What each one now
measures:

- **held**: the run's target. No arm has read one exactly yet (0–1 / 27).
- **trained**: lone identity under the line mix. The start reads 21 / 64,
  the 10-11 arms 1–2.
- **next**: syllables the warm does not train (62 of 64 are J1024 rows). A
  leaves them at the start; B / B′ move them through the factors. 10-11:
  8 → 2 under B / B′.
- **H** is no longer out-of-distribution: the warm trains 26 of its 32 in
  every arm. Score it split: H ∩ 192 (26) with trained, the other 6 (번 성 in
  next, 4 in neither) as the composition check.
- **jamo**: by eye (v4 reads no lone jamo). The start already trails
  j1024_ko there (ㄱ → 그).

Seed 0 first. Seeds 0–3 on held and trained if the arms sit within the
reader's noise.

**Judged on** held strings rising (glyph / jamo F1 over chance, exact)
while trained stays near the start's 21 / 64, and next does not fall under
B / B′ as it did on 10-11.

## 5. Budget (local GPU, through the daemon)

| step | time |
|---|---|
| build | 0 (option 1) / ~5 min CPU (options 2, 3) |
| train | 3 × ~27 min (2.3 it/s on 10-11) |
| render | 3 × ~11 min (238 items an arm, as jamo_j1024_ko_cf's) |
| score | ~3 min (8 arms) |

~2.2 h GPU. The local soup queue holds the GPU until ~19:30 on 10-13, so
queue behind it.

## 6. Code before the run

- `configs/ko_warm_cf_{a,b,bp}.toml`: ko_micro_*'s with `rows_from` /
  `factor_from` moved and the § 3 mix.
- `probes/micro_read.py` `ARMS`: the three arms; the H split in `items` /
  `score`.
- Option 3 only: `merge_all` writes the hosts' own composition under a
  residual (and a test).
