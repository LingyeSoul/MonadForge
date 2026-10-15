# hangul_init — the jamo factors from a glyph prediction (2026-10-15)

User 10-15, after `../cjk_anima_reseed/reports/hira_cold_2026_10_14.md` § 5:
there a PE-Spatial-B16 glyph prediction ranks a hiragana row 2.3 / 81, but the
start buys nothing on the kana read. Every arm already picks its own row
(80 – 81 / 81). Hangul is not saturated (J256_ko_cf: trained ≈ 18 / 56, H
10 / 32), so a start has room here. The lone curriculum
(`jamo_curriculum_2026_10_11.md`) put each jamo's standalone row into its
factor and cost H, because the trained factors were orthogonal to the lone
directions. This start predicts whole syllable blocks instead and projects
them onto the factors.

## 1. Offline (`probes/hangul_init.py`, job `20261015-161228-b95c3f`, 1.7 min)

`results/20261015-1614-hangul_init/`.

- **Ridge.** PE-Spatial-B16 tokens pooled 4×4, fitted on seed_1008's
  katakana (83) and kanji (1 183, the ones `glyph_init`'s cache draws). It
  predicts the 1 056 syllables that j256_ko, j256_ko_cf and j1024_ko train,
  plus H.
- **Faces.** The syllables are drawn in the runs' KO faces (`kozh/` +
  `ko_extra/`, NanumPenScript out).
- **Target.** Each run's trained syllable `e` (offset − the syllables' stick,
  `q̂` out).
- **Ranks.** Rank among the run's syllables, and among the same layout
  (V / H / C ± final) only.
- **Factor cos.** C and V are centred over their used cells first: a shift of
  every C by u and every V by −u composes the same.

Ceiling (independent factor runs, the 256 they share):

| pair | top-1 / 256 | rank | same-layout rank (chance 29) | factor cos C / V / F |
|---|---|---|---|---|
| j256 · j256_cf | 245 | 1.1 | 1.3 | 0.42 / 0.53 / 0.49 |
| j256 · j1024 | 239 | 1.1 | 1.4 | 0.38 / 0.48 / 0.51 |
| j256_cf · j1024 | 222 | 1.7 | 1.7 | 0.36 / 0.44 / 0.48 |

The prediction (no Hangul row seen):

| target | n | top-1 | rank (chance) | same-layout rank (chance) | projected top-1 / rank | factor cos C / V / F (shuffled) |
|---|---|---|---|---|---|---|
| j256 | 256 | 37 | 20.1 (128) | 9.3 (29) | 43 / 19.6 | 0.11 / 0.12 / 0.17 (0.01 / 0.06 / 0.00) |
| j256_cf | 256 | 47 | 19.9 (128) | 8.6 (29) | 46 / 19.5 | 0.13 / 0.11 / 0.14 (0.01 / 0.04 / −0.01) |
| j1024 | 1 024 | 50 | 76.1 (512) | 42.7 (160) | 60 / 66.4 | 0.12 / 0.12 / 0.17 (0.02 / 0.04 / 0.00) |

Shuffled targets rank 86 – 95 / 256.

1. **The identity is more than the layout.** Within the layout the rank is
   9 / 29.
2. **The projection keeps it.** The factor span loses nothing (j1024 top-1
   50 → 60).
3. **The factors point the right way, weakly.** cos 0.11 – 0.17 against
   ≈ 0 for shuffled and for the lone curriculum. That is about ⅓ of the
   independent-run ceiling.
4. **It is weaker than kana's.** The rank is 8 % of n here against 3 %
   there, cross-script and in other faces.

## 2. Arm `jamo_j256_ko_cf_pe`

`configs/jamo_j256_ko_cf_pe.toml` is J256_ko_cf on its own data
(`data_from`), with the same mix, steps and σ / ε draws. The factor start is
`factor_init = { glyph = "pe_spatial", radius = 211.0 }`
(`Jamo.init_glyph`): the least-squares C / V / F are scaled so the composed
rows centred over the run's 584 rows have seed_1008's kanji ball radius, and
b starts at 0. The start: scale × 4.60, radius 211.0, |composed mean| 106.1
(a common part C / V / F carry, outside b), |Δ| 230.

- Train job: `20261015-161623-6109d5`.
- Read (`probes/micro_read.py`, floor / cf / cf_pe; trained, next, held, H):
  render `…-7f6fdb`, score `…-532879`.
- Result: pending.
