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
### Training

Train 101 min. In-box sits under cf from the first step: −0.009 over steps
0 – 1 000, −0.005 to 2 000, −0.001 from 3 000, −0.0004 by 6 850. Out-box
matches cf at every logged step. |row| starts 100 above cf and closes to
≈ 10 (427 against 417).

### Read

`results/20261015-1807-micro-read/`, seed 0, one render an item. drawn is the
mean letters read an image, and the target is one syllable.

| arm | group | exact | drawn | glyph F1 | jamo P / R / F1 |
|---|---|---|---|---|---|
| floor (j256_ko) | trained | 21 / 64 | 2.75 | 0.218 | 0.364 / 0.594 / 0.407 |
| cf | trained | 20 / 64 | 2.98 | 0.251 | 0.407 / 0.620 / 0.449 |
| cf_pe | trained | 18 / 64 | **4.08** | 0.233 | 0.364 / 0.651 / 0.407 |
| floor | next | 9 / 64 | 3.23 | 0.144 | 0.273 / 0.508 / 0.320 |
| cf | next | 13 / 64 | 3.59 | 0.172 | 0.355 / 0.586 / 0.396 |
| cf_pe | next | 11 / 64 | **4.39** | 0.160 | 0.306 / 0.568 / 0.350 |
| floor | H | 6 / 32 | 3.12 | 0.268 | 0.369 / 0.667 / 0.423 |
| cf | H | **10 / 32** | 2.38 | **0.326** | 0.469 / 0.693 / 0.516 |
| cf_pe | H | 7 / 32 | **3.62** | 0.149 | 0.328 / 0.620 / 0.376 |

Held strings: 0 / 27 for both, as for every J arm. Jamo F1 against the other
targets (chance) is flat at 0.10 – 0.12 in every row.

Paired on exact, cf_pe against cf: trained 10 / 12, next 5 / 7, H 4 / 7
(p 0.55 – 0.83).

1. **The glyph start does not help, and it leans the wrong way.** H goes
   10 → 7 and glyph F1 0.326 → 0.149. None of the exact differences is
   significant on one seed.
2. **What moves is how much it draws.** cf_pe reads 3.6 – 4.4 letters an
   image against cf's 2.4 – 3.6, in every group. It loses precision and not
   recall (jamo R holds at 0.57 – 0.65). On H a lone syllable becomes a line
   of text: 은 → `센질거존자여.`, 해 → `랑인드 대`. This is the kana arms'
   `dup` / Latin-filler failure, and the start makes it worse here.
3. **Untested causes.** The start carries a common part |106| that b does
   not hold, in a direction read off the JP rows; the cold `rand` kana arm
   fell back to sentences the same way. Or 211 is too large a start for a
   weak (cos 0.11 – 0.17) direction.

By eye (`sheet_H_0` – `3`), on H:

- **v4 undercounts cf_pe by one.** cf_pe draws 시 large and correct, and v4
  reads it as 지. By eye cf_pe is ≈ 8 / 32.
- **cf_pe alone gets 제 and 세.** cf draws `제언퍼` and 쇄.
- **The extra letters are text boxes and second bubbles** beside or instead
  of the syllable: 은, 해, 요, 소, 할, 록 (`세살라존 래언머주`, `세멀어슨
  러얽미수`). The floor has the same fallback text on 제 and 체; cf_pe falls
  into it more often.
- **문 draws the same sentence template in all three arms**
  (`○질시든 안두`).

Verdict: with the kana arms (`../cjk_anima_reseed/reports/hira_cold_2026_10_14.md`
§ 5), a glyph-predicted start buys loss early and nothing on the renders.
Here it costs H's precision.
