# findings — the J line on KO, phase 1 (2026-10-12)

What the J arms (jamo-factored Hangul rows, cold on seed_1008, lone glyphs in
bubbles / grids / signs) have shown so far, read by v4 per glyph against
`floor` = jamo_j256_ko. The out-of-distribution test is **H**: 32 common
syllables no arm trains, drawn only through the composition b + C[cho, cls] +
V + F. Every read in the table is seed 0, one render per item (cf vs floor
also on seeds 0–3, below); the reports carry the full tables.

## H across the line

| arm | what changed | steps | H exact | ~ | glyph F1 | jamo F1 | report |
|---|---|---|---|---|---|---|---|
| J128 | — | 7 168 | 3 | 2 | 0.06 | 0.27 | `reports/jamo_vl_reads_2026_10_11.md` |
| J256 | set + steps × 2 | 14 336 | 5 | 3 | 0.15 | 0.32 | `reports/jamo_j256_2026_10_11.md` |
| **J256_ko** | the data redrawn (faces, pose) | 14 336 | **8** | 4 | **0.31** | **0.46** | `reports/jamo_j256_ko_2026_10_11.md` |
| J1024_ko | set + steps × 4 (+ 51 jamo) | 60 200 | 7 | 5 | 0.22 | 0.43 | `reports/jamo_j1024_ko_2026_10_12.md` |
| J256_ko_g44 | tier mix: 16 px → grid_44 | 14 336 | 6 | 2 | 0.22 | 0.41 | `reports/jamo_j256_ko_g44_2026_10_12.md` |
| **J256_ko_cf** | + counterfactual syllables (25 %) | 14 336 | **10** | 4 | **0.35** | **0.52** | `reports/jamo_j256_ko_cf_2026_10_12.md` |

(~ = one syllable sharing a jamo with the target; jamo F1 chance ≈ 0.12–0.13
on every arm.)

Seeds 0–3, H / next only (`results/20261012-2005-micro-read/`): floor H 18 /
128, cf **26 / 128** (glyph F1 0.17 → 0.25, jamo F1 0.37 → 0.43); next 17 →
**32 / 256**. Seed 0 is the best seed for both arms.

## Verdict

- **Better data moved H; reshuffling the mix did not.** J256 → J256_ko kept
  the rows, recipe and steps and redrew the data — faces curated
  (NanumPenScript out, the 10 `ko_extra` faces in, thin / display faces
  floored by px) and `pose` (the glyph's ink centred in its bubble / sign
  and tilted with it). H exact 5 → 8, glyph F1 0.15 → 0.31, jamo P and R
  both up with chance flat — a larger step than J128 → J256's doubling.
  Moving the share between tiers of the same data (g44: the 16 px tiers into
  grid_44, larger glyphs) gave nothing: H 8 → 6, trained 18 → 11, next
  flat, and every arm draws the glyph at the same size whatever size it
  trained on.
- **Counterfactual syllables help composition, weakly.** Each trained
  syllable with one jamo taken out (less the initial / less the vowel, the
  rest in place) as 25 % of the items: H 8 → 10 / 32, jamo F1 0.46 → 0.52,
  next 8 → 13 / 64, trained holding (18 → 19) at ~0.75× the exposure to the
  built tiers. The gain sits on syllables J256 did not train (13 → 20 / 93;
  trained ones 19 → 18 / 56) — the first lift on the zero-shot side. Four
  seeds keep the direction at a smaller size: H 14 % → 20 % exact (paired by
  item 8 ahead / 4 behind, t ≈ 1.6), next 7 % → 13 % (15 / 7, t ≈ 2.1).
- **Scale buys trained rows, not OOD.** J1024_ko (rows and steps × 4) reads
  its new rows about as well as J256's own (5 → 17 / 61 new rows, next 8 →
  15 / 64), and the 51 jamo cost nothing to add (~30 drawn). H does not
  move (8 → 7, jamo F1 0.46 → 0.43): more rows of the same data do not lift
  the composition's ceiling. Coverage is what scale is for.
- **A mix that teaches lines costs lone identity.** The micro arms (warm
  from J256_ko on 70 % sentko lines, `reports/micro_2026_10_11.md`) made
  every lone syllable draw a line: trained exact 18 → 1–2 / 64. The mix is
  not free to change once it leaves lone glyphs — the one mix change that
  moved anything moved it down.
- **The held strings are out of reach of lone-glyph data.** 0–1 / 27 under
  every J arm; cf lowers their glyph F1 (0.17 → 0.09), as its tier is lone
  forms on a flat canvas. Strings are the dialogue warm's job.

## Not separated

- **One seed, one render per item** for every arm but cf. Hits swap both
  ways between arms (cf vs floor: 20 cf-only, 14 floor-only of 149), seeds
  spread wider than the steps (cf on H: 10 / 5 / 9 / 2), and v4 itself flips
  ±2 / 32 H reads on identical renders when the batch changes. The cf and
  J256_ko steps move every H column the same way, which is what the verdict
  leans on; cf's holds on four seeds, H short of significance.
- **J256_ko's data change is three things** (faces out / floored, the
  `ko_extra` faces, `pose`); which carried the gain is not read.
- **cf is three things too**: the dropped-jamo forms, a new 25 % flat-canvas
  grid tier at σ 0.3–0.75 in Noto Serif CJK, and 13 % less of the built mix.
  The arm that separates them keeps the tier and draws full syllables.
- **g44 couples size with σ**: cutting the 16 px tiers also cut their
  σ 0.2–0.6 band.
- **J1024 scales set and steps together** (× 4), as J128 → J256 did.

## Open

- `jamo_j256_ko_part` (the 48 partial forms, 15 %): stopped at step 1 525,
  no read.
- The cf-tier-without-cf arm (hosts as full syllables, tier kept).
- The dialogue warm on J1024_ko (or J1024 + cf): coverage from scale,
  composition from cf, strings from a mix that does not cost lone identity.
