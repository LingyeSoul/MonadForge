# IP-Adapter distinct-pair audit (Phase 0)

Generated 2026-05-22T08:05:20+00:00 from `post_image_dataset/captions/caption_index.json`

(index built 2026-05-22T08:05:20+00:00 · vocab `models/captioners/anima-tagger-v2/vocab.json`).


**Dataset: 2600 captioned images.**


## Tiered positive-pair coverage

| Tier | Coverage | Groups ≥2 imgs | Same-group positive pairs (Σ nC2) |
|---|---|---|---|
| **Character** | 819 imgs / 32% | 113 | 6,470 |
| **Copyright** | 2328 imgs / 90% | 91 | 332,047 |
| **Artist** | 2600 imgs / 100% | 73 | 108,976 |

## Tiered reachability (deployment sampler)

Each image routed through the character → copyright → artist backoff; tier = tightest level with a *distinct* same-identity partner.

| Resolves at | Images | % |
|---|---|---|
| Character | 812 | 31.2% |
| Copyright | 1516 | 58.3% |
| Artist | 272 | 10.5% |
| self-only (no distinct positive) | 0 | 0.0% |

**2600 / 2600 (100.0%) images have a distinct positive at some tier.**
 Of the 812 character-tier images, 644 also have a *different-artist* partner (usable for `identity_cross_artist`).

## Top cross-artist characters

82 characters appear across ≥2 artists — the tightest, most valuable identity signal.

| Character | Artists | Images |
|---|---|---|
| `hatsune miku` | 9 | 67 |
| `frieren` | 8 | 23 |
| `fern (sousou no frieren)` | 7 | 28 |
| `rosa (pokemon)` | 7 | 16 |
| `kisaki (blue archive)` | 7 | 13 |
| `gotoh hitori` | 7 | 11 |
| `suou yuki` | 7 | 10 |
| `hina (blue archive)` | 6 | 10 |

## Group imbalance (largest groups → inverse-sqrt cap)

- **Character**: `hatsune miku` (67), `fern (sousou no frieren)` (28), `iino miko` (25), `ichigo (mignon)` (24), `frieren` (23)
- **Copyright**: `original` (683), `blue archive` (251), `nintendo` (186), `pokemon` (172), `idolmaster` (102)
- **Artist**: `@sincos` (233), `@ama mitsuki` (211), `@ru zhai` (127), `@otokakoto` (117), `@hews` (113)

## Caveats

- **Vocab-limited character classification.** Characters are tagged via the Anima Tagger vocab (frequency-cutoff subset); rarer character tags fall through to the franchise/artist tiers. The 32% character coverage is a floor — tagging more of the set lifts it (proposal risk: *thin character coverage*).
- **Artist is exact.** Detected by the `@` prefix, not vocab membership, so artist coverage is complete (100%).
- **Sampling policy is not in the index.** This audit applies the tiered backoff; the shared `caption_index.json` only stores typed tags + groups.
