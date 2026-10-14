# Opus relabel of the low-confidence tail — hito + gelnote (2026-10-14)

Opus agents hand-labelled 1,200 crops: 1,000 from hito and 200 from gelnote. A
second Opus pass then checked every label it would keep against its own crop.
**469 labels are usable.** They are in `manifest_pseudo_opus_relabel.parquet`
(`kind = sfx`, `source = opus`); train with `--extra_manifest pseudo_opus_relabel`.
`hito/opus_manifest.py` rebuilds the manifest.

## Which crops

- **hito (1,000):** the reads in `prelabel_v4lp.jsonl` with `n_tokens ≥ 4`, lowest
  logprob first (cut at −0.544; 738 KO-edition crops, 262 JA-edition). We did not
  cut by logprob alone. Its bottom 1,000 are 99 % 1–2-token reads (`…`, `!`, a
  single kana), and a 16-crop sample of them was 16 / 16 with no text: misregistered
  diff noise. The guards already drop those.
- **gelnote (200):** 100 KO + 100 ZH. They are drawn at random (seed 0) from the
  rows that the stock × hayai25 filter dropped and that have `n_tokens ≥ 4` and no
  runaway. The gelnote sweeps store no logprob, so disagreement stands in for low
  confidence.

The labelling agents never saw the reader's output, so the old read could not
anchor them. Each crop got one status (`ok / no_text / multi / cut / illegible`)
and a K3-style label: hearts as ♡, `…` for exactly 3 dots, KO lines joined with
a space. Every agent refused crops showing sexual content with characters who
look like minors. Those galleries are all marked `excluded`, and the list lives
outside the repo (`hito_ocr/FLAGGED.md`).

## Result

| pool | n | usable | excluded | no_text | multi | cut | illegible | dropped by check |
|---|---|---|---|---|---|---|---|---|
| hito KO-edition | 738 | 289 | 285 | 53 | 33 | 53 | 23 | 2 |
| hito JA-edition | 262 | 7 | 172 | 59 | 7 | 11 | 6 | 0 |
| gelnote KO | 100 | 80 | 1 | 0 | 4 | 10 | 3 | 2 |
| gelnote ZH | 100 | 93 | 0 | 0 | 0 | 7 | 0 | 0 |

- **hito KO = Korean hand-lettered SFX** (출렁, 퓨얏, 띠옹 띠옹 띠옹, 꿈이에혀 …).
  v4's read matched the Opus label on **0 / 345** first-pass `ok` rows. v4 reads
  this lettering as katakana (`キイロゥー`, `ムイラロ`, `カイロッ`), so these are
  the rows v4 cannot produce on its own.
- **hito JA:** the low tail on the JA side is almost all junk. Only 7 / 262 are usable.
- **gelnote:** the stock read already matched on 54 / 175 first-pass `ok` rows.
  Agreement threw those away because the voter (hayai25) misread them, not the
  teacher.
- **Check pass** (473 `ok` rows outside excluded galleries): 454 match, 15 fixed
  (dot counts, a final consonant, a heart count), 4 dropped (text clipped or two
  balloons), 0 labelled from the wrong image. The wrong-image check was needed
  because labelling agents had shared one zoom folder.

Files: `hito_ocr/ko/prelabel_opus.jsonl` and `gelnote_crops/relabel_opus.jsonl`
hold all 1,200 rows. Each row has `status`, `lang`, `label` and `note`, plus the
old v4 / stock read.
