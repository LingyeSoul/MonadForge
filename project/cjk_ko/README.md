# cjk_ko

Hangul rows on KO dialogue: `proposal_jamo.md` § 4 (phase 2) of the reseed
line (`../cjk_anima_reseed/`), opened 2026-10-11 on the `cjk-ko` branch. The
start is `jamo_j256_ko`'s merged rows: seed_1008's JA rows, the 256 syllables
J256_ko trained, and the other 10 916 at their jamo composition
(`reports/jamo_j256_ko_2026_10_11.md`: H exact 8 / 32, jamo F1 0.46).

The line owns its configs, corpus and reads; the builder and trainer are the
reseed line's `reseed/` package, run through `run.py`. Outputs:
`output/cjk_ko/<run>/`.

## Code

| file | what |
|---|---|
| `corpus.py` | CPU: songys Chatbot_data (MIT, `~/ko_corpus/songys_chatbot/`) → `~/ko_corpus/derived/`: `chatbot_lines.tsv` (the line file, `lines = "ko_chat"`), `chatbot_held.tsv` (the ruler's 96 strings, 32 each short 2–4 / mid 5–9 / long 10–20 syllables), `chatbot_rows.txt` (the line file's syllables by frequency). Normalisation and the hold-out rule in its docstring |
| `run.py` | front door: `run.py <run> data [--frac f] \| train` → the reseed line's `run.py` on `configs/<run>.toml` |
| `configs/ko_chat_warm.toml` | warm A: every syllable of the lines free, warm from `cjk_anima_reseed/jamo_j256_ko`, sent_kanji_pres's recipe (lr 2e-4, free_residual 0, 40 steps a row, L_pres λ 10 at σ 0.8–0.9 every 2nd step) |

| `configs/ko_micro_{a,b,bp}.toml` | the micro arms (user 10-11): the 192 most frequent syllables, 20 steps a row (3 840), no L_pres, on the lines of those 192 (`corpus.py --rows 192` → `ko_chat_192`). A = free rows; B = the jamo factors only, warm from j256_ko's (`factor_from`); B′ = B + a per-syllable residual (`factor_residual`). One data build (`ko_micro_a`), B / B′ `data_from` it |
| `configs/jamo_j1024_ko.toml` | the J line to 1024 (user 10-11): jamo_j256_ko's rows + the next 768 chatbot syllables by frequency (KS X 1001, H out; `corpus.py --jamo 1024` → `assets/jamo1024.json`) composed from the jamo factors, plus the 51 compat jamo as free rows (`Jamo`'s own rows); jamo_j256_ko's recipe and data cold on seed_1008, 56 a row (60 200 steps) |
| `configs/jamo_j256_ko_g44.toml` | jamo_j256_ko with the 16 px tiers cut (user 10-12): lone_16 / grid_16 5 % each, grid_44 31.7 %, the rest as built; same rows, recipe, steps |
| `configs/jamo_j256_ko_part.toml` | jamo_j256_ko + the 48 partial forms (user 10-12): the medial vowels ᅡ … ᅵ and finals ᆨ … ᇂ drawn with the initial left out (`jamo.PART_SPELL`, Noto Serif CJK), composed without the absent parts (ᅥ = b + V[ㅓ]); the opt-in `partial` tier at 15 %, J256_ko's mix at 85 %, 56 a row (17 024 steps) |
| `configs/jamo_j256_ko_cf.toml` | jamo_j256_ko + counterfactual syllables (user 10-12; `../cjk_anima_reseed/reseed/counterfactual.py`): each trained syllable less its initial (언 → ㅓ ㄴ, b + V + F) or its vowel (언 → ㅇ ㄴ, b + C + F), the 287 forms on hosts; less the final = the CV syllable as a row (41, H out); the counterfactual tier 25 %, steps held at 14 336 (`steps`) |
| `configs/jamo_j1024_ko_cf.toml` | jamo_j1024_ko + jamo_j256_ko_cf's counterfactual forms at 25 % of the items — 15 % on the flat canvas (`counterfactual`) and 10 % in a scene bubble (user 10-13: `cf_bubble1_52` / `cf_bubble1_32` at 5.5 / 4.5, bubble1_52 / bubble1_32's bands and px_keep, Noto Serif CJK) — steps held at 60 200; `faces.legible` on (user 10-12, `../cjk_anima_reseed/reseed/legible.py`): from 28 font px a face is re-picked for a glyph whose strokes it fuses (median gap < 1.5 px or p10 < 0.8 px) — ~7 % of the 28–40 px draws, the 16 px tiers untouched. Not run yet |
| `probes/micro_read.py` | GPU: `render` (each read item alone in a KO bubble, seed 0, under `floor` = jamo_j256_ko and the micro arms; cached under `output/cjk_ko/micro_read/render/`) and `score` (v4 free read; per arm × group exact and glyph / jamo P R F1 with chance) → `results/<ts>-micro-read/`. Groups (`assets/micro192.json`): every third of the 192 (64), the next 64 by frequency (in no line), the 27 held strings on the 192; `--seeds` reads several renders an item |

```bash
.venv/bin/python project/cjk_ko/corpus.py
.venv/bin/python project/cjk_ko/run.py ko_chat_warm data --frac 0.03   # sizes: sheet_<tier>.png
.venv/bin/python project/cjk_ko/run.py ko_chat_warm data
make daemon-run ARGS="project/cjk_ko/run.py ko_chat_warm train"
.venv/bin/python -m pytest project/cjk_ko/tests
```

## What the reseed package does for a KO run

Added 10-11 (`reseed/config.py` docstring, `table.py`, `recipes.py`); a JA
run draws what it drew.

- `rows_from = "<line>/<run>"` warms from `output/<line>/<run>/trained.pt`.
- A config outside the reseed line writes to `output/<its line>/<run>/`.
- `lines = "ko_chat"` (`LINE_LANG`: korean) gives a `lang` run its windows
  and `sent` lines (no JA dataset text joins them) and stamps
  `glyph_route_ko` in `build.json`, so the trainer routes Hangul tokens per
  syllable.
- `sentko_34` / `sentko_22` (opt-in): a KO line in 1–4 horizontal lines
  broken at its spaces, at sent_34 / sent_22's font px and bands; lengths in
  cells, spaces not counted; 70 % of the lines left-aligned, the rest
  centred (`SENT_H_LEFT`). A KO window (`bubbleN`) is a whole word of the
  lines (`word_pool`: 2–6 syllables, edge `?!…` dropped), never a piece of
  one, drawn horizontal on any scene. Neither caption says `horizontal`.
- A line's spaces pass the encoding check (no T5 row; the routed stream
  drops them); its letters count Hangul syllables.
- L_pres's EN caption names `Korean text` / `korean text` English too.

## Reads

`findings.md`: the J line's verdict so far — better data and counterfactual
syllables move H, tier reshuffles and × 4 scale do not.

- `reports/micro_2026_10_11.md`: the micro arms A / B / B′ — every arm
  learned to draw a line for a lone syllable (exact 18 → 1–2 / 64 against
  j256_ko); B′ leads on the held strings (glyph F1 0.27 vs A 0.18); B / B′
  carry the line habit to untrained syllables. The data mix comes first.
- `reports/jamo_j1024_ko_2026_10_12.md`: the J line to 1024 syllables + the
  51 jamo — trained rows read as J256's own (new rows exact 5 → 17 / 61), the
  jamo draw (~30 / 51, as jamo_lone alone); H flat (8 → 7 / 32): scale buys
  trained rows, not composition. A better start for the dialogue warm.
- `reports/jamo_j256_ko_g44_2026_10_12.md`: the 16 px tiers cut into
  grid_44 — no gain; H and trained fall (H exact 8 → 6 / 32, glyph F1 0.31 →
  0.22). Size and σ band move together in the table, so it does not say
  which carried the loss.
- `reports/jamo_j256_ko_cf_2026_10_12.md`: counterfactual syllables (one
  jamo left out, 25 % of the items) — the first J arm to move H (exact 8 →
  10 / 32, glyph F1 0.31 → 0.35, jamo F1 0.46 → 0.52), next 8 → 13 / 64, the
  gain on syllables J256 did not train (13 → 20 / 93); held strings fall
  (glyph F1 0.17 → 0.09). Seeds 0–3 keep the direction, smaller: H 18 → 26
  / 128, next 17 → 32 / 256 (H short of significance). Forms vs the tier
  itself not separated.

## Open

- **jamo_j1024_ko_cf, ready for a RunPod 5090** (10-12; no 5090 in stock that
  night, not launched). Config and `faces.legible` committed; a 3 % build
  checked locally (3 274 items, 1 718 rows; the bubble1_32 sheet draws no
  fused glyph). On the pod, as J256_ko: the tracked code (`git ls-files -z`)
  plus 1.24 GB of inputs over 8 `rsync -aL --mkpath` shards —
  `../cjk_anima_reseed/assets/fonts/`, `output/cjk_anima_scale/scenes_{s1,
  s1s,s1w,sl1w,ja_comic}/`, `output/cjk_anima_scale/experiments/
  scene_{colorful,bubble_check}.json`, `models/vocab_packs/
  anima_cjk_vocab_pack_punct.{json,safetensors}`,
  `output/cjk_anima_reseed/seed_1008/trained.pt` — and the
  `output/wake_probe/scenes_{s1,s1w,sl1w,ja_comic}` links with plain
  `rsync -a` (the scenes' `scenes.jsonl` paths go through them). Then
  `run.py jamo_j1024_ko_cf data --workers 12 && run.py jamo_j1024_ko_cf
  train`; pull back `trained.pt`, `train_log.json`, `train_record.json`,
  `eval_coverage.json`, the data's manifests and `sheet_*.png`. Estimate:
  ~110 k items, ~10 min build, train ~3.7 h at J256_ko's 4.5 it/s (local:
  J1024_ko took 419.8 min at 2.39 it/s), ~4–4.5 h and ~$5 on SECURE. The
  cf masks need `uharfbuzz` + `freetype-py` (not in pyproject; a `uv sync`
  drops them): `uv pip install uharfbuzz freetype-py` on the pod before
  `data`. Read against jamo_j1024_ko: cf (flat and bubble) and the legible
  re-pick move together.
- **The read.** No KO ruler yet: the 96 held strings are cut out of the data
  now; the renderer and v4 per-glyph read are still to build (the JA ruler's
  shape, `criteria.md`). Also H (the jamo line's 32 held syllables) — they
  are all in the chatbot rows, so warm trains them directly.
- **Mode B**: the factors warm with a per-syllable residual (`proposal_jamo`
  § 4), so syllables outside the lines move too.
- **hitocrawl**: 133 KO editions (3 569 pages) to OCR into a second line
  file — manga register, licence unclear.
- The T5 stream holds no space between KO words (the routed encoder drops
  it); only the Qwen text keeps the word breaks.
