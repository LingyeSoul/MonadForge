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

## Open

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
