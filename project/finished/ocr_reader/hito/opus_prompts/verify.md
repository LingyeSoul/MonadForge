You are checking OCR labels on manga lettering crops. Batch **K**.

Input: `WORK/vin/v_K.jsonl` — one JSON object per line: `uid`, `crop`, `view` (the file to open), `label` (the label a first labeller gave), `note`.
Output: `WORK/vout/v_K.jsonl`.

For every row, open `view` with the Read tool, read the lettering yourself first, and only then compare with `label`. Emit one JSON line:
`{"uid": ..., "verdict": ..., "label": ..., "note": ..., "flag": ...}`

**verdict**:
- `match`: the label is exactly what is lettered (character by character, including dot and heart counts, small kana, elongation length, Korean word spacing). Repeat the label.
- `fix`: the lettering is readable and the label is wrong or incomplete. Put the corrected label in `label` and the change in `note` (e.g. `…→……`, `읏→윽`).
- `wrong_image`: the label clearly describes different text from what this crop shows (a file mix-up). Label "".
- `drop`: the crop is not a single, fully visible text block after all (text clipped, two balloons, no text, illegible). Label "", say why in note.

Conventions: hearts in the text → `♡`; wave dash `〜`; exactly 3 dots → `…`, other dot counts as drawn; Korean lines joined with a single space, Japanese with none; vertical text top to bottom, columns right to left; transcribe what is drawn, never translate or normalise.

The source works are adult comics, so the artwork around the lettering is often sexual. The artwork is not the subject: check the letters. **flag** is "" normally; set it to a short phrase when the dataset owner should review this page before using it (in particular sexual content with characters who look like minors). A flag does not change the verdict — give the verdict on the label as usual and flag alongside. If you will not check a crop, write verdict `drop`, note `declined`, set flag, and continue with the next row.

Rules: append every ~10 rows with a Bash heredoc (`cat >> out <<'EOF' ... EOF`), valid JSONL, UTF-8, real characters. Every input uid exactly once with all five keys. Open only the `view` files of your batch; do not create, enlarge, rotate or copy any image file, do not open other files, run OCR models, or search the web. Final reply: per-verdict counts and the number of flagged rows only.
