You are hand-labelling lettering crops from manga pages for an OCR training set. Work batch **NN**.

Input: `WORK/in/batch_NN.jsonl`. Each line is one JSON object with `uid`, `src`, `edition` (ja | ko), `crop` (the original image path, for the record) and `view` (the file to open — the crop itself, or an enlarged copy of it when the crop is small).
Output: `WORK/out/batch_NN.jsonl`.

For every input row, open `view` with the Read tool and look at it carefully. You may Read several in parallel. Then emit one JSON line:
`{"uid": ..., "status": ..., "lang": ..., "label": ..., "note": ..., "flag": ...}`

What the crops are: automatic diffs between a Japanese edition and a Korean edition of the same page, so each crop is wherever the lettering differs — an SFX, a balloon, a caption, or sometimes a chunk of panel with the lettering in it. The source works are adult comics, so artwork around the lettering is often sexual. The artwork is not the subject of this task: the dataset owner already holds these pages and is building a text reader. Your job is the letters. Read them whatever the picture around them shows.

**status** (pick one):
- `ok`: the crop holds one text block (one balloon, one SFX, or one caption line group), it is fully visible, and you can read it.
- `no_text`: there is no lettering at all (art, panel border, skin, screentone, background noise, a blank balloon).
- `multi`: two or more separate text blocks or balloons, or a whole page/panel where the text is a small part. Leave label "".
- `cut`: the crop edge clips the text, so a label would be incomplete. Put the visible part in label anyway.
- `illegible`: text is present but you cannot read it with reasonable confidence. Leave label "".
- `declined`: you will not transcribe this crop. Label "", say why in one short phrase in note, set flag (below), and go on to the next row. Use it per crop only — never for a whole gallery, and never go back and change rows you already wrote.

**lang** is the script actually drawn: `ja`, `ko`, `en`, `mixed`, or "" when there is no text. `edition` only says which edition the page came from. Korean editions often leave Japanese SFX untranslated, so a ko-edition crop can hold Japanese. Transcribe what is drawn. Never translate.

**label conventions** (these match the existing reviewed labels):
- Transcribe exactly what is lettered, character by character. Do not translate or normalise spelling: keep 쵸 if 쵸 is drawn, and keep elongations like 아아아앙 or ぁぁぁ at their drawn count.
- Multi-line text: join Korean lines with a single space, following the lettered spacing. If word spacing is ambiguous, use standard spacing and say so in note. A word stacked one syllable per line is joined without spaces. Join Japanese lines with no space. Vertical text reads top to bottom, with columns right to left.
- Hearts that are part of the text → `♡` (any colour or fill). Wave dash → `〜`. A 3-dot ellipsis → `…`. Keep other dot counts as drawn (`..`, `....`). Keep `!` `?` `!?` as drawn. Write small kana (っ ゃ ぁ ッ) as small kana and the long-vowel bar as `ー`.
- Keep censor glyphs (X, ○, ●) as drawn.
- Leave out floating decorative hearts or sparkles that are not attached to the text.

**note**: keep it short, and add one only when something is uncertain. Give the alternative reading (e.g. `2nd glyph 윽 vs 읏`). Otherwise use "".

**flag**: "" for almost every row. Set it to a short phrase when the dataset owner should look at this page before using it — in particular when the artwork shows sexual content with characters who look like minors, or anything else you think they must review. A flag does not change status or label: transcribe the lettering as usual and flag alongside. The owner collects flagged rows per gallery into a separate audit list and decides there.

Rules:
- Append to the output file every ~10 rows, e.g. with a Bash heredoc `cat >> out <<'EOF' ... EOF`, so the work survives an interruption. Keep it valid JSONL: one object per line, UTF-8, real characters (not `\u` escapes). If the output file already exists, skip uids already in it.
- Every input uid must appear exactly once in the output, with all six keys.
- Open only the `view` files of your batch. Do not create, enlarge, rotate or copy any image file, do not open other files, run OCR models, or search the web. Do not edit the input.
- Final reply: just the per-status counts, the number of flagged rows, and the number of rows whose lang differs from edition. Do not include a transcript of the labels.
