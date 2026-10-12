# OvisOCR2 — an outside page parser on the crop gates (2026-10-12)

[`ATH-MaaS/OvisOCR2`](https://huggingface.co/ATH-MaaS/OvisOCR2) (commit
`1fc9221`; 0.9 B, Qwen3.5-0.8B base, Apache-2.0) parses whole document pages
to Markdown. It is first on OmniDocBench v1.6 (96.58). The card does not
mention Japanese, Korean, Chinese, vertical text or manga. Reader:
`eval_manga109.OvisReader` (`--reader ovis`), weights in `models/ovis_ocr2`.

The card's recipe is used verbatim: its single page-parse prompt, thinking off,
greedy, `min_pixels` 448² / `max_pixels` 2880², with `<img src="images/bbox_…">`
blocks dropped. Two harness fixes were needed:

- **EOS.** The repo ships no `generation_config.json`, and `config.json`'s eos
  is `<|endoftext|>`. The chat turn ends at `<|im_end|>`, which is the
  tokenizer's eos and what vLLM stops on. Without it, `generate` runs past the
  answer and repeats `<think></think>` blocks until `max_new_tokens`. The first
  K3 / sincos run was invalid for this reason and was re-run.
- **Markdown.** Short crops come back as headings (`## 잠깐만`), so the reader
  strips `#` markers from each line. Line breaks cost nothing because
  `exact_key` is whitespace-blind.

`fla` / `causal-conv1d` are not installed, so the GDN layers run on the
transformers torch path. That is fast enough for crops: 10.5 crops/s on
sincos.

## Crop mode

| gate | stock VL-1.6 | v4 (P3 ep2) | OvisOCR2 | OvisOCR2 vs stock |
|---|---|---|---|---|
| K3 KO exact (n = 48) | 21 | 26 | 12 | 2 / 11, z −2.5 |
| K3 ZH exact (n = 48) | 31 | 30 | **32** | 9 / 8, z +0.2 |
| sincos SFX strict / ♡-blind (n = 597) | — | 346 / 387 | **2 / 6** | — |
| sincos speech strict (n = 272) | — | 123 | 35 | — |

- **It does not read hand-lettered SFX.** On 597 SFX crops it gets 2 right
  (sim 0.098) and runs away on 40. Kana bursts come back as digits or as
  unrelated glyphs: `おっ` → `1`, `ぢゅぽ` → `170`, `へっ？` → `77`.
- **It reads printed balloon text.** ZH matches stock (32 vs 31). It reads
  Korean dialogue when the lettering is clean (`최근 제게 보내는 관심이
  부족하지 않습니까?`), but reads stylised Hangul as Han (`조로` → `圣呈`).
  Several KO crops come back as digit runs.
- Sincos speech, 35 / 272: the speech labels are PP-OCRv6 text (see the
  `eval_sfx.py` docstring), so this measures agreement, not accuracy. Even
  so, it is far below every manga-trained reader.

Verdict: OvisOCR2 is not a candidate for the crop reader.

## Page mode

`ocr/eval_page.py` feeds the 146 labelled sincos pages whole, which is what the
model was built for. A label row counts as **hit** when its `exact_key` is a
substring of the page output's `exact_key`. This is position-blind recall, an
upper bound, and not comparable to crop exact. It is split at key length 3,
because 1–2 char keys can hit by accident. It also counts rows that are hit
only once the `<img>` blocks are kept, i.e. SFX the parser filed as a picture.

Pending — job `20261012-184829-f88741`.
