# MOSS-Image technical report — facts for this line

Source: `MOSS_Image_Technical_Report.pdf` (35 pp., HF Space `assets/`). Page numbers in
parentheses. *Italic* = our inference, not the report's.

## (a) Which checkpoint is Base

- The text says the release is the **Stage 3 EMA checkpoint at step 310,000** (p.14
  §A.2). Table 6's Stage 3 schedule ends at 310k (p.15); Table 5 gives the final text
  limit as "4096 tokens (Stage 3)" (p.14). Prose and Fig. 3c describe three stages,
  Stage 3 = "≤1024², 0.31M steps" (p.5–6).
- **Table 1 is the outlier** (p.6): Stage 3 at 370k steps plus a Stage 4 ("up to
  ~2×1024²", 520k). Nothing else mentions Stage 4.
- Native ceiling per the text: **area ≤ 1024² px, aspect preserved, sides multiples of
  16, text ≤ 4096 tokens** (p.6 §3.3). The human eval (RLA) generated at ~1 MP (p.26).
- *The Space presets (1776×1184, 2880×720, ~2.1 MP) match Table 1's Stage 4, not the
  stated cap: either the weights are a later checkpoint than the prose, or the Space runs
  Base past its training range. Keep LoRA buckets ≤ ~1 MP until Base is compared at
  1 MP vs 2 MP.*

## (b) Training facts a LoRA needs

- **Loss** (eq. 3, p.15): target `ε − z0`, model called at `1000·σ`; per-token mean over
  image tokens only, text tokens excluded; no σ/SNR weighting; images weigh by token
  count. Stage 2 moved from per-sample to per-token normalisation.
- **σ sampling, Stage 3** (Table 6, eq. 4, p.15). Z ~ N(μ, s²), S = 1 − sigmoid(Z); with
  `u = sigmoid(Z)` indexing a *descending* noise grid. **A negative μ pushes σ toward
  noise**: at the final interval median σ ≈ 0.75 (*our arithmetic*).

  | Stage 3 steps | shift | (μ, s) | cutoff | full-noise override |
  |---|---|---|---|---|
  | 1–140k | 3 | (0, 1) | — | 0.01 |
  | 140k–207k | 1 | (−1.45, 1.05) | — | 0.01 |
  | 207k–274k | 1 | (−1.45, 1.05) | 0.95 | 0 |
  | 274k–310k (final) | 1 | (−1.09, 0.95) | 0.95 | 0 |

  Cutoff truncates S at 0.95, then rescales onto the grid {0.001…1}; 1 % uniform mix.
  Stages 1–2 σ sampling is not reported.
- **Resolution per stage** (Table 1, p.6): Stage 1 256-px level, text 320; Stage 2 512
  then 256/512, text 512 → 1024 → 2048; Stage 3 mixed aspect, area ≤ 1024², text 4096.
  Stage 3 packs 8576 tokens per microbatch (p.14–15).
- **Caption sources** (Table 8, p.16): ReLAION, OmniCorpus, DataComp, MedTrinity,
  optical-rs, marqo, git_10m, Megalith, CommonCatalog, TextAtlas5M, HumanCaption,
  textrecap, Google Landmarks, WikiArt, Places365, poster100k. **No booru or anime
  source; character names never mentioned.** Captions "predominantly in English" (p.16).
- **Recaptions**: gemma4 short / medium / long, mean 21 / 77 / 235 tokens, P90
  26 / 97 / 293, max 3126 (p.18 Table 11). 80 % recaption, 20 % original; the three
  lengths 1/3 each (p.6, p.19).
- **Caption content**: visible text transcribed verbatim (no translation; line breaks,
  reading order, position, font, colour kept) (p.18). Line breaks in prompts help text
  rendering (p.9, p.33). Artifacts (watermarks, blur) are captioned, not filtered (p.5).
- **JSON layout captions** (p.19–20): two schemas (scene/element, summary/items); items
  carry a `label` and generic `tags` (not booru). Boxes as normalised [0,1000] yxyx,
  absolute px, or centre-relative px (p.19); centre-relative generalises best to unseen
  resolutions (p.10, p.34). Fields are concatenated, chat-templated, truncated at 4096
  (p.15).
- **Prompt-format eval** (p.9 Fig. 4b, RLA Elo): JSON layout 1623 > direct expansion
  1541 > detailed 1509 > natural 1506 > short 1082.
- **Caption dropout / uncond: not described.** *The pipeline's negative is `""` through
  the chat template; dropout should use that same encoding.*
- **Optimizer / EMA** (Table 7, p.16; p.15): Muon–AdamW, both lr 5e-5; AdamW β
  (0.9, 0.95); wd 0.01 (none on bias/norm/embedding); clip 1.0; warmup 2000. **EMA
  0.9999, and Base is the EMA weights.**

## (c) Inference

- **SFT is not covered by the report.** The pipeline defaults (50 steps, CFG 3, shift 5,
  skip 22) do not appear in it.
- **No rationale for skip-22 or SLG** anywhere.
- App. J case study: 30 steps, shift 3, plain grid (p.33).
- CFG 3 was best in an early 24-layer ablation (p.17), not re-measured on the 5B.
- No few-step / distillation work. No reference-image conditioning or editing (p.10);
  the RoPE `r` axis is reserved for it, hence the 32 unrotated head dims (p.14).

## (d) CJK

- LLTB Chinese short / medium / long 96.3 / 96.7 / 92.4; English 98.0 / 97.1 / 97.1
  (p.7 Table 4). LongTextBench Chinese 97.17 (p.28).
- No dedicated Chinese caption annotation (p.16): the ability comes from verbatim
  transcription plus text-heavy sources (TextAtlas5M, textrecap, poster100k).
- **Japanese and Korean are never mentioned** — the 2026-10-16 JA/KO renders go beyond
  what the report documents.
