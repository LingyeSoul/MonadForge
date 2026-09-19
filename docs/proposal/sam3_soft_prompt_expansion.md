# SAM3 soft prompts — generalise the trick to every concept we ask SAM3 for by text

Status: **PROPOSED (2026-08-27)**. Part A of the original proposal — ship the
learned `anime girl` prompt into `caption-position` with the caformer tagger —
is **done** and documented in
[`docs/experimental/soft_prompt_for_sam.md`](../experimental/soft_prompt_for_sam.md)
(Phase 0, the A0 boys-drift measurement that closed the retrain, the A1/A2
gates, the shipped default). This file now carries only the open follow-up.
Every number below that is not in that doc is a target, not a measurement.

## TL;DR

A soft prompt is 1 k parameters trained in 25 min from SAM3's own filtered
outputs, with A/B sheets as the verdict. Anything we currently ask SAM3 for by
text is a candidate: **speech bubbles** (`configs/sam_mask.yaml` prompts
`speech bubble` / `text bubble` at threshold 0.7 — never measured), **SFX /
free text** (MIT's job today — a prompt that covers it lets `make mask` drop a
backend), **faces / body parts** (the `caption-position` fallback prompts),
and **boy** as its own subject for the region task
([[project_region_v5_slack_pairs_face]] — SAM `boy` prompt trap). Each is the
same three scripts with a different `--init` and target filter. Run them as
one multi-concept sweep; promote whichever clears its gate.

## What we have (verified)

- `bench/sam3_soft_prompt/{common,build_targets,train_soft_prompt,ab_sam3_prompt,pair_negatives}.py`;
  the shipped girl prompt `networks/calibration/sam3_girl_prompt.safetensors`
  (zero-proposal 310 → 4 corpus-wide, 0 degenerate survivors, 0 whole-canvas
  junk; `caption-position` 433 → 439 proposed with 0 wrong new clauses).
- `--prompt_embed` plumbing on `position_captions.py` / `audit_multiview.py` /
  `probe_nms_pairs.py` (`library/preprocess/instance_detection.py::
  resolve_prompt_embed`); `report.json` stamps `prompt_embed_sha256`.
- `ab_position_captions.py` runs a per-side detector when the B prompt
  differs — the consumer-side A/B for any subject-prompt candidate.
- `pair_negatives.py`: tagger-labelled boy / girl boxes on boy-tagged images
  (+ `eval` mode) — the label source for the **boy** row. Measured: the girl
  prompt boxes a drawn boy in 11 / 1045 boy-tagged images and the corpus has
  7 boy-only images, so boy positives must come from those 11 + the region
  task's own crops, not from the caption count axis.
- `scripts/preprocess/generate_masks.py::detect_union` + `configs/sam_mask.yaml`
  per-pattern prompt routing — the bubble / text consumer (becomes per-pattern
  *prompt file* routing).

## Concepts, same trick

Every candidate runs through the same three scripts; only `--init`, the
target filter, and the eval set change. Run as one sweep (`--concept` flag on
`build_targets.py` selecting a filter preset; one job per concept).

| concept | init phrase(s) | pseudo-target source | negatives | eval / gate | consumer |
|---|---|---|---|---|---|
| **speech bubble** | `speech bubble`, `text bubble` | current `sam_mask.yaml` union @0.7 on images where MIT *also* fires inside the box (agreement filter) | images with no MIT text at all | 60 hand-checked bubble images: mask IoU vs hand mask ≥ 0.85, false bubbles ≤ 2 | `make mask` SAM backend |
| **free text / SFX** | `text`, `sound effect text`, `handwritten text` | MIT boxes (independent detector = real labels, not self-labels) | clean images | MIT recall ≥ 0.9 at ≤ 0.1 FP/img → `RUN_MIT_MASK=0` becomes viable | `make mask`, drop a backend |
| **face** | `face`, `anime face` | current part-prompt survivors with fill ≥ 0.3 inside a subject box | — | headless-crop hair attribution: fewer `None` from the tagger's solo gate | `caption-position` part fallback |
| **boy** | `boy`, `anime boy` | tagger-verified boy boxes (`pair_negatives.py`, 11 today) + region-task crops | pure-girl images | held-out boy set: boy recall ≥ 0.9, girl-box FP ≤ 0.05 | region v5 partner masks (today `person − girl`) |
| **girl (region)** | the shipped prompt | — | — | region v5 slack/pair recipe re-run: found-rate vs 88 % baseline | `project/region/` |

Notes that shape the design:

- **Self-labelling is the weak point; prefer an independent detector when one
  exists.** SFX/text has MIT, bubbles have MIT-inside-box agreement, boys have
  the tagger on crops. The girl prompt got away with self-labels because the
  eval set was the disagreement population; every row here needs an
  equivalent held-out set defined *before* training.
- **One prompt per concept, not a shared one.** `language_features` is
  per-prompt; SAM3 grounds one phrase at a time. Multi-concept masks stay
  unions at the caller (as `detect_union` does now).
- **Bubbles are the highest-ROI unknown.** The shipped 0.7 threshold was never
  measured; a 20-image eyeball of current `make mask` output is the Phase 0
  that decides whether this row runs at all.
- **The boy row is data-starved.** 11 verified boy boxes is not a training
  pool; the row runs only if the region task's own crops supply ≥ ~100 more,
  else it stays a tagger post-filter on crops.
- **Input-token variant** (learn a `<anime_girl>` word so it composes) stays
  out of scope — the audit showed composition degrades grounding
  (`multiview_audit.md` §5.3).

## Phases and gates

| phase | work | gate | cost |
|---|---|---|---|
| B0 | bubble Phase 0 eyeball | decides B1 | 20 min |
| B1 | concept sweep (bubble, text, face, boy) | per-row gates above | 4 × 1 h GPU |
| B2 | promote winners into `sam_mask.yaml` rules / region recipe | consumer-side A/B | per row |

Kill conditions: a B1 row fails its gate → record in
`soft_prompt_for_sam.md`, no second attempt without a new label source.

## Risks

- **Prompt ≠ word.** `sam_mask.yaml` rules and mask sidecars must record
  *which* file ran (sha, as `caption-position`'s `report.json` already does),
  or two runs become incomparable.
- **Trunk drift is not on the table.** If a concept fails (bubbles with thin
  outlines, screentone SFX), the answer is a label source, not trunk LoRA —
  the audit's evidence that features exist applies to characters; for text it
  is unverified, so each row carries its own Phase 0.
- **Depth of the daemon queue.** Every step is GPU-bound behind training;
  batch the sweep as one `--queue` chain.
