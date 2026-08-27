# Multi-view audit — untagged `multiple views` in the caption master

Status: **tool built, full sweep run (3008 images, 122 findings); root-cause fix
shipped.** The fill-ratio survivor swap landed in shared `dedupe_detections`
(`597d7894`, knob `--dedupe_fill_ratio`, default **2.0**) — §5 below is the
evidence it rests on and §5.4 is the corpus measurement that fixed the default.
The companion degenerate-proposal guard was measured and **refuted** (§5.4) and
does not ship.

Written up before implementing because the first two explanations of the
5847152 failure were both wrong (§4), and both were wrong in the same way —
asserted from correlation in an existing doc instead of measured.

---

## 1. The blind spot

`is_candidate` (`library/preprocess/position_captions.py:542`) sends a caption to
detection only if it has a layout tag or claims more than one girl. A sheet that
*is* several views of one character but was never tagged `multiple views` claims
`1girl`, so it is skipped as `single-subject` and never looked at.

Over the caption master, that is **2551 of 3007 captions**. Some unknown fraction
of them are mis-tagged.

The second-order damage matters more than the missing tag. If such an image *did*
reach the clause writer, `is_repeated_subject_layout` would return `False` (no
layout tag), so the `view_invariant` gate would not fire and the writer would bind
the character's name and her view-invariant traits **per view** — asserting one
girl as two. The missing tag is what keeps that gate from working.

Founding example: `ama_mitsuki/5847168` — one office lady drawn twice (bent-over
close-up left, seated full body right), captioned `1girl`, no layout tag.

## 2. The tool

| File | Role |
|---|---|
| `library/preprocess/multiview_audit.py` | Orchestration: population filter, detection, verdict |
| `library/preprocess/multiview_sheet.py` | Per-finding contact sheet renderer |
| `scripts/preprocess/audit_multiview.py` | Thin CLI |
| `scripts/preprocess/probe_sam_masks.py` | Diagnostic: dump SAM3's raw masks for one image, colour-coded |
| `scripts/preprocess/probe_nms_pairs.py` | Diagnostic: replay NMS over a corpus, record every suppressed pair's scores + mask fills |

Run: `make daemon-run ARGS="scripts/preprocess/audit_multiview.py [flags]"`
(GPU work from an agent must go through the daemon).

**Population** — exactly the complement of the clause pipeline: every caption
`is_candidate` rejects with reason `single-subject`. Tied to that function's own
reason string so the two cannot drift apart.

**Detection** differs from the clause pipeline in one way on purpose: the
escalation target is forced to `min_instances` (2) rather than the caption's
count. Passing `expected=1` would satisfy the target on the first box and
suppress both the low-threshold retry and the body-part fallback — on the exact
population we are trying to search.

**Evidence model.** Two boxes on a `1girl` caption raises the image; three
signals then argue about what it means, and `--apply` requires two to agree:

1. **Identity agreement** across the per-instance crops (hair / eye / hairstyle,
   plus character name). The only signal that separates "one girl twice" from "a
   second girl" — but it goes silent on a headless crop.
2. **Whole-image `multiple views` head.** Needs no legible crop; the fallback
   when (1) has nothing. Runs on *every* audited image, so a sheet whose views SAM
   merged into one box still surfaces (`tagger-only`, never above `weak`).
3. **People-count head** saying `1girl` while the geometry sees several bodies.

**Verdicts**: `multiple views` / `extra-character` / `unsure` / `count-explained`.

`count-explained` is not a finding: the caption's own `girls + boys` already
covers every box. Needed because the `girl` prompt does not exclude males, so a
`1girl, 1boy` image lands in this population (girls-count is 1) and detects two
bodies that are both already named. Removed 32 of 42 audited rows in the smoke.

**Writes**: `--apply` writes the **caption master** (`image_dataset/`), unlike the
clause rewrite which only touches the derived caption — a missing `multiple views`
is a fact about the picture that every later stage should read down from. Append
at the end of the flat bag, via `compose_caption` so trailing clauses survive.
Default `--apply_verdicts` is `multiple views` only, `--apply_confidence` is
`strong` only. **`image_dataset/` is gitignored** — `report.json` holds the
verbatim before-text and is the only undo. Follow any apply with
`make preprocess-te`.

**Contact sheets** (`<report_dir>/sheets/`, on by default): one PNG per finding —
boxed original, the crops the tagger actually saw colour-matched to their box, the
identity read off each, the verdict and its witnesses, and the proposed caption.
Filenames are `verdict_confidence_stem.png` so a directory listing sorts by
verdict.

## 3. Smoke result — `ama_mitsuki`, 106 images

```
seen 106 → audited 42 → findings 10
  multiple views 8   unsure 2   extra-character 0
  strong 7   weak 3
skipped: already-has-clauses 55, count-explained 32, handled-by-position-captions 9
```

Not validated against ground truth beyond eyeballing the sheets.

## 4. Investigation: `ama_mitsuki/5847152`

### Symptom

Verdict came out `extra-character` on an image that is plainly one girl drawn
twice (close-up foreground + standing full body, same outfit, same hair). The
left crop was a near-white canvas with scattered dark specks, and the tagger read
`red eyes 0.93 / black hair 0.99` off it.

### Two wrong explanations

Both are recorded because both were stated confidently before measuring.

1. **"The identity gate should catch it."** It cannot. The tagger's group heads
   are an **argmax over a softmax**, so they always name a value, and the winner
   is emitted into `kept` by argmax rather than by its own sigmoid threshold —
   filtering on `kept` membership is a no-op. A raw-probability gate at 0.9 was
   added and *does* catch the 5847168 case (headless crop read `blonde hair` 0.54
   / `blue eyes` 0.63 vs 0.978–1.000 on legible crops), but 5847152's bad crop was
   read at **0.93 / 0.99**. A confident answer to a question the crop cannot pose.
2. **"A weak match produces a low score and a bad mask together."** Asserted from
   the correlation in `position_captions.md:463` (~6% of instances recovered in the
   0.35–0.5 band have a broken mask) as if it were a mechanism. It is not what
   happened here.

### The measurement

`scripts/preprocess/probe_sam_masks.py` on the raw image, processor floor 0.3:

```
image 864×1232;  masks (1, 1232, 864) float32, values 0.0–1.0, ALIGNED to image
```

| # | score | box | area frac | mask fill (own box) | what it is |
|---|---|---|---|---|---|
| 0 | 0.633 | `556, 132 → 830, 1148` | 0.261 | 0.460 | standing girl — clean |
| 1 | **0.354** | `-2, 2 → 633, 1222` | 0.728 | **0.560** | **close-up girl — clean** |
| 2 | 0.389 | `1, 7 → 861, 1224` | 0.983 | **0.077** | speckle inside #1 — garbage |

So: masks are full-resolution, correctly aligned, proper probabilities. Nothing
wrong with SAM3's output and nothing wrong with `crop_instance`'s indexing. **The
mask we wanted already existed** (#1).

### Root cause

`dedupe_detections` (`library/preprocess/position_captions.py:618`) is greedy NMS
ranked on **score alone**:

```
sorted:  0.633 (#0)  →  0.389 (#2)  →  0.354 (#1)

#0  keep
#2  IoU vs #0 = 0.265 < 0.65                       keep
#1  IoU vs #0 = 0.080,  IoU vs #2 = 0.728 ≥ 0.65   SUPPRESSED
```

**The garbage proposal outscored the good one by 0.035 and suppressed it.** The
audit's report confirms this from the other end: the two boxes it kept are exactly
#0 and #2.

Containment did not save it either — #1 is 0.991 contained in #2 — but
`--containment_threshold` ships at 1.01 (off) for the reason measured in
`position_captions.md:457`, and that decision is not in question here.

Note what this means for the clause pipeline: **the same bug is live in
`caption-position`**, which shares `dedupe_detections`. Any image where SAM emits
a degenerate duplicate that narrowly outscores a good proposal is being tagged off
the wrong pixels there too. Unmeasured — no idea how often.

### Interim mitigation (already in)

A box below `score_threshold` — i.e. recovered only by the retry escalation — no
longer votes on identity (`reliable=False`), though it still counts as a body. On
5847152 that drops the poisoned read and the other two witnesses carry it to
`multiple views / strong`. This treats the symptom: it discards a bad identity
rather than recovering the good one, and it silences *every* retry-recovered box,
including the ones whose masks are fine.

## 5. The fix (shipped) — mask quality in the duplicate decision

When NMS has already judged two proposals to be **the same object**, choose the
survivor by mask quality rather than by score alone. Here that is 0.077 vs 0.560 —
a 7× gap, no absolute threshold required.

**This is not the settled negative.** `position_captions.md:463` rejects an
**absolute** gate on mask fill, and rejects it on measured grounds: a clean
0.87-score figure sits at fill 0.267, the same as bad ones, so no cut-point
separates them. What shipped is a **relative comparison inside a pair NMS
has already matched** — it never needs a cut-point, and it cannot drop an
instance, only swap which of two duplicates represents it.

How the three open decisions resolved:

- **Where.** Shared `dedupe_detections`, so `caption-position` is fixed too.
  Regression-checked on the clause corpus (`position_swapdiff_{on,off}`) and on
  the audit smoke (`multiview_audit_smoke_swap`).
- **Form.** Swap-on-ratio, no score-margin term: §5.1 measured margin as a
  non-discriminator (the 0.035 pathological margin is matched exactly by a
  benign pair).
- **Metric.** Fill-within-own-box (`mask_box_fill`), compared as a ratio inside
  the matched pair. `position_captions.md` found fill useless as an *absolute*
  cut, which said nothing either way about it as a relative one — §5.1/§5.4
  measured it and it separates cleanly.

### 5.1 Measurement — every NMS-suppressed pair on `ama_mitsuki`

`probe_nms_pairs.py --path_pattern 'ama_mitsuki/*'`, replaying the same greedy NMS
at the floor the audit's retry actually reaches (0.35) with `iou_threshold` 0.65:

```
106 images → 7 suppressions across 7 images   (~6.6% of images)
             4 fill inversions (survivor claims less of its box than the loser)
             1 degenerate survivor
```

| image | kept score | dropped score | margin | kept fill | dropped fill | ratio | IoU |
|---|---|---|---|---|---|---|---|
| **5847152** | 0.389 | 0.354 | **0.035** | **0.077** | **0.560** | **7.3×** | 0.728 |
| 6360109 | 0.490 | 0.356 | 0.135 | 0.383 | 0.464 | 1.2× | 0.889 |
| 5828766 | 0.441 | 0.426 | 0.016 | 0.401 | 0.479 | 1.2× | 0.901 |
| 12971490 | 0.412 | 0.377 | 0.035 | 0.703 | 0.780 | 1.1× | 0.694 |
| 5029937 | 0.393 | 0.367 | 0.025 | 0.293 | 0.174 | — | 0.666 |
| 12971620 | 0.645 | 0.416 | 0.229 | 0.377 | 0.231 | — | 0.650 |
| 12971572 | 0.455 | 0.438 | 0.018 | 0.436 | 0.134 | — | 0.658 |

Three things this settles:

1. **A fill *inversion* is common; a *pathological* one is not.** 4 of 7
   suppressions keep the lower-fill mask, but three of those are 1.1–1.2× — two
   comparable masks on slightly different boxes, where the survivor is fine.
   Swapping on any inversion would churn three good decisions to fix one.
2. **The pathology is an order-of-magnitude outlier.** 7.3× against a benign band
   of 1.1–1.2×. So the discriminator is the **ratio**, and it needs no absolute
   cut-point anywhere — which is what keeps this clear of the settled negative in
   `position_captions.md:463`. Any ratio in roughly [1.5, 5] gives the same answer
   on this sample; the threshold is not a tuned quantity here.
3. **Score margin is not the discriminator and should not be in the rule.**
   5847152's margin (0.035) is matched exactly by a benign pair (12971490, 0.035)
   and beaten by another (5828766, 0.016). Gating on margin would fire on the
   benign ones and add nothing.

**Blast radius is small in both directions**: ~1 image in 106 is affected, so this
is a correctness fix on a rare case, not a throughput win.

Caveats: one artist directory, 7 pairs, **one positive example**. The 1.2 → 7.3
gap is wide but it is a gap in a sample of four. Nothing here says how the
distribution looks on the other 2900 images, and the same probe should be run over
the full corpus before the ratio is written into shared code.

### 5.2 Measurement — mask probe over the 10 smoke findings

Is 5847152 one-sample bias? The batch mask probe (SAM3 floor 0.3, every raw
proposal's score / area / fill-in-own-box, NMS replayed at the audit's 0.35/0.65)
over the smoke's 10 finding images (`mask_probe/smoke_batch.json`, overlays per
stem) splits the question in two:

**The garbage proposal is NOT one-sample.** 3 of 10 findings carry a
near-whole-canvas proposal with an essentially empty mask, all scoring within
±0.04 of the 0.35 retry floor:

| image | score | area frac | fill | outcome |
|---|---|---|---|---|
| 5029937 #2 | 0.338 | 0.980 | 0.000 | below floor by 0.012 — filtered. Counterfactually harmless too: all 3 overlapping real boxes (IoU 0.53–0.79) outscore it, so greedy NMS would have suppressed *it* |
| **5847152 #2** | **0.389** | 0.983 | 0.077 | above floor, **outscores its clean duplicate by 0.035 → suppressed it** (§4) |
| 5847182 #2 | 0.332 | 0.982 | 0.001 | below floor by 0.018 — filtered. Counterfactually it overlaps **nothing** at ≥0.65, so above the floor it would have been **kept as a phantom third body** — a failure shape the pair-relative NMS fix cannot touch; today only the score floor stands between it and the count |

**The harmful *ordering* is still one positive example.** Damage needs two coin
flips to land the same way: score above the floor (missed by 0.012 / 0.018 in the
two harmless cases) *and* — for the §4 shape — outscoring the clean duplicate it
overlaps. The recurrence rate is governed by where SAM3's junk lands relative to
an arbitrary floor, not by the junk being rare.

**The ratio discriminator survives the new data, and so does the settled
negative.** Garbage fills in this set: 0.000–0.077. Every real proposal:
0.148–0.780. The 7.3× pair of §4 was the *least* extreme garbage — the other two
would pair at effectively ∞ — so the [1.5, 5] insensitivity band widens. And the
floor case 6494927 #1 (score 0.455, area 0.930, fill **0.148**) is a *real*
close-up view with a patchy mask, kept and counted correctly (`reliable=False`
kept its identity vote out): an absolute fill cut at any level that catches the
garbage would have to survive real views at 0.148 — the `position_captions.md:463`
negative, re-confirmed. The relative pair rule never sees 6494927 because nothing
collides.

### 5.3 Measured and declined — prompt engineering instead of the NMS fix

Swept 4 alternative SAM3 prompts over the same 10 findings
(`mask_probe/prompt_sweep.json` / `prompt_sweep_en.json`, overlays
`prompt_<tag>.png` per stem):

- **Instruction-style prompts do not act as instructions.** `girl (하나의 신체를
  나누지 말것)` and `girl (do not split one body)` both return **0 proposals on
  all 10 images** — the parenthetical form falls off SAM3's noun-phrase grounding
  distribution entirely.
- **Longer noun phrases trade the garbage for recall collapse.** `whole girl` and
  `full body of a girl` emit zero empty-mask proposals on this set — and lose the
  hard close-up images outright: 5029937 and 6360109 drop to **0 NMS survivors**
  (`whole girl` returns nothing even at floor 0.3; `full body` one box at 0.32,
  under the 0.35 floor), and three more images drop 2 → 1. Undershoot is the
  audit's enemy; disqualifying.
- **`one girl, do not split her body` keeps recall but reproduces the pathology
  in a new costume.** On 5847152 it emits no empty mask, but a whole-canvas-box
  duplicate of the close-up (fill 0.363) outscores (0.547 vs 0.440) and
  suppresses the tight-box version (fill 0.543) — same greedy score-only NMS
  failure, now "box too big" instead of "mask empty". Fill ratio of that pair is
  1.5× — exactly at the proposed band's lower edge, a data point for tuning it.
  5029937 also churns harder (5 proposals, 3 suppressions).

- **`woman` / `female` / `person` inflate scores across the board (0.9+ on the
  easy sheets, better recall on some hard ones) — and the junk inflates with
  them** (`prompt_sweep_wf.json`). On 5847152 **all three** reproduce the exact
  §4 pair: a near-whole-canvas low-fill proposal outscores the clean close-up and
  suppresses it — by margins of 0.008 (`woman`), 0.058 (`female`), **0.184**
  (`person`, junk at 0.637). What was a 0.035 coin-flip under `girl` becomes the
  *reliable* outcome. And on 5847182 the §5.2 phantom-body counterfactual becomes
  real: the empty proposal (fill 0.001–0.014) clears the floor at 0.432–0.478
  under all three and survives NMS as an extra body. `person` also fragments
  5847100 into 8 surviving boxes.

Net: prompting reshuffles where the junk lands; it does not change that the
survivor of an NMS-matched pair is chosen by score alone. The prompt stays
`girl`; the fix belongs in `dedupe_detections`.

**Side-product — the sweep multiplied the positive examples.** Across all 8
prompts × 10 images, 25 suppressed pairs: pathological (clean mask lost to a
low-fill duplicate) ratios are 7.25 / 3.54 / 3.30 / 2.43 / 2.11 / 1.51 / 1.50;
benign inversions top out at **1.33** (was 1.2 in §5.1). The discriminator still
separates, but the §5.1 claim "any ratio in [1.5, 5] gives the same answer"
tightens: across prompts the gap is 1.33 vs 1.50, not 1.2 vs 7.3. Under the
shipping `girl` prompt the wide gap stands (1.21 vs 7.25).

### 5.4 Phase 0 — full-corpus measurement (2026-08-17)

`probe_nms_pairs.py` over the whole resized corpus — 3008 images, 3993
proposals — one SAM3 grounding pass per image at the 0.35 floor, NMS replayed
at both floors (0.5 primary, 0.35 retry) as a pure re-filter, every proposal's
box fill + area fraction recorded. Payload:
`post_image_dataset/captions/nms_pairs_full.json` (daemon job
`20260817-210615-75e671`).

**How often the machinery fires.** Floor 0.5: 28 suppressions on 28 images
(0.9%), 12 fill inversions, **2** degenerate survivors (kept fill < 0.15).
Floor 0.35: 110 suppressions on 108 images (3.6%), 62 inversions, **12**
degenerate-survivor pairs on 11 images (0.37% of the corpus; `5847152` among
them). This answers the previously unmeasured `caption-position` frequency:
shape A corrupts ~0.1% of images at the primary floor and ~0.4% at the retry
floor.

**Gate 1 (ratio separation) — PASS, R = 2.0 confirmed.** At both floors the
ratio axis has an empty band exactly where the proposal wanted to cut:
(1.87, 2.75) at floor 0.35, (1.75, 3.70) at floor 0.5. Every pair at ratio
≥ 2.0 has a degenerate survivor (kept fill ≤ 0.149 — the pathology); every
pair below 2.0 has kept fill ≥ 0.213 (clean-figure territory per the settled
0.267 point) — with one exception where *both* fills are degenerate
(`tottotonero/5661996` at 0.35: 0.142 vs 0.143, ratio 1.00 — no ratio rule can
or should act there). So `R = 2.0` swaps 11 of the 12 degenerate-survivor
pairs at the retry floor (the 12th is the both-degenerate pair) and 2 of 2 at
the primary floor, and touches nothing else. `R = 1.5` would add 4 swaps whose
current survivors are clean (fills 0.254–0.347) — risk without measured
benefit; `R = 3.0` would miss two real pathological pairs
(`pepper0/5853766` 2.82, `fizz_(pixiv34498626)/6584585` 2.75).

**Gate 2 (shape-B fill gap) — FAIL, Phase 2 is dead.** The thin
`ama_mitsuki` margin (garbage ≤ 0.077 vs real ≥ 0.148) does not survive corpus
scale: whole-canvas proposals (area ≥ 0.95, score ≥ 0.35) fill the
0.05–0.15 region continuously — 0.055, 0.059, 0.062, 0.065, 0.071, 0.077,
0.080, 0.087, 0.089, 0.094, 0.096, 0.118, 0.123, 0.127, 0.129, 0.138,
0.146, 0.149… — and several low-fill whole-canvas proposals carry *high*
scores (0.67–0.79), so any cut in this region trades phantom bodies for real
sparse-subject views one-for-one. Per the proposal's pre-registered gate, the
degenerate-proposal guard does not ship; shape B stays handled by the score
floor plus audit spot-checking.

### 5.5 `anime girl` — the one prompt variant §5.3 missed (2026-08-17)

Full-corpus `probe_nms_pairs --prompt "anime girl"`
(`nms_pairs_animegirl.json`), same dual-floor replay as §5.4. Not among the 8
swept prompts, and it breaks §5.3's pattern in one direction while confirming
it in the other:

- **Recall improves dramatically — the first variant that doesn't collapse
  it.** Zero-proposal images at the primary 0.5 floor: 310 → **18** (91 → 6 at
  0.35). 466 images gain at least one NMS survivor at 0.5; **8** lose one.
  The close-up undershoot that disqualified `whole girl`/`full body` does not
  happen here.
- **The junk inflates with it, exactly like `woman`/`person`.** Suppressions
  28 → 86 at 0.5; degenerate survivors 2 → 8 (0.5) and 12 → 21 (0.35);
  whole-canvas fill<0.10 junk at score ≥ 0.5: 53 → 66. `5847182` (the shape-B
  phantom) goes from 3 proposals to 6.
- **R = 2.0 is a `girl`-calibrated fact and does not transfer.** Under
  `anime girl` the empty ratio band around 2.0 disappears (pairs at 1.96,
  1.99, 2.02, 2.28, 2.40 form a continuum), and swaps at 2.0 would hit 3–4
  pairs whose current survivor is clean (kept fill 0.197–0.246). No degenerate
  survivor sits below 2.0, so the swap still catches all of shape A — but the
  "touches nothing else" property is lost.

Net: `anime girl` is the first real candidate for a prompt change — but it is
a *population* change (~15% of images gain survivors of unverified nature,
shape B becomes live, R needs recalibration), not a drop-in. If pursued, it
needs its own gated evaluation: eyeball the gained survivors, re-run the
R sweep under the new prompt, and re-measure the audit/caption-position
verdict deltas. The shipped prompt stays `girl` until then.

### 5.6 Closure — the learned prompt ships (2026-08-27)

§5.5's gated evaluation ran on a *learned* prompt instead of the text
`anime girl` (`docs/experimental/soft_prompt_for_sam.md`): recall of `anime girl`
(zero-proposal 310 → 4) with `girl`'s junk profile (degenerate survivors 0/0,
whole-canvas fill < 0.10 junk 53 → 0). R sweep under it: one pair ≥ 2.0 (2.55,
kept fill 0.274 — clean), so the R = 2.0 swap is inert; the `girl`-calibrated
floors stay. Audit girl vs soft, same tagger: findings 51 → 56, all deltas
weak `unsure` (inset second views, low-score body parts, drawn boys), one
dubious flip on a 0.01-area chibi sticker (`otokakoto/11809823`). It is the
shipped default of this audit and of `caption-position`
(`--prompt_embed none` for the text prompt).

## 6. Verified vs not

Verified:
- SAM3 masks for 5847152 are aligned, full-res, 0–1 probabilities.
- #1 (good, 0.354) is suppressed by #2 (garbage, 0.389) at IoU 0.728.
- The audit kept #0 and #2 — the report's box coordinates match.
- Identity confidence 0.9 separates 5847168's headless crop; it does **not**
  separate 5847152's.

- On `ama_mitsuki`: NMS suppression fires on 7 of 106 images; 4 of those 7 keep
  the lower-fill mask; 1 keeps a degenerate one. Mask-fill **ratio** separates the
  pathological pair (7.3×) from the benign inversions (1.1–1.2×); score margin
  does not separate them at all.

- Full-corpus frequency and the ratio discriminator at scale (§5.4): 3008
  images, degenerate survivors on 0.07% (floor 0.5) / 0.37% (floor 0.35) of
  images; ratio ≥ 2.0 ⟺ degenerate survivor, with an empty band around 2.0 at
  both floors. Shape B's fill gap measured and **refuted** at scale.

Not verified:
- Why SAM3 emitted proposal #2 at all.
- Whether the smoke's 8 `multiple views` calls are all correct — eyeballed, not
  ground-truthed.
- The full-corpus pathology labelling in §5.4 uses kept-fill < 0.15 as the
  proxy for "degenerate survivor" (same proxy as §5.1); the corpus pairs have
  not been individually eyeballed the way the `ama_mitsuki` seven were.
