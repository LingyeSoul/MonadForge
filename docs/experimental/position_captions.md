# Position-aware captions — binding attributes to the subject they belong to

A preprocessing pass that gives multi-subject images spatially bound captions
in the dataset's existing hand-written convention: SAM3 detects the subjects,
they are put in reading order, each mask-blanked crop is tagged by the Anima
Tagger, and the result is **appended** to the caption as trailing clauses.

Status: **v1 shipped and runnable as `make caption-position`.** Dry-run is the
default and **nothing has been applied to the caption master yet** — the
remaining Phase-1 gate is a spot-check of the dry-run report. Design rationale,
Phase-0 probe evidence and the phase plan live in
[`docs/proposal/position_captions.md`](../proposal/position_captions.md); this
doc is the operational one — what it does, how to run it, and what to watch.

## Why

The flat tag bag cannot say *which* attribute belongs to *which* subject:

- `3girls, blonde hair, aqua hair, red hair, …` — who is blonde?
- `1girl, multiple views, maid, playboy bunny, swimsuit, pink jacket, …` —
  four outfit views of one character, every outfit unbound.

The dataset already has a hand-written answer for this (12 captions carry it),
and the base model already obeys it: a Phase-0 probe rendered 24 no-LoRA images
from counterbalanced positional prompts and the tagger scored **48/48 sides
correct** (chance 50%). So clauses reinforce an existing conditioning channel
rather than teaching one from scratch — the reason v1 is worth applying at all.

The gate is the number of **detected** instances, never the girls-count tag: a
`1girl, multiple views` sheet is four bindable subjects and goes through the
same machinery as `3girls`.

### Before / after

```
explicit, 3girls, kisaki (blue archive), black hair, white hair, pink hair, …

  ↓ make caption-position

… , blue archive, @aak. On the left, kisaki (blue archive), black hair, blue
eyes, hair bun, back, double bun, ass, loli. On the middle, white hair, purple
eyes, hair between eyes, underwear only, black bra, navel, black wings,
underwear. On the right, pink hair, blue eyes, ahoge, halo, loli,
heterochromia, standing, black wings.
```

v1 is **purely additive** — the flat bag is byte-identical, so a caption gains
binding without losing anything the model was pretrained on. Moving attributable
tags *out* of the bag is v2 (`--rewrite`, unbuilt, phase-gated on a training
A/B).

## The clause grammar — read this before touching any caption code

The convention delimits clauses with a **period** and tags with **commas**:

```
<flat tag bag>. On the left, akita neru, yellow eyes. On the right, kasane teto.
```

So a plain `caption.split(",")` glues the clause header onto the previous tag
(`"white socks. On the left"`), and every consumer keying off
`tag.startswith("On the ")` then sees **no clauses at all**. That was a live bug
before v1: `anima_smart_shuffle_caption`'s section logic and the
identity-randomize guard both silently saw a flat caption, so the 12
hand-written ground-truth captions had their clause attributes scattered across
the caption and reassigned to the wrong subject in **3 of the 4 default caption
variants**. Fixed alongside v1.

**Never hand-split a caption.** `library/captioning/position_clauses.py` (pure
stdlib, no torch) is the single grammar:

| Function | Use |
|---|---|
| `parse_caption(text) -> ParsedCaption` | `.flat_tags` + `.clauses`; safe on clause-free captions (round-trips to flat tags alone) |
| `compose_caption(flat_tags, clauses)` | inverse; safe to route every caption through |
| `has_clauses(text)` | "does this already carry clauses?" — the prefilter's leave-it-alone check |
| `assign_positions(boxes, size)` / `ordered_indices(...)` | position vocabulary + reading order |

It accepts both written forms (period-delimited, and the comma form where the
header is its own token) and case-insensitive `on the left`; emission is always
the canonical capitalized `On the `. `In the …` (used by a few hand-written
scene-region clauses) parses and round-trips byte-stable but is never emitted.

### Position vocabulary

Rows are clustered first (single-linkage on box center-y, gap `row_tol × H`),
then each row is named left→right — grid sheets interleave badly under pure
x-ordering, and `multiple views` sheets are routinely 2×2.

| N in row | Words |
|---|---|
| 2 | `left`, `right` |
| 3 | `left`, `middle`, `right` |
| ≥4 | `leftmost`, `second from left`, …, `rightmost` |

Rows prefix the horizontal word (`On the top left, …`); a row holding a single
subject reads as the bare row word (`On the top, …`). Up to 3 rows
(`top`/`middle`/`bottom`); beyond that it degrades to plain left→right. 53 of
the 317 current proposals are row-aware.

## Pipeline

Per candidate image (`library/preprocess/position_captions.py`):

1. **Detect** — SAM3 with the text prompt `girl` on the *resized* image (the
   pixels training actually sees) → per-instance boxes + masks + scores. Score
   floor 0.5, greedy IoU dedupe at 0.65, and a **retry at 0.3** when the
   caption's own girls-count says more subjects exist than were found (an
   unconditional low threshold floods grids with part-detections).
2. **Order** — row-cluster, then left→right; positions assigned as above.
3. **Crop + blank** — padded bbox crop (6%), every non-instance pixel blanked
   to white using the instance mask. This is the load-bearing fix from Phase-0
   probe B: without blanking, a neighbor standing inside the padded box
   contributes their hair to this subject's tags, which was *both* of that
   probe's hair-color misses.
4. **Tag** — Anima Tagger per crop, then clause selection (below).
5. **Compose** — `compose_caption(flat_tags, clauses)` written back to the
   caption **master** (`image_dataset/*.txt`), which `preprocess-captions`
   mirrors into `resized/` and the TE step then encodes.

Models are injected as `detect_fn` / `tag_fn` callables, so the orchestration
module imports neither SAM3 nor the tagger and unit-tests with stubs; the CLI
shell (`scripts/preprocess/position_captions.py`) owns argparse + model loading.

## What goes into a clause

Eligibility comes from the tagger's own `groups.yaml`, not substring
heuristics, so the two can't drift:

- **Per-subject groups bind** — hair (color/length/style/accessory), eyes,
  expression, body, all clothing groups, pose/gesture/action (`SUBJECT_GROUPS`).
- **Scene groups never bind** — lighting, background, framing, medium,
  `interaction`, `character_relationship`.
- **Copyright / artist / metadata / deprecated / count / rating are excluded
  outright.** A franchise tag fires on every crop and would otherwise ride the
  ranked path into every clause.
- **Ungrouped tags** — where curated compounds like `pink jacket` live — are
  admitted only when they are both **in the flat bag** and **attributable**
  (kept on exactly one crop).
- **At most one member of an exclusive group** (the softmax /
  `softmax_when_solo` groups), or a contaminated crop emits `green hair, …,
  aqua hair` for one subject.

Emission order: character name → exclusive-group winners (`hair_color`,
`eye_color`, `hair_length`, `hairstyle` — these are `softmax_when_solo`, and a
single-subject crop is exactly the condition under which they fire, which is the
whole point of cropping) → everything else ranked, preferring tags the caption
already curated. Cap 8 tags per clause.

A **character name** needs both floors conjunctively: `--name_confidence` (0.5)
*and* membership in the flat bag — probe B scored names 4/7 on crops, so an
unlisted name is most likely a crop artifact. `--allow_unlisted_names` relaxes
the second. One identity per subject.

### A clause carries only what discriminates

Any tag *every* crop keeps is suppressed (`--keep_shared_tags` disables). On a
`1girl, multiple views` outfit sheet all views are the same character with the
same hair and eyes; repeating `hatsune miku, aqua hair, twintails` four times
binds nothing and crowds out the maid / bunny / bikini that actually tells the
views apart. Those attributes are already in the flat bag and v1 never removes
anything, so nothing is lost. When suppression empties every clause the image is
skipped as `no-discriminative-tags` — the subjects are genuinely
indistinguishable to the tagger and there is nothing to ground.

## Running it

```bash
make caption-position                                  # dry run, whole dataset
make caption-position ARGS="--crops --qwen3 models/text_encoders/qwen_3_06b_base.safetensors"
make caption-position ARGS="--path_pattern 'artist_a/*'"   # scope a slice
make caption-position ARGS="--apply"                   # write (after the review)
make preprocess-te                                     # REQUIRED after --apply
```

GPU job (SAM3 + tagger held resident for the whole sweep), so it is
**daemon-routed** like every other agent-launched GPU work — it queues behind a
live train run instead of OOM-colliding. `--queue` detaches, `--inline` bypasses.

**Dry run is the default and writes nothing.** It emits
`post_image_dataset/captions/position/report.json`:

```
summary: {applied, seen, candidates, proposed, written, skipped{reason: n},
          max_tokens, over_token_budget[]}
images[]: {image, caption_path, status, detected, expected, original, proposed,
           tokens, instances[{position, box, score, tags, crop}]}
```

With `--crops` it also exports the **exact mask-blanked pixels the tagger saw**,
mirroring the dataset layout, named `<stem>_<i>_<position>.png`. That is the
only way to tell a detection miss from a tagging miss when reviewing — read the
proposed clause next to its crop.

`--qwen3 <tokenizer>` adds a token count per proposal and flags anything past
512 (`qwen3_max_token_length` / `t5_max_token_length`). Past that the tail is
truncated **silently** at TE-cache time and, given the padding invariant, simply
never reaches the model. The first sweep found one caption at 522 tokens
(median 214) — check `summary.over_token_budget` before applying.

### Two silent-failure traps in the ops sequence

1. **Caption edits do NOT invalidate the TE caches.** After `--apply` the caches
   still *look* current and training keeps using the pre-clause embeddings until
   an explicit `make preprocess-te`. The script prints this reminder.
2. **`*.variants.txt` sidecars override the CLI dropout rate**, so a stale
   sidecar keeps training the pre-clause caption even after re-encoding.
   `preprocess-te` chains `preprocess-captions` and regenerates them first,
   which is why it — and not a bare TE re-encode — is the required follow-up.

### Current dry-run sweep

3008 images, whole dataset (2026-08-17). `defaults` is the shipped
configuration; `+parts` adds the opt-in `--part_prompts buttocks,hips,thighs`.
The first sweep's numbers are in brackets.

| | defaults | +parts |
|---|---|---|
| candidates (prefilter passed) | 441 | 441 |
| **proposals** | **394** [317] | **404** |
| skip: too-few-instances | 29 [81] | 19 |
| skip: count-mismatch | 17 [19] | 17 |
| skip: already-has-clauses (hand-written) | 15 | 15 |
| skip: too-many-instances | 1 [2] | 1 |
| not a candidate (single-subject) | 2552 | 2552 |

Against the 373-proposal sweep that preceded the three 2026-08-17 changes:
comic pages contribute **+24** candidates that used to be `single-subject`, the
part-prompt fallback **+10**, and the identity gate **0** (it changes what
clauses say, not how many there are). Clause tags rose 7173 → 7671 and identity
contradictions fell 520 → **0**.

Net against the first sweep: 59 of its 102 skips now propose, 3 regress (two to
`--min_area_frac`, one to a lower-floor detection that overshoots the count).
`count-mismatch` barely moves because the class *gained* members as
`too-few-instances` shrank — an image whose retry now finds three boxes for a
`2girls` caption lands here instead. That is the safe direction: a mismatch is a
skip, not a wrong write.

Of the 373 proposals: 249 are 2-subject, 58 are 3, 33 are ≥4; 152 come from
`multiple views` sheets; median 8 tags per clause (i.e. the cap binds). 48
proposals rest on at least one sub-0.5 detection — see the mask caveat below.

### Triaging the skips (2026-08-17)

The first full sweep skipped 81 `too-few-instances` + 19 `count-mismatch`. Both
were investigated end-to-end; four mechanical causes, all fixed.

**`too-few-instances` — under-detection, and the mitigation for it was dead
code.**

1. **The low-threshold retry never ran.** `Sam3Processor` carries its *own*
   `confidence_threshold` (default 0.5) and applies it inside
   `_forward_grounding` — boxes below it never reach the caller. The old
   `detect()` post-filtered the returned list against `retry_score_threshold`,
   which can only ever remove boxes, never add the ones SAM3 already dropped.
   A probe at 0.5 reproduced the reported counts exactly on 20/20 sampled
   failures; re-running at SAM3's real 0.35 floor brought 14 of the 20 to ≥2
   instances. `build_detect_fn` now constructs the processor at the *lowest*
   threshold it may be asked for and memoises the raw detections per image, so
   the retry is a pure re-filter with no second image encode.
2. **Multi-view sheets never even attempted the retry.** `detect_subjects`
   gated it on `if expected and …`, and `caption_subject_count` returns `None`
   for `multiple views` **by design** (the count tag counts characters, not
   views). `None` is falsy, so the entire multi-view population — 35 of the 81
   — skipped the retry before cause 1 could matter. The target is now
   `expected or min_instances`.

Irreducible tail: SAM3 scales every instance probability by one global presence
score, so on some framings (extreme close-up, from-behind, cropped body) *all*
boxes sink together — 2 of the 20 sampled stayed at zero detections even at 0.15.

**`count-mismatch` — the two counts were counting different things.** Lowering
the threshold makes this class *worse*, so it needed the opposite fix.

3. **Nested boxes survive dedupe** (9/19 had a pair at IoMin ≥ 0.7, several at
   exactly 1.00). Plain IoU is blind to nesting, and both over-detection
   families are nested: an **inset** — a character icon drawn on a phone screen
   inside the main subject, IoU 0.003 — and a **group box** spanning every
   subject, IoU 0.44 against each member.

   Suppressing on containment looks like the obvious fix and **was measured to
   be a bad trade**: enabling it broke 34 rows that previously proposed, and an
   ablation over those rows recovered 32 with the rule off. A *real* second
   subject is exactly as nested as a group box — one girl in front of another,
   an embrace, a background figure inside a foreground figure's box — and this
   corpus has far more of those than group boxes. `--containment_threshold`
   ships **off** (`1.01`); a surviving group box costs one `count-mismatch`
   skip, which is the safe direction. Only the inset half is handled
   automatically, by `--min_area_frac` (0.005 of the canvas) — that costs 2 real
   but genuinely tiny subjects across the corpus, and buys back the insets.
4. **Males and open-ended crowds.** `expected` counts girls, but the `girl`
   prompt picks males up **inconsistently** — it found the boy in 7 of the 19
   mismatches and missed him in 89 images that passed, so neither counting nor
   ignoring him works as an equality. The gate is now the range
   `girls … girls + boys` (`caption_boy_count`; unknown counts like
   `multiple boys` drop the upper bound). Separately, `6+girls` was parsed as
   *exactly six* — an open-ended crowd tag no detection can ever match — and now
   returns `None` like `multiple girls`, deferring to detection.

Whole-corpus result with the shipped settings: **59 of the 102 skips now
propose, 3 regress** (table above).

**Known residual — fragmentary masks at the low end.** Roughly 6% of instances
in the recovered 0.35–0.5 band have a broken mask (holes, or a box spanning two
views of a sheet with the mask covering pieces of both), and the blanked crop
then feeds the tagger a mix — one maid-outfit view came back as
`black dress, hood, mask`. Two candidate gates were measured and **both
refuted**: mask *fill* does not separate them (a clean 0.87-score standing
figure sits at 0.267 fill, same as the bad ones), and a row/column *gap* metric
maxes out at 0.106 across all 141 instances because the blobs are diagonally
offset. `main_frac` (largest connected component / mask) does correlate —
`frag>1` is 0/35 above score 0.7 versus 8/53 in [0.4, 0.5) — but it also flags
visually-clean crops whose hair or limbs simply split. No automatic gate is
shipped; the dry-run report carries a per-detection `score`, and the low-score
instances are the ones worth eyeballing. `report.json` also now records
`detections` for **skipped** rows and, with `--crops`, writes a box overlay
under `crops/_skipped/` — previously the rows most needing review were the only
ones with no visual evidence at all.

### Comic pages are a layout, not a subject count (2026-08-17)

`multiple views` was special-cased from the start — its girls-count counts
*characters*, not bindable subjects, so `caption_subject_count` returns `None`
and detection is trusted. **Comic panels are the same thing and were not
handled**: a `1girl, 2koma` page draws the same girl once per panel, so it
failed the prefilter as `single-subject`. 26 corpus images carry a comic-layout
tag and no `multiple views`; **22 were being skipped**, including clean
vertical 2-panel pages whose panels differ exactly the way clauses are good at
(`kase_daiki`, `ie_(raarami)`).

`_PANEL_LAYOUT_TAGS` (`comic`, `silent comic`, `sequential`, `2koma`/`3koma`/
`4koma`, `multiple 4koma`) now joins `multiple views` in `_LAYOUT_TAGS`, and
`is_candidate` returns a distinct `panel-layout` reason so the report keeps the
two apart.

**`page number` is deliberately excluded**, though it tags 15 more images. It
marks a scanned art-book page, not a layout — every one checked was a single
illustration with a number in the margin (`mignon/10831765`). A false signal,
not a weak one. (Separately, `ama_mitsuki/5847152` *is* a real two-view sheet
whose caption simply lacks `multiple views`; no tag rule can recover that.)

**The koma ceiling — a regression this introduced, and the fix.** A layout tag
waives the count check entirely (`expected` is `None` by design), which left
comic pages with no backstop against one subject detected twice.
`kase_daiki/11645055` is a 2-panel page with one girl per panel that SAM3
returns **three** boxes for: the bottom girl split into an overlapping pair at
IoMin 0.99, the second with a shredded mask, which the tagger read as
`glasses, multicolored hair, hood`. `strict_count` used to catch exactly this.

Suppressing the nesting is **not** the fix — measured, comic pages carry a
nested pair at 20% versus 19% for the rest of the corpus, so this is the
pre-existing over-detection residual whose ablation already said the rule costs
more than it buys. Instead `Nkoma` names the panel count, which the code was
throwing away: `caption_panel_ceiling` bounds detections at
`panels × (girls + boys)`. That is generous by construction and still catches
the split (`1girl, 2koma` tops out at 2 against 3 detected). `multiple 4koma`
is anchored out of the regex and plain `comic` has no panel count, so both stay
unbounded like `multiple views`. Over the whole corpus the ceiling changed
**exactly one row** — that one.

| | n |
|---|---|
| comic pages evaluated (were `single-subject`) | 26 |
| proposed | 24 |
| skip: too-few-instances | 1 |
| skip: count-mismatch (the koma ceiling) | 1 |

### The identity gate — the flat bag outranks the crop tagger (2026-08-17)

`select` used to emit the `_PRIORITY_GROUPS` winner (hair color, eye color, hair
length, hairstyle) **unconditionally**, with no check against the caption. That
is a noise amplifier, not a neutral default, because the discriminative rule
suppresses whatever every crop agrees on: a value the crops agree on is dropped,
so a *wrong outlier* is exactly what survives into a clause. On a back view the
eyes are not visible at all, the tagger guesses anyway, and the guess is
promoted precisely because it disagrees with the front view.

Measured over the first full-corpus dry run: **520 of 1600 identity clause tags
(33%) claimed a value the curated caption never listed**, and **114 of 146
single-character `1girl, multiple views` sheets (78%)** bound contradictory hair
or eye colors to different views of the same girl.

The gate: for a group in `_BAG_GATED_GROUPS` (`hair_color`, `eye_color`,
`hair_length`) a clause may carry a value the caption named, or nothing. Nothing
is lost — v1 is additive, so the attribute is still in the flat bag.
`hairstyle` is deliberately **not** gated even though it is a priority group: a
crop legitimately reveals a `hair bun` or `sidelocks` the booru caption never
tagged, and unlike a color that does not contradict what is there.
`body_shape` / `fashion_style` are left out for the same reason.

| | before | after |
|---|---|---|
| identity clause tags contradicting the caption | 520 | **0** |
| `1girl, multiple views` sheets with contradictory identity across views | 114/146 (78%) | 26/153 (**16%**) |
| total clause tags | 7173 | 7255 |
| proposals lost to an emptied clause | — | **0** |

Total clause tags *rose*: a blocked slot refills from the ranked tail, so the
gate trades an invented hair color for a real outfit tag. The residual 26 are
legitimate — the caption named no eye color at all, so differing per-view values
are new information rather than a contradiction, and the gate only fires when
the bag has already spoken for that group. `--ungated_identity` restores the old
behaviour for A/B.

This is **not** the v2 rewrite. v2 *moves* a bound tag out of the flat bag and
is gated on the Phase 3 training A/B; the identity gate only declines to *emit*
a clause tag the caption contradicts, and never touches the bag.

### Body-part detection fallback (2026-08-17)

Some sheets are one small full body plus two or three **headless close-up
panels** — a hip, a backside, a crotch. The `girl` prompt cannot see those at
any threshold, so the image dies on `too-few-instances` with its most
attribute-dense panels never tagged. `ama_mitsuki` is the signature case: 8 of
its 61 candidates skipped this way.

`--part_prompts` runs extra SAM3 prompts as a **second escalation**, under the
same undershoot condition as the low-threshold retry — never on an image the
subject prompt already resolved, where they could only add nested duplicates.
The image encoding is reused (`set_text_prompt` re-grounds against the cached
`backbone_out`), so each extra prompt costs a grounding pass, not a re-encode.

Three things had to be typed to the part pass to make the recovered clauses
usable, and all three were found by looking at the crops:

- **Containment suppression is ON for part boxes** (0.7) even though it ships
  off globally. The global rule is harmful because a *subject* nested in another
  subject is routinely real; a **part** nested in a subject never is — an `ass`
  box inside a girl box is that girl's own backside.
- **Part crops skip mask-blanking.** Blanking exists to stop a neighbor's hair
  bleeding into a subject's padded bbox. On a part box the mask *is* the part,
  so blanking deleted the torn jeans / pantyhose / panties the pass exists to
  recover and handed the tagger a bare skin blob — which came back
  `pink hair, black eyes, nude`. Plain padded bbox instead.
- **Identity groups are suppressed on a part crop** (`allow_identity=False`).
  No head means no evidence; every part crop in the first run invented both a
  hair color and an eye color.
- **Part boxes top up to the target and no further.** A part prompt is a looser
  concept than `girl` and fragments — `thighs` returned four boxes for two
  panels on `ama_mitsuki/6040950`, which would have bound five clauses to a
  three-panel image.

Whole-corpus with `--part_prompts buttocks,hips,thighs`: `too-few-instances`
27 → **18**, proposals 373 → **382**, no regressions (the fallback cannot fire
on an image that already passed). On `ama_mitsuki` alone, 8 → 5 skips.

**It is opt-in and the win is modest.** Of the 3 recovered ama_mitsuki images,
one is clearly good (`5828775` left panel: `ass, white panties, school uniform,
panties, cleft of venus`), one is partial (`6040950` — the part pass filled it
but the full-body figure was never detected at all, so it binds 2 of 3 panels),
and one is poor (`7987088` — both crops are the *same kind* of panel, so
everything true about them is shared and gets suppressed, leaving noise). That
last failure is `discriminative_only` biting in a non-identity group; it is the
same shape as the problem the identity gate fixes and is **not** addressed.

## Knobs

`ARGS="…"` on the make target; every flag has a `--kebab-case` alias.

| Flag | Default | What it does |
|---|---|---|
| `--apply` | off | Write to the caption master (else dry run) |
| `--path_pattern` | `*` | fnmatch glob (`\|` to OR) relative to the resized dir |
| `--crops` | off | Export the mask-blanked crops next to the report |
| `--prompt` | `girl` | SAM3 subject prompt (`person` sweeps the rare on-screen-boy images) |
| `--score_threshold` / `--retry_score_threshold` | 0.5 / 0.35 | Detection floor; retry floor when the count undershoots. These are SAM3's **own** confidence floor, not a post-filter — see the skip-triage section |
| `--iou_threshold` / `--pad` | 0.65 / 0.06 | Dedupe IoU; bbox padding fraction |
| `--containment_threshold` | 1.01 (off) | Suppress a box this nested inside a kept one (intersection over the *smaller* box). Measured harmful on this corpus — see the triage section before enabling |
| `--min_area_frac` | 0.005 | Drop detections below this fraction of the image (insets are not subjects) |
| `--no_blank_crops` | — | Skip mask-blanking (this is what caused probe B's hair-color misses — diagnostic only) |
| `--row_tol` | 0.25 | Row-clustering gap as a fraction of image height |
| `--part_prompts` | off | Comma-separated body-part prompts, tried **only** when the subject prompt undershoots — recovers headless close-up panels. Try `buttocks,hips,thighs`; see below |
| `--part_score_threshold` / `--part_containment_threshold` | 0.5 / 0.7 | Part-box confidence floor; drop a part box this nested inside an already-kept box |
| `--ungated_identity` | — | Let a clause carry a hair/eye color the caption never listed (disables the identity gate below) |
| `--min_instances` / `--max_instances` | 2 / 8 | Instance-count window |
| `--no_strict_count` | — | Propose even when detection disagrees with the girls-count |
| `--max_clause_tags` | 8 | Cap per clause |
| `--name_confidence` / `--allow_unlisted_names` | 0.5 / off | Character-name floors |
| `--keep_shared_tags` | — | Keep tags every crop agrees on (disables the discriminative rule) |
| `--qwen3` / `--max_tokens` | — / 512 | Token-budget column + over-budget flag |

## How clauses behave downstream

- **Caption variants** (`library/preprocess/caption_variants.py`) parse through
  the grammar and treat **each clause as an atomic unit**: kept or dropped whole
  at `clause_dropout_rate` (defaults to `tag_dropout_rate`), tags shuffled
  inside, header never randomized. Per-tag dropout inside a clause would leave a
  half-described position. Clause-free captions keep the historical raw split,
  so v0 stays byte-identical.
- **Order correction** (`correct_caption`) splits clauses off before
  bucket-reordering the flat bag — clauses are already ordered left→right and
  their tags are position-scoped, so reordering them across the caption is
  exactly the shredding the fix removed.
- **Training** sees no new machinery at all: this is a caption-text feature, and
  the clauses ride the ordinary TE path.

## In the GUI

Preprocessing tab → **캡션 편집 / Caption rewriting**, a group split out of the
caching box so the four caption-*text* knobs live together:
`캡션 순서 교정` · `@no-artist 삽입` · `트리거 워드` · `트리거를 맨 앞에 고정`
· **`위치 절 생성 (다중 인물)`** (`caption_position_clauses`, off by default).

Checked, it runs **inline as part of the ordinary Run** — no dry run, no extra
button: `tasks.py preprocess` chains the stage after the VAE cache and before
the caption/TE steps, because it rewrites the caption master those two then
read. The checkbox persists to the GUI variant's `[variant]` table like its
neighbours and rides to `tasks.py` as `CAPTION_POSITION_CLAUSES`, so the
ConfigTab Train auto-chain honours it too.

Two things follow from applying without a review step:

- It **edits the source captions in place**. The pass is idempotent — a caption
  that already carries clauses is skipped by the prefilter — but it is not
  undoable from the GUI. `make caption-position` (dry run, `report.json`,
  `--crops`) is still the way to eyeball proposals first, and is worth doing
  once on a new dataset.
- The TE-cache staleness trap is handled for you here: the stage runs *inside*
  the preprocess chain, so the variants and TE caches are rebuilt from the
  rewritten captions in the same job. It is the standalone `--apply` path that
  needs the manual `make preprocess-te`.

CLI equivalents: `--caption_position_clauses` / `--no_caption_position_clauses`
on `make preprocess`, or `caption_position_clauses = true` in the merged config.

**The Phase-1 spot-check is still owed** (≥90% of proposed clauses right on ~30
reviewed images, weighted toward grids, overlapping pairs and N≥4) — which is
why the checkbox ships off by default.

## Limits / open

- **Hair *length* across crops** — `long hair` vs `medium hair` on two views of
  the same character is crop-scale dependent, and `hair_length` is a priority
  group. The discriminative rule masks most of it (shared → suppressed), but a
  scale artifact that differs between views will bind. Watch it in the
  spot-check.
- **Character names on crops** are the weakest signal (probe B: 4/7), which is
  why they need the flat-bag floor.
- **Boys / POV** are out of v1 by default — `--prompt person` sweeps them
  separately; nothing is hardcoded to `girl`.
- **v2 (`--rewrite`)** — moving attributable tags out of the flat bag — changes
  the token distribution the base model was pretrained on. Probe A validated
  clause *comprehension*, not bag-*removal* tolerance, so it stays phase-gated
  on a training A/B.

## Code map

| Path | Role |
|---|---|
| `library/captioning/position_clauses.py` | Clause grammar (torch-free) — parse / compose / position vocabulary |
| `library/preprocess/position_captions.py` | Pipeline orchestration; models injected as `detect_fn` / `tag_fn` |
| `scripts/preprocess/position_captions.py` | CLI shell — argparse + SAM3/tagger loading |
| `scripts/tasks/preprocess.py::cmd_caption_position` | `make caption-position` (daemon-routed) |
| `library/preprocess/caption_variants.py` | Atomic-clause variant generation |
| `library/captioning/correction.py` | Clause-aware order correction |
| `tests/test_position_captions.py` | 40 unit tests (grammar round-trip, ordering, selection, skip paths) |
| `bench/position_captions/` | Phase-0 probe envelopes (`20260817-1122-autocaption`, `20260817-1123-binding`) |
