# Position-aware captions — binding attributes to the subject they belong to

A preprocessing pass that gives multi-subject images spatially bound captions
in the dataset's existing hand-written convention: SAM3 detects the subjects,
they are put in reading order, each mask-blanked crop is tagged by the Anima
Tagger, and the caption is **rewritten** so each attribute is asserted once, in
the clause of the subject it belongs to.

Status: **v2 shipped and runnable as `make caption-position`.** v2 *replaces* v1
— it is the same pipeline with the flat bag rewritten instead of merely
appended to; `--no_rewrite` keeps the old additive behaviour for the A/B arm.
Dry-run is the default and **nothing has been applied to the caption master
yet** — two gates are still owed (spot-check, then a training A/B), both spelled
out at the bottom of this doc. This is now the **canonical** doc for the
feature: what it does, how to run it, what to watch. The design proposal is
retired at
[`_archive/proposals/position_captions.md`](../../_archive/proposals/position_captions.md)
— Phase-0 probe evidence and the rationale behind each rule, not kept current.

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

explicit, 3girls, kisaki (blue archive), blue archive, @aak. On the left,
kisaki (blue archive), black hair, blue eyes, hair bun, back, double bun, ass,
loli. On the middle, white hair, purple eyes, hair between eyes, underwear
only, black bra, navel, black wings, underwear. On the right, pink hair, blue
eyes, ahoge, halo, loli, heterochromia, standing, black wings.
```

The three hair colors **leave** the flat bag: each is now asserted exactly once,
by the subject that has it. That is the whole feature — a bag that still lists
`black hair, white hair, pink hair` alongside the clauses is still claiming all
three of all three girls, which is the ambiguity the clauses exist to remove.

**This is the convention, not an invention.** Measured across the 14
hand-written ground-truth captions: 244 clause tags, of which 19 also appear in
the flat bag — and **all 19 are character names**. Not one attribute is
duplicated. So the hand-written form is exactly "cast list flat, attributes
bound", which is what v2 emits and what the additive v1 did not.

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
5. **Rewrite** — the tags a clause has earned leave the flat bag (next section),
   and `compose_caption(flat_tags, clauses)` is written back to the caption
   **master** (`image_dataset/*.txt`), which `preprocess-captions` mirrors into
   `resized/` and the TE step then encodes.

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
  outright**, on *every* emission path. A franchise tag fires on every crop and
  would otherwise ride the ranked path into every clause. The check lives in
  `add()` rather than only on the ranked path because an excluded tag can also
  be *grouped*: `light brown hair` is a deprecated alias `groups.yaml` still
  files under `hair_color`, so it rode the exclusive-group step straight into a
  clause on 4 images of the first full-corpus run.
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
views apart. A shared attribute keeps its place in the flat bag, which is where
an attribute that belongs to *everyone* belongs — the rewrite only takes what one
clause alone claims. When suppression empties every clause the image is skipped
as `no-discriminative-tags` — the subjects are genuinely indistinguishable to the
tagger and there is nothing to ground.

### On a repeated-subject layout, only what a view can differ in

Shared-tag suppression is the *right* rule but it is evidence-based, and on a
`multiple views` sheet or a comic page the evidence is a crop tagger disagreeing
with itself. Any `_LAYOUT_TAGS` image is **one character drawn several times**
— a turnaround, an outfit sheet, a girl once per panel — so nothing that belongs
to *her* can discriminate between the subjects. A trait that survived shared-tag
suppression there did so precisely because some crop missed it, and the
discriminative rule then *promotes* the miss.

The multi-view gate (on by default, `--bind_view_traits` reverts) therefore
suppresses at emission time, before any of the rewrite's removal rules see it:

- **The character name.** Every view is the same girl, so binding her name to
  one says the others are somebody else. All 16 `multiple views` rows that got a
  name into a clause on the first full-corpus run were single-character sheets
  (`hatsune miku` bound to 2 of 4 views).
- **Every `_VIEW_INVARIANT_GROUPS` trait** = `_CHARACTER_INVARIANT_GROUPS`
  (hair color/length/style, eyes, face, age, gender, skin, body shape, species,
  animal parts) **+ `body_parts`**. Anatomy is owned by the character the same
  way hair color is — a girl does not grow a navel between panel 1 and panel 3 —
  but its *visibility* genuinely varies with the view, so `body_parts` joins the
  set for this rule **only** and stays freely bindable on a real multi-character
  image.

What is left is what one view or panel has and another does not: outfit, pose,
expression, framing. Measured over the 2026-08-17 full-corpus dry run
(`--attribution_margin 0.35`, 394 proposals):

| | Gated rows | Clause tags dropped | Clauses emptied | Rows falling under `min_instances` |
|---|---|---|---|---|
| `multiple views` | 157 | 1454 / 3201 (45%) | 1 | 0 |
| comic panels only | 23 | 216 / 440 (49%) | 0 | 0 |
| **all layouts** | **180** | **1670 / 3641 (46%)** | **1** | **0** |

(Clause-count figures are an upper bound on the loss: the real run refills the
freed slots from the ranked tail up to `--max_clause_tags`.) Top surviving
groups across the gated set: expression 274, clothing_details 193, underwear
173, pose 166, bottom_clothing 130, top_clothing 118, swimwear 106. Nothing is
destroyed — a tag that never reaches a clause cannot be moved out of the bag, so
every suppressed trait stays asserted, flat.

This is strictly stronger than the corroboration rule under "What leaves the flat
bag" below, which only governs whether a tag may *leave* the bag; here it never
enters the clause. The rules stack: `--bind_view_traits` drops back to the
corroboration-only behaviour, which is what the pre-gate arm did.

## What leaves the flat bag (v2)

A tag moves out of the bag into its clause when **all five** hold. Fail any one
and it stays flat *and* stays bound — i.e. that single tag degrades to v1's
additive behaviour, which is why nothing here can produce a wrong caption, only
a less-resolved one.

| # | Rule | Why |
|---|---|---|
| 1 | **Not a character name** | The cast list stays flat and is bound as well — the hand-written convention, measured (19/19 duplicated ground-truth tags are names, 0 are attributes). The bag answers *who is in this image* and is how a prompt summons them; the clause answers *which one is where* |
| 2 | **Exactly one clause claims it** | Two clauses claiming a tag means it is shared, and a shared attribute belongs to the bag |
| 3 | **Corroboration**, for a character-invariant group | Hair color, eyes, body shape, species … are properties of a *character*, not of a view. On a `1girl, multiple views` sheet they hold in every panel, so moving `aqua hair` into one view would make the caption claim the other views are *not* aqua-haired. Such a tag moves only when the bag names **≥2 values of that group** — i.e. the caption is already enumerating per-subject values. Outfit / pose / expression carry no such implication and move freely |
| 4 | **Exclusive keep** | No *other* crop kept the tag. A crop that reached the tag's own calibrated threshold has the attribute, whatever the clause builder later did with it — kept twice but bound once is a selection artifact (clause budget, discriminative filter, view gate), not an attribution. This is the tagger's own per-tag decision answering the question rule 5 can only approximate |
| 5 | **Relative attribution margin** (`--attribution_margin`, 0.25) | The runner-up's probability must fall below `(1 - margin)` of the winner's, so a tag the tagger *nearly* kept on the second subject stays in the bag. Measured **relative to the winner**, not as an absolute gap — see below |

Rule 3's exception: booru tags a **single** character with two hair colors when
the hair is two-toned, so `multicolored hair` / `two-tone hair` / `gradient
hair` / `heterochromia` (and friends) in the bag pin that group flat — the "≥2
values" evidence is explained without a second subject. Those markers are
ungrouped in `groups.yaml`, so they are matched by name.

Rule 3 is deliberately **evidence-based rather than count-based**: 219 of the 373
first-sweep proposals carry no girls-count tag at all, so a `detected ==
characters` gate would have pinned nearly the whole corpus.

#### Why the margin is relative, and why rule 4 exists (2026-08-17)

Rules 4–5 replaced a single **absolute** gap test (`winner - runner_up ≥ 0.35`).
That test asked a question the numbers cannot answer: the tagger's decision
boundaries are **per-tag F1 thresholds spanning ~0.05–0.85**, so a fixed
probability gap is a different — and mostly impossible — test for every tag.
Re-scoring the 89 margin pins of the `ama_mitsuki` sweep against those
thresholds:

- **77 of 89** pinned tags had a runner-up the tagger **did not keep**. Rule 4's
  own premise — "a tag the tagger nearly kept on a second subject" — described
  only the other 12.
- Low-threshold tags could never clear the gap no matter how one-sided the call:
  `sleeves past fingers` (threshold 0.05) pinned at winner **0.342** vs
  runner-up **0.000**; `pulling own clothes` 0.254 vs 0.001; `upshorts` 0.234 vs
  0.001; `drink` 0.132 vs 0.000; `vest` 0.064 vs 0.000. A hard-zero runner-up is
  not a shared attribute.
- Meanwhile a high-threshold tag cleared the gap on genuinely ambiguous calls,
  because 0.35 is cheap when both probabilities live near 1.

The fix splits the question in two. Rule 4 takes the *categorical* half from the
tagger itself (`kept`, i.e. the per-tag threshold), which needs no tuning. Rule
5 keeps a *graded* guard for the runner-up that fell just short, but scores it as
`1 - runner_up/winner`, which is scale-free and so means the same thing across
the vocabulary. At the shipped 0.25 the pin population drops **89 → 28** (16
margin + 12 exclusive-keep) and moved tags rise **309 → 370** on the same
corpus, with the residual pins landing exactly where they should: `high heels`
0.800 vs 1.000, `shoes` 0.719 vs 0.960, `black jacket` 0.765 vs 0.863, plus the
genuinely-shared `long sleeves` / `jewelry` / `school uniform` / `sitting`. The
motivating case was `ama_mitsuki/12948301` — `underwear`, visible on the left
girl only, pinned flat at a 0.262 absolute gap because the right (clothed) crop
carried the tagger's generic 0.581 prior against a 0.600 threshold.

`--attribution_margin 0.0` reduces to rule 4 alone (trust the thresholds); the
old behaviour is not recoverable by a flag, and should not be.

The dry-run report records both sides per image — `moved[{tag, position,
margin}]` and `pinned{tag: rule}` — so a reviewer can see exactly why a tag
stayed. `summary.pinned_tags` aggregates the rules corpus-wide.

### Backing it out

The rewrite **moves** tags; it never deletes them, so a rewritten caption still
contains every tag it started with. `make caption-position ARGS="--flatten
--apply"` merges every clause back into its flat bag and drops the clauses —
text only, no SAM3, no tagger, no images. That is both the undo for an `--apply`
run and the way to build the clause-free control corpus for a training A/B.
Tag *order* is not restored byte-for-byte (a moved tag comes back at the end),
and `correct_caption` re-buckets it anyway. Note it flattens **hand-written**
clauses too — it cannot tell them apart — which is a real loss of curation on
those 14 captions.

## Running it

```bash
make caption-position                                  # dry run, whole dataset
make caption-position ARGS="--crops --qwen3 models/text_encoders/qwen_3_06b_base.safetensors"
make caption-position ARGS="--path_pattern 'artist_a/*'"   # scope a slice
make caption-position ARGS="--apply"                   # write (after the review)
make preprocess-te                                     # REQUIRED after --apply
make caption-position ARGS="--no_rewrite --apply"      # additive v1 (A/B arm)
make caption-position ARGS="--flatten --apply"         # undo: clauses → flat bag
```

GPU job (SAM3 + tagger held resident for the whole sweep), so it is
**daemon-routed** like every other agent-launched GPU work — it queues behind a
live train run instead of OOM-colliding. `--queue` detaches, `--inline` bypasses.

**Dry run is the default and writes nothing.** It emits
`post_image_dataset/captions/position/report.json`:

```
summary: {applied, rewrite, attribution_margin, seen, candidates, proposed,
          written, rewritten, moved_tags, pinned_tags{rule: n},
          skipped{reason: n}, max_tokens, over_token_budget[]}
images[]: {image, caption_path, status, detected, expected, original, proposed,
           tokens, instances[{position, box, score, tags, crop}],
           moved[{tag, position, margin}], pinned{tag: rule}}
```

With `--crops` it also exports the **exact mask-blanked pixels the tagger saw**,
mirroring the dataset layout, named `<stem>_<i>_<position>.png`. That is the
only way to tell a detection miss from a tagging miss when reviewing — read the
proposed clause next to its crop.

`--qwen3 <tokenizer>` adds a token count per proposal and flags anything past
512 (`qwen3_max_token_length` / `t5_max_token_length`). Past that the tail is
truncated **silently** at TE-cache time and, given the padding invariant, simply
never reaches the model. The current sweep is clear (median 198, max 491, none
over budget) — the additive v1 arm had one caption at 524. Check
`summary.over_token_budget` before applying anyway.

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

### What the rewrite does to the corpus (paired v1 / v2 sweep, 2026-08-17)

Both arms run the same code over the same 3008 images, differing only in
`--no_rewrite`. **The clauses come out byte-identical in both** — asserted over
all 394 proposals — which is the separation the design intends: the rewrite
decides what the *bag* keeps, never what a clause says.

| | v1 (`--no_rewrite`) | v2 (default) |
|---|---|---|
| proposals | 394 | 394 |
| captions whose bag changed | — | **380** |
| flat-bag tags across the proposals | 18334 | **16429** (−1905, −10.4%) |
| tags moved per caption (median / max) | — | 5 / 13 |
| caption tokens, median / max | 213 / **524** | 198 / **491** |
| captions over the 512-token budget | **1** | **0** |

The token column is a real ops win, not a rounding effect: v1's single
over-budget caption (`kat_(bu-kunn)/dan_8451598`) would have had its tail
silently truncated at TE-cache time, and stating each attribute once instead of
twice puts it back under the cap. Mean saving 14.5 tokens (6.5%).

Which rule pinned the tags that stayed flat, corpus-wide. **These counts predate
the relative margin** (rules 4–5 above) — that change cut the `margin` class by
~2/3 on the re-scored `ama_mitsuki` slice, so the corpus-wide `margin` row here
should be read as the old absolute-gap behaviour pending a full re-sweep:

| rule | tags |
|---|---|
| `margin` (runner-up crop too close) | 730 |
| `multi-clause` (shared — belongs to the bag) | 431 |
| `sole-value` (character-invariant, one value in the bag) | 404 |
| `character-name` (the cast list stays flat) | 54 |
| `two-tone-marker` (`multicolored hair` etc. explains the second value) | 30 |

So 1905 of the 3554 bag tags a clause claimed actually moved (54%) — the rest
degrade to v1's duplicate-assert, which is the safe direction by construction.
The margin is the single biggest brake; moved tags clear it by a median of 0.69
(floor 0.35), so the population that moves is not marginal.

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
is lost — a value no clause carries is a value the rewrite never takes, so it
stays in the flat bag. `hairstyle` is deliberately **not** gated even though it is a priority group: a
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

The gate is upstream of the v2 rewrite and load-bearing for it: the rewrite can
only remove what a clause carries, so gating emission to values the caption
already named is what keeps a hallucinated hair color from *replacing* the real
one in the bag.

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
| `--bind_view_traits` | — | On a repeated-subject layout (`multiple views` / comic panels), let a clause carry the character's name and traits. Gated by default — every view is the same girl; see the section above |
| `--no_rewrite` | — | Additive v1: append the clauses, leave the flat bag untouched (so every bound attribute is asserted twice). The A/B control arm |
| `--attribution_margin` | 0.25 | How far the winning crop must clear every other **relative to its own probability** (`1 - runner_up/winner`) before a tag may **leave** the bag, on top of the hard rule that no other crop kept it. `0.0` trusts the tagger's per-tag thresholds alone. The clause carries the tag either way |
| `--flatten` | off | Inverse pass — merge clauses back into the bag and drop them. Text only (no models). The undo, and the clause-free A/B corpus |
| `--qwen3` / `--max_tokens` | — / 512 | Token-budget column + over-budget flag |

## How clauses behave downstream

- **Caption variants** (`library/preprocess/caption_variants.py`) parse through
  the grammar and treat **each clause as an atomic unit**: kept or dropped whole
  at `clause_dropout_rate` (defaults to `tag_dropout_rate`), tags shuffled
  inside, header never randomized. Per-tag dropout inside a clause would leave a
  half-described position. Clause-free captions keep the historical raw split,
  so v0 stays byte-identical.

  **v2 changes what a dropped clause costs.** Under the additive v1 the
  attributes were still in the flat bag, so dropping a clause removed only the
  *binding*; under v2 they are nowhere else, so a dropped clause drops that
  subject's attributes from the variant entirely. The variant is still a
  truthful (if less complete) caption — this is correlated tag-dropout, not
  corruption — but it is a stronger perturbation than the same rate applied
  per-tag. Set `clause_dropout_rate = 0.0` to keep every variant fully bound;
  that is the conservative setting on a rewritten corpus.
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

- It **rewrites the source captions in place**, and under v2 that includes
  taking bound tags out of the flat bag. The pass is idempotent — a caption that
  already carries clauses is skipped by the prefilter — and it is reversible
  from the CLI (`make caption-position ARGS="--flatten --apply"`), but there is
  no undo button in the GUI. `make caption-position` (dry run, `report.json`,
  `--crops`) is still the way to eyeball proposals first, and is worth doing
  once on a new dataset.
- The TE-cache staleness trap is handled for you here: the stage runs *inside*
  the preprocess chain, so the variants and TE caches are rebuilt from the
  rewritten captions in the same job. It is the standalone `--apply` path that
  needs the manual `make preprocess-te`.

CLI equivalents: `--caption_position_clauses` / `--no_caption_position_clauses`
on `make preprocess`, or `caption_position_clauses = true` in the merged config.

**The spot-check is still owed** (≥90% of proposed clauses right on ~30 reviewed
images, weighted toward grids, overlapping pairs and N≥4; under v2, also that no
reviewed image *lost* an attribute that belonged to a subject it was taken
from) — which is why the checkbox ships off by default.

## Limits / open

- **Hair *length* across crops** — `long hair` vs `medium hair` on two views of
  the same character is crop-scale dependent, and `hair_length` is a priority
  group. The discriminative rule masks most of it (shared → suppressed), but a
  scale artifact that differs between views will bind. Watch it in the
  spot-check.
- **Character names on crops** are the weakest signal (probe B: 4/7), which is
  why they need the flat-bag floor.
- **Boys / POV** are out of the default sweep — `--prompt person` sweeps them
  separately; nothing is hardcoded to `girl`.
- **Bag-removal tolerance is the open risk of v2.** Probe A validated clause
  *comprehension* (48/48 sides correct) — it did not validate that removing a
  tag from the flat bag is safe for a model pretrained on flat bags. The five
  rules above bound *which* tags move, not whether the model likes the resulting
  distribution; that is what the training A/B below answers.
- **Is the margin in the right place?** The absolute-gap version was measurably
  *not* (see "Why the margin is relative"); the relative one at 0.25 is now
  calibrated against one artist slice, not the corpus. The report carries the
  per-move margin on the same scale as the knob, so retune it against a
  full-corpus spot-check rather than by guess.
- **`sole-value` on non-identity invariants.** `body_shape` / `skin` /
  `face_features` are in the invariant set, so a `2girls` caption naming one
  `large breasts` keeps it flat even when only one girl has it. Safe, and the
  class most likely to be over-pinned — count it in the spot-check before
  loosening.

## Owed work — the two gates before this is on by default

The design proposal is retired (`_archive/proposals/position_captions.md`); what
it left open lives here.

**1. Spot-check the dry run.** ~30 proposed clauses from
`post_image_dataset/captions/position/report.json`, weighted toward grids,
overlapping pairs and N≥4, read against the exported `--crops`. Exit: clause
proposals right at **≥90%**, count-disagreement skip rate reported, and — v2's
own criterion — **no reviewed image losing an attribute that belonged to a
subject it was taken from**. Until this passes, no `--apply` and the GUI
checkbox stays off.

**2. Training A/B.** `--apply`, regenerate variants + TE caches, train the
standard LoRA recipe on a multi-girl-dense slice against a twin control on the
same seed. Compare **renders, not raw ΔW cosines** — the paired-ΔW chaos floor
makes absolute cosines unreadable without a twin control. Eval: a probe-A-style
render test through the trained LoRA (does binding survive the artist style?)
plus multi-character sample grids. Exit: binding ≥ control, no regression on
single-girl renders.

Three corpora are available for that A/B at one preprocess each — v2 (default),
v1 (`--no_rewrite`), clause-free (`--flatten`) — so *do clauses help* and *does
bag-removal hurt* are separable in one experiment instead of sequential phases.

## Code map

| Path | Role |
|---|---|
| `library/captioning/position_clauses.py` | Clause grammar (torch-free) — parse / compose / `flatten_caption` / position vocabulary |
| `library/preprocess/position_captions.py` | Pipeline orchestration (`plan_bag_removals` = the v2 rules, `flatten_captions` = the undo); models injected as `detect_fn` / `tag_fn` |
| `scripts/preprocess/position_captions.py` | CLI shell — argparse + SAM3/tagger loading |
| `scripts/tasks/preprocess.py::cmd_caption_position` | `make caption-position` (daemon-routed) |
| `library/preprocess/caption_variants.py` | Atomic-clause variant generation |
| `library/captioning/correction.py` | Clause-aware order correction |
| `tests/test_position_captions.py` | 65 unit tests (grammar round-trip, ordering, selection, skip paths, the four rewrite rules) |
| `bench/position_captions/` | Phase-0 probe envelopes (`20260817-1122-autocaption`, `20260817-1123-binding`) |
