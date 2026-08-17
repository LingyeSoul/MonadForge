# Position-aware captions — binding attributes to the subject they belong to

A preprocessing pass that gives multi-subject images spatially bound captions
in the dataset's existing hand-written convention: SAM3 detects the subjects,
they are put in reading order, each mask-blanked crop is tagged by the Anima
Tagger, and the caption is **rewritten** so each attribute is asserted once, in
the clause of the subject it belongs to.

Status: **v2 shipped and runnable as `make caption-position`.** v2 *replaces* v1
— same pipeline, but the flat bag is rewritten instead of merely appended to;
`--no_rewrite` keeps the additive v1 behaviour for the A/B arm. Dry run is the
default and **nothing has been applied to the caption master yet** — two gates
are still owed (spot-check, then a training A/B), spelled out at the bottom.

This is the canonical doc: what it does, how to run it, what to watch. It is
kept deliberately short — the measurement logs behind each rule (sweep tables,
skip triage, refuted alternatives) live in git history and in the retired design
proposal [`_archive/proposals/position_captions.md`](../../_archive/proposals/position_captions.md).

## Why

The flat tag bag cannot say *which* attribute belongs to *which* subject:

- `3girls, blonde hair, aqua hair, red hair, …` — who is blonde?
- `1girl, multiple views, maid, playboy bunny, swimsuit, pink jacket, …` —
  four outfit views of one character, every outfit unbound.

The dataset already has a hand-written answer (14 captions carry it), and the
base model already obeys it: a Phase-0 probe rendered no-LoRA images from
counterbalanced positional prompts and the tagger scored **48/48 sides correct**
(chance 50%). Clauses reinforce an existing conditioning channel rather than
teaching one from scratch.

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
by the subject that has it. A bag that still lists `black hair, white hair, pink
hair` alongside the clauses is still claiming all three of all three girls,
which is the ambiguity the clauses exist to remove.

**This is the convention, not an invention.** Across the 14 hand-written
ground-truth captions: 244 clause tags, of which 19 also appear in the flat bag
— and **all 19 are character names**. Not one attribute is duplicated. The
hand-written form is exactly "cast list flat, attributes bound", which is what
v2 emits and the additive v1 did not.

## The clause grammar — read this before touching any caption code

The convention delimits clauses with a **period** and tags with **commas**:

```
<flat tag bag>. On the left, akita neru, yellow eyes. On the right, kasane teto.
```

So a plain `caption.split(",")` glues the clause header onto the previous tag
(`"white socks. On the left"`), and every consumer keying off
`tag.startswith("On the ")` then sees **no clauses at all**. That was a live bug
before v1: `anima_smart_shuffle_caption`'s section logic and the
identity-randomize guard both silently saw a flat caption, scattering the
hand-written clause attributes across the caption in 3 of the 4 default variants.

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
the canonical capitalized `On the `. `In the …` (a few hand-written scene-region
clauses) parses and round-trips byte-stable but is never emitted.

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
(`top`/`middle`/`bottom`); beyond that it degrades to plain left→right.

## Pipeline

Per candidate image (`library/preprocess/position_captions.py`):

1. **Detect** — SAM3 with the text prompt `girl` on the *resized* image (the
   pixels training actually sees) → per-instance boxes + masks + scores. Score
   floor 0.5, greedy IoU dedupe at 0.65, and a **retry at 0.35** when the count
   undershoots (an unconditional low threshold floods grids with
   part-detections). The retry target is `expected or min_instances`, so
   multi-view sheets — whose `caption_subject_count` is `None` by design —
   attempt it too. `build_detect_fn` constructs the SAM3 processor at the
   *lowest* threshold it may be asked for and memoises raw detections per image,
   so the retry is a pure re-filter, not a second image encode (SAM3 applies its
   own `confidence_threshold` internally — a post-filter can only ever remove
   boxes).
2. **Order** — row-cluster, then left→right; positions assigned as above.
3. **Crop + blank** — padded bbox crop (6%), every non-instance pixel blanked to
   white using the instance mask. Load-bearing: without blanking, a neighbor
   standing inside the padded box contributes their hair to this subject's tags.
4. **Tag** — Anima Tagger per crop, then clause selection (below).
5. **Rewrite** — the tags a clause has earned leave the flat bag, and
   `compose_caption(flat_tags, clauses)` is written back to the caption
   **master** (`image_dataset/*.txt`), which `preprocess-captions` mirrors into
   `resized/` and the TE step then encodes.

Models are injected as `detect_fn` / `tag_fn` callables, so the orchestration
module imports neither SAM3 nor the tagger and unit-tests with stubs; the CLI
shell (`scripts/preprocess/position_captions.py`) owns argparse + model loading.

**Layout tags decouple the girls-count from the bindable-subject count**
(`_LAYOUT_TAGS`): `multiple views` **and** comic pages (`comic`, `silent comic`,
`sequential`, `Nkoma`, `multiple 4koma`) — a `1girl, 2koma` page is two
subjects, so the count check is waived and detection is trusted. `Nkoma` names
the panel count, which restores a `panels × (girls + boys)` ceiling
(`caption_panel_ceiling`) so one girl detected twice still skips. `page number`
is deliberately **not** a layout tag — it marks a scanned art-book page, not a
panel grid.

The strict-count gate is a **range**, `girls … girls + boys` (`caption_boy_count`;
unknown counts like `multiple boys` drop the upper bound), because the `girl`
prompt picks males up inconsistently. Open-ended crowd tags (`6+girls`,
`multiple girls`) return `None` and defer to detection.

## What goes into a clause

Eligibility comes from the tagger's own `groups.yaml`, not substring heuristics,
so the two can't drift:

- **Per-subject groups bind** — hair (color/length/style/accessory), eyes,
  expression, body, all clothing groups, pose/gesture/action (`SUBJECT_GROUPS`).
- **Scene groups never bind** — lighting, background, framing, medium,
  `interaction`, `character_relationship`.
- **Copyright / artist / metadata / deprecated / count / rating are excluded
  outright**, on *every* emission path — the check lives in `add()`, not only on
  the ranked path, because an excluded tag can also be *grouped* (`light brown
  hair` is a deprecated alias `groups.yaml` still files under `hair_color`, and
  it rode the exclusive-group step straight into clauses).
- **Ungrouped tags** — where curated compounds like `pink jacket` live — are
  admitted only when they are both **in the flat bag** and **attributable**
  (kept on exactly one crop).
- **At most one member of an exclusive group** (softmax / `softmax_when_solo`),
  or a contaminated crop emits `green hair, …, aqua hair` for one subject.

Emission order: character name → exclusive-group winners (`hair_color`,
`eye_color`, `hair_length`, `hairstyle`) → everything else ranked, preferring
tags the caption already curated. Cap 8 tags per clause.

A **character name** needs both floors conjunctively: `--name_confidence` (0.5)
*and* membership in the flat bag — names are the weakest crop signal, so an
unlisted one is most likely a crop artifact. `--allow_unlisted_names` relaxes the
second. One identity per subject.

### Move, don't invent — the flat bag fills a clause first

Candidates are ranked once (the order above), then admitted in **two passes**:
everything already in the flat bag, and only then up to `--max_novel_tags` (**1**)
tags the caption never contained. Emission order is still the ranking, so a
clause reads name → hair → eyes → the rest.

The reason is mechanical: a novel clause tag cannot be a **move**.
`plan_bag_removals` only removes what is in the bag, so a tag the caption never
had is a pure v1-style addition — it disambiguates nothing and asserts something
the curated caption declined to. Bag-blind selection spent ~46% of the clause
budget that way. Measured on one artist slice, budget 1 vs 8 cuts clause tags
583 vs 983 and novel tags 115 vs 515 while leaving **the moves byte-identical**:
the budget removes padding, it does not buy bindings.

`--max_novel_tags 0` never invents at all; `8` is the bag-blind A/B arm. The
report carries `clause_tags` / `novel_tags` / `reuse_ratio` in its summary.

**Refinement collapse is deliberately not built** (`breasts` → `large breasts`).
A safe same-group rule catches ~3% of it, and the looser rule also collapses
`on stomach` → `stomach` and `looking back` → `back`. Doing it properly needs a
tag *implication* table, which neither `taxonomy.py` nor `groups.yaml` has. With
a novel budget of 1 it is moot — don't re-propose without that table.

### A clause carries only what discriminates

Any tag *every* crop keeps is suppressed (`--keep_shared_tags` disables). On a
`1girl, multiple views` outfit sheet all views are the same character with the
same hair and eyes; repeating `hatsune miku, aqua hair, twintails` four times
binds nothing and crowds out the maid / bunny / bikini that actually tells the
views apart. A shared attribute keeps its place in the flat bag — where an
attribute belonging to *everyone* belongs. When suppression empties every clause
the image is skipped as `no-discriminative-tags`.

### On a repeated-subject layout, only what a view can differ in

Shared-tag suppression is the right rule but it is evidence-based, and on a
`multiple views` sheet or a comic page the evidence is a crop tagger disagreeing
with itself. Any `_LAYOUT_TAGS` image is **one character drawn several times**,
so nothing that belongs to *her* can discriminate between the subjects — a trait
that survived shared-tag suppression there did so precisely because some crop
*missed* it, and the discriminative rule then promotes the miss.

The multi-view gate (on by default, `--bind_view_traits` reverts) suppresses at
emission time, before any removal rule sees it:

- **The character name.** Every view is the same girl, so binding her name to one
  says the others are somebody else.
- **Every `_VIEW_INVARIANT_GROUPS` trait** = `_CHARACTER_INVARIANT_GROUPS` (hair
  color/length/style, eyes, face, age, gender, skin, body shape, species, animal
  parts) **+ `body_parts`**. Anatomy is owned by the character the same way hair
  color is, but its *visibility* genuinely varies with the view — so `body_parts`
  joins the set for this rule **only** and stays freely bindable on a real
  multi-character image.

What is left is what one view or panel has and another does not: outfit, pose,
expression, framing. It drops ~46% of clause tags on gated rows and empties
essentially none of them — a suppressed trait stays asserted, flat, and the
freed slots refill from the ranked tail.

This is strictly stronger than the corroboration rule below, which only governs
whether a tag may *leave* the bag; here it never enters the clause.

### The identity gate — the flat bag outranks the crop tagger

For a group in `ClauseVocabulary.gated_groups()` a clause may carry a value the
caption named, or nothing. Emitting the crop's winner unconditionally is a noise
*amplifier*, not a neutral default: the discriminative rule suppresses whatever
every crop agrees on, so a wrong outlier is exactly what survives — on a back
view the eyes are not visible, the tagger guesses anyway, and the guess is
promoted precisely because it disagrees with the front view. Measured: a third of
identity clause tags claimed a value the caption never listed, and 78% of
single-character `multiple views` sheets bound contradictory hair or eye colors
to views of the same girl. With the gate, both go to zero, and total clause tags
*rise* — a blocked slot refills from the ranked tail, trading an invented hair
color for a real outfit tag.

**The gated set is derived, not hand-picked**: `_BAG_GATED_GROUPS` (`hair_color`,
`eye_color`, `hair_length`) **plus every exclusive subject group the checkpoint
declares**. An exclusive (softmax) group holds exactly one value by construction,
so a crop naming a second is a contradiction rather than extra detail — the same
argument the original three were picked on. Hand-picking left 150 contradictions
on a full sweep (103 `body_shape` — bag `flat chest`, clause `large breasts`; 39
`fashion_style`; `fox girl` → `cat girl`). Deriving it from `groups.yaml` also
means the gate cannot drift from the tagger. `hair_length` is the one member that
is not exclusive, which is why the hand list survives alongside the derivation.
`hairstyle` is deliberately **not** gated even though it is a priority group: a
crop legitimately reveals a `hair bun` the booru caption never tagged, and unlike
a color that contradicts nothing.

An exclusive slot also prefers a **kept tag the caption already named** over the
crop's softmax winner. Taking the winner unconditionally let the gate reject it
and emit *nothing*, losing a bindable tag to a rejected guess.

The gate is upstream of the v2 rewrite and load-bearing for it: the rewrite can
only remove what a clause carries, so gating emission to values the caption
already named is what keeps a hallucinated hair color from *replacing* the real
one in the bag. `--ungated_identity` restores the old behaviour for A/B.

### Body-part detection fallback (opt-in)

Some sheets are one small full body plus two or three **headless close-up
panels** — a hip, a backside, a crotch. The `girl` prompt cannot see those at any
threshold, so the image dies on `too-few-instances` with its most attribute-dense
panels never tagged.

`--part_prompts buttocks,hips,thighs` runs extra SAM3 prompts as a **second
escalation**, under the same undershoot condition as the low-threshold retry —
never on an image the subject prompt already resolved, where they could only add
nested duplicates. The image encoding is reused (`set_text_prompt` re-grounds
against the cached `backbone_out`), so each prompt costs a grounding pass, not a
re-encode. Four things are typed differently for part boxes:

- **Containment suppression is ON** (0.7) even though it ships off globally: a
  *subject* nested in another subject is routinely real, a **part** nested in a
  subject never is.
- **Part crops skip mask-blanking.** On a part box the mask *is* the part, so
  blanking deleted the torn jeans / pantyhose the pass exists to recover and
  handed the tagger a bare skin blob (`pink hair, black eyes, nude`).
- **Identity groups are suppressed** (`allow_identity=False`) — no head means no
  evidence.
- **Part boxes top up to the target and no further** — a part prompt fragments
  (`thighs` returning four boxes for two panels).

It is opt-in and the win is modest (~10 images corpus-wide, one clean recovery in
three on the signature artist). Its residual failure — two crops of the *same*
kind of panel, so everything true about them is shared and suppressed — is the
same shape as the problem the identity gate fixes and is **not** addressed.

## What leaves the flat bag (v2)

A tag moves out of the bag into its clause when **all five** hold. Fail any one
and it stays flat *and* stays bound — i.e. that single tag degrades to v1's
additive behaviour, which is why nothing here can produce a wrong caption, only a
less-resolved one.

| # | Rule | Why |
|---|---|---|
| 1 | **Not a character name** | The cast list stays flat and is bound as well — the hand-written convention, measured (19/19 duplicated ground-truth tags are names, 0 are attributes). The bag answers *who is in this image* and is how a prompt summons them; the clause answers *which one is where* |
| 2 | **Exactly one clause claims it** | Two clauses claiming a tag means it is shared, and a shared attribute belongs to the bag |
| 3 | **Corroboration**, for a character-invariant group | Hair color, eyes, body shape, species … are properties of a *character*, not of a view. On a `1girl, multiple views` sheet they hold in every panel, so moving `aqua hair` into one view would claim the other views are *not* aqua-haired. Such a tag moves only when the bag names **≥2 values of that group** — i.e. the caption is already enumerating per-subject values |
| 4 | **Exclusive keep** | No *other* crop kept the tag. A crop that reached the tag's own calibrated threshold has the attribute, whatever the clause builder later did with it — kept twice but bound once is a selection artifact (clause budget, discriminative filter, view gate), not an attribution. This is the tagger's own per-tag decision answering the question rule 5 can only approximate |
| 5 | **Relative attribution margin** (`--attribution_margin`, 0.25) | The runner-up's probability must fall below `(1 - margin)` of the winner's, so a tag the tagger *nearly* kept on the second subject stays in the bag |

Rule 3's exception: booru tags a **single** character with two hair colors when
the hair is two-toned, so `multicolored hair` / `two-tone hair` / `gradient hair`
/ `heterochromia` in the bag pin that group flat — the "≥2 values" evidence is
explained without a second subject. Those markers are ungrouped in `groups.yaml`,
so they are matched by name. Rule 3 is evidence-based rather than count-based
because most proposals carry no girls-count tag at all, so a `detected ==
characters` gate would pin nearly the whole corpus.

**Why the margin is relative.** Rules 4–5 replaced a single **absolute** gap test
(`winner - runner_up ≥ 0.35`), which asked a question the numbers cannot answer:
the tagger's decision boundaries are **per-tag F1 thresholds spanning ~0.05–0.85**,
so a fixed probability gap is a different — and mostly impossible — test for every
tag. It pinned `sleeves past fingers` (threshold 0.05) at winner 0.342 vs
runner-up **0.000** while waving through genuinely ambiguous high-threshold calls;
77 of 89 pins on the re-scored slice had a runner-up the tagger never kept at all.
The fix splits the question: rule 4 takes the *categorical* half from the tagger
itself (needs no tuning), rule 5 keeps a *graded* guard scored as
`1 - runner_up/winner`, which is scale-free and so means the same thing across the
vocabulary. The split took pins 89 → 28 and moves 309 → 370 on the same corpus.
`--attribution_margin 0.0` reduces to rule 4 alone; the absolute behaviour is not
recoverable by a flag, and should not be.

The dry-run report records both sides per image — `moved[{tag, position, margin}]`
and `pinned{tag: rule}` — so a reviewer can see exactly why a tag stayed.
`summary.pinned_tags` aggregates the rules corpus-wide.

### Backing it out

The rewrite **moves** tags; it never deletes them, so a rewritten caption still
contains every tag it started with. `make caption-position ARGS="--flatten
--apply"` merges every clause back into its flat bag and drops the clauses — text
only, no SAM3, no tagger, no images. That is both the undo for an `--apply` run
and the way to build the clause-free control corpus for a training A/B. Tag
*order* is not restored byte-for-byte (a moved tag comes back at the end), and
`correct_caption` re-buckets it anyway. Note it flattens **hand-written** clauses
too — it cannot tell them apart — a real loss of curation on those 14 captions.

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
          skipped{reason: n}, clause_tags, novel_tags, reuse_ratio,
          max_tokens, over_token_budget[]}
images[]: {image, caption_path, status, detected, expected, original, proposed,
           tokens, instances[{position, box, score, tags, crop}],
           moved[{tag, position, margin}], pinned{tag: rule}}
```

With `--crops` it also exports the **exact mask-blanked pixels the tagger saw**,
mirroring the dataset layout, named `<stem>_<i>_<position>.png` — the only way to
tell a detection miss from a tagging miss when reviewing. Skipped rows record
their `detections` too and get a box overlay under `crops/_skipped/`.

`--qwen3 <tokenizer>` adds a token count per proposal and flags anything past 512
(`qwen3_max_token_length`). Past that the tail is truncated **silently** at
TE-cache time and, given the padding invariant, never reaches the model. v2 helps
here — asserting each attribute once instead of twice saves ~6.5% of tokens and
took the corpus's one over-budget caption back under the cap — but check
`summary.over_token_budget` before applying anyway.

### Two silent-failure traps in the ops sequence

1. **Caption edits do NOT invalidate the TE caches.** After `--apply` the caches
   still *look* current and training keeps using the pre-clause embeddings until
   an explicit `make preprocess-te`. The script prints this reminder.
2. **`*.variants.txt` sidecars override the CLI dropout rate**, so a stale
   sidecar keeps training the pre-clause caption even after re-encoding.
   `preprocess-te` chains `preprocess-captions` and regenerates them first, which
   is why it — and not a bare TE re-encode — is the required follow-up.

### Reading the skip reasons

| Reason | What it means |
|---|---|
| `single-subject` | Not a candidate: one subject, no layout tag |
| `already-has-clauses` | Hand-written clauses — left alone |
| `too-few-instances` | Detection undershot even after the retry (and the part fallback, if enabled) |
| `count-mismatch` | Detection disagrees with the girls…girls+boys range, or busts the koma ceiling |
| `too-many-instances` | Past `--max_instances` |
| `no-discriminative-tags` | Every clause emptied — the subjects are indistinguishable to the tagger |

A mismatch is a **skip, not a wrong write**, which is the safe direction; lowering
the detection floor trades `too-few-instances` for `count-mismatch`.

**Two settled negatives — don't re-propose without new evidence:**

- **Containment suppression ships off** (`--containment_threshold 1.01`). It
  looks like the obvious fix for nested boxes (insets, group boxes), and it was
  measured to break 34 previously-proposing rows to recover ~2. A real second
  subject is exactly as nested as a group box — one girl in front of another, an
  embrace — and this corpus has far more of those. Only the inset half is handled
  automatically, by `--min_area_frac` (0.005 of the canvas).
- **No automatic gate on fragmentary masks.** ~6% of instances recovered in the
  0.35–0.5 band have a broken mask, and the blanked crop feeds the tagger a mix.
  Mask *fill* does not separate them (a clean 0.87-score figure sits at the same
  0.267 fill as the bad ones), row/column *gap* maxes out uselessly because the
  blobs are diagonally offset, and `main_frac` correlates but also flags visually
  clean crops whose hair or limbs simply split. The report carries a per-detection
  `score`; the low-score instances are the ones worth eyeballing.

## Knobs

`ARGS="…"` on the make target; every flag has a `--kebab-case` alias.

| Flag | Default | What it does |
|---|---|---|
| `--apply` | off | Write to the caption master (else dry run) |
| `--path_pattern` | `*` | fnmatch glob (`\|` to OR) relative to the resized dir |
| `--crops` | off | Export the mask-blanked crops next to the report |
| `--prompt` | `girl` | SAM3 subject prompt (`person` sweeps the rare on-screen-boy images) |
| `--score_threshold` / `--retry_score_threshold` | 0.5 / 0.35 | Detection floor; retry floor when the count undershoots. These are SAM3's **own** confidence floor, not a post-filter |
| `--iou_threshold` / `--pad` | 0.65 / 0.06 | Dedupe IoU; bbox padding fraction |
| `--containment_threshold` | 1.01 (off) | Suppress a box this nested inside a kept one (intersection over the *smaller* box). Measured harmful on this corpus — see above before enabling |
| `--min_area_frac` | 0.005 | Drop detections below this fraction of the image (insets are not subjects) |
| `--no_blank_crops` | — | Skip mask-blanking (diagnostic only — it is what causes cross-subject hair bleed) |
| `--row_tol` | 0.25 | Row-clustering gap as a fraction of image height |
| `--part_prompts` | off | Comma-separated body-part prompts, tried **only** when the subject prompt undershoots — recovers headless close-up panels. Try `buttocks,hips,thighs` |
| `--part_score_threshold` / `--part_containment_threshold` | 0.5 / 0.7 | Part-box confidence floor; drop a part box this nested inside an already-kept box |
| `--ungated_identity` | — | Let a clause carry a value the caption never listed for a gated group (the identity trio plus every exclusive subject group) |
| `--min_instances` / `--max_instances` | 2 / 8 | Instance-count window |
| `--no_strict_count` | — | Propose even when detection disagrees with the girls-count |
| `--max_clause_tags` | 8 | Cap per clause |
| `--max_novel_tags` | 1 | How many tags a clause may introduce that the caption never contained — the bag fills it first. `0` never invents; `8` is the bag-blind A/B arm |
| `--name_confidence` / `--allow_unlisted_names` | 0.5 / off | Character-name floors |
| `--keep_shared_tags` | — | Keep tags every crop agrees on (disables the discriminative rule) |
| `--bind_view_traits` | — | On a repeated-subject layout, let a clause carry the character's name and view-invariant traits (disables the multi-view gate) |
| `--no_rewrite` | — | Additive v1: append the clauses, leave the flat bag untouched (every bound attribute asserted twice). The A/B control arm |
| `--attribution_margin` | 0.25 | How far the winning crop must clear every other **relative to its own probability** (`1 - runner_up/winner`) before a tag may **leave** the bag, on top of the hard rule that no other crop kept it. `0.0` trusts the per-tag thresholds alone. The clause carries the tag either way |
| `--flatten` | off | Inverse pass — merge clauses back into the bag and drop them. Text only (no models). The undo, and the clause-free A/B corpus |
| `--qwen3` / `--max_tokens` | — / 512 | Token-budget column + over-budget flag |

## How clauses behave downstream

- **Caption variants** (`library/preprocess/caption_variants.py`) parse through
  the grammar and treat **each clause as an atomic unit**: kept or dropped whole
  at `clause_dropout_rate` (defaults to `tag_dropout_rate`), tags shuffled
  inside, header never randomized. Per-tag dropout inside a clause would leave a
  half-described position. Clause-free captions keep the historical raw split, so
  v0 stays byte-identical.

  **v2 changes what a dropped clause costs.** Under the additive v1 the
  attributes were still in the flat bag, so dropping a clause removed only the
  *binding*; under v2 they are nowhere else, so a dropped clause drops that
  subject's attributes from the variant entirely. Still a truthful (if less
  complete) caption — correlated tag-dropout, not corruption — but a stronger
  perturbation than the same rate applied per-tag. Set `clause_dropout_rate = 0.0`
  to keep every variant fully bound; that is the conservative setting on a
  rewritten corpus.
- **Order correction** (`correct_caption`) splits clauses off before
  bucket-reordering the flat bag — clauses are already ordered left→right and
  their tags are position-scoped, so reordering them across the caption is
  exactly the shredding the grammar fix removed.
- **Auto-tagging** (`make caption-autotag --mode merge`) round-trips position
  clauses verbatim and counts their bound tags as present, so a merge after
  `caption-position` cannot re-flatten a binding back into the bag.
- **Training** sees no new machinery at all: this is a caption-text feature, and
  the clauses ride the ordinary TE path.

## Turning it on in the pipeline

Off by default in all three surfaces; each of them runs the stage **with
`--apply`** (no dry run) inline in `make preprocess`, after the VAE cache and
before the caption/TE steps — because it rewrites the caption master those two
then read, and the same job re-encodes, so the TE-staleness trap is handled for
you. Only the standalone `--apply` path needs the manual `make preprocess-te`.

The two caption-only entry points honour the same knob: `make preprocess-captions`
(mirror + variant sidecars) and `make preprocess-te` (which chains it) run the
stage before they read the master, so a caption-only re-encode can never mirror a
pre-clause caption. In the full chain it still runs exactly once — the "already
ran" mark rides on the caption-config dict `cmd_preprocess` threads down into
`cmd_preprocess_te` → `cmd_preprocess_captions`, so the later calls no-op instead
of re-paying the SAM3 + tagger load. The GPU order stays VAE → SAM3/tagger
because `cmd_preprocess` keeps its own early call in that slot.

| Surface | How |
|---|---|
| Config | `caption_position_clauses = true` in `configs/preprocess.toml` (user-owned, survives `make update`) |
| CLI | `make preprocess ARGS="--caption_position_clauses"` / `--no_caption_position_clauses` |
| Env | `CAPTION_POSITION_CLAUSES=1` |
| GUI | Preprocessing tab → **캡션 편집 / Caption rewriting** → `위치 절 생성 (다중 인물)` |

Precedence is env → config, with the CLI flag winning over both; the GUI always
exports the env var (persisting the checkbox to its variant's `[variant]` table),
so it wins over `preprocess.toml` and the ConfigTab Train auto-chain honours it.

Two things follow from applying without a review step: it **rewrites the source
captions in place** (under v2 that includes taking bound tags out of the flat
bag), and there is no undo button in the GUI. The pass is idempotent — a caption
that already carries clauses is skipped by the prefilter — and reversible from the
CLI (`--flatten --apply`), but `make caption-position` (dry run, `report.json`,
`--crops`) is still the way to eyeball proposals first, and is worth doing once on
a new dataset. **The spot-check below is still owed**, which is why every surface
ships off.

## Limits / open

- **Hair *length* across crops** — `long hair` vs `medium hair` on two views of
  the same character is crop-scale dependent. The discriminative rule and the
  multi-view gate mask most of it, but a scale artifact on a real
  multi-character image will bind. Watch it in the spot-check.
- **Character names on crops** are the weakest signal (probe B: 4/7), which is
  why they need the flat-bag floor.
- **Boys / POV** are out of the default sweep — `--prompt person` sweeps them
  separately; nothing is hardcoded to `girl`.
- **Under-detection has an irreducible tail.** SAM3 scales every instance
  probability by one global presence score, so on some framings (extreme
  close-up, from-behind, cropped body) *all* boxes sink together and no threshold
  recovers them.
- **Bag-removal tolerance is the open risk of v2.** The probe validated clause
  *comprehension* (48/48 sides correct) — it did not validate that removing a tag
  from the flat bag is safe for a model pretrained on flat bags. The five rules
  bound *which* tags move, not whether the model likes the resulting
  distribution; that is what the training A/B answers.
- **Is the margin in the right place?** The absolute-gap version measurably was
  not; the relative one at 0.25 is calibrated against one artist slice, not the
  corpus. The report carries the per-move margin on the same scale as the knob —
  retune it against a full-corpus spot-check rather than by guess.
- **`sole-value` on non-identity invariants.** `body_shape` / `skin` /
  `face_features` are in the invariant set, so a `2girls` caption naming one
  `large breasts` keeps it flat even when only one girl has it. Safe, and the
  class most likely to be over-pinned — count it in the spot-check before
  loosening.
  
## Code map

| Path | Role |
|---|---|
| `library/captioning/position_clauses.py` | Clause grammar (torch-free) — parse / compose / `flatten_caption` / position vocabulary |
| `library/preprocess/position_captions.py` | Pipeline orchestration (`plan_bag_removals` = the v2 rules, `flatten_captions` = the undo); models injected as `detect_fn` / `tag_fn` |
| `scripts/preprocess/position_captions.py` | CLI shell — argparse + SAM3/tagger loading |
| `scripts/tasks/preprocess.py::cmd_caption_position` | `make caption-position` (daemon-routed) + the in-chain stage |
| `library/preprocess/caption_variants.py` | Atomic-clause variant generation |
| `library/captioning/correction.py` | Clause-aware order correction |
| `tests/test_position_captions.py` | Unit tests (grammar round-trip, ordering, selection, skip paths, the rewrite rules) |
| `bench/position_captions/` | Phase-0 probe envelopes (`20260817-1122-autocaption`, `20260817-1123-binding`) |
