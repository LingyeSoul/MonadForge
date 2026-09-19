# Position-aware captions — natural caption enhance → smart caption rewrite

> **RETIRED 2026-08-17 — both proposed features are built and shipped.** This is
> the design record: why clauses, the Phase-0 probe evidence, and the reasoning
> behind each rule. It is **not** the operational contract and will not be kept
> current — read
> [`docs/experimental/position_captions.md`](../../docs/experimental/position_captions.md)
> for how to run it, what the knobs do, and the two gates still owed (the
> dry-run spot-check and the training A/B, both carried over there).
>
> One rule in here was **overruled by measurement** during the v2 build and is
> corrected in place below: Feature 2's sketch moved character names out of the
> flat bag. The hand-written ground truth keeps them (19/19 of its duplicated
> clause tags are names), so the shipped rewrite pins the cast list flat.

Status: **Phase 0 DONE (both probes green) — Feature 1 (v1) shipped, then
Feature 2 (v2) shipped on top and it REPLACED v1** as the default behaviour of
`make caption-position` (2026-08-17). Probes + result envelopes live in
`bench/position_captions/` (runs `20260817-1122-autocaption`,
`20260817-1123-binding`). Code: `library/captioning/position_clauses.py`
(clause grammar + `flatten_caption`), `library/preprocess/position_captions.py`
(pipeline + `plan_bag_removals`), `scripts/preprocess/position_captions.py`
(CLI), tests in `tests/test_position_captions.py`. Dry-run is the default on the
CLI; **nothing has been applied to the caption master yet** — that is the
remaining gate (spot-check the dry-run report, then `--apply`). The pass is also
wired as an opt-in preprocess stage (`caption_position_clauses`, GUI
Preprocessing tab → 캡션 편집 box, off by default) which applies inline; ops
contract in
[`docs/experimental/position_captions.md`](../../docs/experimental/position_captions.md).

## TL;DR

Two staged captioning features that give multi-subject images spatially bound
captions in the dataset's existing hand-written convention
(`On the left, <tags>. On the middle, <tags>. …`):

1. **Natural caption enhance** (v1, additive): for captions carrying
   `multiple views` or a girls-count > 1, detect `girl` instances with SAM3,
   order them left→right, tag each mask-blanked crop with the Anima Tagger,
   and **append** positional clauses to the caption. The flat tag bag is
   untouched.
2. **Smart caption rewrite** (v2): additionally **move** each tag that is
   attributable to exactly one instance out of the flat bag into its positional
   clause, so the duplication between bag and clause is resolved and the flat bag
   stops asserting unbound attributes.

**v2 replaced v1** rather than shipping beside it. The decision rests on a
measurement neither feature's design anticipated: across the 14 hand-written
ground-truth captions, 244 clause tags, only 19 also appear in the flat bag, and
**all 19 are character names** — no attribute is ever duplicated. The
hand-written convention *is* the rewrite (cast list flat, attributes bound), so
v1's additive form was the deviation, not the safe default. v1 survives as
`--no_rewrite` for the A/B arm.

## Why

The flat tag bag cannot express which attribute belongs to which subject:

- `channel_(caststation)/channel6` — `3girls, blonde hair, aqua hair, red
  hair, …`: who is blonde? (This image already has hand-written clauses —
  they are the convention v1 automates.)
- `channel_(caststation)/8090164` — `1girl, multiple views, maid, playboy
  bunny, swimsuit, pink jacket, …`: four outfit views of one character, all
  outfits unbound. Same ambiguity, same fix, and the gate must therefore be
  **detected instance count**, not the girls-count tag.

Populations at time of writing: **258** multi-girl captions (12 with
hand-written clauses = ground truth), **~350** `multiple views` captions.

## Phase 0 evidence (both probes green)

**Probe A — the base model already obeys positional clauses.**
`probe_binding.py` rendered 24 no-LoRA images from counterbalanced prompts
(`…, {a} hair, {b} hair. On the left, {a} hair. On the right, {b} hair.`,
4 color pairs × both orders × 3 seeds), split each render into halves, and
asked the tagger which color won per side: **48/48 sides correct (chance
50%)**. Spot-checked renders confirm honest wins (two girls, correct color on
the prompted side, either order). Positional captions therefore *reinforce an
existing base-model capability* — there is no teach-from-scratch risk, and
LoRA training on such captions inherits a working conditioning channel.

**Probe B — the detection→crop→tag pipeline works on first contact.**
`probe_autocaption.py` on the 12 ground-truth images + the 8090164 showcase:

- Instance count vs hand-written clause count: **10/12**. Misses: `chicke2`
  (a two-view sheet whose left view is an extreme close-up SAM scores < 0.5
  for `girl`) and `4615461` (2 of 3 found).
- Hair-color-to-position: **8/10**. Both misses are *neighbor hair intruding
  into the bbox crop* (plus genuinely borderline colors: near-black vs brown,
  pale pink vs white).
- Character-name-to-position: 4/7 — names are the weakest signal on crops.
- The 8090164 proposal came out exactly right: maid → bunny leotard → pink
  jacket → one-piece swimsuit, left to right, from a `1girl` image.

The two systematic errors have known mechanical fixes, wired into v1 below:
mask-blanking (SAM already returns per-instance masks; blank non-instance
pixels before tagging) kills crop contamination, and a lower score threshold
with a count-consistency check recovers weak detections.

## Feature 1 — natural caption enhance (v1)

Pipeline per image (candidates prefiltered by caption: `multiple views` OR
girls-count tag > 1; skip any caption that already contains `On the `):

1. **Detect**: SAM3 `girl` on the resized image → per-instance boxes + masks
   + scores (`Sam3Processor.set_text_prompt`, same API as
   `scripts/preprocess/generate_masks.py`). Score filter (default 0.5, retry
   pass at 0.3 when detected count < caption girls-count), greedy IoU dedupe
   (0.65).
2. **Order**: left→right by box center-x. Position vocabulary: N=2 →
   `left/right`; N=3 → `left/middle/right`; N≥4 → `leftmost / second from
   left / … / rightmost`. **Row-aware extension for grid sheets**: cluster
   center-y first; if >1 row, prefix `top`/`bottom` (`on the top left, …`) —
   multiple-views sheets are often 2×2, and pure x-ordering interleaves rows.
3. **Crop + blank**: padded bbox crop (6%), non-instance pixels blanked to
   white using the instance mask (the probe-B contamination fix).
4. **Tag**: Anima Tagger per crop. Clause tags, in emission order: character
   name (must clear `--name_confidence` **and** appear in the flat bag, unless
   `--allow_unlisted_names` — probe B scored names 4/7, so an unlisted name is
   most likely a crop artifact), then the exclusive-group winners
   (`hair_color`, `eye_color`, `hair_length`, `hairstyle` — these are
   `softmax_when_solo`, and a single-subject crop is exactly the condition
   under which they fire), then everything else ranked, preferring tags the
   caption already curated. Cap 8 tags/clause (`--max_clause_tags`).

   Which tags are *eligible* comes from the tagger's own `groups.yaml`
   (`SUBJECT_GROUPS`), not substring heuristics: a tag in a per-subject group
   (hair / clothing / expression / pose / …) binds; a tag in a scene group
   (lighting, background, framing, medium, `interaction`,
   `character_relationship`) never does. Copyright / artist / metadata /
   count / rating tags are excluded outright. Ungrouped tags — where the
   curated compounds like `pink jacket` live — are admitted only when they are
   both **in the flat bag** and **attributable** (kept on exactly one crop).
   At most one member of an exclusive group per clause, so a crop that keeps
   two hair colors can't emit `green hair, …, aqua hair` for one subject.

   **A clause carries only what tells its subject apart.** Any tag *every*
   crop keeps is suppressed (`--keep_shared_tags` disables). On a `1girl,
   multiple views` outfit sheet all four views are the same character with the
   same hair and eyes; repeating `hatsune miku, aqua hair, twintails` four
   times binds nothing and crowds out the maid / bunny / bikini that actually
   distinguishes them. Those shared attributes are already in the flat bag —
   v1 is additive and never removes anything — so nothing is lost. When
   suppression empties every clause the image is skipped
   (`skip:no-discriminative-tags`): the subjects are genuinely
   indistinguishable to the tagger and there is nothing to ground.
5. **Append**: `<flat caption>. On the left, <tags>. On the right, <tags>.`
   Written to the caption master (`image_dataset/*.txt`); `make
   preprocess-captions` mirrors it into `post_image_dataset/resized/*.txt` the
   same way existing caption tooling does.

Gate on **detected** count ≥ 2. If detection and the girls-count tag disagree
after the retry pass (probe-B: 2/13), *skip and log* — never write clauses we
can't ground (`--no_strict_count` overrides). A `multiple views` caption
deliberately has **no** expected count: the girls-count tags how many
characters are drawn, while each view is its own bindable subject, so gating
`1girl, multiple views` on the count tag would skip every outfit sheet.

**Ops sequencing (both are silent-failure traps):** caption edits do NOT
invalidate TE caches — an `--apply` run must be followed by an explicit `make
preprocess-te` (the script prints this); and `*.variants.txt` tag-dropout
sidecars OVERRIDE the CLI rate, so variants must be regenerated after the
rewrite. `preprocess-te` chains `preprocess-captions` and does both.

**Clause-shredding fix (shipped alongside v1).** The convention delimits
clauses with `.` and tags with `,`, so the caption's *comma* split glues the
header onto the previous tag (`"white socks. On the left"`). Every consumer
keying off `tag.startswith("On the ")` — `anima_smart_shuffle_caption`'s
section logic, the identity-randomize guard — therefore saw **no sections at
all**, and the shuffled variants scattered clause attributes across the whole
caption and reassigned them to the wrong subject. Verified on the 12
hand-written ground-truth captions before the fix. `library.captioning.
position_clauses` is now the single clause grammar; `generate_caption_variants`
parses through it and treats **each clause as an atomic unit** (kept or dropped
whole at `clause_dropout_rate`, defaulting to `tag_dropout_rate`, shuffled
inside), and `correct_caption` splits clauses off before bucket-reordering the
flat bag. Without this, v1's clauses would have been shredded in 3 of the 4
default caption variants.

**Code layout** (per the layering contract): clause grammar in
`library/captioning/position_clauses.py` (torch-free, shared by the variant
generator / order-correction / pipeline), orchestration in
`library/preprocess/position_captions.py` (models injected as `detect_fn` /
`tag_fn`, so it imports neither SAM3 nor the tagger and unit-tests with stubs),
thin argparse shell `scripts/preprocess/position_captions.py`, `make
caption-position` target, daemon-routed (GPU: SAM3 + tagger). Dry run is the
default: `report.json` (proposed clause + per-instance boxes/scores/tags per
image, plus a token-budget column with `--qwen3`) and, with `--crops`, the
exact mask-blanked pixels the tagger saw; `--apply` writes.

## Feature 2 — smart caption rewrite (v2, shipped as the default)

v1 leaves each bound attribute in two places (flat bag + clause). v2 resolves
the duplication by **moving** tags into clauses. A flat tag moves when all four
of these hold; failing any one leaves it flat *and* bound, i.e. v1's behaviour
for that one tag — so the failure mode is a less-resolved caption, never a wrong
one:

1. **Not a character name.** The cast list stays flat and is also bound. This is
   the one place the shipped rule contradicts this proposal's original sketch
   (`3girls, <scene tags>. On the left, akita neru, …`): the hand-written
   ground truth keeps names in both places, unanimously, and it is also what a
   prompt needs to summon the characters at inference.
2. **Exactly one clause claims it.** Shared attributes belong to the bag.
3. **Corroboration for character-invariant groups** (hair, eyes, body shape,
   species, age …). These are properties of a *character*, not of a view, so on
   a `1girl, multiple views` sheet they hold in every panel and moving one into
   a single view would deny it of the others. Such a tag moves only when the bag
   names **≥2 values of that group** — the caption is then already enumerating
   per-subject values. Exception: a two-tone marker (`multicolored hair`,
   `heterochromia`, …) explains two values with one character, and pins the
   group. Outfit / pose / expression carry no invariance and move freely.
   Evidence-based rather than count-based on purpose: 219 of the 373 first-sweep
   proposals carry no girls-count tag at all, so a `detected == characters` gate
   would have pinned nearly the whole corpus.
4. **Attribution margin** (`--attribution_margin`, default 0.35): the winning
   crop must clear every other crop's probability for the tag, read off the
   tagger's full `scores` (not the thresholded `kept`) precisely so a runner-up
   that fell just under the keep threshold still blocks the move.

Scene/meta tags, count/rating/artist/copyright never enter a clause in the first
place and so never move. Result for channel6-style images: `3girls, akita neru,
hatsune miku, kasane teto, <scene tags>. On the left, akita neru, blonde hair,
…` — every attribute asserted exactly once, bound.

**The open risk is bag-removal tolerance**, unchanged from the original
framing: probe A validated clause *comprehension*, not that removing tags from
the flat bag is safe for a model pretrained on flat bags. The four rules bound
*which* tags move, not whether the model likes the resulting distribution. That
is what the Phase-2 A/B answers — with `--no_rewrite` (v1) as one control arm
and `--flatten` (clauses merged back, no clauses at all) as the other.

**The rewrite is reversible.** A moved tag is still in the caption, inside a
clause; `flatten_caption` / `make caption-position ARGS="--flatten --apply"`
merges every clause back into the bag. Tag order is not restored byte-for-byte
(the corrector re-buckets it anyway), and hand-written clauses are flattened
too, which is a real loss on those 14 captions.

## Phases

- **Phase 0 — probes: DONE** (numbers above).
- **Phase 1 — implementation + dry-run sweep: IMPLEMENTED (v1 then v2), sweep
  run, skips triaged, spot-check pending.** Pipeline built + tested (65 unit
  tests);
  the clause-shredding fix landed alongside. The first sweep's 102 skips were
  investigated 2026-08-17 and four mechanical causes fixed (dead retry, retry
  never attempted for multi-view, girls-only count vs. a prompt that catches
  males, `6+girls` read as exactly six) — 373 proposals now, up from 317, net
  +59/−3. Full mechanism table, plus the two refuted mask-quality gates and the
  measured-harmful containment rule, in
  [`docs/experimental/position_captions.md`](../../docs/experimental/position_captions.md#triaging-the-skips-2026-08-17).

  Three fixes landed 2026-08-17 on top of that sweep, all found by reading the
  report against the exported crops (`ama_mitsuki`, whose sheets are a full body
  next to headless close-up panels). Whole-corpus net: proposals 373 → **394**
  (404 with the opt-in part prompts), identity contradictions 520 → **0**:

  - **Comic pages are a layout, not a subject count.** `multiple views` was
    special-cased from the start; comic panels are the same thing and were not.
    A `1girl, 2koma` page draws the same girl once per panel, so 22 of the 26
    comic-layout images failed the prefilter as `single-subject`. `page number`
    is excluded — it marks a scanned art-book page, not a layout. The waiver
    also removed the count backstop, letting one girl detected twice through
    (`kase_daiki/11645055`, an overlapping pair at IoMin 0.99); `Nkoma` names
    the panel count, so `caption_panel_ceiling` bounds detections at
    `panels × (girls + boys)`. That ceiling changed exactly one corpus row.

  - **Identity gate** (default on). `select` emitted the hair/eye/hair-length
    winner with no check against the caption, and the discriminative rule then
    promoted whichever value was *wrong* — a value every crop agrees on is
    suppressed, so an outlier is what survives. 520 of 1600 identity clause tags
    (33%) contradicted the caption; 78% of single-character `multiple views`
    sheets bound conflicting hair or eye colors to views of the same girl. Now
    **0** and **16%**, at zero cost in proposals (total clause tags rose, 7173 →
    7255 — a blocked slot refills from the ranked tail). `--ungated_identity`
    reverts. It is upstream of v2 and load-bearing for it: the rewrite removes
    only what a clause carries, so gating emission to values the caption already
    named is what stops a hallucinated hair color from *replacing* the real one.
  - **Body-part detection fallback** (`--part_prompts`, opt-in). Recovers
    headless panels the `girl` prompt cannot see at any threshold:
    `too-few-instances` 27 → 18, proposals 373 → 382. Required three part-typed
    rules (containment on, no mask-blanking, no identity groups) plus a top-up
    cap; the win is real but modest, and 1 of 3 spot-checked recoveries is still
    poor. Details and the failure analysis in the experimental doc.

  **v2 landed on top of that and became the default** (Feature 2 above). A
  paired sweep — same code, same 3008 images, `--no_rewrite` the only difference
  — measured what the rewrite changes:

  | | v1 (`--no_rewrite`) | v2 (default) |
  |---|---|---|
  | proposals / clause text | 394 | 394, **byte-identical** |
  | captions whose bag changed | — | 380 |
  | flat-bag tags across the proposals | 18334 | 16429 (−1905, −10.4%) |
  | caption tokens, median / max | 213 / 524 | 198 / 491 |
  | captions over the 512-token budget | 1 | **0** |

  The identical clauses are the design assertion holding: the rewrite decides
  what the bag keeps, never what a clause says. 1905 of the 3554 bag tags a
  clause claimed actually moved (54%); the rest were pinned by `margin` (730),
  `multi-clause` (431), `sole-value` (404), `character-name` (54) and
  `two-tone-marker` (30), and each of those degrades to v1's duplicate-assert
  rather than to a wrong caption. As a side effect the one caption v1 pushed
  past the TE token cap — a silent truncation — now fits.

  Remaining exit criterion: spot-check
  ~30 proposed clauses from `post_image_dataset/captions/position/report.json`
  (weighted toward grids, overlapping pairs, N≥4) against the exported crops —
  clause proposals right at ≥90%, count-disagreement skip rate reported, and
  (v2) no reviewed image losing an attribute that belonged to a subject it was
  taken from. **No `--apply` yet.**

  Run it:

  ```
  make caption-position ARGS="--crops --qwen3 models/text_encoders/qwen_3_06b_base.safetensors"
  make caption-position ARGS="--apply"   # after the spot-check
  make preprocess-te                     # REQUIRED after --apply
  ```
- **Phase 2 — apply + train A/B**: `--apply`, regenerate variants + TE
  caches, train the standard LoRA recipe on a multi-girl-dense slice, twin
  control on unmodified captions (same seed — mind the paired-ΔW chaos
  floor: compare renders, not raw ΔW cosines). Eval: probe-A-style render
  test through the trained LoRA (does binding hold under the artist style?)
  + multi-character sample grids. Exit: binding ≥ control, no regression on
  single-girl renders. Three corpora are now available for the same protocol
  and cost one preprocess each: v2 (default), v1 (`--no_rewrite`), and
  clause-free (`--flatten`) — so "do clauses help" and "does bag-removal hurt"
  are separable in one experiment rather than sequential phases.

## Open questions — resolved in the build

1. **TE token budget** — *instrumented*, and v2 shrinks it. `--qwen3 <tokenizer>` adds a token
   count per proposal; the report's `summary.max_tokens` /
   `over_token_budget` flag anything past 512 (the `qwen3_max_token_length` /
   `t5_max_token_length` default), which the padding invariant would truncate
   silently. Read it off the dry-run report before applying.
2. **Clause dropout semantics** — *whole clause, as recommended*. A clause is
   kept or dropped as a unit at `clause_dropout_rate` (defaults to
   `tag_dropout_rate`), tags shuffled inside, header never randomized. Per-tag
   dropout inside a clause would leave a half-described position. **v2 changes
   what dropping one costs**: the attributes are no longer duplicated in the
   bag, so a dropped clause drops that subject's attributes from the variant —
   correlated tag-dropout, still a truthful caption, but a stronger perturbation
   than the same rate per-tag. `clause_dropout_rate = 0.0` is the conservative
   setting on a rewritten corpus.
3. **Character names in clauses** — *both floors, conjunctively*, **and the name
   stays in the flat bag**. A name needs `--name_confidence` (0.5) **and**
   membership in the flat bag; one identity per subject.
   `--allow_unlisted_names` relaxes the second. The rewrite never takes a name
   out of the bag: the hand-written ground truth keeps the cast list flat *and*
   binds it (19/19 of its duplicated clause tags are names), and that flat list
   is what a prompt uses to summon the characters. In practice the
   discriminative rule also drops the name on single-character `multiple
   views` sheets, where it is the same in every clause anyway.
4. **Boys/POV** — *left out of the default sweep, but not hardcoded*: `--prompt` defaults to
   `girl`, matching the hand-written convention. Pass `--prompt person` to
   sweep the rare on-screen-boy images separately.
5. **Vertical layouts** — *covered*. Rows are clustered before ordering, so a
   grid gets `top left / top right / bottom left / bottom right` and a
   single-subject row gets the bare row word (`On the top, …`). Up to 3 rows
   (`top/middle/bottom`); beyond that it degrades to plain left→right.

## Remaining open questions (Phase 2+)

1. **Does hair *length* survive cropping?** `long hair` vs `medium hair` on
   two views of the same character is crop-scale dependent, and `hair_length`
   is in the priority group list. The discriminative rule masks most of it
   (shared → suppressed), but a scale artifact that differs between views will
   bind. Worth a look in the spot-check.
2. **Bag-removal tolerance** — the open risk of v2, and the reason the Phase-2
   A/B carries three arms. Probe A validated clause *comprehension*, not that a
   model pretrained on flat bags tolerates tags leaving them. The four move
   rules bound *which* tags go, not whether the resulting distribution trains
   well.
3. **Is the margin at the right place?** 730 tags — the largest pinned class —
   stay flat because the runner-up crop scored within 0.35. That is deliberately
   conservative; the dry-run report carries the per-move margin, so the knob can
   be retuned against the spot-check rather than guessed at.
4. **`sole-value` on non-identity invariants.** `body_shape` / `skin` /
   `face_features` are in the invariant set, so a `2girls` caption naming one
   `large breasts` keeps it flat even when only one girl has it. Safe, but it is
   the class most likely to be over-pinned — worth counting in the spot-check
   before loosening.
