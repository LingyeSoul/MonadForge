# Position-aware captions — natural caption enhance → smart caption rewrite

Status: **Phase 0 DONE (both probes green) — Phase 1 (v1) IMPLEMENTED and
shipped as `make caption-position`.** Probes + result envelopes live in
`bench/position_captions/` (runs `20260817-1122-autocaption`,
`20260817-1123-binding`). v1 code: `library/captioning/position_clauses.py`
(clause grammar), `library/preprocess/position_captions.py` (pipeline),
`scripts/preprocess/position_captions.py` (CLI), tests in
`tests/test_position_captions.py`. Dry-run is the default; **nothing has been
applied to the caption master yet** — that is the remaining Phase 1 gate
(spot-check the dry-run report, then `--apply`).

## TL;DR

Two staged captioning features that give multi-subject images spatially bound
captions in the dataset's existing hand-written convention
(`On the left, <tags>. On the middle, <tags>. …`):

1. **Natural caption enhance** (v1, additive): for captions carrying
   `multiple views` or a girls-count > 1, detect `girl` instances with SAM3,
   order them left→right, tag each mask-blanked crop with the Anima Tagger,
   and **append** positional clauses to the caption. The flat tag bag is
   untouched.
2. **Smart caption rewrite** (v2, advanced): additionally **move** each tag
   that is attributable to exactly one instance out of the flat bag into its
   positional clause, so the duplication between bag and clause is resolved
   and the flat bag stops asserting unbound attributes.

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

## Feature 2 — smart caption rewrite (v2)

v1 leaves each bound attribute in two places (flat bag + clause). v2 resolves
the duplication by **moving** tags into clauses:

- A flat tag moves into a clause when the tagger keeps it on **exactly one**
  crop (with margin) — i.e. it is attributable. `blonde hair` kept only on
  the left crop → leaves the bag, lives in `On the left, …`.
- Tags kept on multiple crops (shared uniform, `smile`), scene/meta tags,
  count/rating/artist/copyright, and anything unattributable stay flat.
- Result for channel6-style images: `3girls, <scene tags>. On the left,
  akita neru, blonde hair, …` — each attribute asserted exactly once, bound.

This is the riskier half: it changes the flat-bag token distribution the base
model was pretrained on (probe A validated clause *comprehension*, not
bag-*removal* tolerance). Ship it behind a flag, phase-gated on a training
A/B (Phase 3), never as a default until the A/B clears.

## Phases

- **Phase 0 — probes: DONE** (numbers above).
- **Phase 1 — v1 implementation + dry-run sweep: IMPLEMENTED, sweep run,
  spot-check pending.** Pipeline built + tested (40 unit tests); the
  clause-shredding fix landed alongside. Remaining exit criterion: spot-check
  ~30 proposed clauses from `post_image_dataset/captions/position/report.json`
  (weighted toward grids, overlapping pairs, N≥4) against the exported crops —
  clause proposals right at ≥90%, count-disagreement skip rate reported. **No
  `--apply` yet.**

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
  single-girl renders.
- **Phase 3 — v2 behind `--rewrite`**: same A/B protocol, v2 vs v1 corpus.

## Open questions — resolved in v1

1. **TE token budget** — *instrumented*. `--qwen3 <tokenizer>` adds a token
   count per proposal; the report's `summary.max_tokens` /
   `over_token_budget` flag anything past 512 (the `qwen3_max_token_length` /
   `t5_max_token_length` default), which the padding invariant would truncate
   silently. Read it off the dry-run report before applying.
2. **Clause dropout semantics** — *whole clause, as recommended*. A clause is
   kept or dropped as a unit at `clause_dropout_rate` (defaults to
   `tag_dropout_rate`), tags shuffled inside, header never randomized. Per-tag
   dropout inside a clause would leave a half-described position.
3. **Character names in clauses** — *both floors, conjunctively*. A name needs
   `--name_confidence` (0.5) **and** membership in the flat bag; one identity
   per subject. `--allow_unlisted_names` relaxes the second. In practice the
   discriminative rule also drops the name on single-character `multiple
   views` sheets, where it is the same in every clause anyway.
4. **Boys/POV** — *left out of v1, but not hardcoded*: `--prompt` defaults to
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
2. **v2 (`--rewrite`) bag-removal tolerance** — unchanged from below: probe A
   validated clause *comprehension*, not bag-*removal*, so it stays
   phase-gated on the Phase 3 A/B.
