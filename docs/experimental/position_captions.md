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

3008 images, whole dataset, defaults:

| | n |
|---|---|
| candidates (prefilter passed) | 419 |
| **proposals** | **317** |
| skip: too-few-instances | 81 |
| skip: count-mismatch | 19 |
| skip: already-has-clauses (hand-written) | 15 |
| skip: too-many-instances | 2 |
| not a candidate (single-subject) | 2574 |

Count-mismatch skip rate is 4.5% of candidates (probe B saw 15% on its
13-image sample). Of the 317 proposals: 229 are 2-subject, 54 are 3, 34 are ≥4;
137 come from `multiple views` sheets; median 8 tags per clause (i.e. the cap
binds).

## Knobs

`ARGS="…"` on the make target; every flag has a `--kebab-case` alias.

| Flag | Default | What it does |
|---|---|---|
| `--apply` | off | Write to the caption master (else dry run) |
| `--path_pattern` | `*` | fnmatch glob (`\|` to OR) relative to the resized dir |
| `--crops` | off | Export the mask-blanked crops next to the report |
| `--prompt` | `girl` | SAM3 subject prompt (`person` sweeps the rare on-screen-boy images) |
| `--score_threshold` / `--retry_score_threshold` | 0.5 / 0.3 | Detection floor; retry floor when the count undershoots |
| `--iou_threshold` / `--pad` | 0.65 / 0.06 | Dedupe IoU; bbox padding fraction |
| `--no_blank_crops` | — | Skip mask-blanking (this is what caused probe B's hair-color misses — diagnostic only) |
| `--row_tol` | 0.25 | Row-clustering gap as a fraction of image height |
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
