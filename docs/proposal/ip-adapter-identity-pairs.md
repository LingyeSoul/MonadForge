# IP-Adapter revival — distinct-pair (identity) training

Status: **proposal** (2026-05-22). Builds directly on `docs/proposal/ip-adapter-0502.md`
and `docs/experimental/ip-adapter.md`. Does not change the architecture; changes the
**training contract** (which reference image the IP path sees).

## TL;DR

IP-Adapter today is **self-paired**: reference image == VAE target == the same
image. The model can satisfy the loss by copying pixel detail from the reference
straight into the output, which is a shortcut that doesn't require learning
*identity*. The 0502 debugging trail diagnosed the residual wall as "narrow
signal on a collapsed manifold" (participation ratio 6.2) and attacked it on the
**representation** side — centroid subtraction, PE-LoRA. Those help the encoder
separate images; they do nothing about the copy shortcut in the **objective**.

This proposal supplies the other half, which the 0502 notes already pointed at
("most likely *paired-but-different data is needed*, not more LR"): train with a
reference that is a **different image of the same identity** as the target. The
only signal the IP path can then carry that consistently lowers loss is what is
*invariant across the pair* — identity — because the pixel-level specifics
(pose, crop, background, exact lighting) differ between reference and target and
are useless to copy.

The data supports it without any new preprocessing: PE features are already
cached per image, so "pairing" just means **loading a different stem's
`_anima_pe.safetensors` into `batch["ip_features"]`**, decoupled from the VAE
target.

## Why self-pairing is the residual bottleneck

The cross-attention IP path adds `scale * SDPA(text_q, ip_k, ip_v)` to the text
result, where `ip_k/ip_v` come from PE features of the reference. Under
self-pairing the reference *is* the target, so the IP path's lowest-loss
behavior is to encode and re-emit the target's own appearance — including detail
that the caption doesn't mention. This is a genuine loss reduction (hence Run #1's
`epoch_baseline_no_ip_delta` going positive, +0.00015 → +0.00045), but it is
**copy, not abstraction**. At inference the user hands a *new* reference and
expects the model to transfer identity/style onto a *different* prompt — a
generalization the self-paired objective never asked for.

The 0502 feature analysis is consistent with this: crop recall@1 = 0.22 ruled
out *pixel* memorization, but "narrow signal on a collapsed manifold" is exactly
what a copy-shortcut objective produces — the per-image delta the resampler is
asked to extract is small *because the task doesn't force it to find the
identity axis*. Distinct pairs change the task so the identity axis is the only
thing worth extracting.

This is orthogonal to, and composes with, the representation-side fixes already
in the tree:

| Lever | Attacks | Already shipped? |
|---|---|---|
| Centroid subtraction | input representation (zero-mean the collapsed manifold) | yes |
| PE-LoRA | encoder adaptation to anime distribution | yes (`pe_lora_enabled`) |
| **Distinct pairs (this)** | **the objective (removes the copy shortcut)** | **no** |

## Data: the pairing is feasible (measured)

Walked all 2,600 captions in `image_dataset/`, classified tags via the Anima
Tagger vocab (`models/captioners/anima-tagger-v2/vocab.json`):

| Grouping level | Coverage | Groups w/ ≥2 imgs | Same-group positive pairs (Σ nC2) |
|---|---|---|---|
| **Character** (tightest) | 819 imgs / **32%** | 113 | 6,501 |
| **Franchise / copyright** | 2,328 imgs / **90%** | 91 | 332,047 |
| **Artist** (loosest) | 2,600 imgs / **100%** | 71 | 108,799 |

Top cross-artist characters: `frieren` (8 artists / 23 imgs), `hatsune miku`
(8 / 67), `fern (sousou no frieren)` (7 / 28), `gotoh hitori` (7 / 12),
`kisaki (blue archive)` (7 / 13) — 80 characters appear across ≥2 artists.

The three levels nest into a **tiered positive sampler**: prefer same-character;
if the target's character is a singleton or untagged, back off to same-franchise;
then same-artist; then self. Every image therefore has *some* positive at a
known tightness, and the tightness level is logged per sample so we can ablate
its effect.

The "1girl" axis the original brainstorm mentioned is a dead lever here — 91% of
the set is `1girl`, so it carries no contrast. Identity (character → franchise →
artist) is the axis with real structure.

## Design

### Caption index (Phase 0, no training) — promoted to a shared artifact

The per-image tag parse is **not** IP-Adapter-specific, so it is built as a
general dataset artifact and the IP-Adapter pairing only *consumes* it. Two
pieces, cleanly split:

1. **`preprocess/build_caption_index.py`** (`make caption-index`): walks caption
   sidecars, classifies each comma-separated tag into character / copyright /
   artist / count via the tagger vocab (artist additionally by the `@` prefix,
   which is exact and not bounded by the vocab cutoff), and writes
   **`post_image_dataset/captions/caption_index.json`**:

   ```json
   {
     "meta":  {"vocab_path": "...", "vocab_mtime": "...", "n_images": 2600, ...},
     "image_meta": {"<stem>": {"path": "...", "character": [...],
                                "copyright": [...], "artist": [...], "count": [...]}},
     "groups": {
       "character": {"frieren": ["<stem>", ...], ...},
       "copyright": {"sousou no frieren": [...], ...},
       "artist":    {"@sincos": [...], ...}
     }
   }
   ```

   Cheap (~1 s), method-agnostic, lives beside the VAE/PE caches under
   `post_image_dataset/` (not the checkpoint), regenerated when the dataset or
   vocab changes. **It encodes no sampling policy** — `level_priority`,
   `ip_pair_prob`, cross-artist are method config, not disk state. Reusable by
   artist balancing (Room C below), dataset analytics, any future
   identity-aware method.

2. **`bench/ip_adapter/pair_audit.py`** owns the IP-Adapter *policy* (the
   tiered character → copyright → artist backoff), reads the shared index, and
   emits `bench/ip_adapter/pair_audit.md` — the coverage table, tiered
   reachability, cross-artist characters, and group-imbalance numbers.

### Dataset hook (Phase 1)

The reference contributes **only `ip_features`** — not VAE latent, not caption.
So the change is local to how `batch["ip_features"]` is populated; the target
path (VAE latent + caption + bucketing) is untouched.

In `library/datasets/base.py::__getitem__`, where it currently calls
`_try_load_ip_features(image_info.absolute_path)`, route through a new
`_resolve_reference_stem(target_stem)` when `ip_pair_mode != "self"`:

1. With probability `1 - ip_pair_prob`, return the target itself (self-pair —
   keeps the base behavior in the mix; reference recipes warm up better with
   some self-pairs early).
2. Otherwise walk `level_priority`: pick a *different* stem sharing the target's
   character; if none, franchise; if none, artist; if none, self. Optionally
   require a *different artist* (`ip_pair_cross_artist=true`) to force pure
   identity transfer (style must be dropped) vs. identity+style preservation.
3. Load that stem's cached `_anima_pe.safetensors` (the existing
   `_try_load_ip_features` path already resolves nested cache dirs by stem — so
   this is a stem swap, nothing more).

New `network_args` / dataset knobs:

| Knob | Default | Meaning |
|---|---|---|
| `ip_pair_mode` | `self` | `self` \| `identity` \| `identity_cross_artist` |
| `ip_pair_prob` | `0.8` | fraction of steps that draw a *distinct* reference |
| `ip_pair_index` | `post_image_dataset/captions/caption_index.json` | shared caption index (see Phase 0) |
| `ip_pair_min_level` | `artist` | loosest level allowed before falling back to self |

`ip_pair_mode` defaults to `self` so existing runs are bit-identical until opted in.

### Caption-leakage guard

`keep_tokens=3` preserves the character token in both captions, so the model
could still identify via *text* and leave the IP path idle. On distinct-pair
steps, drop the character/copyright tokens from the **target** caption with
probability `ip_pair_caption_strip_p` (default 0.5). This forces the identity
information to flow through the IP image path, not the text path — the whole
point of the method. (Independent of the existing `caption_dropout_rate`.)

### Validation (the success signal)

Wire in **backstop D from 0502** plus a matched-distinct baseline, in
`IPAdapterMethodAdapter.validation_baselines()`:

- `no_ip` — existing.
- `matched_distinct` — reference is a held-out *different* image of the target's
  identity (the deployment condition).
- `shuffled_ref` — reference is an unrelated image.

Healthy result: `matched_distinct` beats both `no_ip` and `shuffled_ref`. That is
the thing self-pairing could never show, and the kill criterion if it fails.

Note (from memory `project_fm_val_loss_uninformative`): FM-MSE val deltas don't
track perceptual quality on Anima. Treat the validation Δ as a *necessary,
not sufficient* gate, and confirm with the existing `make exp-test-ip` debug
ladder (reproduction → crank → style-transfer) plus a small paired-CMMD on
held-out identities.

## Phasing

- **Phase 0 — index + audit. ✅ done (2026-05-22).** Built the shared
  `caption_index.json` (`make caption-index`) and `bench/ip_adapter/pair_audit.py`
  → `pair_audit.md` on the *live* 2,600-image dataset. Coverage reproduces the
  measured table (character 819/32% · copyright 2328/90% · artist 2600/100%;
  113/91/73 groups≥2). **100% of images reach a distinct positive at some tier**
  (31% character · 58% copyright · 11% artist · 0 self-only), 82 cross-artist
  characters, 644/812 character-tier images have a different-artist partner.
  Gate cleared. Pure data, no GPU.
- **Phase 1 — pairing + self-fallback, small run. ✅ implemented (2026-05-22;
  no-PE-LoRA fast path).** Mechanism landed:
  - `library/datasets/identity_pairs.py::IdentityPairSampler` — tiered
    character → copyright → artist back-off + cross-artist + shuffled negative,
    consuming the shared `caption_index.json` (unit-tested in
    `tests/test_identity_pairs.py`).
  - `library/datasets/base.py` — `setup_identity_pairs()` +
    `_load_ip_features_for_stem()` resolve a *distinct* reference's nested PE
    cache; `__getitem__` swaps `ip_features` (B=1, so distinct-aspect refs are
    fine) and, for validation, also emits `ip_features_shuffled`. Caption-strip
    guard implemented but **inert while TE outputs are cached** (warns once);
    enabling it (Phase 2) needs `cache_text_encoder_outputs=false`.
  - `train.py::assert_extra_args` wires the knobs onto train + val datasets;
    new args in `library/anima/training.py` (`--ip_pair_mode` /
    `--ip_pair_prob` / `--ip_pair_min_level` / `--ip_pair_caption_strip_p`).
    `ip_pair_index` is an internal default (`post_image_dataset/captions/
    caption_index.json`), not a user-facing knob.
  - `networks/methods/ip_adapter.py` — `matched_distinct` is the validation
    primary; `validation_baselines()` returns `no_ip` + `shuffled_ref` (the
    latter sourcing `ip_features_shuffled` via `_ref_override`).
  - `library/training/validation.py::_run_validation_baselines` — **re-wires
    the in-loop FM-MSE baseline-delta pass** (the old machinery was orphaned
    when CMMD replaced the FM-val loop); runs independently of CMMD with
    per-item RNG re-seeding so primary + baselines share noise. Logs
    `loss/validation/baseline_{no_ip,shuffled_ref}[_delta]`.
  - `configs/methods/ip_adapter.toml` flipped to the recipe:
    `pe_lora_enabled=false`, `ip_features_cache_to_disk=true`,
    `cache_latents=true`, `ip_pair_mode="identity"`.

  **NOT using PE-LoRA** (deliberate — the live PE encoder forward/backward is
  the slow part; distinct-pairing attacks the *objective* and runs entirely off
  cached features). PE-LoRA stays a composable lever for later.

  Gate: `matched_distinct` Δ > `shuffled_ref` Δ and both signs sane, compared
  against the self-paired Run #1 baseline. NB the FM-MSE delta is
  necessary-not-sufficient on Anima (`project_fm_val_loss_uninformative`) —
  confirm wins with CMMD + the `exp-test-ip` ladder. Run: `make exp-ip-adapter`
  (defaults to `identity`; set `ip_pair_mode=self` to A/B against self-pairing).
- **Phase 2 — full budget + ablation.** Full data, 30 epochs. Ablate
  `self` vs `identity` vs `identity_cross_artist` and `ip_pair_caption_strip_p`
  ∈ {0, 0.5}. Visual eval on held-out identities (face/outfit consistency on a
  new pose). Pick the mode; update `configs/methods/ip_adapter.toml` + docs.

## Risks & open questions

- **Group imbalance.** `original` / big franchises dominate. Cap per-group draw
  frequency in the sampler (inverse-sqrt of group size) so a few large franchises
  don't swamp the tail. (Mirrors the artist-balancing idea — Room C.)
- **Cross-artist forces style drop.** `identity_cross_artist` teaches the IP path
  to carry identity *without* the source artist's style — great for "character in
  a new style" but it removes the style-transfer use case. Hence both modes are
  knobs, not a single default; Phase 2 decides which ships (or whether to mix).
- **Caption still leaks identity.** The strip guard is probabilistic, not total.
  If `matched_distinct` ≈ `no_ip` even with strip=1.0, the bottleneck is upstream
  (resampler/encoder), and we're back to the representation levers — but now we'd
  *know* that, which the self-paired setup can't tell us.
- **B=1 is unaffected.** The pair lives inside one sample (target latent + a
  distinct reference's PE features), so this needs no batching change and doesn't
  reintroduce the in-batch-negative problem that killed the contrastive objective.
- **Thin character coverage (32%).** Mitigated by the franchise/artist fallback
  tiers (90% / 100%), but the *tightest* signal is character-only. If Phase 1
  shows the win concentrated on character-tier samples, it's worth tagging more
  of the set with the Anima Tagger to lift the 32%.

## What this does NOT do

- Does not touch the network architecture, the gate, the centroid, or PE-LoRA —
  all of those stay and compose.
- Does not revive the contrastive/InfoNCE objective (dead at B=1 — this is a
  reconstruction objective with a smarter reference, not a contrastive one).
- Does not replace EasyControl/DirectEdit or claim to — it's an IP-Adapter-only
  change to the image-prompt pathway.

## Reference points

- Prior debugging trail & diagnosis: `docs/proposal/ip-adapter-0502.md`
- Architecture: `docs/experimental/ip-adapter.md`, `networks/methods/ip_adapter.py`
- Feature analysis (PR=6.2): `bench/ip_adapter/analysis.md`
- Dataset hook: `library/datasets/base.py::__getitem__` / `_try_load_ip_features`
- Validation infra: `IPAdapterMethodAdapter.validation_baselines()`
- Shared caption index: `preprocess/build_caption_index.py` (`make caption-index`)
  → `post_image_dataset/captions/caption_index.json`
- Phase-0 audit: `bench/ip_adapter/pair_audit.py` → `bench/ip_adapter/pair_audit.md`
- Tiered grouping source: tagger vocab categories in
  `models/captioners/anima-tagger-v2/vocab.json`
