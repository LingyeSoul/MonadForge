# DirectEdit editing v2 — closing the source-prompt gap

Companion to [`directedit_editing.md`](./directedit_editing.md). v1 wired the
DirectEdit primitive (invert + edit + ΔZ residual). v2 is about the part v1
explicitly punted on: **where ψ_src comes from for an external image**, now
that we have a working primitive to test against.

## What we learned testing v1

Two empirical findings from running `make exp-test-directedit` against held-out
images and `--cached_embed` sanity passes:

1. **DirectEdit reconstruction is robust to prompt corruption.** Feeding a
   shuffled or tag-dropped ψ_src still preserves ~99 % of the source image
   pixel-wise. The ΔZ_t residual anchors the trajectory hard enough that
   even a meaningfully wrong inversion prompt reconstructs cleanly. (The
   `cached_embed_variants=all` sweep already confirms this for v0..vN
   re-encodings of the same caption — same conclusion under stronger
   corruption.)

2. **Edit leverage collapses when ψ_src is wrong, even though pixels look
   fine.** The edit pass at ψ_tar = ψ_src + edit-tag fails to apply the
   intended change when ψ_src doesn't actually describe the image's content.
   wd-tagger output, in particular, is bad enough at this that it's a
   blocker — false negatives on character-defining tags and false positives
   from manga-page artifacts (already partially mitigated by
   `DEFAULT_TAG_BLOCKLIST` in `library/captioning/wd_tagger.py`, but only
   for the worst offenders) leave ψ_src structurally far from the image's
   "true" Anima embedding.

Together these say: **inversion fidelity is a solved problem; edit fidelity
is bottlenecked entirely on ψ_src quality**. v1's caption-fallback path is
the live bottleneck, and wd-tagger is the wrong tool for it.

## Diagnosis — why wd-tagger fails as ψ_src

wd-tagger was trained on a Danbooru distribution that is *related to* but
*not the same as* Anima's training distribution. Concretely:

- **Distribution mismatch.** Anima's training captions are curated for
  this model (specific tag selection, ordering conventions, stylistic
  vocabulary). wd-tagger doesn't know our caption norms — it produces a
  tag set that is plausible-looking but distributionally off when fed
  through T5.
- **Manga-page bias.** wd-tagger fires `transparent_background`,
  `monochrome`, `greyscale`, `no_humans` on stylized character art with
  high frequency. The blocklist patches the worst cases but the
  underlying "this looks like a scan, not a render" prior is structural.
- **Threshold brittleness.** A single global `general_threshold=0.35`
  drops character-relevant low-confidence tags while keeping
  high-confidence-but-irrelevant ones. There's no per-image calibration.
- **Tag vocabulary gap.** Anima caption vocabulary includes terms that
  aren't in wd-tagger's label set at all (training-distribution-specific
  composite tags, style markers).

The net effect is that ψ_src = T5(wd_tags) lives off Anima's training-time
embedding manifold, and the edit delta `ψ_tar − ψ_src` doesn't isolate the
intended change cleanly — it also has to traverse the off-manifold offset.

## Options for v2

Five paths, ordered roughly by build cost. They are not mutually exclusive
— a v2 ship could use (A) as the default and (D) as a "high-quality, slow"
fallback.

### (A) Custom tagger trained on Anima's caption distribution

Take a small frozen vision encoder (PE-Core-L14-336 — already cached in
the IP-Adapter pipeline) and train a multi-label classifier head whose
label set is exactly Anima's training-caption vocabulary. Loss is the
standard BCE-with-logits over the label vector built from the
`image_dataset/{stem}.txt` sidecars. We have ~thousands of (image,
tag-list) pairs already on disk; this is a one-evening training run.

Pros:
- Tag vocabulary matches Anima's training distribution by construction —
  no captioner-shape bias.
- Cheap to train, cheap to run (frozen vision tower + linear head).
- Drop-in replacement for `WDTagger` in `scripts/edit.py`.
- Threshold per tag can be calibrated on a held-out split rather than
  inheriting wd-tagger's global default.

Cons:
- Still produces a *tag list* and routes it through T5. The tag → T5
  pipeline is whatever it is; if T5's interpretation of a tag string
  differs subtly from how it was trained on those same tags during DiT
  training, ψ_src is still slightly off-manifold. (This is a smaller
  effect than wd-tagger's distribution gap, but not zero.)
- Doesn't help on novel content that didn't appear in Anima's training
  set — falls back to the closest-tag distribution.

### (B) Captioner + Anima-style rewriter

Use a strong VLM (Qwen2-VL, Florence-2, JoyCaption) to produce natural
language, then a lightweight rewriter (small LM or rule-based) to
re-shape it into Anima's caption style. The rewriter could even be a
second pass of the captioner with an in-context-learning prompt that
shows ~5 Anima caption examples.

Pros:
- Vocabulary not bounded by a fixed label set.
- Standard, off-the-shelf components.

Cons:
- Two-stage failure surface — captioner errors compound with rewriter
  errors.
- VLMs hallucinate; in stylized art domains the failure modes are hard
  to characterize.
- Doesn't ship in our model bundle — adds a second large model dep
  (Qwen2-VL is ~2B params at the small end).

### (C) img2emb (per [`img2emb_plan.md`](./img2emb_plan.md))

The full proposal: image → K continuous post-T5 tokens, supervised by FM
loss through the frozen DiT, trained with implicit alignment against the
frozen vision encoder. Bypasses tags entirely.

Pros:
- ψ_src lives on Anima's training-time embedding manifold by
  construction (FM loss is the right signal — see the EOSTok ablation
  cited in `img2emb_plan.md`).
- Embedding-arithmetic UI for edits becomes natural (Component 3 of v1).

Cons:
- **Track record is bad.** The archived `archive/img2emb/` design didn't
  work at quality. The revival plan addresses the structural reason
  (direct alignment was wrong; FM-only is right) but it's still
  unproven in our codebase. IP-Adapter's manifold collapse on the same
  cached PE features (see `project_pe_feature_diagnostics` memory) is a
  closely related failure mode.
- High build cost. New trainer, bench harness, sweep on `λ_align`,
  probably weeks of training+iteration.
- Hard to debug — there's no intermediate human-readable artifact like
  a tag list. When an edit fails, you can't read off what ψ_src thought
  the image was.

### (D) Embedding inversion at edit time (no training)

The archived `archive/inversion/invert_embedding.py` already does this:
gradient descent on ψ_src to minimize FM loss for the source image
through the frozen DiT. By construction the result is on-manifold (it's
optimized against the same loss the model was trained with). Cost is
minutes per image, not milliseconds.

Pros:
- Zero training. Code already exists.
- Highest possible ψ_src fidelity — it's literally the embedding the
  model would have used to generate this image.
- Useful as a *training target* for img2emb if (C) goes ahead — one
  inverted ψ_src per held-out image gives a clean supervisory signal.

Cons:
- ~1–5 minutes per image at 28 steps × hundreds of GD iterations.
  Unacceptable as the default ψ_src path; usable as a "premium" mode.
- Requires the user to wait, which is a UX problem in ComfyUI.

### (E) Hybrid: tagger seed + short embedding refinement

Use (A) to produce a fast initial ψ_src, then run K=20–50 GD steps of
embedding inversion to refine it onto the manifold. K=200 is the
existing inversion default; K=20 might be enough as a "polish" pass
since the seed is already in a sensible region.

Pros:
- Bounds the inference cost (a few seconds extra, not minutes).
- Combines (A)'s vocabulary alignment with (D)'s manifold guarantee.

Cons:
- Adds a tunable (K_refine) and another path through the inference
  pipeline.
- Only worth doing if (A) alone proves insufficient — premature if (A)
  hasn't been tried yet.

## Recommendation

Phase v2 in this order:

### Phase v2.0 — Drop wd-tagger, ship custom tagger

Train (A). The training data is already on disk under `image_dataset/`
and `post_image_dataset/lora/{stem}_anima_te.safetensors` (the cached T5
embeddings can also serve as a regression target if BCE-on-tags
underperforms — see "decisions" below). The vision tower is already
cached via `make preprocess-pe`. This is the smallest possible change
that addresses the actual blocker observed in testing.

Concretely:

- New `library/captioning/anima_tagger.py` mirroring the `WDTagger`
  interface (`predict_caption(pil_img) → tag_string`). Frozen PE-Core
  trunk + small projection head + per-tag sigmoid + per-tag threshold
  vector calibrated on a held-out split.
- New `scripts/train_anima_tagger.py`. Trains in <1 hour on the
  existing dataset. Bench harness drops in `bench/anima_tagger/` per
  the standard envelope.
- Swap `WDTagger` → `AnimaTagger` in
  `scripts/experimental_tasks/inference.py::cmd_test_directedit` (and
  expose a CLI flag in `scripts/edit.py` so users can pick which
  tagger).
- Eyeball test: rerun the same set of `make exp-test-directedit`
  invocations that motivated this proposal. Edit success rate should
  jump.

This is shippable on its own. If it closes the gap, v2 is done.

### Phase v2.1 — Embedding inversion as a "premium" path (optional)

If Phase v2.0 still misses on rare/un-tagged content, expose (D) as an
opt-in `--ψ_src_mode invert` flag in `scripts/edit.py`. Reuse the
existing `archive/inversion/invert_embedding.py` largely as-is; just
move it out of `archive/` and wire it into the edit pipeline. No new
training. Ships behind a "slow but accurate" UI label.

### Phase v2.2 — Reconsider img2emb only if both above are insufficient

If (A) and (D) together don't cover the use cases (specifically: large
volume editing of external content where (D) is too slow and (A) is too
narrow), revisit (C). At that point we'd have:

- A working `AnimaTagger` to seed it
- A working `invert_embedding` to *generate ground-truth ψ_src* on a
  held-out subset for distillation supervision
- Empirical evidence about exactly which content (A) misses, which
  scopes the img2emb training distribution

That is a much stronger starting position than building (C) speculatively.

## Decisions to make before starting Phase v2.0

1. **Tagger output shape.** Three options:
   - Sigmoid over Anima's full caption tag vocabulary (BCE loss). Most
     interpretable; threshold-tunable. **Recommended.**
   - Direct regression to the cached T5 `crossattn_emb` mean-pooled
     vector. Skips T5 entirely at inference. Concerning because it's
     direct alignment to a positional target — exactly the failure
     mode the EOSTok paper documented for img2emb. Don't.
   - Joint: sigmoid head + auxiliary regression head, sigmoid is the
     primary inference path. Slightly more code, marginal expected gain.
2. **Vocabulary scope.** Use only tags that appear in ≥N captions
   (N=5? 10?) to cap the head dimension. Long-tail tags hurt
   calibration and rarely matter for editing.
3. **Training split.** Hold out ~5 % of the dataset for threshold
   calibration and bench. Don't reuse training images for either.
4. **Threshold calibration.** Per-tag F1-optimal threshold on the
   held-out split. Some tags want 0.2, others want 0.6 — global 0.35
   is a wd-tagger inheritance, not a principled choice.
5. **Caption normalization.** Anima's training captions have a stable
   format (comma-separated, specific ordering, underscore/space
   conventions). The tagger should emit that exact format, not "tags
   sorted alphabetically by confidence" — the latter is what
   wd-tagger does, and T5 is sensitive to ordering.

## What this changes about v1

Concrete edits to v1's text:

- v1 §"Component 2 — img2emb as the inversion-prompt provider": img2emb
  is no longer the *primary* provider. The "Captioner → T5" row of the
  ψ_src table moves from "fallback" to "default", with an Anima-trained
  tagger filling the captioner slot. img2emb stays in the table as a
  future-work row.
- v1 §"Phase 2 — img2emb revival" and §"Phase 3 — full pipeline" become
  conditional on Phase v2.2's "if (A) and (D) are insufficient" gate.
- v1 §"Open design questions" #4 ("Captioner fallback") is answered by
  Phase v2.0: the captioner is our own Anima-trained tagger, not
  Qwen2-VL or BLIP-2.

The DirectEdit primitive itself (v1 Component 1) is unchanged — testing
confirmed it works.

## Why this matters

The v1 testing data is the strongest signal we've gotten in a while
about which way to invest:

- **DirectEdit invert/edit machinery is good enough.** The hard,
  research-y part of v1 worked. Don't over-engineer it.
- **Tagging on Anima's distribution is the live bottleneck.** It's
  also the cheapest thing to fix. Fix the cheap thing first.
- **img2emb's failure mode is structural, not a v1 bug.** Spending a
  month on (C) before trying (A) repeats the same mistake the archived
  img2emb design made: assuming the hard problem (manifold-correct
  embeddings) needs the hard solution (FM-loss-through-DiT training)
  when the actual gap (vocabulary distribution) has a much simpler
  remedy.

If Phase v2.0 ships and the edit rate is still bad after that, **then**
we have learned something specific about what's missing, and (C) starts
from a stronger position. Until that data exists, (C) is speculative.

## Non-goals

- Replacing wd-tagger as a *general-purpose* booru tagger. Anima-tagger
  is for Anima editing only; wd-tagger stays available for users who
  want booru-distribution tags for unrelated reasons.
- Training a captioner that competes with VLMs on natural-language
  description. Tag-list output is fine — that's what Anima's training
  pipeline expects.
- Fully solving "edit any external image perfectly". (D) is the
  upper-bound fallback; (A) closes most of the gap; some content
  (extreme out-of-distribution photos, unusual styles) will always
  edit worse than in-distribution character art.
