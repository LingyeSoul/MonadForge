"""Position-aware caption rewrite (v2) — detect subjects, bind tags to sides.

Orchestration for ``make caption-position``: for every multi-subject image,
detect the ``girl`` instances, order them into reading order, tag each
mask-blanked crop, and rewrite the caption in the dataset's hand-written
convention::

    <flat tag bag>. On the left, akita neru, yellow eyes. On the right, ...

**v2 moves an attributable tag out of the flat bag into its clause** rather than
asserting it twice: ``2girls, blonde hair, aqua hair`` becomes ``2girls. On the
left, blonde hair. On the right, aqua hair.`` — each attribute stated exactly
once, bound to the subject it belongs to. That is the whole point of the feature;
the additive v1 (clause appended, bag untouched) left the bag still claiming
every attribute of every subject, which is the ambiguity clauses exist to
resolve. ``rewrite=False`` restores v1 for the A/B arm.

This module owns the *pipeline*: knobs, the detect→crop→tag→compose pass over
one image, and the dataset walk that applies it. The pieces it drives live with
their own kind:

* :mod:`library.captioning.position_clauses` — clause grammar and position words
* :mod:`library.captioning.caption_layout` — count/layout prefilter
* :mod:`library.captioning.clause_vocabulary` — which tags may enter a clause
* :mod:`library.captioning.clause_rewrite` — which bag tags a clause may take
* :mod:`library.preprocess.instance_detection` — boxes, NMS, crops

Takes its two models as injected callables (``detect_fn``/``tag_fn``), staying
import-free of SAM3/the tagger; ``scripts/preprocess/position_captions.py``
owns argparse + model loading.

**A clause moves tags; it does not invent them.** The crop decides *where* an
attribute belongs, the caption decides *what* is in the image — so the bag fills
each clause first and only ``max_novel_tags`` (1) slots are left for something
the caption never contained. A novel clause tag cannot be a move
(:func:`plan_bag_removals` only removes what is in the bag), so bag-blind
selection spent 46% of the clause budget on assertions the curated caption never
made and that bind nothing.

Measured on ``ama_mitsuki`` (55 proposed images, 131 clauses) against the same
pass with ``max_novel_tags=8``: clause tags 983 → 583, novel 515 → 115, reuse
0.476 → 0.803. The bound bag tags and all 370 moves are **byte-identical** — the
budget removes only the novel padding. It does not buy extra bindings, and it
was never crowding them out: the candidate ranking already put bag tags ahead of
novel ones, so the clause cap was reached only after the bag was exhausted. What
it buys is a caption that asserts 400 fewer unverified things and is 40%
shorter.

Three rules bound what may leave the bag, because a wrong move is worse than a
wrong clause (it makes the caption assert that the *other* subjects lack the
attribute):

* **Character-invariant groups need corroboration.** Hair color, eyes, body
  shape, species … are properties of a *character*, not of a view, so on a
  ``1girl, multiple views`` sheet they are true of every panel. Such a tag may
  only move when the bag names **two or more** values of that group (see
  ``_CHARACTER_INVARIANT_GROUPS``) — i.e. the caption is already enumerating
  per-subject values and binding them loses nothing.
* **Exclusive keep.** No other crop may have *kept* the tag. This is the
  tagger's own calibrated per-tag threshold answering "does this subject have
  it too", which is the question the rule is actually asking.
* **Relative attribution margin.** The runner-up crop's probability must fall
  below ``(1 - attribution_margin)`` of the winner's, so a tag the tagger
  *nearly* kept on a second subject stays in the bag (and stays duplicated in
  the clause). Relative, not an absolute gap: per-tag thresholds span
  ~0.05–0.85, so an absolute gap is a different test for every tag.

On a **repeated-subject layout** — ``multiple views`` or a comic-panel page, the
``_LAYOUT_TAGS`` set — a third, stricter rule applies one level earlier, to what
may **enter** a clause at all. The subjects there are one character drawn
several times, so her name and her traits (appearance *and* anatomy,
``_VIEW_INVARIANT_GROUPS``) discriminate nothing and are dropped from every
clause; a view or panel keeps only what one can differ in — outfit, pose,
expression, framing. ``multi_view_gate=False`` reverts it.

Layering: this module holds the "drive the primitives over a dataset" logic and
takes its two models as **injected callables** (``detect_fn`` / ``tag_fn``), so
it imports neither SAM3 nor the tagger and stays unit-testable with stubs. The
entry point ``scripts/preprocess/position_captions.py`` owns argparse + model
loading.

Two systematic errors Phase-0 probe B found are fixed here mechanically:

* **crop contamination** — a neighbor's hair bleeding into the padded bbox made
  the crop tagger call the wrong hair color. SAM3 already returns a per-instance
  mask, so non-instance pixels are blanked before tagging.
* **weak detections** — an extreme close-up scored below the 0.5 gate. When
  fewer subjects were detected than we have reason to expect (the caption's own
  count, or failing that ``min_instances``), the detection is retried at a lower
  threshold. Note the threshold has to reach the *detector* — SAM3 applies its
  own confidence floor before returning boxes, so post-filtering the result at a
  lower number is a no-op. See ``build_detect_fn`` in the CLI.
* **headless panels** — a close-up of a hip or a backside is a bindable panel
  that the ``girl`` prompt cannot see at *any* threshold. Under the same
  undershoot condition, opt-in ``part_prompts`` run a second grounding pass over
  the already-encoded image and their boxes are merged in without displacing a
  subject. Off by default (``part_prompts=()``).

The gate is the number of **detected** instances (≥2), never the girls-count
tag: a ``1girl, multiple views`` outfit sheet is four bindable subjects, a
``1girl, 2koma`` comic page is two, and both are handled by exactly the same
machinery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Mapping

from PIL import Image

from library.captioning.caption_layout import (
    caption_boy_count,
    caption_panel_ceiling,
    caption_subject_count,
    is_candidate,
    is_repeated_subject_layout,
)
from library.captioning.clause_rewrite import MovedTag, RemovalPlan, plan_bag_removals
from library.captioning.clause_vocabulary import (
    ClauseGroups,
    ClauseVocabulary,
    load_clause_groups,
    load_clause_vocabulary,
)
from library.captioning.position_clauses import (
    PositionClause,
    assign_positions,
    compose_caption,
    flatten_caption,
    has_clauses,
    ordered_indices,
    parse_caption,
)
from library.preprocess.instance_detection import (
    Detection,
    box_area,
    box_containment,
    box_iou,
    crop_instance,
    dedupe_detections,
    drop_small_boxes,
    mask_box_fill,
    mask_containment,
    merge_part_detections,
)

# Convenience re-exports: the canonical homes are the modules imported above,
# but every consumer of this pipeline (the CLI, the A/B tool, the NMS probe, the
# multiview audit, the tests) reaches for them on this module. Listed in
# ``__all__`` so they read as the public surface rather than dead imports.
__all__ = [
    "ClauseGroups",
    "ClauseVocabulary",
    "Detection",
    "ImageProposal",
    "InstanceProposal",
    "MovedTag",
    "PositionCaptionOptions",
    "PositionCaptionStats",
    "RemovalPlan",
    "box_area",
    "box_containment",
    "box_iou",
    "caption_boy_count",
    "caption_panel_ceiling",
    "caption_subject_count",
    "crop_instance",
    "dedupe_detections",
    "detect_subjects",
    "drop_small_boxes",
    "flatten_captions",
    "is_candidate",
    "is_repeated_subject_layout",
    "load_clause_groups",
    "load_clause_vocabulary",
    "mask_box_fill",
    "mask_containment",
    "merge_part_detections",
    "plan_bag_removals",
    "propose_for_image",
    "run_position_captions",
]


# ---------------------------------------------------------------------------
# Proposals
# ---------------------------------------------------------------------------


@dataclass
class InstanceProposal:
    position: str
    box: list[int]
    score: float
    tags: list[str]
    crop: str | None = None
    source: str = "subject"
    # How many of ``tags`` the flat bag did not already contain. The reuse
    # ratio is the headline number for this feature — a clause tag that came
    # from the bag is a candidate *move*, a novel one can only ever be an
    # addition — so it is reported per instance rather than recomputed offline.
    novel: int = 0


@dataclass
class ImageProposal:
    image: str
    caption_path: str
    status: str
    detected: int = 0
    expected: int | None = None
    original: str = ""
    proposed: str | None = None
    instances: list[InstanceProposal] = field(default_factory=list)
    # Boxes as detected, recorded even when a gate rejects the image — the
    # skipped rows are exactly the ones a reviewer needs evidence for, and
    # ``instances`` is only populated once every gate has passed.
    detections: list[dict] = field(default_factory=list)
    tokens: int | None = None
    # v2 bookkeeping: which bag tags the clauses took, and which reached a clause
    # but stayed flat (tag → the rule that pinned it). Both empty under
    # ``rewrite=False``.
    moved: list[dict] = field(default_factory=list)
    pinned: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "proposed"


@dataclass
class PositionCaptionStats:
    seen: int = 0
    candidates: int = 0
    proposed: int = 0
    written: int = 0
    rewritten: int = 0
    moved_tags: int = 0
    # Clause composition: how many tags the clauses carry in total and how many
    # of those the caption never had. ``clause_tags - novel_tags`` is reuse.
    clause_tags: int = 0
    novel_tags: int = 0
    pinned_tags: dict[str, int] = field(default_factory=dict)
    skipped: dict[str, int] = field(default_factory=dict)

    def skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    def pin(self, reason: str) -> None:
        self.pinned_tags[reason] = self.pinned_tags.get(reason, 0) + 1


@dataclass(frozen=True)
class PositionCaptionOptions:
    """Knobs for one pass. Defaults are the shipped v2 recipe."""

    prompt: str = "girl"
    score_threshold: float = 0.5
    retry_score_threshold: float = 0.35
    # Body-part fallback: extra SAM3 prompts run *only* when the subject prompt
    # undershoots, to recover headless close-up panels. Empty tuple = off, which
    # is the default — on a sheet the subject prompt already resolved, part
    # boxes only add nested duplicates. See ``merge_part_detections``.
    part_prompts: tuple[str, ...] = ()
    part_score_threshold: float = 0.5
    part_containment_threshold: float = 0.7
    iou_threshold: float = 0.65
    # Containment suppression is OFF by default: measured, it costs far more
    # than it buys (see ``box_containment``).
    containment_threshold: float = 1.01
    # On by default, unlike its box counterpart — see ``mask_containment``. Two
    # boxes nest identically whether the inner one is a fragment or a second
    # girl in front of the first; their masks do not.
    mask_containment_threshold: float = 0.8
    # Mask-quality tie-break inside an NMS-matched pair — see
    # ``dedupe_detections``. 2.0 sits in the measured empty ratio band
    # (multiview_audit.md §5.4); 0 disables (score-only survivor).
    dedupe_fill_ratio: float = 2.0
    min_area_frac: float = 0.005
    pad: float = 0.06
    blank_crops: bool = True
    row_tol: float = 0.25
    max_clause_tags: int = 8
    # How many tags the caption never contained a single clause may introduce.
    # The rest of its budget is filled from the flat bag, because only a bag tag
    # can actually *move* — a novel one is a pure v1-style addition. Measured on
    # ama_mitsuki, going from 8 to 1 cut novel clause tags 515 → 115 (reuse 0.476
    # → 0.803) with the moved set unchanged. 0 = never invent;
    # ``max_clause_tags`` = the old bag-blind behaviour.
    max_novel_tags: int = 1
    name_confidence: float = 0.5
    allow_unlisted_names: bool = False
    min_instances: int = 2
    max_instances: int = 8
    strict_count: bool = True
    discriminative_only: bool = True
    bag_gated_identity: bool = True
    # On a repeated-subject layout (``multiple views`` / comic panels), keep the
    # character's own traits — and her name — out of every clause: they belong
    # to the girl, not to a view of her.
    multi_view_gate: bool = True
    # Let a clause say which *view* it describes (`ass focus`, `close-up`,
    # `full body`). False is the pre-2026-08-19 behaviour, kept for the A/B.
    bind_framing: bool = True
    # Let a view layout's clause carry anatomy (`ass`, `thighs`) — what is
    # *visible* in that panel. False re-gates it, the pre-2026-08-19 behaviour.
    bind_view_anatomy: bool = True
    # Bag-tag keep relaxation: a tag already in the flat bag can only MOVE into
    # a clause, never be invented — the curated caption corroborates it, so the
    # crop tagger only has to *localize* it, and its per-tag F1 threshold may be
    # relaxed for exactly that population. 1.0 = off (a bag tag needs the
    # tagger's own keep decision, the pre-relaxation behaviour). Applied to
    # every crop before the attributable/shared census, so a rival crop's
    # borderline score also BLOCKS a move the strict kept set would have waved
    # through. 0.35 shipped as the default 2026-08-19 (the "aggressive" A/B
    # arm): it is what recovers pose tags (`lying`, `on back`, `sleeping`)
    # whose scores collapse when mask-blanking removes the scene context.
    bag_relax: float = 0.35
    # Extra relaxation multiplier per word beyond the first (compounds with
    # ``bag_relax``): `black panties` is more specific than `panties`, so a
    # sub-threshold hit on it is less likely to be noise. 1.0 = off.
    bag_word_relax: float = 0.85
    # v2: move an attributable tag out of the flat bag into its clause. False is
    # the additive v1 behaviour (bag untouched), kept for the training A/B.
    rewrite: bool = True
    # How far the winning crop must clear every other crop, *relative to its own
    # probability* (``1 - rival/winner``), before a tag may leave the bag — on
    # top of the hard rule that no other crop kept it. Only the removal is gated
    # — a tag that fails still enters its clause, so the caption degrades to v1
    # for it. 0.0 = trust the tagger's thresholds alone.
    attribution_margin: float = 0.25


def detect_subjects(
    image: Image.Image,
    detect_fn: Callable[[Image.Image, float], list[Detection]],
    options: PositionCaptionOptions,
    expected: int | None,
    part_detect_fn: Callable[[Image.Image, str, float], list[Detection]] | None = None,
) -> list[Detection]:
    """Detect + dedupe, with two escalations when the count falls short.

    ``detect_fn(image, score_threshold)`` returns raw detections. Neither
    escalation is unconditional — they fire only when we detected fewer subjects
    than we have reason to expect, because on an image the subject prompt
    already resolved they can only add duplicates:

    1. **Lower the score threshold** — recovers an extreme close-up.
    2. **Body-part prompts** (``part_detect_fn``, when supplied) — recovers a
       headless panel the subject prompt can't see at any threshold. Merged via
       :func:`~library.preprocess.instance_detection.merge_part_detections`,
       never displacing a subject box.

    The target is ``expected or min_instances``, **not** ``expected`` alone: a
    ``multiple views`` sheet reports ``expected=None`` on purpose (the count tag
    counts characters, not views), and gating on truthiness used to skip the
    retry for that entire population — 35 of the 81 ``too-few-instances`` skips
    in the first full-corpus run.
    """

    def run(threshold: float) -> list[Detection]:
        dets = dedupe_detections(
            detect_fn(image, threshold),
            options.iou_threshold,
            options.containment_threshold,
            options.dedupe_fill_ratio,
            options.mask_containment_threshold,
        )
        return drop_small_boxes(dets, image.size, options.min_area_frac)

    dets = run(options.score_threshold)
    target = expected or options.min_instances
    if len(dets) < target and options.retry_score_threshold < options.score_threshold:
        retry = run(options.retry_score_threshold)
        if len(retry) > len(dets):
            dets = retry

    if len(dets) >= target or part_detect_fn is None or not options.part_prompts:
        return dets

    parts: list[Detection] = []
    for prompt in options.part_prompts:
        parts.extend(part_detect_fn(image, prompt, options.part_score_threshold))
    parts = drop_small_boxes(parts, image.size, options.min_area_frac)
    merged = merge_part_detections(
        dets,
        parts,
        iou_threshold=options.iou_threshold,
        containment_threshold=options.part_containment_threshold,
    )
    # Top up to the target, no further. A part prompt is a looser concept than
    # ``girl`` and fragments: on ama_mitsuki/6040950 ``thighs`` returned four
    # boxes for two panels, which would have bound five clauses to a three-panel
    # image. Taking only the highest-scoring boxes needed to clear the gate
    # bounds that, and an image the part pass cannot fill still skips.
    return merged[: max(target, len(dets))]


def _relax_bag_keeps(
    kept_sets: list[dict[str, float]],
    score_sets: list[dict[str, float]],
    predictions: list[Mapping[str, object]],
    flat_bag: frozenset[str],
    options: PositionCaptionOptions,
) -> None:
    """Admit sub-threshold flat-bag tags into each crop's kept set, in place.

    The bag is the curated ground truth: a bag tag can only *move* into a
    clause (never be invented), so the crop tagger's job for it is attribution,
    not detection, and its calibrated keep threshold may be relaxed by
    ``bag_relax`` (times ``bag_word_relax`` per word beyond the first — a more
    specific tag is less likely to clear on noise). Runs on every crop before
    the attributable/shared census, which cuts both ways: a rival crop's
    borderline score now also blocks a move the strict kept sets would have
    granted. Needs the per-tag thresholds ``AnimaTagger.predict`` attaches;
    silently a no-op per crop when a stub ``tag_fn`` omits them.
    """
    relax = options.bag_relax
    word_relax = options.bag_word_relax
    if relax >= 1.0 and word_relax >= 1.0:
        return
    for kept, scores, pred in zip(kept_sets, score_sets, predictions):
        thresholds = pred.get("thresholds") or {}
        for tag in flat_bag:
            if tag in kept or tag not in scores or tag not in thresholds:
                continue
            floor = thresholds[tag] * relax * word_relax ** (len(tag.split()) - 1)
            if scores[tag] >= floor:
                kept[tag] = float(scores[tag])


def propose_for_image(
    image: Image.Image,
    caption: str,
    *,
    detect_fn: Callable[[Image.Image, float], list[Detection]],
    tag_fn: Callable[[Image.Image], Mapping[str, object]],
    vocabulary: ClauseVocabulary,
    options: PositionCaptionOptions,
    crop_sink: Callable[[int, str, Image.Image], str] | None = None,
    part_detect_fn: Callable[[Image.Image, str, float], list[Detection]] | None = None,
) -> ImageProposal:
    """Build the clause proposal for one image. Never writes any caption."""
    parsed = parse_caption(caption)
    flat_bag = parsed.tag_keys
    expected = caption_subject_count(caption)

    proposal = ImageProposal(
        image="",
        caption_path="",
        status="proposed",
        expected=expected,
        original=caption,
    )

    dets = detect_subjects(image, detect_fn, options, expected, part_detect_fn)
    proposal.detected = len(dets)
    proposal.detections = [
        {
            "box": [int(v) for v in d.box],
            "score": round(float(d.score), 3),
            "source": d.source,
        }
        for d in dets
    ]
    if len(dets) < options.min_instances:
        proposal.status = "skip:too-few-instances"
        return proposal
    if len(dets) > options.max_instances:
        proposal.status = "skip:too-many-instances"
        return proposal
    # Detection and the caption's own count must agree, or we would be writing
    # clauses we cannot ground — probe B saw this on 2/13. Skip and log.
    #
    # "Agree" is a range, not equality: ``expected`` counts girls, while the
    # ``girl`` prompt picks up males inconsistently (it found the boy in 7 of
    # the 19 first-run mismatches and missed him in 89 that passed). Anything
    # from girls to girls+boys is consistent with the caption.
    if options.strict_count and expected:
        boys = caption_boy_count(caption)
        upper = None if boys is None else expected + boys
        if len(dets) < expected or (upper is not None and len(dets) > upper):
            proposal.status = "skip:count-mismatch"
            return proposal
    # A layout tag waives the check above (``expected`` is None by design), which
    # leaves a comic page with no backstop against one subject detected twice.
    # An ``Nkoma`` tag names the panel count and restores a generous ceiling.
    if options.strict_count and not expected:
        ceiling = caption_panel_ceiling(caption)
        if ceiling is not None and len(dets) > ceiling:
            proposal.status = "skip:count-mismatch"
            return proposal

    order = ordered_indices([d.box for d in dets], image.size, row_tol=options.row_tol)
    dets = [dets[i] for i in order]
    positions = assign_positions(
        [d.box for d in dets], image.size, row_tol=options.row_tol
    )

    # Mask-blanking is a *subject*-crop fix (it stops a neighbor's hair bleeding
    # into the padded bbox). On a part box the mask IS the part, so blanking
    # deletes the panel's content — the torn jeans, the pantyhose, the panties,
    # i.e. exactly the tags the part pass exists to recover — and hands the
    # tagger a bare skin blob. Part crops therefore take the plain padded bbox.
    crops = [
        crop_instance(
            image,
            d,
            pad=options.pad,
            blank=options.blank_crops and d.source == "subject",
        )
        for d in dets
    ]
    predictions = [tag_fn(crop) for crop in crops]
    kept_sets = [dict(p.get("kept") or {}) for p in predictions]
    score_sets = [dict(p.get("scores") or {}) for p in predictions]
    _relax_bag_keeps(kept_sets, score_sets, predictions, flat_bag, options)
    # A tag only *this* crop keeps is attributable to it; one every crop keeps
    # discriminates nothing, so it stays in the flat bag instead of padding
    # every clause identically.
    counts: dict[str, int] = {}
    for kept in kept_sets:
        for tag in kept:
            counts[tag] = counts.get(tag, 0) + 1
    attributable = frozenset(t for t, n in counts.items() if n == 1)
    shared = frozenset(t for t, n in counts.items() if n == len(kept_sets))
    view_invariant = options.multi_view_gate and is_repeated_subject_layout(caption)

    for i, (det, kept, pred) in enumerate(zip(dets, kept_sets, predictions)):
        tags = vocabulary.select(
            kept,
            dict(pred.get("groups") or {}),
            flat_bag=flat_bag,
            attributable=attributable,
            shared=shared,
            max_tags=options.max_clause_tags,
            name_confidence=options.name_confidence,
            allow_unlisted_names=options.allow_unlisted_names,
            discriminative_only=options.discriminative_only,
            allow_identity=det.source == "subject",
            bag_gated_identity=options.bag_gated_identity,
            view_invariant=view_invariant,
            bind_framing=options.bind_framing,
            bind_view_anatomy=options.bind_view_anatomy,
            max_novel_tags=options.max_novel_tags,
        )
        crop_name = crop_sink(i, positions[i], crops[i]) if crop_sink else None
        proposal.instances.append(
            InstanceProposal(
                position=positions[i],
                box=[int(v) for v in det.box],
                score=round(float(det.score), 3),
                tags=tags,
                crop=crop_name,
                source=det.source,
                novel=sum(1 for t in tags if t.strip().lower() not in flat_bag),
            )
        )

    clauses = [
        PositionClause(position=inst.position, tags=tuple(inst.tags))
        for inst in proposal.instances
        if inst.tags
    ]
    if len(clauses) < options.min_instances:
        # Every crop tagged identically — the subjects are genuinely
        # indistinguishable to the tagger, so there is nothing to bind.
        proposal.status = "skip:no-discriminative-tags"
        return proposal

    flat = list(parsed.flat_tags)
    if options.rewrite:
        plan = plan_bag_removals(
            parsed.flat_tags,
            [inst.tags for inst in proposal.instances],
            [inst.position for inst in proposal.instances],
            kept_sets,
            score_sets,
            vocabulary=vocabulary,
            margin=options.attribution_margin,
        )
        proposal.pinned = dict(plan.blocked)
        taken = {m.tag.strip().lower() for m in plan.moved}
        remaining = [t for t in flat if t.strip().lower() not in taken]
        # A caption that is nothing but clauses has no scene, rating or count
        # left to condition on. Unreachable in practice (those tags never enter a
        # clause) but the rewrite removes text, so it is asserted, not assumed.
        if remaining:
            flat = remaining
            proposal.moved = [
                {"tag": m.tag, "position": m.position, "margin": m.margin}
                for m in plan.moved
            ]

    proposal.proposed = compose_caption(flat, clauses)
    return proposal


# ---------------------------------------------------------------------------
# Review artifacts
# ---------------------------------------------------------------------------


def _crop_sink(crops_dir: Path, rel: Path) -> Callable[[int, str, Image.Image], str]:
    """Save each crop under ``crops_dir`` mirroring the dataset layout.

    The dry-run review artifact: the reviewer reads a proposed clause next to
    the exact pixels the tagger saw, which is the only way to tell a detection
    miss from a tagging miss.
    """
    target = crops_dir / rel.parent

    def sink(index: int, position: str, crop: Image.Image) -> str:
        target.mkdir(parents=True, exist_ok=True)
        name = f"{rel.stem}_{index}_{position.replace(' ', '-')}.png"
        crop.save(target / name)
        return str((target / name).relative_to(crops_dir))

    return sink


def _save_skip_overlay(
    crops_dir: Path, rel: Path, image: Image.Image, proposal: ImageProposal
) -> None:
    """Draw the detected boxes over a skipped image, under ``_skipped/``.

    A skip produces no crops (``crop_sink`` runs only once every gate passes),
    which left the dry-run report with zero visual evidence for exactly the rows
    a reviewer has to adjudicate — is this an over-detection, a missing subject,
    or a wrong caption count? The overlay answers that at a glance.
    """
    from PIL import ImageDraw

    target = crops_dir / "_skipped" / rel.parent
    target.mkdir(parents=True, exist_ok=True)
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for i, det in enumerate(proposal.detections):
        box = det["box"]
        draw.rectangle(box, outline=(255, 0, 0), width=4)
        draw.text(
            (box[0] + 6, box[1] + 6), f"{i}:{det['score']:.2f}", fill=(255, 255, 0)
        )
    status = proposal.status.removeprefix("skip:")
    canvas.save(target / f"{rel.stem}_{status}.png")


# ---------------------------------------------------------------------------
# Dataset passes
# ---------------------------------------------------------------------------


def _iter_captions(
    resized_dir: Path,
    source_dir: Path,
    path_pattern: str | None,
    stats: PositionCaptionStats,
    progress: Callable[[int, int, str], None] | None = None,
) -> Iterator[tuple[Path, Path, Path, str]]:
    """Yield ``(image_path, rel, dst_caption, caption)`` for the walked tree.

    Shared by both passes so they can't disagree on which caption file is
    authoritative: the derived caption (already order-corrected, and carrying an
    earlier run's clauses so ``is_candidate`` skips it) with the master as the
    read-only fallback for an image the caption step has not mirrored yet.
    ``stats.seen``, the ``no-caption`` skip, and the progress callback (over
    *every* walked image, not only the captioned ones) are handled here.
    """
    from library.preprocess._dataset import walk_images

    images = walk_images(resized_dir, recursive=True, pattern=path_pattern)
    stats.seen = len(images)
    for index, image_path in enumerate(images, 1):
        rel = image_path.relative_to(resized_dir).with_suffix(".txt")
        dst_caption = resized_dir / rel
        caption_path = dst_caption if dst_caption.exists() else source_dir / rel
        if progress is not None:
            progress(index, len(images), str(rel))
        if not caption_path.exists():
            stats.skip("no-caption")
            continue
        caption = caption_path.read_text(encoding="utf-8").strip()
        yield image_path, rel, dst_caption, caption


def run_position_captions(
    *,
    resized_dir: Path,
    source_dir: Path,
    detect_fn: Callable[[Image.Image, float], list[Detection]],
    tag_fn: Callable[[Image.Image], Mapping[str, object]],
    vocabulary: ClauseVocabulary,
    options: PositionCaptionOptions | None = None,
    path_pattern: str | None = None,
    apply: bool = False,
    crops_dir: Path | None = None,
    token_count_fn: Callable[[str], int] | None = None,
    progress: Callable[[int, int, str], None] | None = None,
    part_detect_fn: Callable[[Image.Image, str, float], list[Detection]] | None = None,
) -> tuple[list[ImageProposal], PositionCaptionStats]:
    """Walk the resized tree, propose clauses, and (with ``apply``) write them.

    **The caption master is never touched.** Clauses are a *derived* caption
    layer, so the rewrite lands next to the resized image (``resized_dir/<rel>``)
    — the same file ``preprocess-captions`` writes and the TE step encodes. The
    hand-written caption under ``source_dir`` (``image_dataset/``) stays exactly
    as the user left it; it is only the *read* fallback for an image whose
    resized caption has not been mirrored yet. Detection runs on the resized
    image either way, because that is the pixel data training actually sees.

    Two things make the derived layer safe to write into. The mirror pass
    (:func:`library.captioning.preprocess.write_corrected_preprocess_captions`)
    re-attaches clauses it finds on a destination caption whose master has none,
    so a later ``preprocess-captions`` re-corrects the flat bag instead of
    dropping the clauses; and the write bumps the caption's mtime, which is what
    the TE cache staleness check keys on — so the next TE pass re-encodes rather
    than silently keeping a pre-clause cache. Any ``{stem}.variants.txt`` sidecar
    is dropped here for the same reason: the sidecar wins over ``{stem}.txt`` at
    encode time, so a stale one would train the pre-clause caption.

    Under v2 the write is a **rewrite**, not an append — a bound tag leaves the
    flat bag. It is recoverable (:func:`flatten_captions`), and the master always
    holds the pre-clause caption, but it is not a no-op, which is why ``apply``
    defaults off.
    """
    options = options or PositionCaptionOptions()
    stats = PositionCaptionStats()
    rows: list[ImageProposal] = []

    walked = list(_iter_captions(resized_dir, source_dir, path_pattern, stats))
    for index, (image_path, rel, dst_caption, caption) in enumerate(walked, 1):
        if progress is not None:
            progress(index, len(walked), str(rel))
        ok, reason = is_candidate(caption)
        if not ok:
            stats.skip(reason)
            continue
        stats.candidates += 1

        crop_sink = _crop_sink(crops_dir, rel) if crops_dir is not None else None
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        proposal = propose_for_image(
            image,
            caption,
            detect_fn=detect_fn,
            tag_fn=tag_fn,
            vocabulary=vocabulary,
            options=options,
            crop_sink=crop_sink,
            part_detect_fn=part_detect_fn,
        )
        proposal.image = str(image_path.relative_to(resized_dir))
        proposal.caption_path = str(rel)
        rows.append(proposal)

        if not proposal.ok:
            stats.skip(proposal.status.removeprefix("skip:"))
            if crops_dir is not None:
                _save_skip_overlay(crops_dir, rel, image, proposal)
            continue
        stats.proposed += 1
        stats.clause_tags += sum(len(i.tags) for i in proposal.instances)
        stats.novel_tags += sum(i.novel for i in proposal.instances)
        if proposal.moved:
            stats.rewritten += 1
            stats.moved_tags += len(proposal.moved)
        for reason in proposal.pinned.values():
            stats.pin(reason)
        if token_count_fn is not None and proposal.proposed:
            proposal.tokens = token_count_fn(proposal.proposed)
        if apply:
            _write_derived_caption(dst_caption, proposal.proposed)
            stats.written += 1

    return rows, stats


def _write_derived_caption(dst_caption: Path, text: str) -> None:
    """Write a caption into the resized tree and drop its variant sidecar.

    The sidecar is the encode source of truth when present, so leaving a
    pre-clause one behind would keep training the pre-clause caption however
    fresh ``{stem}.txt`` is. Dropping it makes the TE step either regenerate the
    variants in-process or pick up the one the next caption pass writes.
    """
    from library.preprocess.caption_variants import variants_sidecar_path

    dst_caption.parent.mkdir(parents=True, exist_ok=True)
    dst_caption.write_text(text, encoding="utf-8")
    sidecar = variants_sidecar_path(dst_caption)
    if sidecar.exists():
        sidecar.unlink()


def flatten_captions(
    *,
    resized_dir: Path,
    source_dir: Path,
    path_pattern: str | None = None,
    apply: bool = False,
) -> tuple[list[dict], PositionCaptionStats]:
    """Undo a rewrite: merge every caption's clauses back into its flat bag.

    The v2 rewrite *moves* tags rather than deleting them, so a clause-free
    caption is recoverable from the text alone — no SAM3, no tagger, no pixels.
    Two uses: backing out an ``--apply`` run, and building the clause-free
    control corpus for a training A/B.

    Reads and writes the same derived caption as :func:`run_position_captions`
    (``resized_dir/<rel>``, falling back to the master only for the read), so
    ``path_pattern`` means the same thing in both and the nested-symlink layout
    of ``image_dataset/`` is never globbed.

    Hand-written clauses are flattened too — the pass cannot tell them from
    generated ones. In the derived layer that is recoverable (the master still
    holds them, and the next mirror re-writes them), but on a caption whose
    clauses only ever existed here it is a real loss of curation, hence the
    dry-run default.
    """
    stats = PositionCaptionStats()
    rows: list[dict] = []
    for _, rel, dst_caption, original in _iter_captions(
        resized_dir, source_dir, path_pattern, stats
    ):
        if not has_clauses(original):
            stats.skip("no-clauses")
            continue
        stats.candidates += 1
        flattened = flatten_caption(original)
        if flattened == original:
            stats.skip("unchanged")
            continue
        stats.proposed += 1
        rows.append(
            {
                "caption_path": str(rel),
                "original": original,
                "proposed": flattened,
            }
        )
        if apply:
            _write_derived_caption(dst_caption, flattened)
            stats.written += 1
    return rows, stats
