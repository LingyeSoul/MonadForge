"""Position-aware caption enhance (v1) — detect subjects, bind tags to sides.

Orchestration for ``make caption-position``: for every multi-subject image,
detect the ``girl`` instances, order them into reading order, tag each
mask-blanked crop, and **append** positional clauses to the caption in the
dataset's hand-written convention::

    <flat tag bag>. On the left, akita neru, yellow eyes. On the right, ...

v1 is purely **additive** — the flat tag bag is left exactly as it was, so a
caption gains binding without losing anything the model was pretrained on.
(Moving attributable tags *out* of the bag is v2; see
``docs/proposal/position_captions.md``.)

Layering: this module holds the "drive the primitives over a dataset" logic and
takes its two models as **injected callables** (``detect_fn`` / ``tag_fn``), so
it imports neither SAM3 nor the tagger and stays unit-testable with stubs. The
entry point ``scripts/preprocess/position_captions.py`` owns argparse + model
loading.

Two systematic errors Phase-0 probe B found are fixed here mechanically:

* **crop contamination** — a neighbor's hair bleeding into the padded bbox made
  the crop tagger call the wrong hair color. SAM3 already returns a per-instance
  mask, so non-instance pixels are blanked before tagging.
* **weak detections** — an extreme close-up scored below the 0.5 gate. When the
  caption's own girls-count says more subjects exist than were detected, the
  detection is retried at a lower threshold.

The gate is the number of **detected** instances (≥2), never the girls-count
tag: a ``1girl, multiple views`` outfit sheet is four bindable subjects and is
handled by exactly the same machinery.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

from library.captioning.position_clauses import (
    PositionClause,
    assign_positions,
    compose_caption,
    has_clauses,
    ordered_indices,
    parse_caption,
)
from library.captioning.taxonomy import is_artist_tag, is_count_tag, is_rating_tag

# ---------------------------------------------------------------------------
# Clause vocabulary
# ---------------------------------------------------------------------------

# Tag groups that describe *one subject*. A tag in any of these binds to a
# position; everything else in the taxonomy (lighting, background, framing,
# medium, interaction, …) describes the scene or the relation between subjects
# and stays in the flat bag. Drawn from the tagger's own ``groups.yaml`` rather
# than substring heuristics so the two can't drift.
SUBJECT_GROUPS = frozenset(
    {
        # identity / face
        "eye_color",
        "eye_shape",
        "face_features",
        "expression",
        "emotion",
        "age",
        "gender",
        "skin",
        "body_shape",
        "body_parts",
        "species_nonhuman",
        "animal_parts",
        # hair
        "hair_color",
        "hair_length",
        "hairstyle",
        "hair_accessory",
        # clothing
        "top_garment",
        "top_clothing",
        "bottom_clothing",
        "dress_onepiece",
        "underwear",
        "swimwear",
        "school_uniform",
        "traditional_clothing",
        "costume",
        "clothing_details",
        "fashion_style",
        "footwear",
        "socks_stockings",
        "gloves_acc",
        "earrings_necklace",
        "hat",
        "eyewear",
        "bag",
        # what the subject is doing
        "pose",
        "gesture",
        "daily_action",
    }
)

# Emitted first, in this order, when present — the attributes that actually
# disambiguate a subject. Everything else follows, ranked.
_PRIORITY_GROUPS = ("hair_color", "eye_color", "hair_length", "hairstyle")

_GIRLS_COUNT_RE = re.compile(r"^(\d+)\+?girls?$")
_MULTI_VIEW_TAGS = frozenset({"multiple views", "multiple_views"})


@dataclass(frozen=True)
class ClauseVocabulary:
    """Which tags may enter a clause, and in what order.

    ``tag_to_group`` comes from the tagger checkpoint's ``groups.yaml``;
    ``characters`` / ``excluded`` from its ``vocab.json``. Tags with no group are
    admitted only through the *attributable + in the caption* path (see
    :meth:`select`) — that is what lets a curated compound like ``pink jacket``
    bind while keeping ungrouped scene tags (``simple background``) out.

    ``excluded`` holds the categories that describe the *image*, not a subject
    inside it: copyright, artist, metadata, deprecated. A franchise tag like
    ``vocaloid`` fires on every crop and would otherwise ride the ranked path
    into a clause.

    ``exclusive_groups`` are the softmax / softmax_when_solo groups — at most one
    of their members may enter a clause, or a crop that keeps two hair colors
    emits ``green hair, …, aqua hair`` for one subject.
    """

    characters: frozenset[str] = frozenset()
    excluded: frozenset[str] = frozenset()
    exclusive_groups: frozenset[str] = frozenset()
    tag_to_group: Mapping[str, str] = field(default_factory=dict)

    def group_of(self, tag: str) -> str | None:
        return self.tag_to_group.get(tag)

    def is_subject_tag(self, tag: str) -> bool:
        return self.group_of(tag) in SUBJECT_GROUPS

    def is_scene_tag(self, tag: str) -> bool:
        """Grouped, but into a group that describes the scene, not a subject."""
        group = self.group_of(tag)
        return group is not None and group not in SUBJECT_GROUPS

    def select(
        self,
        kept: Mapping[str, float],
        groups: Mapping[str, str | None],
        *,
        flat_bag: frozenset[str],
        attributable: frozenset[str],
        shared: frozenset[str],
        max_tags: int,
        name_confidence: float,
        allow_unlisted_names: bool,
        discriminative_only: bool = True,
    ) -> list[str]:
        """Clause tags for one crop, ordered most-disambiguating first.

        ``kept`` / ``groups`` are the crop tagger's output; ``flat_bag`` is the
        image's existing caption (the curated ground truth for *what* is in the
        image — the crop only decides *where*); ``attributable`` is the set of
        tags this crop is the **only** one to keep; ``shared`` is the set *every*
        crop keeps.

        **A clause only carries what tells its subject apart.** With
        ``discriminative_only`` (the default), ``shared`` tags are suppressed:
        on a ``1girl, multiple views`` outfit sheet every view is the same
        character with the same hair, so repeating ``hatsune miku, aqua hair,
        twintails`` four times binds nothing and crowds out the maid / bunny /
        swimsuit that actually distinguishes the views. Those shared attributes
        are already in the flat bag — v1 is additive and never removes them —
        so nothing is lost by leaving them there.
        """
        out: list[str] = []
        seen: set[str] = set()
        taken_groups: set[str] = set()
        blocked = shared if discriminative_only else frozenset()

        def add(tag: str) -> bool:
            if not tag or tag in seen or tag in blocked:
                return False
            group = self.group_of(tag)
            if group in self.exclusive_groups and group in taken_groups:
                return False  # one hair color / one eye color per subject
            seen.add(tag)
            if group:
                taken_groups.add(group)
            out.append(tag)
            return True

        # 1. Character name. A name the caption never claimed is a crop
        #    hallucination, so by default it must appear in the flat bag.
        names = sorted(
            (t for t in kept if t in self.characters and kept[t] >= name_confidence),
            key=lambda t: -kept[t],
        )
        for name in names:
            if allow_unlisted_names or name in flat_bag:
                add(name)  # no-op when the name is shared by every crop
                break  # one identity per subject

        # 2. Exclusive-group winners (hair color, eye color, …). These are the
        #    softmax_when_solo groups, and a single-subject crop is exactly the
        #    condition under which they fire — the whole point of cropping.
        for group in _PRIORITY_GROUPS:
            winner = groups.get(group)
            if winner is None:
                # Group didn't fire (contaminated / multi-person crop): fall
                # back to the highest-probability kept member of that group.
                members = sorted(
                    (t for t in kept if self.group_of(t) == group),
                    key=lambda t: -kept[t],
                )
                winner = members[0] if members else None
            if winner:
                add(winner)

        # 3. Everything else, preferring tags the caption already curated.
        rest = [
            t
            for t in kept
            if t not in seen
            and not is_count_tag(t)
            and not is_rating_tag(t)
            and not is_artist_tag(t)
            and t not in self.characters
            and t not in self.excluded
            and not self.is_scene_tag(t)
            and (self.is_subject_tag(t) or (t in flat_bag and t in attributable))
        ]
        rest.sort(key=lambda t: (t not in flat_bag, -kept[t]))
        for tag in rest:
            if len(out) >= max_tags:
                break
            add(tag)
        return out[:max_tags]


def load_clause_vocabulary(ckpt_dir: str | Path) -> ClauseVocabulary:
    """Build a :class:`ClauseVocabulary` from a tagger checkpoint directory."""
    import json

    from library.captioning import tag_groups as tg

    ckpt = Path(ckpt_dir)
    with open(ckpt / "vocab.json", encoding="utf-8") as f:
        vocab = json.load(f)
    characters = frozenset(
        t["name"] for t in vocab["tags"] if t.get("category") == "character"
    )
    excluded = frozenset(
        t["name"]
        for t in vocab["tags"]
        if t.get("category") in {"copyright", "artist", "metadata", "deprecated"}
    )
    groups_path = ckpt / "groups.yaml"
    tag_to_group: Mapping[str, str] = {}
    exclusive: frozenset[str] = frozenset()
    if groups_path.exists():
        groups = tg.load_groups(groups_path)
        tag_to_group = dict(groups.tag_to_group)
        exclusive = frozenset(
            g.name for g in groups.groups if g.mode in {"softmax", "softmax_when_solo"}
        )
    return ClauseVocabulary(
        characters=characters,
        excluded=excluded,
        exclusive_groups=exclusive,
        tag_to_group=tag_to_group,
    )


# ---------------------------------------------------------------------------
# Candidate prefilter
# ---------------------------------------------------------------------------


def caption_subject_count(caption: str) -> int | None:
    """How many bindable subjects the caption itself claims, if it says.

    ``Ngirls`` gives a number. ``None`` means "more than one, count unknown" —
    the count-consistency check then trusts detection instead of skipping. A
    ``multiple views`` sheet is always ``None`` even when it also carries a
    girls-count: the count tags how many *characters* are drawn, while each view
    is its own bindable subject (``1girl, multiple views`` is routinely four).
    """
    tags = {t.strip().lower() for t in parse_caption(caption).flat_tags}
    if tags & _MULTI_VIEW_TAGS:
        return None
    counts = [int(m.group(1)) for t in tags if (m := _GIRLS_COUNT_RE.match(t))]
    if "multiple girls" in tags and not counts:
        return None
    return max(counts) if counts else 0


def is_candidate(caption: str) -> tuple[bool, str]:
    """Should this caption go through detection? Returns ``(ok, reason)``."""
    if has_clauses(caption):
        return False, "already-has-clauses"
    tags = {t.strip().lower() for t in parse_caption(caption).flat_tags}
    if tags & _MULTI_VIEW_TAGS:
        return True, "multiple-views"
    expected = caption_subject_count(caption)
    if expected is None or expected > 1:
        return True, "multi-girl"
    return False, "single-subject"


# ---------------------------------------------------------------------------
# Detection plumbing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Detection:
    """One detected subject: box in pixels, score, and optional instance mask."""

    box: tuple[float, float, float, float]
    score: float
    mask: np.ndarray | None = None


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def dedupe_detections(
    detections: Iterable[Detection], iou_threshold: float
) -> list[Detection]:
    """Greedy IoU suppression, highest score first."""
    ranked = sorted(detections, key=lambda d: -d.score)
    keep: list[Detection] = []
    for det in ranked:
        if all(box_iou(det.box, k.box) < iou_threshold for k in keep):
            keep.append(det)
    return keep


def crop_instance(
    image: Image.Image,
    det: Detection,
    *,
    pad: float = 0.06,
    blank: bool = True,
    blank_color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Padded bbox crop with every non-instance pixel blanked out.

    Blanking is the probe-B contamination fix: without it a neighbor standing
    inside the padded box contributes their hair/outfit to this subject's tags,
    which was both of that probe's hair-color misses. Falls back to a plain crop
    when the detector supplied no mask.
    """
    width, height = image.size
    x1, y1, x2, y2 = det.box
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    box = (
        max(0, int(x1 - px)),
        max(0, int(y1 - py)),
        min(width, int(x2 + px)),
        min(height, int(y2 + py)),
    )
    if det.mask is None or not blank:
        return image.crop(box)
    mask = np.asarray(det.mask)
    if mask.ndim == 3:
        mask = mask[0]
    keep = mask[box[1] : box[3], box[0] : box[2]] > 0.5
    pixels = np.asarray(image.crop(box).convert("RGB")).copy()
    pixels[~keep] = blank_color
    return Image.fromarray(pixels)


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
    tokens: int | None = None

    @property
    def ok(self) -> bool:
        return self.status == "proposed"


@dataclass
class PositionCaptionStats:
    seen: int = 0
    candidates: int = 0
    proposed: int = 0
    written: int = 0
    skipped: dict[str, int] = field(default_factory=dict)

    def skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1


@dataclass(frozen=True)
class PositionCaptionOptions:
    """Knobs for one pass. Defaults are the shipped v1 recipe."""

    prompt: str = "girl"
    score_threshold: float = 0.5
    retry_score_threshold: float = 0.3
    iou_threshold: float = 0.65
    pad: float = 0.06
    blank_crops: bool = True
    row_tol: float = 0.25
    max_clause_tags: int = 8
    name_confidence: float = 0.5
    allow_unlisted_names: bool = False
    min_instances: int = 2
    max_instances: int = 8
    strict_count: bool = True
    discriminative_only: bool = True


def detect_subjects(
    image: Image.Image,
    detect_fn: Callable[[Image.Image, float], list[Detection]],
    options: PositionCaptionOptions,
    expected: int | None,
) -> list[Detection]:
    """Detect + dedupe, with the low-threshold retry when the count falls short.

    ``detect_fn(image, score_threshold)`` returns raw detections. The retry only
    fires when the caption claims a specific count we missed — an unconditional
    low threshold would flood grids with duplicate part-detections.
    """
    dets = dedupe_detections(
        detect_fn(image, options.score_threshold), options.iou_threshold
    )
    if (
        expected
        and len(dets) < expected
        and options.retry_score_threshold < options.score_threshold
    ):
        retry = dedupe_detections(
            detect_fn(image, options.retry_score_threshold), options.iou_threshold
        )
        if len(retry) > len(dets):
            dets = retry
    return dets


def propose_for_image(
    image: Image.Image,
    caption: str,
    *,
    detect_fn: Callable[[Image.Image, float], list[Detection]],
    tag_fn: Callable[[Image.Image], Mapping[str, object]],
    vocabulary: ClauseVocabulary,
    options: PositionCaptionOptions,
    crop_sink: Callable[[int, str, Image.Image], str] | None = None,
) -> ImageProposal:
    """Build the clause proposal for one image. Never writes any caption."""
    parsed = parse_caption(caption)
    flat_bag = frozenset(t.strip().lower() for t in parsed.flat_tags)
    expected = caption_subject_count(caption)

    proposal = ImageProposal(
        image="",
        caption_path="",
        status="proposed",
        expected=expected,
        original=caption,
    )

    dets = detect_subjects(image, detect_fn, options, expected)
    proposal.detected = len(dets)
    if len(dets) < options.min_instances:
        proposal.status = "skip:too-few-instances"
        return proposal
    if len(dets) > options.max_instances:
        proposal.status = "skip:too-many-instances"
        return proposal
    # Detection and the caption's own count must agree, or we would be writing
    # clauses we cannot ground — probe B saw this on 2/13. Skip and log.
    if options.strict_count and expected and len(dets) != expected:
        proposal.status = "skip:count-mismatch"
        return proposal

    order = ordered_indices([d.box for d in dets], image.size, row_tol=options.row_tol)
    dets = [dets[i] for i in order]
    positions = assign_positions(
        [d.box for d in dets], image.size, row_tol=options.row_tol
    )

    crops = [
        crop_instance(image, d, pad=options.pad, blank=options.blank_crops)
        for d in dets
    ]
    predictions = [tag_fn(crop) for crop in crops]
    kept_sets = [dict(p.get("kept") or {}) for p in predictions]
    # A tag only *this* crop keeps is attributable to it. One that *every* crop
    # keeps discriminates nothing — the same character in four outfit views
    # scores the same name, hair, and eyes on all four — so it stays in the flat
    # bag rather than padding every clause identically.
    counts: dict[str, int] = {}
    for kept in kept_sets:
        for tag in kept:
            counts[tag] = counts.get(tag, 0) + 1
    attributable = frozenset(t for t, n in counts.items() if n == 1)
    shared = frozenset(t for t, n in counts.items() if n == len(kept_sets))

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
        )
        crop_name = crop_sink(i, positions[i], crops[i]) if crop_sink else None
        proposal.instances.append(
            InstanceProposal(
                position=positions[i],
                box=[int(v) for v in det.box],
                score=round(float(det.score), 3),
                tags=tags,
                crop=crop_name,
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
    proposal.proposed = compose_caption(parsed.flat_tags, clauses)
    return proposal


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
) -> tuple[list[ImageProposal], PositionCaptionStats]:
    """Walk the resized tree, propose clauses, and (with ``apply``) write them.

    Captions are written back to the **master** under ``source_dir``
    (``image_dataset/``), which is what ``preprocess-captions`` mirrors into
    ``resized/`` and the TE step then encodes. Detection runs on the *resized*
    image because that is the pixel data training actually sees.

    Caption edits do **not** invalidate the TE caches — the caller must follow
    an ``apply`` run with ``make preprocess-te`` (which regenerates the variant
    sidecars first). That silent-failure trap is why ``apply`` defaults off.
    """
    from library.preprocess._dataset import walk_images

    options = options or PositionCaptionOptions()
    stats = PositionCaptionStats()
    rows: list[ImageProposal] = []

    images = walk_images(resized_dir, recursive=True, pattern=path_pattern)
    stats.seen = len(images)

    for index, image_path in enumerate(images, 1):
        rel = image_path.relative_to(resized_dir).with_suffix(".txt")
        caption_path = source_dir / rel
        if progress is not None:
            progress(index, len(images), str(rel))
        if not caption_path.exists():
            stats.skip("no-caption")
            continue
        caption = caption_path.read_text(encoding="utf-8").strip()
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
        )
        proposal.image = str(image_path.relative_to(resized_dir))
        proposal.caption_path = str(rel)
        rows.append(proposal)

        if not proposal.ok:
            stats.skip(proposal.status.removeprefix("skip:"))
            continue
        stats.proposed += 1
        if token_count_fn is not None and proposal.proposed:
            proposal.tokens = token_count_fn(proposal.proposed)
        if apply:
            caption_path.write_text(proposal.proposed, encoding="utf-8")
            stats.written += 1

    return rows, stats
