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

# The groups you read off a **face**. A crop without a head in it has no
# evidence for any of them, so they are suppressed on a body-part crop
# (``allow_identity=False``).
_IDENTITY_GROUPS = frozenset({"hair_color", "eye_color", "hair_length", "hairstyle"})

# Groups where the flat bag outranks the crop tagger. Once the caption names a
# hair color, a clause may only pick from the colors it named — a crop claiming
# a value the curated caption never listed is a hallucination, and
# discriminative-only *promotes* exactly those (a value shared by every crop is
# suppressed, so a wrong outlier is what survives). Measured over the first
# full-corpus dry run: 520 of 1600 identity clause tags claimed a value the
# caption contradicted, 33% of the total.
#
# ``hairstyle`` is deliberately NOT gated even though it is a priority group:
# a crop legitimately reveals a ``hair bun`` or ``sidelocks`` the booru caption
# never bothered to tag, and unlike a color it does not contradict what is
# there. ``body_shape`` and ``fashion_style`` are left out for the same reason —
# a per-subject value the bag omitted is real information, not a contradiction.
_BAG_GATED_GROUPS = ("hair_color", "eye_color", "hair_length")

_GIRLS_COUNT_RE = re.compile(r"^(\d+)girls?$")
_BOYS_COUNT_RE = re.compile(r"^(\d+)boys?$")
# ``6+girls`` is an open-ended crowd tag, not the number six — matching it
# exactly is impossible, so it is treated like ``multiple girls``: count
# unknown, trust detection.
_OPEN_GIRLS_RE = re.compile(r"^\d+\+girls?$")
_OPEN_BOYS_RE = re.compile(r"^\d+\+boys?$")
# ``2koma`` / ``4koma`` name the panel count. Deliberately anchored, so the
# open-ended ``multiple 4koma`` does not match and stays unbounded.
_KOMA_COUNT_RE = re.compile(r"^(\d+)koma$")
_MULTI_VIEW_TAGS = frozenset({"multiple views", "multiple_views"})

# Panel layouts: a comic page draws the same character once per panel, so like
# ``multiple views`` its girls-count counts *characters*, not bindable subjects
# — ``1girl, 2koma`` is routinely two. Without this a comic page fails the
# candidate prefilter as ``single-subject``: 22 of the corpus's 26 comic pages
# that carry no ``multiple views`` tag, including clean vertical 2-panel pages
# whose panels differ exactly the way clauses are good at.
#
# ``page number`` is deliberately **excluded** despite tagging 15 more images.
# It marks a scanned art-book page, not a layout — the images it catches are
# single illustrations with a number in the margin (``mignon/10831765``), so it
# is a false signal, not a weak one.
_PANEL_LAYOUT_TAGS = frozenset(
    {
        "comic",
        "silent comic",
        "silent_comic",
        "sequential",
        "2koma",
        "3koma",
        "4koma",
        "multiple 4koma",
        "multiple_4koma",
    }
)

# Every layout tag that decouples the girls-count from the bindable-subject
# count. Both branches of the prefilter and the count check read this.
_LAYOUT_TAGS = _MULTI_VIEW_TAGS | _PANEL_LAYOUT_TAGS


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
        allow_identity: bool = True,
        bag_gated_identity: bool = True,
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

        ``allow_identity=False`` suppresses the hair/eye/hairstyle groups
        entirely. It is set for a **body-part crop**, which has no head in it:
        those groups have no evidence to read, the tagger emits a guess anyway,
        and discriminative-only then *promotes* the guess precisely because it
        disagrees with the full-body crop. Measured on ama_mitsuki, every part
        crop came back with a hair color and an eye color, all invented.

        ``bag_gated_identity`` (on by default) applies the milder form of the
        same rule to *every* crop: for a group in ``_BAG_GATED_GROUPS`` the flat
        bag outranks the tagger, so a clause carries a hair color the caption
        named or none at all. See that constant for the measurement.
        """
        out: list[str] = []
        seen: set[str] = set()
        taken_groups: set[str] = set()
        blocked = shared if discriminative_only else frozenset()
        # Which identity groups the caption has already spoken for. A crop may
        # only pick from those members; see ``bag_gated`` in ``add``.
        bag_members = (
            {
                group: {t for t in flat_bag if self.group_of(t) == group}
                for group in _BAG_GATED_GROUPS
            }
            if bag_gated_identity
            else {}
        )

        def add(tag: str) -> bool:
            if not tag or tag in seen or tag in blocked:
                return False
            group = self.group_of(tag)
            if group in self.exclusive_groups and group in taken_groups:
                return False  # one hair color / one eye color per subject
            if not allow_identity and group in _IDENTITY_GROUPS:
                return False  # no head in this crop — nothing to read it off
            if bag_members.get(group) and tag not in flat_bag:
                return False  # the caption named this attribute; it wins
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
    the count-consistency check then trusts detection instead of skipping.

    A **layout** tag (``_LAYOUT_TAGS``: ``multiple views`` or a comic-panel tag)
    always forces ``None`` even when the caption also carries a girls-count,
    because the count tags how many *characters* are drawn while each view or
    panel is its own bindable subject — ``1girl, multiple views`` is routinely
    four, ``1girl, 2koma`` is two.

    ``multiple girls`` and the open-ended ``N+girls`` crowd tag are ``None`` too
    — an exact match against "six or more" can only ever fail.
    """
    tags = {t.strip().lower() for t in parse_caption(caption).flat_tags}
    if tags & _LAYOUT_TAGS:
        return None
    counts = [int(m.group(1)) for t in tags if (m := _GIRLS_COUNT_RE.match(t))]
    if not counts and (
        "multiple girls" in tags or any(map(_OPEN_GIRLS_RE.match, tags))
    ):
        return None
    return max(counts) if counts else 0


def caption_panel_ceiling(caption: str) -> int | None:
    """Most bindable subjects an ``Nkoma`` page can hold, or ``None`` if unbounded.

    A layout tag makes :func:`caption_subject_count` return ``None`` — the
    girls-count no longer bounds anything, because the same girl is drawn once
    per panel. That waives the count check entirely, and on a comic page the
    check is exactly what used to catch a subject detected twice:
    ``kase_daiki/11645055`` is a 2-panel page with one girl per panel that SAM3
    returns **three** boxes for, the bottom girl split into an overlapping pair
    at IoMin 0.99 with a shredded mask on the second.

    ``Nkoma`` names the panel count, so the ceiling is
    ``panels × (girls + boys)`` — every panel drawing every character at once.
    That is generous by construction and still catches the split: a
    ``1girl, 2koma`` page tops out at 2. Plain ``comic`` carries no panel count
    and stays unbounded, as does ``multiple views``.

    ``None`` whenever any term is unknown (no koma tag, or an open-ended crowd
    count) — an unbounded check can only produce false skips.
    """
    tags = {t.strip().lower() for t in parse_caption(caption).flat_tags}
    panels = [int(m.group(1)) for t in tags if (m := _KOMA_COUNT_RE.match(t))]
    if not panels:
        return None
    girls = [int(m.group(1)) for t in tags if (m := _GIRLS_COUNT_RE.match(t))]
    if not girls and ("multiple girls" in tags or any(map(_OPEN_GIRLS_RE.match, tags))):
        return None
    boys = caption_boy_count(caption)
    if boys is None:
        return None
    # A page with no counted character at all still draws somebody per panel.
    per_panel = max(max(girls, default=0) + boys, 1)
    return max(panels) * per_panel


def caption_boy_count(caption: str) -> int | None:
    """How many *male* subjects the caption claims — the count check's slack.

    The SAM3 ``girl`` prompt does not reliably exclude males: on the same corpus
    it detects the boy in some images and not in others, so neither counting him
    nor ignoring him works as an equality. The count gate therefore accepts the
    range ``girls .. girls + boys``. ``None`` = "some boys, count unknown",
    which drops the upper bound entirely.
    """
    tags = {t.strip().lower() for t in parse_caption(caption).flat_tags}
    counts = [int(m.group(1)) for t in tags if (m := _BOYS_COUNT_RE.match(t))]
    if not counts and ("multiple boys" in tags or any(map(_OPEN_BOYS_RE.match, tags))):
        return None
    return max(counts) if counts else 0


def is_candidate(caption: str) -> tuple[bool, str]:
    """Should this caption go through detection? Returns ``(ok, reason)``."""
    if has_clauses(caption):
        return False, "already-has-clauses"
    tags = {t.strip().lower() for t in parse_caption(caption).flat_tags}
    if tags & _MULTI_VIEW_TAGS:
        return True, "multiple-views"
    if tags & _PANEL_LAYOUT_TAGS:
        return True, "panel-layout"
    expected = caption_subject_count(caption)
    if expected is None or expected > 1:
        return True, "multi-girl"
    return False, "single-subject"


# ---------------------------------------------------------------------------
# Detection plumbing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Detection:
    """One detected subject: box in pixels, score, and optional instance mask.

    ``source`` records which detector pass produced the box — ``"subject"`` for
    the ``girl`` prompt, the part prompt itself for a body-part fallback box.
    Carried into the report so a reviewer can tell the two apart.
    """

    box: tuple[float, float, float, float]
    score: float
    mask: np.ndarray | None = None
    source: str = "subject"


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def box_area(box: Sequence[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def box_containment(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over the *smaller* box — how nested the pair is.

    IoU is blind to nesting: a box wholly inside another scores ``area_small /
    area_large``, which is tiny exactly when the size gap is large. Both
    over-detection families this pipeline hits are nested, not overlapping — an
    inset (a character icon on a phone screen inside the main subject, IoU
    0.003) and a *group* box spanning every subject (IoU 0.44 vs. each member).
    Containment scores both at ~1.0.

    **Suppressing on it is nonetheless off by default**, because a *real* second
    subject is just as nested: one girl standing in front of another, an
    embrace, a background figure inside a foreground figure's box. Ablated over
    the 34 rows that regressed when it was first enabled, 32 recover with the
    rule off — the corpus has far more legitimately-nested subjects than group
    boxes. Kept as an opt-in knob; the inset half of the problem is handled by
    :func:`drop_small_boxes` instead, and a surviving group box costs one
    ``count-mismatch`` skip, which is the safe direction.
    """
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    smallest = min(box_area(a), box_area(b))
    return (ix * iy) / smallest if smallest > 0 else 0.0


def dedupe_detections(
    detections: Iterable[Detection],
    iou_threshold: float,
    containment_threshold: float = 1.01,
) -> list[Detection]:
    """Greedy IoU + containment suppression, highest score first.

    A threshold above 1.0 disables the containment rule — a box can never be
    more than fully inside another — leaving plain-IoU behaviour.
    """
    ranked = sorted(detections, key=lambda d: -d.score)
    keep: list[Detection] = []
    for det in ranked:
        if any(
            box_iou(det.box, k.box) >= iou_threshold
            or box_containment(det.box, k.box) >= containment_threshold
            for k in keep
        ):
            continue
        keep.append(det)
    return keep


def merge_part_detections(
    subjects: Sequence[Detection],
    parts: Iterable[Detection],
    *,
    iou_threshold: float,
    containment_threshold: float,
) -> list[Detection]:
    """Add body-part boxes that the subject prompt missed, never displacing one.

    The failure this exists for: a sheet whose panels are headless close-ups of
    a hip / crotch / backside next to one small full body. SAM3's ``girl``
    prompt sees only the full body, so the image dies on ``too-few-instances``
    with its two most attribute-dense panels never tagged.

    Containment is applied here even though :func:`dedupe_detections` leaves it
    off by default, and the asymmetry is the point. That rule is off globally
    because a *subject* nested in another subject is routinely real (one girl in
    front of another). A **part** nested in a subject never is — an ``ass`` box
    inside a girl box is that same girl's backside, a second position for a
    subject that already has one. Typing the rule to the part pass gets the
    duplicate suppression without the 32 real subjects the global rule cost.

    Subjects are kept unconditionally and win every tie; parts are considered
    highest-score first and tested against everything kept so far, so two part
    boxes on the same panel collapse to one.
    """
    keep = list(subjects)
    for det in sorted(parts, key=lambda d: -d.score):
        if any(
            box_iou(det.box, k.box) >= iou_threshold
            or box_containment(det.box, k.box) >= containment_threshold
            for k in keep
        ):
            continue
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
    source: str = "subject"


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
    min_area_frac: float = 0.005
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
    bag_gated_identity: bool = True


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

    1. **Lower the score threshold** — recovers an extreme close-up scored under
       the 0.5 gate.
    2. **Body-part prompts** (``part_detect_fn(image, prompt, threshold)``, when
       supplied) — recovers a panel the subject prompt cannot see at any
       threshold because it has no head. Merged via
       :func:`merge_part_detections`, which never displaces a subject box.

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


def drop_small_boxes(
    detections: Iterable[Detection],
    image_size: tuple[int, int],
    min_area_frac: float,
) -> list[Detection]:
    """Discard boxes too small to be a bindable subject.

    A detection covering 0.3% of the canvas is an inset — a character drawn on a
    phone screen, a poster, a chibi in a corner — not a subject a position clause
    can meaningfully describe.
    """
    if min_area_frac <= 0:
        return list(detections)
    floor = min_area_frac * image_size[0] * image_size[1]
    return [d for d in detections if box_area(d.box) >= floor]


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
    flat_bag = frozenset(t.strip().lower() for t in parsed.flat_tags)
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
            allow_identity=det.source == "subject",
            bag_gated_identity=options.bag_gated_identity,
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
        if token_count_fn is not None and proposal.proposed:
            proposal.tokens = token_count_fn(proposal.proposed)
        if apply:
            caption_path.write_text(proposal.proposed, encoding="utf-8")
            stats.written += 1

    return rows, stats
