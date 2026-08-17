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

Nothing is destroyed by the move — a moved tag is still in the caption, inside a
clause — and :func:`library.captioning.position_clauses.flatten_caption` merges
it back, so an ``--apply`` run is reversible.

Two rules bound what may leave the bag, because a wrong move is worse than a
wrong clause (it makes the caption assert that the *other* subjects lack the
attribute):

* **Character-invariant groups need corroboration.** Hair color, eyes, body
  shape, species … are properties of a *character*, not of a view, so on a
  ``1girl, multiple views`` sheet they are true of every panel. Such a tag may
  only move when the bag names **two or more** values of that group (see
  ``_CHARACTER_INVARIANT_GROUPS``) — i.e. the caption is already enumerating
  per-subject values and binding them loses nothing.
* **Attribution margin.** The winning crop's probability must clear every other
  crop's by ``attribution_margin``, so a tag the tagger nearly kept on a second
  subject stays in the bag (and stays duplicated in the clause).

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
    flatten_caption,
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

# Groups whose value belongs to a **character**, not to a view of one. The v2
# rewrite treats them specially: on a ``1girl, multiple views`` sheet every panel
# is the same girl, so binding ``aqua hair`` to one view and removing it from the
# bag makes the caption claim the other views are *not* aqua-haired. Outfit /
# pose / expression / framing groups carry no such implication — a maid view and
# a bunny view genuinely differ — so they move freely.
#
# The corroboration rule: a tag in one of these groups may leave the bag only
# when the bag names **≥2 distinct values of that group**. A caption listing
# ``black hair, white hair, pink hair`` is already enumerating per-subject values
# and gains from binding them; a caption listing one hair color is describing the
# character, and that value stays flat. Deliberately evidence-based rather than
# count-based: 219 of the 373 first-sweep proposals carry no girls-count tag at
# all, so a ``detected == characters`` gate would pin nearly everything.
_CHARACTER_INVARIANT_GROUPS = frozenset(
    {
        "hair_color",
        "hair_length",
        "hairstyle",
        "eye_color",
        "eye_shape",
        "face_features",
        "age",
        "gender",
        "skin",
        "body_shape",
        "species_nonhuman",
        "animal_parts",
    }
)

# On a repeated-subject layout (``_LAYOUT_TAGS`` — a ``multiple views`` sheet or
# a comic page) the subjects are not different characters; they are the *same*
# character drawn from several angles, in several outfits, or once per panel. No
# clause there may carry a trait the character owns. That is a stronger rule than
# the corroboration gate above, which only governs whether a tag may LEAVE the
# bag: here the tag may not enter the clause at all, because a per-view emission
# is either redundant (every view really does share it, and discriminative-only
# was supposed to catch that) or a crop hallucination (a view the tagger
# disagreed with, which discriminative-only actively *promotes*).
#
# ``body_parts`` joins the character-invariant groups for this rule and this rule
# only. Anatomy is owned by the character the same way hair color is — a girl
# does not grow a navel between panel 1 and panel 3 — but its *visibility*
# genuinely varies with the view, so it stays freely bindable on a real
# multi-character image. Measured on the first full-corpus v2 dry run, the 157
# ``multiple views`` proposals emitted 3201 clause tags of which 1429 (45%) were
# view-invariant: 445 ``body_parts``, 330 ``hairstyle``, 127 ``eye_color``, 118
# ``body_shape``, 113 ``hair_color``, 105 ``hair_length``, the rest spread over
# skin / animal_parts / face_features / age, plus 25 character names. The 23
# comic-panel proposals ran 33% view-invariant on the same measure. What
# survives — outfit, pose, expression, framing — is exactly what one view or
# panel has and another does not.
_VIEW_INVARIANT_GROUPS = _CHARACTER_INVARIANT_GROUPS | {"body_parts"}

# …and the exception to the corroboration rule. Booru tags a *single* character
# with two hair colors when the hair itself is two-toned, so the "≥2 values"
# evidence is explained without there being two subjects. These markers are
# ungrouped in ``groups.yaml`` (checked), hence a plain name set rather than
# group membership: when one is in the bag, that group is pinned flat.
_MULTI_VALUE_MARKERS: Mapping[str, frozenset[str]] = {
    "hair_color": frozenset(
        {
            "multicolored hair",
            "two-tone hair",
            "gradient hair",
            "streaked hair",
            "colored inner hair",
            "split-color hair",
            "rainbow hair",
            "colored tips",
            "multicolored bangs",
            "alternate hair color",
        }
    ),
    "eye_color": frozenset(
        {
            "heterochromia",
            "multicolored eyes",
            "gradient eyes",
            "two-tone eyes",
        }
    ),
}

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
        view_invariant: bool = False,
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

        ``view_invariant`` is the repeated-subject-layout form, and it is the
        strongest of the three: the subjects are one character drawn several
        times, so the clause drops the character name **and** every
        ``_VIEW_INVARIANT_GROUPS`` trait — appearance and anatomy alike — and
        keeps only what a view or panel can differ in. See that constant for the
        measurement; :func:`is_repeated_subject_layout` decides when it applies.
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
            # Copyright / artist / metadata / deprecated describe the *image*.
            # Checked here rather than only on the ranked path below, because
            # they can be grouped: ``light brown hair`` is a deprecated alias
            # that ``groups.yaml`` still files under ``hair_color``, so it rode
            # the priority path straight into a clause on 4 images of the first
            # full-corpus dry run.
            if tag in self.excluded:
                return False
            group = self.group_of(tag)
            if group in self.exclusive_groups and group in taken_groups:
                return False  # one hair color / one eye color per subject
            if not allow_identity and group in _IDENTITY_GROUPS:
                return False  # no head in this crop — nothing to read it off
            if view_invariant and group in _VIEW_INVARIANT_GROUPS:
                return False  # same girl in every view/panel — the bag owns this
            if bag_members.get(group) and tag not in flat_bag:
                return False  # the caption named this attribute; it wins
            seen.add(tag)
            if group:
                taken_groups.add(group)
            out.append(tag)
            return True

        # 1. Character name. A name the caption never claimed is a crop
        #    hallucination, so by default it must appear in the flat bag.
        #
        #    Skipped entirely on a repeated-subject layout: every view is the
        #    same girl, so a bound name says the *other* views are somebody
        #    else. Shared-tag suppression already hides the name when all crops
        #    agree — which means the only names that got through were the ones a
        #    crop missed. All 16 such ``multiple views`` rows in the first
        #    full-corpus dry run were single-character sheets (``hatsune miku``
        #    bound to 2 of 4 views).
        names = (
            []
            if view_invariant
            else sorted(
                (
                    t
                    for t in kept
                    if t in self.characters and kept[t] >= name_confidence
                ),
                key=lambda t: -kept[t],
            )
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


def is_repeated_subject_layout(caption: str) -> bool:
    """Is this one character drawn several times, rather than several characters?

    Any ``_LAYOUT_TAGS`` member says yes — the same set that decouples the
    girls-count from the bindable-subject count in :func:`caption_subject_count`,
    and for the same reason. ``multiple views`` is the clean case (an outfit
    sheet, a turnaround), but an ``Nkoma`` page or a ``comic`` is the same
    situation panel-by-panel: the girl in panel 3 is the girl in panel 1, drawn
    again. Whatever belongs to *her* therefore discriminates nothing between
    subjects, and :meth:`ClauseVocabulary.select` drops the whole class
    (``view_invariant``).

    A comic can of course introduce a new character mid-page, which a turnaround
    cannot — but that only makes a bound trait *sometimes* right instead of
    never, and the tags this suppresses are the ones the crop tagger is worst at
    (a name or a hair color the other panels' crops disagreed with). The bag
    keeps every one of them either way; only the per-panel binding is dropped.
    """
    tags = {t.strip().lower() for t in parse_caption(caption).flat_tags}
    return bool(tags & _LAYOUT_TAGS)


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


@dataclass(frozen=True)
class MovedTag:
    """One flat-bag tag the rewrite bound to a position and removed from the bag."""

    tag: str
    position: str
    margin: float


@dataclass(frozen=True)
class RemovalPlan:
    """What the rewrite moves, and why it declined to move the rest.

    ``blocked`` maps a bag tag that *reached* a clause but stays flat to the rule
    that kept it there — the review artifact for tuning the two safety rules.
    """

    moved: tuple[MovedTag, ...] = ()
    blocked: Mapping[str, str] = field(default_factory=dict)


def _score_of(
    scores: Mapping[str, float], kept: Mapping[str, float], tag: str
) -> float:
    """This crop's probability for ``tag``.

    ``predict`` returns ``scores`` for the *whole* vocabulary, which is what the
    margin needs — the runner-up crop's probability is interesting precisely when
    it fell below the keep threshold. Falls back to ``kept`` (0.0 for a crop that
    did not keep the tag) when a caller supplies no ``scores``, which only the
    unit-test stubs do.
    """
    if tag in scores:
        return float(scores[tag])
    return float(kept.get(tag, 0.0))


def plan_bag_removals(
    flat_tags: Sequence[str],
    clause_tags: Sequence[Sequence[str]],
    positions: Sequence[str],
    kept_sets: Sequence[Mapping[str, float]],
    score_sets: Sequence[Mapping[str, float]],
    *,
    vocabulary: ClauseVocabulary,
    margin: float,
) -> RemovalPlan:
    """Decide which flat-bag tags the clauses have earned the right to take.

    A tag moves out of the bag when all four hold:

    1. **It is not a character name.** The cast list stays flat and is *also*
       bound — the hand-written convention, measured (see the ``character-name``
       branch below).
    2. **It reached exactly one clause.** Two clauses claiming it means the
       attribute is shared, and a shared attribute belongs to the bag.
    3. **Corroboration**, for a character-invariant group: the bag names ≥2
       values of that group, with no two-tone marker to explain them away. See
       ``_CHARACTER_INVARIANT_GROUPS``.
    4. **Margin**: the winning crop beats every other crop's probability for the
       tag by ``margin``. A tag the tagger nearly kept on a second subject is a
       shared attribute the threshold happened to split, and removing it would
       make the caption deny it of that subject.

    Failing any of them is not an error — the tag simply stays in the bag *and*
    in its clause, which is exactly v1's additive behaviour for that one tag.
    """
    bag: dict[str, str] = {}
    for tag in flat_tags:
        bag.setdefault(tag.strip().lower(), tag)

    where: dict[str, list[int]] = {}
    for i, tags in enumerate(clause_tags):
        for tag in tags:
            where.setdefault(tag.strip().lower(), []).append(i)

    # Census of the bag: which tags are characters (rule 1) and how many values
    # of each invariant group it names (rule 3).
    values_per_group: dict[str, set[str]] = {}
    names_in_bag: set[str] = set()
    for key in bag:
        group = vocabulary.group_of(key)
        if group in _CHARACTER_INVARIANT_GROUPS:
            values_per_group.setdefault(group, set()).add(key)
        if key in vocabulary.characters:
            names_in_bag.add(key)
    pinned_groups = {
        group for group, markers in _MULTI_VALUE_MARKERS.items() if markers & bag.keys()
    }

    moved: list[MovedTag] = []
    blocked: dict[str, str] = {}
    for key, indices in sorted(where.items()):
        if key not in bag:
            continue  # the clause tag was never in the bag — nothing to move
        if len(indices) != 1:
            blocked[key] = "multi-clause"
            continue
        group = vocabulary.group_of(key)
        if key in names_in_bag:
            # The cast list stays flat — this is the hand-written convention,
            # measured rather than assumed: across the 14 ground-truth captions,
            # 19 of 244 clause tags are also in the bag and **all 19 are
            # character names**; not one non-name attribute is duplicated. The
            # bag answers "who is in this image" (and is how a prompt summons
            # them), the clause answers "which one is where".
            blocked[key] = "character-name"
            continue
        if group in _CHARACTER_INVARIANT_GROUPS:
            if group in pinned_groups:
                blocked[key] = "two-tone-marker"
                continue
            if len(values_per_group.get(group, ())) < 2:
                blocked[key] = "sole-value"
                continue
        winner = indices[0]
        mine = _score_of(score_sets[winner], kept_sets[winner], key)
        rival = max(
            (
                _score_of(score_sets[j], kept_sets[j], key)
                for j in range(len(clause_tags))
                if j != winner
            ),
            default=0.0,
        )
        if mine - rival < margin:
            blocked[key] = "margin"
            continue
        moved.append(
            MovedTag(
                tag=bag[key],
                position=positions[winner],
                margin=round(mine - rival, 3),
            )
        )
    return RemovalPlan(moved=tuple(moved), blocked=blocked)


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
    # On a repeated-subject layout (``multiple views`` / comic panels), keep the
    # character's own traits — and her name — out of every clause: they belong
    # to the girl, not to a view of her.
    multi_view_gate: bool = True
    # v2: move an attributable tag out of the flat bag into its clause. False is
    # the additive v1 behaviour (bag untouched), kept for the training A/B.
    rewrite: bool = True
    # How far the winning crop must clear every other crop before a tag is
    # allowed to *leave* the bag. Only the removal is gated — a tag that fails
    # the margin still enters its clause, so the caption degrades to v1 for it.
    attribution_margin: float = 0.35


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
    score_sets = [dict(p.get("scores") or {}) for p in predictions]
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

    Under v2 the write is a **rewrite**, not an append — a bound tag leaves the
    flat bag. It is still recoverable (:func:`flatten_captions`), but it is not a
    no-op on the master, which is the other reason ``apply`` defaults off.

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
        if proposal.moved:
            stats.rewritten += 1
            stats.moved_tags += len(proposal.moved)
        for reason in proposal.pinned.values():
            stats.pin(reason)
        if token_count_fn is not None and proposal.proposed:
            proposal.tokens = token_count_fn(proposal.proposed)
        if apply:
            caption_path.write_text(proposal.proposed, encoding="utf-8")
            stats.written += 1

    return rows, stats


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

    Walks ``resized_dir`` and maps to the caption master exactly like
    :func:`run_position_captions`, so ``path_pattern`` means the same thing in
    both and the nested-symlink layout of ``image_dataset/`` is never globbed.

    Hand-written clauses are flattened too — the pass cannot tell them from
    generated ones, and that is a real loss of curation, hence the dry-run
    default.
    """
    from library.preprocess._dataset import walk_images

    stats = PositionCaptionStats()
    rows: list[dict] = []
    images = walk_images(resized_dir, recursive=True, pattern=path_pattern)
    stats.seen = len(images)
    for image_path in images:
        rel = image_path.relative_to(resized_dir).with_suffix(".txt")
        caption_path = source_dir / rel
        if not caption_path.exists():
            stats.skip("no-caption")
            continue
        original = caption_path.read_text(encoding="utf-8").strip()
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
            caption_path.write_text(flattened, encoding="utf-8")
            stats.written += 1
    return rows, stats
