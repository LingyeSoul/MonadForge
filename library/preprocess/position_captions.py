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
it back, so an ``--apply`` run is reversible. It is also written to the
**derived** caption layer (``post_image_dataset/resized/<rel>.txt``), never to
the hand-written master under ``image_dataset/``: clauses are generated data,
and the master stays the user's file. See :func:`run_position_captions` for how
the mirror pass keeps them across a re-correct.

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

# Tag groups that bind to a position; everything else (lighting, background,
# medium, interaction, …) stays in the flat bag. Drawn from the tagger's own
# ``groups.yaml`` rather than substring heuristics so the two can't drift.
# Almost all of them describe *one subject*; `framing` is the exception and
# describes one *view* — see its entry below.
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
        # how *this view* is cropped. The odd one out: not an attribute of the
        # girl but of the panel showing her, and the only group that answers
        # "this view is a headless close-up, not a whole figure". On a sheet of
        # one full body plus a hip/backside panel it is the tag that tells them
        # apart, and the tagger reads it off the crop cleanly (measured on
        # `ama_mitsuki`: `ass focus` 0.54-0.77 on body-part panels vs 0.000 on
        # the full-body ones, `full body` 0.87-1.0 the other way).
        "framing",
    }
)

# Grouped under `framing` but describing the *page*, not one view: `solo focus`
# is a statement about the other characters, `size difference` about a pair,
# `white border` about the canvas. Binding one to a view would be wrong, and v2
# would then MOVE it out of the flat bag — a caption regression, not merely a
# less-resolved clause. Kept out via `is_subject_tag`/`is_scene_tag` so they
# stay flat exactly as they are today.
_PAGE_LEVEL_FRAMING = frozenset({"solo focus", "size difference", "white border"})

# Emitted first, in this order, when present — the attributes that actually
# disambiguate a subject. Everything else follows, ranked.
#
# `framing` rides here, last, for the novel budget rather than the emission
# order: candidates are admitted bag-first and then only `max_novel_tags` (1)
# novel tags in candidate order, so a framing tag the caption never named loses
# the slot to whatever the crop scored higher — measured on `ama_mitsuki`, a
# hallucinated `torn clothes` beat `ass focus` (0.774) on a backside panel and
# the clause said nothing about being a backside panel. It sits after the
# identity groups because on a real multi-character image hair and eyes
# disambiguate harder; on a view layout those are gated out and framing leads
# by itself, which is exactly the population it exists for.
_PRIORITY_GROUPS = ("hair_color", "eye_color", "hair_length", "hairstyle", "framing")

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
# These three are the *hand-picked* additions. The gate's main body is derived,
# not listed: **every exclusive (softmax) subject group the vocabulary declares**
# is gated the same way, because an exclusive group holds exactly one value by
# construction — so a crop naming a second one is not extra detail, it is a
# contradiction. Hand-picking three left the rest open, and the 2026-08-17
# full-corpus dry run emitted 150 clause tags that contradicted a value the bag
# already named: 103 ``body_shape`` (bag ``flat chest`` → clause ``large
# breasts``), 39 ``fashion_style`` (``nude`` → ``completely nude``), plus
# ``species_nonhuman`` (``fox girl`` → ``cat girl``), ``age`` and ``gesture``.
# ``hair_length`` is the one member here that is *not* exclusive in
# ``groups.yaml``, which is why the hand list survives the derivation.
#
# Multi-value groups stay ungated, deliberately: ``hairstyle`` legitimately
# reveals a ``hair bun`` or ``sidelocks`` the booru caption never bothered to
# tag, and unlike a color it does not contradict what is there. See
# ``_gated_groups``.
_BAG_GATED_GROUPS = frozenset({"hair_color", "eye_color", "hair_length"})

# Exclusive groups the bag gate must NOT claim. The gate's premise — "the group
# holds one value, so a second contradicts the first" — holds for a *subject*
# attribute (a girl has one hair color) and fails for `framing`, which is a
# property of the view rather than the girl: a sheet's bag legitimately says
# `full body` for the standing panel while the backside panel is `ass focus`,
# and both are true at once. Without this exemption the gate silently pins
# every clause to whatever framing the caption happened to name first, so
# adding `framing` to SUBJECT_GROUPS would be inert.
_UNGATED_EXCLUSIVE_GROUPS = frozenset({"framing"})

# Groups whose value belongs to a **character**, not to a view of one — on a
# `1girl, multiple views` sheet, binding `aqua hair` to one view and removing it
# from the bag would claim the other views aren't aqua-haired. Outfit / pose /
# expression / framing carry no such implication, so those move freely.
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

# On a repeated-subject layout (`_LAYOUT_TAGS`) the subjects are one character
# drawn several times, so no clause may carry a trait the character owns — a
# stricter rule than the corroboration gate above (that one governs whether a
# tag may LEAVE the bag; this one blocks it from ENTERING a clause at all).
# Measured 45% of `multiple views` clause tags were view-invariant pre-gate (33%
# for comic panels); see docs/experimental/position_captions.md.
_VIEW_INVARIANT_GROUPS = _CHARACTER_INVARIANT_GROUPS

# `body_parts` used to join the set above and no longer does (`--gate_view_anatomy`
# restores it). Anatomy is owned by the character the way hair color is, but
# unlike hair color its *visibility* is a fact about the view: on a sheet of one
# girl from the front and the same girl from behind, `ass` is true of exactly one
# panel. Gating it cost the clause the one thing that separated those two views.
# The set is named rather than inlined so the A/B has something to flip.
_VIEW_ANATOMY_GROUPS = frozenset({"body_parts"})

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

    def gated_groups(self) -> frozenset[str]:
        """Groups where a clause may only pick a value the flat bag already named.

        The hand-picked ``_BAG_GATED_GROUPS`` plus every **exclusive** subject
        group the checkpoint declares. Derived rather than listed so the gate
        cannot drift from ``groups.yaml``: whatever the tagger models as a
        softmax over one subject is, by construction, a group where a second
        value is a contradiction rather than extra detail — minus
        ``_UNGATED_EXCLUSIVE_GROUPS``, where that reasoning does not hold
        because the group describes the view rather than the subject.
        """
        derived = _BAG_GATED_GROUPS | (self.exclusive_groups & SUBJECT_GROUPS)
        return derived - _UNGATED_EXCLUSIVE_GROUPS

    def is_subject_tag(self, tag: str) -> bool:
        if tag in _PAGE_LEVEL_FRAMING:
            return False
        return self.group_of(tag) in SUBJECT_GROUPS

    def is_scene_tag(self, tag: str) -> bool:
        """Grouped, but into a group that describes the scene, not a subject."""
        if tag in _PAGE_LEVEL_FRAMING:
            return True  # filed under `framing`, but about the page
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
        bind_framing: bool = True,
        bind_view_anatomy: bool = True,
        max_novel_tags: int = 1,
    ) -> list[str]:
        """Clause tags for one crop, ordered most-disambiguating first.

        ``kept`` / ``groups`` are the crop tagger's output; ``flat_bag`` is the
        image's existing caption (the curated ground truth for *what* is in the
        image — the crop only decides *where*); ``attributable`` is the set of
        tags this crop is the **only** one to keep; ``shared`` is the set *every*
        crop keeps.

        **The bag fills the clause first.** Candidates are ranked once, then
        admitted in two passes: everything already in ``flat_bag``, and only
        then up to ``max_novel_tags`` tags the caption never contained. The
        emitted order is the ranking, not the admission order, so a clause still
        reads hair-color-first.

        Suppression knobs, weakest to strongest: ``discriminative_only``
        (default) drops ``shared`` tags — a `multiple views` sheet repeats the
        same character/hair on every view, crowding out the outfit that
        differs. ``allow_identity=False`` (body-part crops) drops
        hair/eye/hairstyle outright — no head, no evidence. ``bag_gated_identity``
        (default) makes the flat bag outrank the tagger for any
        :meth:`gated_groups` member, once the bag has spoken for it.
        ``view_invariant`` (repeated-subject layout) is strongest: drops the
        name and every ``_VIEW_INVARIANT_GROUPS`` trait, keeping only what a
        view/panel can differ in — plus ``_VIEW_ANATOMY_GROUPS`` when
        ``bind_view_anatomy`` is off, which is what that gate used to include.
        """
        out: list[str] = []
        seen: set[str] = set()
        taken_groups: set[str] = set()
        blocked = shared if discriminative_only else frozenset()
        invariant_groups = _VIEW_INVARIANT_GROUPS
        if not bind_view_anatomy:
            invariant_groups = invariant_groups | _VIEW_ANATOMY_GROUPS
        # Gated groups the caption has already spoken for — see the
        # ``bag_members`` test in ``add``.
        bag_members = (
            {
                group: {t for t in flat_bag if self.group_of(t) == group}
                for group in self.gated_groups()
            }
            if bag_gated_identity
            else {}
        )

        def add(tag: str) -> bool:
            if not tag or tag in seen or tag in blocked:
                return False
            # Checked here (not only on the ranked path below) because an
            # excluded tag can still be grouped, e.g. a deprecated alias filed
            # under hair_color — it must not ride the priority path in. Same
            # shape for the page-level framing tags: `framing` is a priority
            # group, and the priority step reads the group's winner straight
            # off ``groups`` without ever consulting ``is_scene_tag``.
            if tag in self.excluded or tag in _PAGE_LEVEL_FRAMING:
                return False
            group = self.group_of(tag)
            if group in self.exclusive_groups and group in taken_groups:
                return False  # one hair color / one eye color per subject
            if not allow_identity and group in _IDENTITY_GROUPS:
                return False  # no head in this crop — nothing to read it off
            if view_invariant and group in invariant_groups:
                return False  # same girl in every view/panel — the bag owns this
            if not bind_framing and group == "framing":
                return False  # A side of the framing A/B
            if bag_members.get(group) and tag not in flat_bag:
                return False  # the caption named this attribute; it wins
            seen.add(tag)
            if group:
                taken_groups.add(group)
            out.append(tag)
            return True

        # ---- Rank the candidates (this is the *emitted* order) --------------
        candidates: list[str] = []

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
                candidates.append(name)
                break  # one identity per subject

        # 2. Exclusive-group winners (hair color, eye color, …). These are the
        #    softmax_when_solo groups, and a single-subject crop is exactly the
        #    condition under which they fire — the whole point of cropping.
        for group in _PRIORITY_GROUPS:
            members = sorted(
                (t for t in kept if self.group_of(t) == group),
                key=lambda t: -kept[t],
            )
            # A kept member the caption already named outranks the softmax
            # winner. Without this the gate and the winner fight each other: on
            # a gated group a novel winner is rejected by ``add`` and the group
            # then emits nothing, even though the crop also kept the very value
            # the bag listed. Reuse is the whole point, so it wins the slot.
            winner = next((t for t in members if t in flat_bag), None)
            if winner is None:
                # Group didn't fire (contaminated / multi-person crop): fall
                # back to the highest-probability kept member of that group.
                winner = groups.get(group) or (members[0] if members else None)
            if winner:
                candidates.append(winner)

        # 3. Everything else, ranking tags the caption already curated first.
        rest = [
            t
            for t in kept
            if t not in candidates
            and not is_count_tag(t)
            and not is_rating_tag(t)
            and not is_artist_tag(t)
            and t not in self.characters
            and t not in self.excluded
            and not self.is_scene_tag(t)
            and (self.is_subject_tag(t) or (t in flat_bag and t in attributable))
        ]
        rest.sort(key=lambda t: (t not in flat_bag, -kept[t]))
        candidates.extend(rest)

        # ---- Admit: the bag first, then a bounded number of novel tags ------
        rank = {tag: i for i, tag in enumerate(candidates)}
        novel_budget = max(0, max_novel_tags)
        for reuse_pass in (True, False):
            for tag in candidates:
                if len(out) >= max_tags:
                    break
                if (tag in flat_bag) is not reuse_pass:
                    continue
                if not reuse_pass and novel_budget <= 0:
                    break
                if add(tag) and not reuse_pass:
                    novel_budget -= 1
        out.sort(key=lambda t: rank[t])
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


def mask_box_fill(det: Detection) -> float | None:
    """Fraction of the detection's own box that its mask actually claims.

    ``None`` when the detection carries no mask (stub tests, part boxes) so
    callers can fall back to score-only behaviour.
    """
    if det.mask is None:
        return None
    mask = np.asarray(det.mask)
    if mask.ndim == 3:
        mask = mask[0]
    height, width = mask.shape
    x1, y1, x2, y2 = det.box
    window = (
        mask[
            max(0, int(y1)) : min(height, int(y2)),
            max(0, int(x1)) : min(width, int(x2)),
        ]
        > 0.5
    )
    return float(window.mean()) if window.size else 0.0


def dedupe_detections(
    detections: Iterable[Detection],
    iou_threshold: float,
    containment_threshold: float = 1.01,
    fill_ratio_threshold: float = 0.0,
) -> list[Detection]:
    """Greedy IoU + containment suppression, highest score first.

    A threshold above 1.0 disables the containment rule — a box can never be
    more than fully inside another — leaving plain-IoU behaviour.

    ``fill_ratio_threshold`` > 0 enables the mask-quality tie-break
    (``docs/experimental/multiview_audit.md`` §5.4 fixed the default at
    2.0): when a candidate collides with a kept box — the pair already judged
    to be the same object — and the candidate's :func:`mask_box_fill` beats the
    kept box's by at least this ratio, the candidate *replaces* the kept box
    instead of being dropped. SAM3's score is box-level confidence and says
    nothing about mask coherence, so a near-empty duplicate can outscore the
    clean mask by a hair and hand every downstream consumer a blank crop.
    Instance count is invariant by construction — the swap only changes which
    of two matched duplicates represents the object. This is deliberately NOT
    an absolute fill gate (settled negative: clean figures live at fill ~0.27);
    the ratio only ever compares the two halves of one matched pair. When
    either mask is missing, the pair falls back to score-only suppression.
    Single-pass: the swapped-in geometry is not re-checked against other kept
    boxes (bounded, order-stable; no cascade observed over the full corpus).
    """
    ranked = sorted(detections, key=lambda d: -d.score)
    keep: list[Detection] = []
    for det in ranked:
        hit = next(
            (
                i
                for i, k in enumerate(keep)
                if box_iou(det.box, k.box) >= iou_threshold
                or box_containment(det.box, k.box) >= containment_threshold
            ),
            None,
        )
        if hit is None:
            keep.append(det)
            continue
        if fill_ratio_threshold > 0:
            kept_fill = mask_box_fill(keep[hit])
            det_fill = mask_box_fill(det)
            if (
                kept_fill is not None
                and det_fill is not None
                and det_fill / max(kept_fill, 1e-9) >= fill_ratio_threshold
            ):
                keep[hit] = det
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
    """One flat-bag tag the rewrite bound to a position and removed from the bag.

    ``margin`` is the *relative* slack the move cleared, ``1 - rival/winner``
    over the two crops' probabilities — the same scale as
    ``attribution_margin``, so a reviewer can read a report row against the knob
    that let it through.
    """

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

    A tag moves out of the bag when all five hold:

    1. **It is not a character name.** The cast list stays flat and is *also*
       bound — the hand-written convention, measured (see the ``character-name``
       branch below).
    2. **It reached exactly one clause.** Two clauses claiming it means the
       attribute is shared, and a shared attribute belongs to the bag.
    3. **Corroboration**, for a character-invariant group: the bag names ≥2
       values of that group, with no two-tone marker to explain them away. See
       ``_CHARACTER_INVARIANT_GROUPS``.
    4. **Exclusive keep**: no *other* crop kept the tag. A crop that reached the
       tag's own calibrated threshold has the attribute, whatever the clause
       builder later did with it — a tag kept twice but bound once is a
       selection artifact (clause budget, discriminative filter, view gate), not
       an attribution.
    5. **Relative margin**: the runner-up's probability is below
       ``(1 - margin)`` of the winner's. Scored *relative to the winner*, not as
       an absolute gap, because the tagger's decision boundaries are per-tag and
       span ~0.05–0.85: an absolute gap pins a 0.34-vs-**0.000** call on a
       0.05-threshold tag (``sleeves past fingers``) while passing a
       0.99-vs-0.60 call on a high-threshold one. The ratio is scale-free, so
       one knob means the same thing across the vocabulary.

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
        others = [j for j in range(len(clause_tags)) if j != winner]
        if any(key in kept_sets[j] for j in others):
            blocked[key] = "multi-kept"
            continue
        mine = _score_of(score_sets[winner], kept_sets[winner], key)
        rival = max(
            (_score_of(score_sets[j], kept_sets[j], key) for j in others),
            default=0.0,
        )
        slack = 1.0 - rival / mine if mine > 0.0 else 0.0
        if slack < margin:
            blocked[key] = "margin"
            continue
        moved.append(
            MovedTag(
                tag=bag[key],
                position=positions[winner],
                margin=round(slack, 3),
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
            options.dedupe_fill_ratio,
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
    from library.preprocess._dataset import walk_images

    options = options or PositionCaptionOptions()
    stats = PositionCaptionStats()
    rows: list[ImageProposal] = []

    images = walk_images(resized_dir, recursive=True, pattern=path_pattern)
    stats.seen = len(images)

    for index, image_path in enumerate(images, 1):
        rel = image_path.relative_to(resized_dir).with_suffix(".txt")
        dst_caption = resized_dir / rel
        # Prefer the derived caption (already order-corrected, and carrying an
        # earlier run's clauses so `is_candidate` skips it) and fall back to the
        # master for an image the caption step has not mirrored yet.
        caption_path = dst_caption if dst_caption.exists() else source_dir / rel
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
    from library.preprocess._dataset import walk_images

    stats = PositionCaptionStats()
    rows: list[dict] = []
    images = walk_images(resized_dir, recursive=True, pattern=path_pattern)
    stats.seen = len(images)
    for image_path in images:
        rel = image_path.relative_to(resized_dir).with_suffix(".txt")
        dst_caption = resized_dir / rel
        caption_path = dst_caption if dst_caption.exists() else source_dir / rel
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
            _write_derived_caption(dst_caption, flattened)
            stats.written += 1
    return rows, stats
