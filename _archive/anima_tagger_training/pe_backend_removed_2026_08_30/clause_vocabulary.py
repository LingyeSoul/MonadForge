"""Which tags may enter a position clause, and in what order.

The tag-*group* policy behind the position-clause rewrite
(``make caption-position``): the subject/scene split, the gates that bound what
a clause may assert, and :class:`ClauseVocabulary`, which ranks and admits one
crop's tags. The clause *grammar* (parse/compose/position words) lives in
:mod:`library.captioning.position_clauses`, the bag→clause move rules in
:mod:`library.captioning.clause_rewrite`, and the pixels-and-detection half of
the pipeline in :mod:`library.preprocess.position_captions`.

The policy itself is data, not code: :class:`ClauseGroups` is loaded from
``configs/clause_vocabulary.yaml`` (:data:`CLAUSE_GROUPS_CONFIG`), whose group
names come from the tagger checkpoint's own ``groups.yaml`` — so the rules can
be read and A/B'd without editing Python, and cannot drift from the model that
produced the tags. A group name the checkpoint doesn't declare is warned about
at load: a typo would otherwise silently disable a gate.

Pure stdlib + yaml by design (no torch): every caption-side consumer must import
without a model. Per-rule evidence and the knob table live in
``docs/experimental/position_captions.md``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import yaml

from library.captioning.taxonomy import is_artist_tag, is_count_tag, is_rating_tag
from library.env import resolve_under_home

logger = logging.getLogger(__name__)

# Repo-relative, resolved against the repo home (not the CWD) so an embedder
# importing this from anywhere gets the same rules — see library.env.
CLAUSE_GROUPS_CONFIG = "configs/clause_vocabulary.yaml"


@dataclass(frozen=True)
class ClauseGroups:
    """The clause policy as loaded from ``configs/clause_vocabulary.yaml``.

    Group-name sets, except :attr:`page_level_framing` and
    :attr:`multi_value_markers`, which name individual *tags*. Field-by-field
    rationale lives in the YAML — it is the source of truth a reviewer reads.
    """

    subject: frozenset[str] = frozenset()
    page_level_framing: frozenset[str] = frozenset()
    # Ordered: the emission order of the priority step.
    priority: tuple[str, ...] = ()
    identity: frozenset[str] = frozenset()
    bag_gated: frozenset[str] = frozenset()
    ungated_exclusive: frozenset[str] = frozenset()
    character_invariant: frozenset[str] = frozenset()
    view_invariant: frozenset[str] = frozenset()
    view_anatomy: frozenset[str] = frozenset()
    multi_value_markers: Mapping[str, frozenset[str]] = field(default_factory=dict)

    def unknown_groups(self, declared: frozenset[str]) -> frozenset[str]:
        """Group names in this policy that the checkpoint doesn't declare."""
        named = (
            self.subject
            | frozenset(self.priority)
            | self.identity
            | self.bag_gated
            | self.ungated_exclusive
            | self.character_invariant
            | self.view_invariant
            | self.view_anatomy
        )
        return frozenset(named - declared)


def load_clause_groups(path: str | Path = CLAUSE_GROUPS_CONFIG) -> ClauseGroups:
    """Read a clause-policy YAML. See ``configs/clause_vocabulary.yaml``.

    ``view_invariant_groups`` defaults to ``character_invariant_groups`` when
    absent — the shipped configuration, where the two sets are the same rule
    read at two different strengths (may a tag LEAVE the bag vs may it ENTER a
    clause at all).
    """
    with open(resolve_under_home(path), encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    def names(key: str) -> frozenset[str]:
        return frozenset(str(v) for v in (raw.get(key) or ()))

    character_invariant = names("character_invariant_groups")
    return ClauseGroups(
        subject=names("subject_groups"),
        page_level_framing=names("page_level_framing"),
        priority=tuple(str(v) for v in (raw.get("priority_groups") or ())),
        identity=names("identity_groups"),
        bag_gated=names("bag_gated_groups"),
        ungated_exclusive=names("ungated_exclusive_groups"),
        character_invariant=character_invariant,
        view_invariant=(
            names("view_invariant_groups")
            if raw.get("view_invariant_groups")
            else character_invariant
        ),
        view_anatomy=names("view_anatomy_groups"),
        multi_value_markers={
            str(group): frozenset(str(t) for t in (tags or ()))
            for group, tags in (raw.get("multi_value_markers") or {}).items()
        },
    )


@lru_cache(maxsize=None)
def default_clause_groups() -> ClauseGroups:
    """The shipped policy, read once per process."""
    return load_clause_groups()


@dataclass(frozen=True)
class ClauseVocabulary:
    """Which tags may enter a clause, and in what order.

    ``tag_to_group`` from the tagger checkpoint's ``groups.yaml``;
    ``characters``/``excluded`` from its ``vocab.json``. Ungrouped tags are
    admitted only via the attributable + in-the-caption path (see
    :meth:`select`) — lets a curated compound like ``pink jacket`` bind while
    keeping ungrouped scene tags out. ``excluded`` = image-level categories
    (copyright/artist/metadata/deprecated) that would otherwise ride the ranked
    path in. ``exclusive_groups`` = softmax groups where only one member may
    enter a clause. ``clause_groups`` = the policy from
    ``configs/clause_vocabulary.yaml``; pass a :func:`load_clause_groups` result
    to run an alternative rule set.
    """

    characters: frozenset[str] = frozenset()
    excluded: frozenset[str] = frozenset()
    exclusive_groups: frozenset[str] = frozenset()
    tag_to_group: Mapping[str, str] = field(default_factory=dict)
    clause_groups: ClauseGroups = field(default_factory=default_clause_groups)

    def group_of(self, tag: str) -> str | None:
        return self.tag_to_group.get(tag)

    def gated_groups(self) -> frozenset[str]:
        """Groups where a clause may only pick a value the flat bag already named.

        The configured ``bag_gated_groups`` plus every **exclusive** subject
        group the checkpoint declares. Derived rather than listed so the gate
        cannot drift from ``groups.yaml``: whatever the tagger models as a
        softmax over one subject is, by construction, a group where a second
        value is a contradiction rather than extra detail — minus
        ``ungated_exclusive_groups``, where that reasoning does not hold because
        the group describes the view rather than the subject.
        """
        cfg = self.clause_groups
        derived = cfg.bag_gated | (self.exclusive_groups & cfg.subject)
        return derived - cfg.ungated_exclusive

    def is_subject_tag(self, tag: str) -> bool:
        if tag in self.clause_groups.page_level_framing:
            return False
        return self.group_of(tag) in self.clause_groups.subject

    def is_scene_tag(self, tag: str) -> bool:
        """Grouped, but into a group that describes the scene, not a subject."""
        if tag in self.clause_groups.page_level_framing:
            return True  # filed under `framing`, but about the page
        group = self.group_of(tag)
        return group is not None and group not in self.clause_groups.subject

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

        ``kept``/``groups`` = crop tagger output; ``flat_bag`` = curated caption
        (what's in the image; crop only decides *where*); ``attributable`` =
        tags only this crop kept; ``shared`` = tags every crop kept.

        Candidates are ranked once, admitted bag-first then up to
        ``max_novel_tags`` novel (caption never had) — a novel tag can never
        later be *moved* by
        :func:`library.captioning.clause_rewrite.plan_bag_removals`, so this
        bounds dead-weight invention without ever rescuing a crowded-out bag tag.

        Suppression knobs, weakest to strongest: ``discriminative_only``
        (default) drops ``shared`` tags — a `multiple views` sheet repeats the
        same character/hair on every view, crowding out the outfit that
        differs. ``allow_identity=False`` (body-part crops) drops
        hair/eye/hairstyle outright — no head, no evidence. ``bag_gated_identity``
        (default) makes the flat bag outrank the tagger for any
        :meth:`gated_groups` member, once the bag has spoken for it.
        ``view_invariant`` (repeated-subject layout) is strongest: drops the
        name and every configured view-invariant trait, keeping only what a
        view/panel can differ in — plus the view-anatomy groups when
        ``bind_view_anatomy`` is off, which is what that gate used to include.
        """
        cfg = self.clause_groups
        out: list[str] = []
        seen: set[str] = set()
        taken_groups: set[str] = set()
        blocked = shared if discriminative_only else frozenset()
        invariant_groups = cfg.view_invariant
        if not bind_view_anatomy:
            invariant_groups = invariant_groups | cfg.view_anatomy
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
            if tag in self.excluded or tag in cfg.page_level_framing:
                return False
            group = self.group_of(tag)
            if group in self.exclusive_groups and group in taken_groups:
                return False  # one hair color / one eye color per subject
            if not allow_identity and group in cfg.identity:
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
        #    Skipped entirely on a repeated-subject layout: every view is the
        #    same girl, so a bound name would claim the other views are someone
        #    else.
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
        for group in cfg.priority:
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


def load_clause_vocabulary(
    ckpt_dir: str | Path,
    clause_groups: ClauseGroups | None = None,
) -> ClauseVocabulary:
    """Build a :class:`ClauseVocabulary` from a tagger checkpoint directory.

    ``clause_groups`` overrides the shipped policy (see
    :func:`load_clause_groups`). Whatever policy is used is checked against the
    checkpoint's declared groups: an unknown name is a typo that would silently
    disable a gate, so it is logged rather than left to be discovered in a dry
    run's diff.
    """
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
    policy = clause_groups or default_clause_groups()
    groups_path = ckpt / "groups.yaml"
    tag_to_group: Mapping[str, str] = {}
    exclusive: frozenset[str] = frozenset()
    if groups_path.exists():
        groups = tg.load_groups(groups_path)
        tag_to_group = dict(groups.tag_to_group)
        exclusive = frozenset(
            g.name for g in groups.groups if g.mode in {"softmax", "softmax_when_solo"}
        )
        unknown = policy.unknown_groups(frozenset(g.name for g in groups.groups))
        if unknown:
            logger.warning(
                "%s names %d group(s) absent from %s — they match nothing and "
                "silently disable their rule: %s",
                CLAUSE_GROUPS_CONFIG,
                len(unknown),
                groups_path,
                ", ".join(sorted(unknown)),
            )
    return ClauseVocabulary(
        characters=characters,
        excluded=excluded,
        exclusive_groups=exclusive,
        tag_to_group=tag_to_group,
        clause_groups=policy,
    )
