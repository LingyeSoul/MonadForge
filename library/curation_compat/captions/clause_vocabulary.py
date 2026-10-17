"""Which tags may enter a position clause, and in what order.

The subject/scene split, the gates that bound what a clause may assert, and
:class:`ClauseVocabulary`, which ranks and admits one crop's tags.

The policy is data: :class:`ClauseGroups` is loaded from
``configs/clause_vocabulary.yaml``, whose group names come from the tagger
checkpoint's own ``groups.yaml``. A name the checkpoint doesn't declare is
warned about at load — a typo would otherwise silently disable a gate.

Pure stdlib + yaml (no torch). Per-rule evidence and the knob table live in
``docs/position_captions.md``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import yaml

from library.curation_compat._env import resolve_path
from library.curation_compat.captions.taxonomy import is_artist_tag, is_count_tag, is_rating_tag

logger = logging.getLogger(__name__)

# Resolved against the curation home (``anime_tools._env.curation_home``): a
# ``configs/clause_vocabulary.yaml`` there overrides the package default at
# :data:`PACKAGED_CLAUSE_GROUPS_CONFIG`.
CLAUSE_GROUPS_CONFIG = "configs/clause_vocabulary.yaml"
PACKAGED_CLAUSE_GROUPS_CONFIG = (
    Path(__file__).resolve().parent / "data" / "clause_vocabulary.yaml"
)


def clause_groups_config_path(path: str | Path = CLAUSE_GROUPS_CONFIG) -> Path:
    """``<home>/configs/clause_vocabulary.yaml`` when present, else the copy
    shipped inside the package."""
    resolved = resolve_path(path)
    if resolved.exists() or Path(path) != Path(CLAUSE_GROUPS_CONFIG):
        return resolved
    return PACKAGED_CLAUSE_GROUPS_CONFIG


@dataclass(frozen=True)
class ClauseGroups:
    """The clause policy as loaded from ``configs/clause_vocabulary.yaml``.

    Group-name sets, except :attr:`page_level_framing` and
    :attr:`multi_value_markers`, which name individual *tags*.
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
    """Read a clause-policy YAML.

    ``view_invariant_groups`` defaults to ``character_invariant_groups`` — the
    same rule at two strengths (may a tag LEAVE the bag vs ENTER a clause).
    """
    with open(clause_groups_config_path(path), encoding="utf-8") as f:
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


@cache
def default_clause_groups() -> ClauseGroups:
    """The shipped policy, read once per process."""
    return load_clause_groups()


@dataclass(frozen=True)
class ClauseVocabulary:
    """Which tags may enter a clause, and in what order.

    ``tag_to_group`` from the tagger checkpoint's ``groups.yaml``;
    ``characters``/``excluded`` from its ``vocab.json``. Ungrouped tags are
    admitted only via the attributable + in-the-caption path (:meth:`select`).
    ``excluded`` = image-level categories. ``exclusive_groups`` = softmax groups
    where only one member may enter a clause.
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
        group the checkpoint declares, minus ``ungated_exclusive_groups``, where
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
        (what's in the image; the crop only decides *where*); ``attributable`` =
        tags only this crop kept; ``shared`` = tags every crop kept.

        Candidates are ranked once, admitted bag-first then up to
        ``max_novel_tags`` novel ones; a novel tag can never later be *moved*
        by ``clause_rewrite.plan_bag_removals``.

        Suppression knobs, weakest to strongest: ``discriminative_only`` drops
        ``shared`` tags; ``allow_identity=False`` (body-part crops) drops
        hair/eye/hairstyle, there being no head to read them off;
        ``bag_gated_identity`` makes the flat bag outrank the tagger for any
        :meth:`gated_groups` member it has spoken for; ``view_invariant``
        (repeated-subject layout) is strongest, dropping the name and every
        view-invariant trait — plus the view-anatomy groups when
        ``bind_view_anatomy`` is off.
        """
        cfg = self.clause_groups
        out: list[str] = []
        seen: set[str] = set()
        taken_groups: set[str] = set()
        blocked = shared if discriminative_only else frozenset()
        invariant_groups = cfg.view_invariant
        if not bind_view_anatomy:
            invariant_groups = invariant_groups | cfg.view_anatomy
        # Gated groups the caption has already spoken for.
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
            # Checked here, not only on the ranked path below: an excluded tag
            # can still be grouped (a deprecated alias filed under hair_color)
            # and must not ride the priority path in. Same for page-level
            # framing, since `framing` is a priority group.
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
        #    Skipped on a repeated-subject layout, where a bound name would
        #    claim the other views are someone else.
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

        # 2. Exclusive-group winners (hair color, eye color, …) — the
        #    softmax_when_solo groups, which fire on a single-subject crop.
        for group in cfg.priority:
            members = sorted(
                (t for t in kept if self.group_of(t) == group),
                key=lambda t: -kept[t],
            )
            # A kept member the caption already named outranks the softmax
            # winner: otherwise, on a gated group, ``add`` rejects the novel
            # winner and the group emits nothing.
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

    ``clause_groups`` overrides the shipped policy. Whichever policy is used is
    checked against the checkpoint's declared groups; an unknown name is logged,
    since it would silently disable a gate.
    """
    from library.curation_compat.captions import tag_groups as tg
    from library.curation_compat.captions.vocab_io import (
        load_vocab,
        names_by_category,
        names_in_categories,
    )

    ckpt = Path(ckpt_dir)
    vocab = load_vocab(ckpt)
    characters = frozenset(names_by_category(vocab, ("character",))["character"])
    excluded = names_in_categories(
        vocab, ("copyright", "artist", "metadata", "deprecated")
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
