"""v2's move rules: which flat-bag tags a clause has earned the right to take.

The rewrite half of the position-clause pipeline. v1 was additive (a tag stayed
in the bag *and* appeared in its clause); v2 **moves** an attributable tag so
each attribute is asserted exactly once — the hand-written convention. This
module owns the five rules that bound a move and the report of what they
blocked; :mod:`library.captioning.clause_vocabulary` decides what may enter a
clause in the first place, and
:mod:`library.preprocess.position_captions` drives both over the dataset.

Text and scores only — no pixels, no model. Per-rule evidence lives in
``docs/experimental/position_captions.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from library.captioning.clause_vocabulary import ClauseVocabulary


@dataclass(frozen=True)
class MovedTag:
    """One flat-bag tag the rewrite bound to a position and removed from the bag.

    ``margin`` is the relative slack the move cleared (``1 - rival/winner``),
    same scale as ``attribution_margin``.
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

    ``scores`` (whole-vocabulary) is what the margin needs, since the runner-up's
    probability matters even below its keep threshold. Falls back to ``kept``
    (0.0 if absent) when a caller supplies no ``scores`` (unit-test stubs only).
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

    A tag moves out of the bag when all five hold: (1) not a character name —
    the cast list stays flat *and* bound; (2) reached exactly one clause — two
    means shared, so it belongs to the bag; (3) corroboration, for a
    character-invariant group — the bag names >=2 values of that group with no
    two-tone marker explaining them away (``character_invariant_groups`` /
    ``multi_value_markers`` in ``configs/clause_vocabulary.yaml``);
    (4) exclusive keep — no *other* crop kept the tag, else it's a selection
    artifact, not an attribution; (5) relative margin — the runner-up's
    probability is below ``(1 - margin)`` of the winner's (relative, not
    absolute, since per-tag thresholds span ~0.05-0.85).

    Failing any rule is not an error — the tag stays in the bag *and* in its
    clause, i.e. v1's additive behaviour for that one tag.
    """
    cfg = vocabulary.clause_groups
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
        if group in cfg.character_invariant:
            values_per_group.setdefault(group, set()).add(key)
        if key in vocabulary.characters:
            names_in_bag.add(key)
    pinned_groups = {
        group
        for group, markers in cfg.multi_value_markers.items()
        if markers & bag.keys()
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
            # The cast list stays flat: the bag answers "who is in this image"
            # (and is how a prompt summons them), the clause "which one is where".
            blocked[key] = "character-name"
            continue
        if group in cfg.character_invariant:
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
