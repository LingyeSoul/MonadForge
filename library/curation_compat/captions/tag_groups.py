"""Anima caption tag-groups — structural typing on top of the flat tag vocab.

``tag_groups.yaml`` says how tags relate: which sets are mutually exclusive on
one subject (eye/hair color), which are always exclusive (rating), and which are
flat multi-label. It is loaded once, intersected with the kept vocab, and the
resolved index sets are written into ``vocab.json``.

YAML schema
-----------

::

    version: 1

    eye_color:
      mode: softmax_when_solo            # | softmax | multilabel
      description: "Primary eye color"   # optional
      escape: [heterochromia, ...]       # optional — disables softmax routing
      tags:                              # canonical space form (matches vocab)
        - blue eyes
        - ...

Mode semantics
~~~~~~~~~~~~~~

* ``softmax_when_solo`` — K-way CE on the group's logits when the image is
  single-subject **and** no ``escape`` tag fires; per-tag BCE otherwise. The
  gating is the trainer's job; the loader only exposes the structure.
* ``softmax`` — always K-way CE, for genuinely exclusive groups like rating.
* ``multilabel`` — sigmoid/BCE per tag; listed for introspection only.

Validation: each tag name appears in at most one group, and names use the
canonical *space* form. Names absent from the kept vocab (below ``min_freq``)
are dropped at :func:`resolve_groups` time, so the YAML stays stable across
min_freq changes.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

# Allowed values for a group's ``mode`` field; the loader rejects typos.
GROUP_MODES: frozenset[str] = frozenset(
    {
        "softmax_when_solo",
        "softmax",
        "multilabel",
    }
)

# ``sentinel: true`` on a softmax group adds a synthetic "none of this group"
# class: the vocab build appends a slot named ``sentinel_tag_name(group)``, CE
# supervises it on applicable samples with no member label (flipping the gate
# from "exactly one member" to "at most one"), and decode emits nothing when it
# wins the argmax. The angle brackets keep the name off any caption tag.
_SENTINEL_FMT = "<none:{group}>"

# ``tags: ["$category:artist"]`` expands at vocab-build time to every kept vocab
# tag of that category, for groups whose membership is the whole category.
_CATEGORY_MEMBER_PREFIX = "$category:"


def sentinel_tag_name(group_name: str) -> str:
    """Synthetic vocab-tag name for a group's "none of these" class."""
    return _SENTINEL_FMT.format(group=group_name)


def is_sentinel_name(tag_name: str) -> bool:
    """True for names produced by :func:`sentinel_tag_name` — never emitted."""
    return tag_name.startswith("<none:") and tag_name.endswith(">")


@dataclass(frozen=True)
class TagGroup:
    """One typed group of tag names (resolved indices come later)."""

    name: str
    mode: str
    description: str
    escape: tuple[str, ...]
    tags: tuple[str, ...]
    sentinel: bool = False


@dataclass(frozen=True)
class TagGroups:
    """All groups + a tag → group reverse map.

    The reverse map masks each group's tags out of the residual BCE head, so a
    tag is supervised by exactly one loss term.
    """

    version: int
    groups: tuple[TagGroup, ...]
    tag_to_group: Mapping[str, str]

    def by_name(self, name: str) -> TagGroup | None:
        for g in self.groups:
            if g.name == name:
                return g
        return None

    def to_dict(self) -> dict:
        """Round-trippable dict for snapshotting into the checkpoint dir."""
        out: dict = {"version": self.version}
        for g in self.groups:
            body: dict = {"mode": g.mode, "tags": list(g.tags)}
            if g.description:
                body["description"] = g.description
            if g.escape:
                body["escape"] = list(g.escape)
            if g.sentinel:
                body["sentinel"] = True
            out[g.name] = body
        return out


def _group_from_body(name: str, body: object, tag_to_group: dict[str, str]) -> TagGroup:
    """Validate one ``name: body`` group entry and register its tags.

    Shared by :func:`load_groups` (YAML) and :func:`from_dict` (snapshot):
    known mode, mapping body, sentinel×mode, cross-group tag uniqueness.
    Mutates ``tag_to_group``.
    """
    if not isinstance(body, dict):
        # ValueError, not TypeError: every validation in this module raises one
        # class so a caller loading a config can catch it once.
        raise ValueError(  # noqa: TRY004
            f"group {name!r}: expected a mapping with 'mode' / 'tags', "
            f"got {type(body).__name__}"
        )
    mode = str(body.get("mode", "")).strip()
    if mode not in GROUP_MODES:
        raise ValueError(f"group {name!r}: mode={mode!r} not in {sorted(GROUP_MODES)}")
    description = str(body.get("description", "") or "")
    escape = tuple(str(t) for t in (body.get("escape") or []))
    tags = tuple(str(t) for t in (body.get("tags") or []))
    sentinel = bool(body.get("sentinel", False))
    if sentinel and mode == "multilabel":
        raise ValueError(
            f"group {name!r}: sentinel=true only makes sense on softmax modes"
        )

    for t in tags:
        existing = tag_to_group.get(t)
        if existing is not None:
            raise ValueError(f"tag {t!r} listed under both {existing!r} and {name!r}")
        tag_to_group[t] = name

    return TagGroup(
        name=name,
        mode=mode,
        description=description,
        escape=escape,
        tags=tags,
        sentinel=sentinel,
    )


def load_groups(path: str | Path) -> TagGroups:
    """Load a ``tag_groups.yaml`` into a :class:`TagGroups`.

    Empty ``tags:`` lists are allowed.
    """
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    version = int(raw.pop("version", 1))

    groups: list[TagGroup] = []
    tag_to_group: dict[str, str] = {}

    for name, body in raw.items():
        groups.append(_group_from_body(name, body, tag_to_group))

    return TagGroups(version=version, groups=tuple(groups), tag_to_group=tag_to_group)


def from_dict(d: dict) -> TagGroups:
    """Inverse of :meth:`TagGroups.to_dict` — load a snapshot from JSON/YAML dict."""
    version = int(d.get("version", 1))
    groups: list[TagGroup] = []
    tag_to_group: dict[str, str] = {}
    for name, body in d.items():
        if name == "version":
            continue
        groups.append(_group_from_body(name, body, tag_to_group))
    return TagGroups(version=version, groups=tuple(groups), tag_to_group=tag_to_group)


@dataclass(frozen=True)
class ResolvedGroup:
    """A :class:`TagGroup` projected onto a built vocab's tag indices.

    ``tag_indices`` and ``escape_indices`` are sorted and disjoint with every
    other group's ``tag_indices``; names absent from the vocab are dropped.
    """

    name: str
    mode: str
    description: str
    tag_indices: tuple[int, ...]
    escape_indices: tuple[int, ...]
    # Names kept for snapshot/debug; dropped names are omitted.
    tag_names: tuple[str, ...]
    escape_names: tuple[str, ...]
    # Vocab index of the group's synthetic "none of these" class, when the
    # group declares ``sentinel: true`` AND the vocab carries the slot. Also
    # folded into ``tag_indices``, always last.
    sentinel_index: int | None = None


def resolve_groups(
    groups: TagGroups,
    vocab_tag_to_idx: Mapping[str, int],
) -> tuple[tuple[ResolvedGroup, ...], dict[str, str]]:
    """Project ``groups`` onto a built vocab's ``tag_to_idx`` map.

    Returns ``(resolved_groups, dropped)``; ``dropped`` maps each YAML-listed
    name that didn't survive the vocab cut to a short reason.
    """
    resolved: list[ResolvedGroup] = []
    dropped: dict[str, str] = {}
    for g in groups.groups:
        kept_tags: list[tuple[int, str]] = []
        for t in g.tags:
            idx = vocab_tag_to_idx.get(t)
            if idx is None:
                dropped[t] = "not_in_vocab"
                continue
            kept_tags.append((idx, t))
        kept_escape: list[tuple[int, str]] = []
        for t in g.escape:
            idx = vocab_tag_to_idx.get(t)
            if idx is None:
                dropped[t] = "not_in_vocab"
                continue
            kept_escape.append((idx, t))
        kept_tags.sort()
        kept_escape.sort()
        sentinel_index: int | None = None
        if g.sentinel and g.mode in ("softmax", "softmax_when_solo"):
            s_name = sentinel_tag_name(g.name)
            sentinel_index = vocab_tag_to_idx.get(s_name)
            if sentinel_index is None:
                # Vocab built without the slot — degrade to exactly-one
                # behaviour rather than error.
                dropped[s_name] = "sentinel_slot_not_in_vocab"
            else:
                kept_tags.append((sentinel_index, s_name))
                kept_tags.sort()
        resolved.append(
            ResolvedGroup(
                name=g.name,
                mode=g.mode,
                description=g.description,
                tag_indices=tuple(i for i, _ in kept_tags),
                tag_names=tuple(n for _, n in kept_tags),
                escape_indices=tuple(i for i, _ in kept_escape),
                escape_names=tuple(n for _, n in kept_escape),
                sentinel_index=sentinel_index,
            )
        )
    return tuple(resolved), dropped


def resolved_to_dict(resolved: tuple[ResolvedGroup, ...]) -> list[dict]:
    """Round-trippable list-of-dicts for embedding into ``vocab.json``."""
    return [
        {
            "name": g.name,
            "mode": g.mode,
            "description": g.description,
            "tag_indices": list(g.tag_indices),
            "tag_names": list(g.tag_names),
            "escape_indices": list(g.escape_indices),
            "escape_names": list(g.escape_names),
            "sentinel_index": g.sentinel_index,
        }
        for g in resolved
    ]


def resolved_from_dict(raw: Iterable[Mapping]) -> tuple[ResolvedGroup, ...]:
    """Inverse of :func:`resolved_to_dict` — read ``vocab.json[groups]`` back.

    Already projected onto vocab indices, so no vocab or YAML is needed. Absent
    optional keys fall back to the defaults, so an older build's dict revives.
    """
    return tuple(
        ResolvedGroup(
            name=str(g["name"]),
            mode=str(g["mode"]),
            description=str(g.get("description", "")),
            tag_indices=tuple(int(i) for i in g.get("tag_indices") or ()),
            tag_names=tuple(str(n) for n in g.get("tag_names") or ()),
            escape_indices=tuple(int(i) for i in g.get("escape_indices") or ()),
            escape_names=tuple(str(n) for n in g.get("escape_names") or ()),
            sentinel_index=(
                None if g.get("sentinel_index") is None else int(g["sentinel_index"])
            ),
        )
        for g in raw
    )


def expand_category_members(
    groups: TagGroups,
    name_to_category: Mapping[str, str],
) -> TagGroups:
    """Expand ``$category:<cat>`` member entries against a built vocab.

    Run this before :func:`resolve_groups` AND before snapshotting to the
    checkpoint dir: the inference wrapper re-resolves names, never markers.
    Expanded names claimed by another group raise, same as the loader.
    """
    out_groups: list[TagGroup] = []
    tag_to_group: dict[str, str] = {}
    for g in groups.groups:
        expanded: list[str] = []
        for t in g.tags:
            if t.startswith(_CATEGORY_MEMBER_PREFIX):
                cat = t[len(_CATEGORY_MEMBER_PREFIX) :]
                expanded.extend(
                    sorted(n for n, c in name_to_category.items() if c == cat)
                )
            else:
                expanded.append(t)
        for t in expanded:
            existing = tag_to_group.get(t)
            if existing is not None:
                raise ValueError(
                    f"tag {t!r} listed under both {existing!r} and {g.name!r} "
                    f"after $category expansion"
                )
            tag_to_group[t] = g.name
        out_groups.append(
            TagGroup(
                name=g.name,
                mode=g.mode,
                description=g.description,
                escape=g.escape,
                tags=tuple(expanded),
                sentinel=g.sentinel,
            )
        )
    return TagGroups(
        version=groups.version, groups=tuple(out_groups), tag_to_group=tag_to_group
    )
