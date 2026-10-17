"""Anima caption tag-rules — replacements, removals, clothing-base dedup.

The ``rules.yaml`` snapshot in the tagger checkpoint dir is the canonical source
at runtime. Three rule families:

* ``replacements``: whole-string ``str.replace`` applied before tokenization —
  HTML-entity decode, rating normalization onto Anima's band (see
  ``taxonomy.LEGACY_RATING_ALIASES``), artist-alias rewrites. Snapshots
  predating the band carry the old ``questionable → sensitive`` collapse and
  stay valid: :meth:`AnimaTagger.tag` holds the rating slot out of
  ``apply_rules``.
* ``aliases``: tag-level renames applied to the split tag list — a retired
  booru name onto the live one (``silver hair → grey hair``). Unlike
  ``replacements`` this never touches a longer tag the name is a prefix of,
  and a caption that already carries the target keeps one copy.
* ``remove``: tag literals that are unconditionally stripped.
* dedup map: ``{base: {variants}}`` — drop ``bra`` when ``black bra`` is
  already in the caption.

Order inside :func:`apply_rules`: aliases, then remove, then dedup — so a
removal or a dedup base is spelled in live names only.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class TagRules:
    """Compiled rule set ready to apply to caption strings or tag lists."""

    replacements: tuple[tuple[str, str], ...]
    remove: frozenset
    dedup: dict[str, frozenset]
    # Consulted before the booru tag cache in vocab.categorize(), for tags the
    # cache mis-types; only tags the curator lists explicitly.
    category_overrides: dict[str, str]
    # Substring patterns suppressed from the build-time "top-20 uncategorized"
    # coverage log. Logging filter only — does not change categorization.
    # Case-sensitive (booru tags are lowercase).
    coverage_ignore: tuple[str, ...]
    # Tag-level renames of retired booru names onto live ones; applied first in
    # apply_rules. Last so positional construction of the older fields holds.
    aliases: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Round-trippable dict for snapshotting into the checkpoint dir."""
        out: dict = {
            "replacements": dict(self.replacements),
            "remove": sorted(self.remove),
        }
        if self.aliases:
            out["aliases"] = dict(self.aliases)
        if self.category_overrides:
            out["category_overrides"] = dict(self.category_overrides)
        if self.coverage_ignore:
            out["coverage_ignore"] = list(self.coverage_ignore)
        out.update({base: sorted(variants) for base, variants in self.dedup.items()})
        return out


# Top-level YAML keys that are NOT dedup base→variants entries.
_RESERVED_KEYS = frozenset(
    {
        "replacements",
        "aliases",
        "remove",
        "category_overrides",
        "coverage_ignore",
    }
)


def load_rules(path: str | Path) -> TagRules:
    """Load a ``tag_rules.yaml`` into a :class:`TagRules`; the YAML and the
    ``to_dict`` JSON snapshot are the same mapping."""
    with open(path, encoding="utf-8") as f:
        return from_dict(yaml.safe_load(f) or {})


def from_dict(d: dict) -> TagRules:
    """Inverse of :meth:`TagRules.to_dict` — load a snapshot from JSON."""
    repl_map = d.get("replacements", {}) or {}
    aliases = {str(k): str(v) for k, v in (d.get("aliases", {}) or {}).items()}
    remove = frozenset(d.get("remove", []) or [])
    overrides = dict(d.get("category_overrides", {}) or {})
    coverage_ignore = tuple(str(s) for s in (d.get("coverage_ignore", []) or []))
    dedup = {
        str(base): frozenset(variants)
        for base, variants in d.items()
        if base not in _RESERVED_KEYS
    }
    replacements = tuple((str(k), str(v)) for k, v in repl_map.items())
    return TagRules(
        replacements=replacements,
        aliases=aliases,
        remove=remove,
        dedup=dedup,
        category_overrides={str(k): str(v) for k, v in overrides.items()},
        coverage_ignore=coverage_ignore,
    )


def apply_replacements(content: str, rules: TagRules) -> str:
    """Apply whole-string replacements (rating collapse, HTML decode, alias)."""
    for find, replace in rules.replacements:
        if find in content:
            content = content.replace(find, replace)
    return content


def parse_caption(content: str, rules: TagRules) -> list[str]:
    """Split a raw caption string into a clean tag list under ``rules``,
    keeping the survivors in their original order."""
    content = apply_replacements(content, rules).strip()
    if not content:
        return []
    tags = [t.strip() for t in content.split(",")]
    tags = [t for t in tags if t]
    return apply_rules(tags, rules)


def apply_aliases(tags: Iterable[str], aliases: dict[str, str]) -> list[str]:
    """Rename aliased tags in place, keeping order; a target the list already
    carries (or reaches twice) survives once, at its first position."""
    if not aliases:
        return list(tags)
    out: list[str] = []
    seen: set[str] = set()
    for t in tags:
        t = aliases.get(t, t)
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def apply_rules(tags: Iterable[str], rules: TagRules) -> list[str]:
    """Alias retired names, drop ``remove``-listed tags and dedup base tags
    whose variants fired."""
    tag_list = apply_aliases(tags, rules.aliases)
    tag_set: set[str] = set(tag_list)
    to_remove: set[str] = set(tag_set & rules.remove)
    for base, variants in rules.dedup.items():
        if base in tag_set and tag_set & variants:
            to_remove.add(base)
    if not to_remove:
        return tag_list
    return [t for t in tag_list if t not in to_remove]
