"""Read side of the tagger checkpoint's ``vocab.json`` (torch-free).

Written by ``anime_tools/tagger/cli/vocab.py``. The schema:

``tags``
    ``[{name, index, category, freq, median_pos}, …]`` — one row per kept tag,
    ``index`` is the model's output slot. ``category`` is the booru tag type
    (``general`` / ``character`` / ``copyright`` / ``artist`` / ``count`` /
    ``metadata`` / ``deprecated``); exactly one per tag.
``ratings``
    Ordered rating labels for the rating head.
``people_count_labels``
    Ordered people-count buckets, or absent/empty for a checkpoint built
    without the head.
``groups``
    :func:`anime_tools.captions.tag_groups.resolved_to_dict` output — the typed
    tag groups projected onto ``index`` slots, as
    :class:`~anime_tools.captions.tag_groups.ResolvedGroup` dataclasses (only
    :meth:`GroupRouter.from_vocab` turns them into tensors, keeping this module
    torch-free). Absent for a vocab built without ``--groups``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

from library.curation_compat._json import read_json
from library.curation_compat.captions.tag_groups import ResolvedGroup, resolved_from_dict

__all__ = [
    "load_vocab",
    "names_by_category",
    "names_in_categories",
    "resolved_groups",
]


def load_vocab(path: str | Path) -> dict:
    """Load a ``vocab.json``; ``path`` may be the file or its checkpoint dir."""
    p = Path(path)
    if p.is_dir():
        p = p / "vocab.json"
    return read_json(p)


def names_by_category(
    vocab: Mapping,
    categories: Iterable[str] | None = None,
    *,
    key: Callable[[str], str] | None = None,
) -> dict[str, set[str]]:
    """``category -> {tag name}`` over ``vocab["tags"]``.

    ``categories`` restricts and pre-seeds the result (an axis with no tags
    still comes back as an empty set); ``None`` returns every category present.
    ``key`` folds each name on the way in — the vocab is the authority on a
    tag's *exact* spelling, so leave it unset unless you mean to fold.
    """
    wanted = None if categories is None else tuple(categories)
    out: dict[str, set[str]] = {} if wanted is None else {c: set() for c in wanted}
    for entry in vocab.get("tags") or ():
        cat = entry.get("category")
        if wanted is not None and cat not in out:
            continue
        out.setdefault(str(cat), set()).add(
            key(entry["name"]) if key else entry["name"]
        )
    return out


def names_in_categories(
    vocab: Mapping,
    categories: Iterable[str],
    *,
    key: Callable[[str], str] | None = None,
) -> frozenset[str]:
    """Union of :func:`names_by_category` over ``categories``."""
    sets = names_by_category(vocab, categories, key=key)
    return frozenset().union(*sets.values()) if sets else frozenset()


def resolved_groups(vocab: Mapping) -> tuple[ResolvedGroup, ...]:
    """``vocab["groups"]`` as :class:`ResolvedGroup`s (empty when absent)."""
    return resolved_from_dict(vocab.get("groups") or ())
