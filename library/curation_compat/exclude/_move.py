"""The move engine: taking one image out of the live trees, and putting it back.

Torch-free, and the only thing in the package that writes. Both directions are
one gesture with one rule about collisions — a slot whose live path is occupied
again is **left** under ``_excluded`` and reported, because the copy there is the
older one and choosing between them is a curation decision this is not.

:class:`Result` is what one call did, so neither the CLI's line nor the GUI's
answer has to re-derive it from the ledger.
"""

from __future__ import annotations

import shutil
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from library.curation_compat import workspace as WS
from library.curation_compat.exclude._artifacts import artifacts
from library.curation_compat.exclude._ledger import (
    Entry,
    Trees,
    read_entries,
    rel_key,
    write_entries,
)

__all__ = ["Result", "exclude_many", "exclude_one", "restore_many", "restore_one"]


@dataclass
class Result:
    """What one exclude or restore actually did — the CLI's line and the GUI's
    answer, so neither has to re-derive it from the ledger."""

    rel: str
    action: str
    """``excluded`` / ``restored`` / ``already-excluded`` / ``not-excluded``."""
    moved: tuple[str, ...] = ()
    """Slots that changed tree."""
    skipped: tuple[str, ...] = ()
    """Slots left where they were, because the live tree already holds that path
    (a resize that ran without the ledger, say). Never overwritten — the copy in
    ``_excluded`` is the older one, and clobbering is not a curation decision."""
    entry: Entry | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rel": self.rel,
            "action": self.action,
            "moved": list(self.moved),
            "skipped": list(self.skipped),
            "entry": self.entry.to_dict() if self.entry else None,
        }


def _move(src: Path, dst: Path) -> None:
    """``shutil.move``, not ``rename``: the trees are normally one filesystem,
    but a ``dst`` root on another mount must not make this a crash."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def _prune(directory: Path, stop: Path) -> None:
    """Remove ``directory`` and its now-empty parents, up to but never including
    ``stop``. What keeps a restored tree from leaving a skeleton of empty folders
    behind under ``_excluded``."""
    while directory != stop and directory.is_relative_to(stop):
        try:
            directory.rmdir()
        except OSError:  # not empty, or gone already — and that ends the walk
            return
        directory = directory.parent


def exclude_one(
    trees: Trees, rel: str, *, note: str = "", apply: bool = True
) -> Result:
    """Take one image out of the pipeline.

    Every file it has moves under ``_excluded`` and the rel goes into the ledger.
    An image with nothing in the workspace yet — excluded before it was ever
    resized — is still recorded: the ledger is the state, and the point is that
    ``resize`` never makes those files in the first place.

    Re-excluding keeps the original ``at`` (that is when it left the dataset) and
    unions what moved, so a file that appeared since is swept up too.
    """
    return exclude_many(trees, [rel], note=note, apply=apply)[0]


def exclude_many(
    trees: Trees, rels: Sequence[str], *, note: str = "", apply: bool = True
) -> list[Result]:
    """:func:`exclude_one` over several images, against one ledger.

    The ledger is read once and written once however many images move, because
    it is a single JSON file: excluding a hundred images one call at a time is a
    hundred full reads and a hundred full rewrites of the same document.
    """
    entries = read_entries(trees.excluded)
    results = [_exclude(trees, entries, rel, note=note, apply=apply) for rel in rels]
    if apply and results:
        write_entries(trees.excluded, entries)
    return results


def _exclude(
    trees: Trees,
    entries: dict[str, Entry],
    rel: str,
    *,
    note: str,
    apply: bool,
) -> Result:
    """One exclusion against an already-loaded ledger, which it mutates in
    place; writing it back is the caller's."""
    key = rel_key(rel)
    was = entries.get(key)

    moved: list[str] = []
    for slot, live in artifacts(trees, key):
        if apply:
            if not live.is_file():  # vanished since the walk above
                continue
            # The destination is inside ``_excluded``, which this module owns, so
            # a leftover there is ours to replace.
            _move(live, trees.excluded / slot)
        moved.append(slot)

    entry = Entry(
        rel=key,
        at=was.at if was else time.time(),
        note=note or (was.note if was else ""),
        moved=tuple(dict.fromkeys((*(was.moved if was else ()), *moved))),
    )
    if apply:
        entries[key] = entry
    return Result(
        rel=key,
        action="already-excluded" if was and not moved else "excluded",
        moved=tuple(moved),
        entry=entry,
    )


def restore_one(trees: Trees, rel: str, *, apply: bool = True) -> Result:
    """Put one image back, and take its rel out of the ledger.

    A slot whose live path is occupied again is **left** under ``_excluded`` and
    reported: the file there is the older one, and choosing between them is a
    curation decision this is not. The rel leaves the ledger either way — the
    image is no longer excluded, whatever is on disk.
    """
    return restore_many(trees, [rel], apply=apply)[0]


def restore_many(
    trees: Trees, rels: Sequence[str], *, apply: bool = True
) -> list[Result]:
    """:func:`restore_one` over several images, against one ledger — the
    inverse of :func:`exclude_many`, and read and written the same once."""
    entries = read_entries(trees.excluded)
    results = [_restore(trees, entries, rel, apply=apply) for rel in rels]
    # A rel that was never excluded took nothing out of the ledger, and a ledger
    # nothing changed is one this should not have written at all.
    if apply and any(r.action != "not-excluded" for r in results):
        write_entries(trees.excluded, entries)
    return results


def _restore(
    trees: Trees, entries: dict[str, Entry], rel: str, *, apply: bool
) -> Result:
    """One restore against an already-loaded ledger, which it mutates in place."""
    key = rel_key(rel)
    was = entries.get(key)
    if was is None:
        return Result(rel=key, action="not-excluded")

    moved: list[str] = []
    skipped: list[str] = []
    for slot in was.moved:
        tree, _, tail = slot.partition("/")
        if not tail or tree not in WS.EXCLUDED_TREES:
            continue
        src = trees.excluded / slot
        dst = trees.live(tree) / tail
        if not src.is_file():
            continue
        if dst.exists():
            skipped.append(slot)
            continue
        if apply:
            _move(src, dst)
            _prune(src.parent, trees.excluded)
        moved.append(slot)

    if apply:
        del entries[key]
    return Result(
        rel=key,
        action="restored",
        moved=tuple(moved),
        skipped=tuple(skipped),
        entry=was,
    )
