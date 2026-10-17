"""Cycle-safe directory walking for the project's symlinked dataset trees."""

from __future__ import annotations

import os
from typing import Iterator, Union

__all__ = ["safe_walk"]


def safe_walk(
    top: Union[str, os.PathLike], *, followlinks: bool = True
) -> Iterator[tuple[str, list[str], list[str]]]:
    """``os.walk`` that follows symlinks but never revisits a directory.

    Dataset roots here are symlinks to (sometimes cross-linked) artist dirs,
    so callers need ``followlinks=True`` to descend at all — but plain
    ``os.walk`` has no cycle detection, so a back-link loops forever (the
    classic "preprocess/training just hangs" failure). This tracks each
    directory's real path and prunes already-visited children from
    ``dirnames`` in place, breaking cycles and de-duping diamond joins.
    """
    seen: set[str] = {os.path.realpath(top)}
    for dirpath, dirnames, filenames in os.walk(top, followlinks=followlinks):
        kept: list[str] = []
        for d in dirnames:
            real = os.path.realpath(os.path.join(dirpath, d))
            if real in seen:
                continue
            seen.add(real)
            kept.append(d)
        dirnames[:] = kept  # prune in place so os.walk won't re-descend a cycle
        yield dirpath, dirnames, filenames
