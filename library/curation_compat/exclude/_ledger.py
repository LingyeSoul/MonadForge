"""The exclusion ledger — the state an exclusion *is*. Torch-free.

**The ledger is the state.** Every stage but ``resize`` walks
``workspace/resized/``, so an image whose resized copy has left that tree is
already invisible to all of them — but ``resize`` would put it straight back on
the next preflight, which is why it reads :func:`excluded_rels` and adds them to
its own ``--skip``. That one chokepoint is the whole enforcement; nothing else
has to know.

The ledger is keyed by the image's path relative to the **source** tree
(``char_aki/a.jpg``), which is what ``resize``'s ``--skip`` names and what a
sidebar row is — :func:`rel_key` is the one spelling of that key. The files it
records are keyed by directory + stem instead (:mod:`._artifacts`), because
resize re-encodes (``a.jpg`` → ``a.png``) and each tree spells the tail its own
way.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from library.curation_compat import workspace as WS
from library.curation_compat._env import resolve_path
from library.curation_compat._json import read_json, write_json

__all__ = [
    "MANIFEST_VERSION",
    "Entry",
    "ExclusionError",
    "Trees",
    "excluded_rels",
    "manifest_path",
    "read_entries",
    "rel_key",
    "write_entries",
]

MANIFEST_VERSION = 1
"""Bumped when the ledger's shape changes. An older file is read as it is —
there is no v0, and a reader that could not parse one refuses rather than
guessing (:class:`ExclusionError`)."""


class ExclusionError(ValueError):
    """A bad rel, or a ledger that will not parse.

    Unparseable is an error and never an empty ledger: "nothing is excluded" is
    the answer that would put every excluded image back through the pipeline on
    the next preflight.
    """


@dataclass(frozen=True)
class Entry:
    """One excluded image: what it was, when, why, and what moved with it."""

    rel: str
    """The image's path relative to the source tree — the ``resize --skip``
    spelling, forward slashes, extension kept."""
    at: float
    """Unix time of the *first* exclusion. Re-excluding an image already in the
    ledger keeps this, since that is when it left the dataset."""
    note: str = ""
    """Free text from the curator; the GUI shows it on the item panel."""
    moved: tuple[str, ...] = ()
    """Every file that moved, as a slot: ``<tree>/<path under that tree>``, with
    ``<tree>`` one of :data:`~anime_tools.workspace.EXCLUDED_TREES`. The slot is
    both where the file sits under ``_excluded`` and how to put it back."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "rel": self.rel,
            "at": self.at,
            "note": self.note,
            "moved": list(self.moved),
        }

    @classmethod
    def from_dict(cls, raw: Any) -> Entry | None:
        """One ledger row, or ``None`` for a row with no usable ``rel``."""
        if not isinstance(raw, dict):
            return None
        rel = str(raw.get("rel") or "").strip()
        if not rel:
            return None
        try:
            at = float(raw.get("at") or 0.0)
        except (TypeError, ValueError):
            at = 0.0
        moved = raw.get("moved")
        return cls(
            rel=rel,
            at=at,
            note=str(raw.get("note") or ""),
            moved=tuple(str(m) for m in moved) if isinstance(moved, list) else (),
        )


@dataclass(frozen=True)
class Trees:
    """The three live trees an exclusion empties, and the tree it fills.

    Field names are :data:`~anime_tools.workspace.EXCLUDED_TREES` plus
    ``excluded``, so a slot's first component names the attribute to put the file
    back into and the mapping is never written down twice.
    """

    resized: Path
    masks: Path
    ocr: Path
    excluded: Path

    @classmethod
    def build(
        cls,
        *,
        dst: str | Path = WS.RESIZED,
        masks: str | Path = WS.MASKS,
        ocr: str | Path = WS.OCR,
        excluded: str | Path = WS.EXCLUDED,
    ) -> Trees:
        """The workspace defaults, each anchored under the curation home."""
        return cls(
            resized=resolve_path(dst),
            masks=resolve_path(masks),
            ocr=resolve_path(ocr),
            excluded=resolve_path(excluded),
        )

    def live(self, tree: str) -> Path:
        """The live tree a slot's first component names."""
        if tree not in WS.EXCLUDED_TREES:
            raise ExclusionError(f"not an excluded tree: {tree!r}")
        return getattr(self, tree)


def manifest_path(excluded: str | Path) -> Path:
    return Path(excluded) / WS.EXCLUDED_MANIFEST


def read_entries(excluded: str | Path) -> dict[str, Entry]:
    """The ledger, keyed by rel. A ledger that is not there is an empty one; a
    ledger that will not parse is an error."""
    path = manifest_path(excluded)
    if not path.is_file():
        return {}
    try:
        data = read_json(path)
    except (OSError, ValueError) as e:
        raise ExclusionError(f"unreadable exclusion ledger {path}: {e}") from e
    if not isinstance(data, dict):
        raise ExclusionError(f"not an exclusion ledger: {path}")
    rows = data.get("excluded")
    entries = [Entry.from_dict(r) for r in rows] if isinstance(rows, list) else []
    return {e.rel: e for e in entries if e is not None}


def write_entries(excluded: str | Path, entries: dict[str, Entry]) -> Path:
    """Write the ledger, rels sorted so a diff of it reads as a diff.

    The file stays even when it is empty: an ``_excluded`` directory with no
    ledger in it would read as a tree nobody wrote.
    """
    return write_json(
        manifest_path(excluded),
        {
            "version": MANIFEST_VERSION,
            "excluded": [entries[rel].to_dict() for rel in sorted(entries)],
        },
    )


def excluded_rels(excluded: str | Path) -> tuple[str, ...]:
    """Every excluded rel, sorted — the ``resize --skip`` list."""
    return tuple(sorted(read_entries(excluded)))


def rel_key(rel: str | Path) -> str:
    """A caller's rel as the ledger spells it: forward slashes, relative, no
    ``..``. Refused rather than normalised, since a rel is a dataset path and a
    surprising one would move the wrong file."""
    s = str(rel).replace("\\", "/")
    win, posix = PureWindowsPath(s), PurePosixPath(s)
    # The anchor test is Windows' on every host, because it is the strict one:
    # `Path("/abs").is_absolute()` is False on Windows (no drive), so a rooted
    # rel slips past `is_absolute` there and `tree / rel` then lands on the
    # drive root -- outside the dataset. Windows rules refuse a drive ("C:/x"),
    # a UNC share ("//srv/x") and a bare leading slash alike.
    if (
        win.drive
        or win.root
        or not posix.parts
        or any(part in ("..", ".") for part in posix.parts)
    ):
        raise ExclusionError(f"bad relative path: {rel!r}")
    return posix.as_posix()
