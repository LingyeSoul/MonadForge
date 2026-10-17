"""Previous versions of one caption — the ``{stem}.history.txt`` sidecar.

Every write pushes the text it replaced here, oldest first, and the GUI's
caption ladder expands them into one badge apiece (``revised@1``,
``revised@2`` …) at the end of the badge row, each drawn as the diff of what it
changed into rather than as a text. It uses
:mod:`anime_tools.captions._sidecar`'s format; what is
local is the vocabulary — a record is a superseded caption, carrying a sequence,
a moment and a hand — and the fact that a file with no entries left is deleted
rather than written empty. The sidecar sits beside the caption, so a writer
needs no second path.

Torch-free, stdlib-only, and import-light: :mod:`anime_tools.stages.replay`
imports it from inside a function.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from library.curation_compat.captions._sidecar import (
    read_rows,
    sidecar_header,
    sidecar_path,
    write_rows,
)

HISTORY_SIDECAR_SUFFIX = ".history.txt"
HISTORY_FIELDS = 4
"""``seq ⇥ when ⇥ by ⇥ text`` — and the text may hold tabs, being last."""

HISTORY_LIMIT = 10
"""How many superseded versions one caption keeps.

The oldest entries fall off the front; sequence numbers do **not** renumber, so
a badge cannot come to mean a different version.
"""


@dataclass(frozen=True)
class HistoryEntry:
    """One superseded caption: what it said, when it stopped saying it, and who
    replaced it.

    ``label`` is the badge — the rung's name with the sequence appended
    (``revised@3``).
    """

    seq: int
    at: str
    by: str
    text: str

    def label(self, kind: str) -> str:
        return f"{kind}@{self.seq}"

    def note(self) -> str:
        """The badge's tooltip line: who replaced it, and when."""
        return f"{self.by} · {self.at}" if self.by else self.at


def history_sidecar_path(caption_path: Path) -> Path:
    """``{stem}.history.txt`` beside a caption (or its image).

    A multi-dot stem survives (``a.b.txt`` → ``a.b.history.txt``).
    """
    return sidecar_path(caption_path, HISTORY_SIDECAR_SUFFIX)


def read_history(path: Path) -> list[HistoryEntry]:
    """Parse a history sidecar into an ordered list, oldest first.

    ``path`` is the *sidecar*; a missing one is simply no history. Records
    arrive with the format's own tolerance, plus one more: a record whose
    sequence is not an integer is dropped like a malformed line.
    """
    if not path.is_file():
        return []
    out: list[HistoryEntry] = []
    for seq, at, by, text in read_rows(path, HISTORY_FIELDS):
        try:
            n = int(seq)
        except ValueError:
            continue
        out.append(HistoryEntry(seq=n, at=at.strip(), by=by.strip(), text=text))
    return out


def write_history(path: Path, entries: list[HistoryEntry]) -> None:
    """Write the sidecar, or delete it when there is nothing left to say.

    A caption with no superseded versions has no history, so the sidecar is
    removed rather than left as an empty file.
    """
    if not entries:
        if path.is_file():
            path.unlink()
        return
    write_rows(
        path,
        sidecar_header("history"),
        ((str(e.seq), e.at, e.by, e.text) for e in entries),
    )


def push_history(
    caption_path: Path,
    text: str,
    *,
    by: str = "",
    limit: int = HISTORY_LIMIT,
) -> HistoryEntry | None:
    """Record ``text`` as a superseded version of ``caption_path``.

    Called with the text the caller is *about to overwrite*, so the newest entry
    is always the immediately previous caption. Returns the entry recorded, or
    ``None`` when there was nothing to record: an empty text, or a text the
    newest entry already holds.

    Sequence numbers continue from the highest ever recorded rather than from
    the list length, so trimming the front cannot make two versions share a
    badge.
    """
    body = (text or "").strip()
    if not body:
        return None
    sidecar = history_sidecar_path(caption_path)
    entries = read_history(sidecar)
    if entries and entries[-1].text == body:
        return None
    entry = HistoryEntry(
        seq=(entries[-1].seq + 1) if entries else 1,
        at=datetime.now(UTC).astimezone().strftime("%Y-%m-%d %H:%M"),
        by=by,
        text=body,
    )
    entries.append(entry)
    write_history(sidecar, entries[-max(1, int(limit)) :])
    return entry


def drop_history(caption_path: Path) -> None:
    """Delete a caption's history. Only for a caption that is itself being
    deleted (an Undo of a run that *created* one)."""
    sidecar = history_sidecar_path(caption_path)
    if sidecar.is_file():
        sidecar.unlink()
