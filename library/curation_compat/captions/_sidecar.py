"""The tab-delimited caption sidecar format shared by ``.variants.txt`` and
``.history.txt``: a comment header, then one tab-delimited record per line.

Two rules live here rather than in both callers:

* **The multi-dot stem.** Sidecar names are built with ``with_name``, never
  ``with_suffix``, so ``a.b.png`` yields ``a.b.variants.txt``.
* **The hand-edit tolerance.** Readers skip blank lines, comment lines and any
  line with the wrong field count rather than raising, and strip ``\\r``; a
  damaged sidecar must cost the sidecar, never a run.

Stdlib-only on purpose: :mod:`anime_tools.stages.replay` must stay importable
without :mod:`anime_tools.captions` behind it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path


def sidecar_header(kind: str) -> str:
    """The comment line every caption sidecar opens with.

    ``kind`` is the sidecar's own word (``variants``, ``history``). Readers skip
    *any* ``#`` line, so this text is for a person, not a parser.
    """
    return f"# anima caption {kind} — auto-generated, do not hand-edit"


def sidecar_path(path: Path, suffix: str) -> Path:
    """``{stem}{suffix}`` beside ``path`` — an image or its ``.txt`` caption.

    ``with_name``, not ``with_suffix``, so a multi-dot stem survives:
    ``a.b.png`` → ``a.b.variants.txt``. An extensionless path keeps its whole
    name.
    """
    stem = path.name[: -len(path.suffix)] if path.suffix else path.name
    return path.with_name(stem + suffix)


def read_rows(path: Path, fields: int) -> list[list[str]]:
    """Parse a sidecar into its records, in file order.

    ``fields`` is the record arity; the split is bounded at that many, so only
    the first ``fields - 1`` tabs are delimiters and the last field may contain
    tabs. Lines with any other field count are skipped, as are blank and comment
    lines. Fields arrive verbatim. ``path`` must exist — whether a missing
    sidecar is empty or an error is the caller's call.
    """
    out: list[list[str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip("\r")
        if not line or line.lstrip().startswith("#"):
            continue
        parts = line.split("\t", fields - 1)
        if len(parts) != fields:
            continue
        out.append(parts)
    return out


def render_rows(header: str, rows: Iterable[Sequence[str]]) -> str:
    """The sidecar's text: ``header`` and one tab-joined line per row.

    Fields are joined, not escaped — a caption never contains a tab. The
    trailing newline is part of the format. What :func:`write_rows` writes, for
    a caller that publishes the text somewhere other than a sidecar of its own.
    """
    lines = [header]
    lines += ["\t".join(row) for row in rows]
    return "\n".join(lines) + "\n"


def write_rows(path: Path, header: str, rows: Iterable[Sequence[str]]) -> None:
    """Write :func:`render_rows`' text to ``path``, creating the parent dir."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_rows(header, rows), encoding="utf-8")
