"""Reading and writing the package's JSON files, in one shape.

* ``ensure_ascii=False``, so non-ASCII directory names stay readable.
* ``encoding="utf-8"`` — a bare ``open(path)`` reads in the platform's locale
  codepage, which is not UTF-8 on Windows.
* the parent directory is created before the write.
* the write is atomic: a temp file beside the target, then ``os.replace``. A
  reader sees the old file or the new one, never a half-written one, so a crash
  mid-write cannot truncate a ``report.json`` or the GUI's settings blob.

``indent=2`` is the canonical shape, overridable. No trailing newline — skip
checks compare these files byte for byte.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

__all__ = ["read_json", "write_json"]


def read_json(path: str | Path) -> Any:
    """Parse a UTF-8 JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any, *, indent: int | None = 2) -> Path:
    """Write ``payload`` as UTF-8 JSON, creating the parent directory.

    Atomic: the bytes land in a temp file in the same directory (so the rename
    stays on one filesystem) and ``os.replace`` swaps it in. Two writers of the
    same path therefore lose one write rather than interleaving into a file that
    parses as neither.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=indent, ensure_ascii=False)
    fd, name = tempfile.mkstemp(dir=p.parent, prefix=f".{p.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(name, p)
    except BaseException:
        # Never leave the dot-file behind for the next walk to trip over.
        Path(name).unlink(missing_ok=True)
        raise
    return p
