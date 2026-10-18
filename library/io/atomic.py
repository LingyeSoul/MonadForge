"""One-file atomic text write: stage, fsync, replace.

``library/`` must not import the WebUI's storage layer (the dependency runs the
other way), so the simple single-file form lives here. The WebUI keeps its
transactional multi-file writer on top of its own staging, because that one also
rolls several files back together.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` without ever exposing a partial file.

    The staged file is flushed and fsynced before the replace, so a crash either
    leaves the previous contents or the complete new ones.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
