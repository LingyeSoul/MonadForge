"""Read helpers for the per-job ``progress.jsonl`` + ``stdout.log``.

The daemon never pipes a child's stdout — it tails files (the payoff of the
Phase-0 file-based progress decision: a re-attached orphan the daemon didn't
spawn can still be followed). These helpers are deliberately tiny and
exception-swallowing; a missing/half-written line is normal while the trainer
appends.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional


def last_event(progress_path: Optional[str]) -> Optional[dict]:
    """Parse the last complete JSON line of ``progress.jsonl`` (or ``None``)."""
    if not progress_path:
        return None
    p = Path(progress_path)
    if not p.is_file():
        return None
    last = None
    try:
        with open(p, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    last = json.loads(line)
                except ValueError:
                    continue  # half-written tail line
    except OSError:
        return None
    return last


def last_ckpt_path(progress_path: Optional[str]) -> Optional[str]:
    """The ``path`` of the most recent ``ckpt`` event, if any."""
    if not progress_path:
        return None
    p = Path(progress_path)
    if not p.is_file():
        return None
    found: Optional[str] = None
    try:
        with open(p, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or '"ckpt"' not in line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if rec.get("ev") == "ckpt" and rec.get("path"):
                    found = rec["path"]
    except OSError:
        return None
    return found


def follow(path: Path, *, from_start: bool = True) -> Iterator[str]:
    """Generator yielding lines as they're appended (``tail -f``).

    Yields existing content first when ``from_start`` is set, then blocks for
    new lines. The caller drives cadence (it sleeps between empty reads) and is
    responsible for stopping — this never returns on its own. Windows opens
    files shared-read by default so this works while the trainer writes.
    """
    import time

    while not path.exists():
        yield ""  # let the caller poll / decide to give up
        time.sleep(0.3)
    with open(path, "r", encoding="utf-8") as fh:
        if not from_start:
            fh.seek(0, 2)
        while True:
            line = fh.readline()
            if line:
                yield line
            else:
                yield ""  # heartbeat tick so the caller can check liveness
                time.sleep(0.3)
