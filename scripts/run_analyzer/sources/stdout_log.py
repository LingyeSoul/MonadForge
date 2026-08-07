"""stdout.log source — raw console capture with structured extraction.

Primary use: warning/error extraction, run markers, and a searchable tail.
The webui has its own tqdm regex parser (``webui/services/training_log_parser.py``)
for legacy runs; for daemon-era jobs the progress JSONL already carries the
structured data, so this source is a *supplement*, not the primary path.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Optional

_LINE_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+(WARNING|ERROR|INFO|DEBUG)\s+(.+?)\s+(?:.*?\s+)?([\w.]+:\d+)?\s*$"
)


@dataclass
class StdoutRun:
    lines: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    markers: list[str] = field(default_factory=list)  # training-identifying lines


def parse(path: str, max_lines: int = 20000) -> Optional[StdoutRun]:
    if not os.path.isfile(path):
        return None
    out = StdoutRun()
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            raw = fh.readlines()
    except OSError:
        return None
    lines = [l.rstrip("\n") for l in raw]
    if len(lines) > max_lines:
        out.lines = lines[-max_lines:]
    else:
        out.lines = lines
    for l in out.lines:
        if "WARNING" in l:
            out.warnings.append(l[:400])
        elif "ERROR" in l or "Traceback" in l:
            out.errors.append(l[:400])
        if "saving checkpoint" in l or "steps:" in l or "train.py" in l or "epoch " in l:
            out.markers.append(l[:400])
    return out
