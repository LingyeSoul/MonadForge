"""Progress JSONL source — parses the daemon-injected per-job structured stream.

Event schema (library/training/progress.py)::

    run_start / step / val / ckpt / sample / log / run_end
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class JsonlRun:
    run: Optional[str] = None
    method: Optional[str] = None
    preset: Optional[str] = None
    total_steps: Optional[int] = None
    total_epochs: Optional[int] = None
    pid: Optional[int] = None
    log_dir: Optional[str] = None
    sampling_enabled: bool = False
    run_end_status: Optional[str] = None
    run_end_final_step: Optional[int] = None
    run_end_error: Optional[str] = None
    run_end_extra: dict = field(default_factory=dict)
    # tag -> [[global_step, value], ...] (kept in arrival order, gs ascending)
    series: dict[str, list[list[float]]] = field(default_factory=dict)
    # parallel to series points: gs -> epoch (only for points carrying loss)
    gs_epoch: dict[int, int] = field(default_factory=dict)
    ckpts: list[dict] = field(default_factory=list)
    samples: list[dict] = field(default_factory=list)
    vals: list[dict] = field(default_factory=list)
    logs: list[dict] = field(default_factory=list)
    first_step_ts: Optional[float] = None
    last_step_ts: Optional[float] = None
    first_step: Optional[int] = None
    last_step: Optional[int] = None
    error_count: int = 0


def parse(path: str) -> Optional[JsonlRun]:
    """Parse a progress.jsonl into a :class:`JsonlRun` (None if unusable)."""
    if not os.path.isfile(path):
        return None
    out = JsonlRun()
    try:
        fh = open(path, encoding="utf-8")
    except OSError:
        return None
    with fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            kind = ev.get("ev")
            if kind == "run_start":
                out.run = ev.get("run")
                out.method = ev.get("method")
                out.preset = ev.get("preset")
                out.total_steps = ev.get("total_steps")
                out.total_epochs = ev.get("total_epochs")
                out.pid = ev.get("pid")
                out.log_dir = ev.get("log_dir")
                out.sampling_enabled = bool(ev.get("sampling_enabled", False))
            elif kind == "step":
                gs = ev.get("global_step")
                ep = ev.get("epoch")
                if gs is None:
                    continue
                ts = ev.get("ts")
                if out.first_step_ts is None:
                    out.first_step_ts = ts
                    out.first_step = gs
                out.last_step_ts = ts
                out.last_step = gs
                for key, val in ev.items():
                    if key in ("ev", "ts", "global_step", "epoch"):
                        continue
                    if isinstance(val, (int, float)) and not isinstance(val, bool):
                        out.series.setdefault(key, []).append([gs, float(val)])
                if isinstance(ep, int) and "loss/current" in ev:
                    out.gs_epoch[gs] = ep
            elif kind == "val":
                out.vals.append(
                    {
                        "global_step": ev.get("global_step"),
                        "epoch": ev.get("epoch"),
                        "cmmd": ev.get("cmmd"),
                        "val_step": ev.get("val_step"),
                        "ts": ev.get("ts"),
                    }
                )
            elif kind == "ckpt":
                out.ckpts.append(
                    {
                        "global_step": ev.get("global_step"),
                        "path": ev.get("path"),
                        "ts": ev.get("ts"),
                    }
                )
            elif kind == "sample":
                out.samples.append(
                    {
                        "global_step": ev.get("global_step"),
                        "epoch": ev.get("epoch"),
                        "path": ev.get("path"),
                        "prompt": ev.get("prompt"),
                        "ts": ev.get("ts"),
                    }
                )
            elif kind == "log":
                out.logs.append(
                    {
                        "level": ev.get("level"),
                        "logger": ev.get("logger"),
                        "msg": ev.get("msg"),
                        "ts": ev.get("ts"),
                    }
                )
                if ev.get("level") == "ERROR":
                    out.error_count += 1
            elif kind == "run_end":
                out.run_end_status = ev.get("status")
                out.run_end_final_step = ev.get("final_step")
                out.run_end_error = ev.get("error")
                out.run_end_extra = {
                    k: v for k, v in ev.items() if k not in ("ev", "ts", "status", "final_step", "error")
                }
    return out
