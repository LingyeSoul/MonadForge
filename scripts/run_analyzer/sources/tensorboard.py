"""TensorBoard source — reads the trainer's scalar events via EventAccumulator.

Location: ``<log_dir>/network_train/events.out.tfevents.*``. Carries the same
``logs`` dict the progress sink flushes (loss/lr/norms/vr), at the same
``log_every_n_steps`` cadence, plus any series the JSONL path dropped.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TbRun:
    # tag -> [[step, value], ...]
    series: dict[str, list[list[float]]] = field(default_factory=dict)
    # steps that carry a value in any series, ascending
    steps: list[int] = field(default_factory=list)
    file: Optional[str] = None


def parse(log_dir: str, network_subdir: str = "network_train") -> Optional[TbRun]:
    """Parse tfevents under ``log_dir/network_train`` (None if absent)."""
    net_dir = os.path.join(log_dir, network_subdir)
    if not os.path.isdir(net_dir):
        return None
    files = sorted(glob.glob(os.path.join(net_dir, "events.out.tfevents.*")))
    if not files:
        return None
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return None
    try:
        ea = EventAccumulator(
            net_dir,
            size_guidance={"scalars": 0, "tensors": 0, "histograms": 0, "images": 0},
        )
        ea.Reload()
    except Exception:
        return None
    tags = ea.Tags().get("scalars", [])
    out = TbRun(file=files[0])
    events = {}
    for tag in tags:
        try:
            vals = [[int(s.step), float(s.value)] for s in ea.Scalars(tag)]
        except Exception:
            continue
        if vals:
            events[tag] = vals
            out.series[tag] = vals
    if events:
        all_steps = sorted({s for vals in events.values() for s, _ in vals})
        out.steps = all_steps
    return out if out.series else None
