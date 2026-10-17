"""Shared job telemetry for the independent Qwen and MOSS training lines.

The daemon owns the path. Standalone CLI runs keep their existing stdout/JSON
reports, and these events never opt a model into Anima's continuation protocol.
Importing this module does not import torch or either model implementation.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass
class TrainingProgress:
    sink: Any = None
    global_step: int = 0
    started: bool = False

    def start(self, *, total_steps: int, total_epochs: int) -> None:
        if self.started:
            return
        self.started = True
        if self.sink is not None:
            self.sink.run_start(
                total_steps=total_steps,
                total_epochs=total_epochs,
                pid=os.getpid(),
                sampling_enabled=False,
            )
            self.sink.attach_log_mirror()

    def step(
        self,
        *,
        global_step: int,
        epoch: int,
        loss: float,
        average_loss: float,
        lr: float,
        **metrics: float,
    ) -> None:
        self.global_step = global_step
        if self.sink is not None:
            self.sink.log(
                {
                    "loss/current": loss,
                    "loss/average": average_loss,
                    "lr": lr,
                    **metrics,
                },
                global_step=global_step,
                epoch=epoch,
            )

    def checkpoint(self, path: str | Path) -> None:
        if self.sink is not None:
            self.sink.ckpt(global_step=self.global_step, path=str(path))


@contextmanager
def training_progress(model_type: str, output: str) -> Iterator[TrainingProgress]:
    job_dir = os.environ.get("ANIMA_DAEMON_JOB_DIR")
    if not job_dir:
        yield TrainingProgress()
        return

    # The generic writer is imported only in a training process, never while
    # constructing browser request forms or launching a standalone CLI.
    from library.training.progress import ProgressSink, run_scope

    sink = ProgressSink(
        str(Path(job_dir) / "progress.jsonl"),
        run=Path(output).stem,
        method=f"{model_type}-train",
        preset=None,
    )
    progress = TrainingProgress(sink)
    with run_scope(sink, final_step=lambda: progress.global_step):
        try:
            yield progress
        finally:
            # Cache discovery or model preflight may fail before totals exist.
            # Emit one complete lifecycle even for that early failure; on the
            # normal path start() already wrote the real training denominator.
            if not progress.started:
                progress.start(total_steps=0, total_epochs=0)
