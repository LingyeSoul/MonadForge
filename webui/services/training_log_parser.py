"""Stateful parser that extracts structured training metrics from tqdm / epoch output lines."""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# tqdm progress bar line:
# steps:  45%|████████▌           | 450/1000 [05:23<06:39,  1.38it/s, avr_loss=0.0234, lr=1.23e-04, router_H=1.872]
# Note: tqdm shows "s/it" when slow, "it/s" when fast — match both.
_TQDM_RE = re.compile(
    r"steps:\s+(\d+)%\|.*?\|\s+(\d+)/(\d+)\s+"
    r"\[(.+?)<(.+?),\s+([\d.]+)\s*((?:it/s|s/it))"
    r"(?:,\s*avr_loss=([\d.]+))?"
    r"(?:,\s*lr=([0-9.eE+\-]+))?"
    r"(?:,\s*router_H=([\d.]+))?"
    r"(?:,\s*Keys Scaled=(\d+))?"
    r"(?:,\s*Average key norm=([\d.]+))?"
)

# Direct step progress line from training loop (stdout):
# step 450/1000 loss=0.02345 lr=1.23e-04 speed=1.380
_STEP_RE = re.compile(
    r"^step\s+(\d+)/(\d+)"
    r"(?:\s+loss=([\d.]+))?"
    r"(?:\s+lr=([0-9.eE+\-]+))?"
    r"(?:\s+speed=([\d.]+))?"
)

# epoch 3/10
_EPOCH_RE = re.compile(r"^epoch\s+(\d+)/(\d+)\s*$")

# saving checkpoint: output/ckpt/...
_CKPT_RE = re.compile(r"saving checkpoint:")

# Loss value from explicit log lines (fallback if tqdm doesn't report)
_LOSS_RE = re.compile(r"(?:avr_loss|loss)[=:]?\s*([\d.]+)")

_MAX_HISTORY = 2000


@dataclass
class TrainingMetrics:
    step: int = 0
    total_steps: int = 0
    epoch: int = 0
    total_epochs: int = 0
    avr_loss: float = 0.0
    loss_history: list[float] = field(default_factory=list)
    step_history: list[int] = field(default_factory=list)
    lr: float = 0.0
    speed: str = ""
    elapsed: str = ""
    eta: str = ""
    router_h: float | None = None
    keys_scaled: int | None = None
    avg_key_norm: float | None = None
    checkpoint_saved: bool = False
    events: list[dict] = field(default_factory=list)

    def snapshot(self) -> dict:
        """Return a JSON-serializable snapshot of current metrics."""
        return {
            "step": self.step,
            "total_steps": self.total_steps,
            "epoch": self.epoch,
            "total_epochs": self.total_epochs,
            "avr_loss": self.avr_loss,
            "loss_history": list(self.loss_history),
            "step_history": list(self.step_history),
            "lr": self.lr,
            "speed": self.speed,
            "elapsed": self.elapsed,
            "eta": self.eta,
            "router_h": self.router_h,
            "keys_scaled": self.keys_scaled,
            "avg_key_norm": self.avg_key_norm,
            "checkpoint_saved": self.checkpoint_saved,
            "events": list(self.events),
        }


class TrainingLogParser:
    """Stateful parser that feeds raw stdout lines and emits structured metrics updates."""

    def __init__(self) -> None:
        self.metrics = TrainingMetrics()
        self._dirty = False

    def feed(self, line: str) -> bool:
        """Feed a raw stdout line. Returns True if metrics were updated."""
        self._dirty = False
        stripped = line.strip()

        # Try tqdm progress bar first (most informative)
        m = _TQDM_RE.search(stripped)
        if m:
            self._parse_tqdm(m)
            return self._dirty

        # Try direct step progress line (stdout fallback when tqdm stderr is lost)
        m = _STEP_RE.match(stripped)
        if m:
            self._parse_step(m)
            return self._dirty

        # Log step-like lines that didn't match (debug)
        if stripped.startswith("step "):
            print(f"[parser] Step line NOT matched: {stripped!r}", flush=True)

        # Try epoch marker
        m = _EPOCH_RE.match(stripped)
        if m:
            self._parse_epoch(m)
            return self._dirty

        # Try checkpoint marker
        if _CKPT_RE.search(stripped):
            self._parse_checkpoint(stripped)
            return self._dirty

        return False

    def _parse_tqdm(self, m: re.Match) -> None:
        step = int(m.group(2))
        total = int(m.group(3))
        elapsed = m.group(4).strip()
        eta = m.group(5).strip()
        speed_num = m.group(6).strip()
        speed_unit = m.group(7).strip()  # "it/s" or "s/it"

        self.metrics.step = step
        self.metrics.total_steps = total
        self.metrics.elapsed = elapsed
        self.metrics.eta = eta
        self.metrics.speed = f"{speed_num} {speed_unit}"

        if m.group(8):  # avr_loss
            loss = float(m.group(8))
            self.metrics.avr_loss = loss
            # Append to rolling history (downsample if exceeding max)
            if not self.metrics.step_history or step != self.metrics.step_history[-1]:
                self.metrics.loss_history.append(loss)
                self.metrics.step_history.append(step)
                if len(self.metrics.loss_history) > _MAX_HISTORY:
                    # Keep every Nth point to stay under limit
                    stride = len(self.metrics.loss_history) // _MAX_HISTORY + 1
                    self.metrics.loss_history = self.metrics.loss_history[::stride]
                    self.metrics.step_history = self.metrics.step_history[::stride]

        if m.group(9):  # lr
            self.metrics.lr = float(m.group(9))
        if m.group(10):  # router_H
            self.metrics.router_h = float(m.group(10))
        if m.group(11):  # Keys Scaled
            self.metrics.keys_scaled = int(m.group(11))
        if m.group(12):  # Average key norm
            self.metrics.avg_key_norm = float(m.group(12))

        self._dirty = True

    def _parse_step(self, m: re.Match) -> None:
        """Parse direct step progress: ``step 450/1000 loss=0.02345 lr=1.23e-04 speed=1.380``."""
        step = int(m.group(1))
        total = int(m.group(2))
        self.metrics.step = step
        self.metrics.total_steps = total

        if m.group(3):  # loss
            loss = float(m.group(3))
            self.metrics.avr_loss = loss
            if not self.metrics.step_history or step != self.metrics.step_history[-1]:
                self.metrics.loss_history.append(loss)
                self.metrics.step_history.append(step)
                if len(self.metrics.loss_history) > _MAX_HISTORY:
                    stride = len(self.metrics.loss_history) // _MAX_HISTORY + 1
                    self.metrics.loss_history = self.metrics.loss_history[::stride]
                    self.metrics.step_history = self.metrics.step_history[::stride]

        if m.group(4):  # lr
            self.metrics.lr = float(m.group(4))
        if m.group(5):  # speed
            self.metrics.speed = m.group(5)

        self._dirty = True

    def _parse_epoch(self, m: re.Match) -> None:
        epoch = int(m.group(1))
        total = int(m.group(2))
        self.metrics.epoch = epoch
        self.metrics.total_epochs = total
        self.metrics.events.append(
            {
                "type": "epoch",
                "epoch": epoch,
                "total_epochs": total,
                "step": self.metrics.step,
                "elapsed": self.metrics.elapsed,
            }
        )
        self._dirty = True

    def _parse_checkpoint(self, line: str) -> None:
        self.metrics.checkpoint_saved = True
        self.metrics.events.append(
            {
                "type": "checkpoint",
                "step": self.metrics.step,
                "epoch": self.metrics.epoch,
                "elapsed": self.metrics.elapsed,
                "detail": line.strip(),
            }
        )
        self._dirty = True
