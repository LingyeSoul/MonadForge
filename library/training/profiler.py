from __future__ import annotations

import logging
import time
from collections import defaultdict

logger = logging.getLogger(__name__)


class StepProfiler:
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self._starts: dict[str, float] = {}
        self._totals: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)
        self._step_count: int = 0
        self._enabled: bool = True

    def start(self, section: str) -> None:
        if not self._enabled:
            return
        self._starts[section] = time.perf_counter()

    def end(self, section: str) -> None:
        if not self._enabled:
            return
        start = self._starts.pop(section, None)
        if start is None:
            return
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self._totals[section] += elapsed_ms
        self._counts[section] += 1

    def step_end(self) -> bool:
        if not self._enabled:
            return False
        self._step_count += 1
        if self._step_count % self.window_size == 0:
            self._report()
            return True
        return False

    def _report(self) -> None:
        if not self._totals:
            return
        total_ms = sum(self._totals.values())
        if total_ms == 0:
            return

        lines = [f"Step profiler (last {self.window_size} steps, {self._step_count} total):"]
        lines.append(f"  {'Section':<20s} {'Avg ms':>10s} {'%':>8s} {'Calls':>8s}")
        lines.append(f"  {'-'*20} {'-'*10} {'-'*8} {'-'*8}")

        for section in sorted(self._totals):
            avg_ms = self._totals[section] / self._counts[section]
            pct = self._totals[section] / total_ms * 100.0
            lines.append(
                f"  {section:<20s} {avg_ms:>10.2f} {pct:>7.1f}% {self._counts[section]:>8d}"
            )

        total_calls = max(self._counts.values())
        lines.append(f"  {'TOTAL':<20s} {total_ms / total_calls:>10.2f} {'100.0%':>8s}")

        logger.info("\n".join(lines))

    def reset(self) -> None:
        self._starts.clear()
        self._totals.clear()
        self._counts.clear()
        self._step_count = 0
