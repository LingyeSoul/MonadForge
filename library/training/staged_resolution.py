"""Compatibility shorthand for in-process staged-resolution training.

``--staged_resolution`` expands the legacy side/ratio strings into the generic
percent-based ``stage_schedule`` consumed by the training loop. Dataset row 0
is stage 0, row 1 is stage 1, and so on. Unlike the former implementation this
does not launch independent subprocesses, so optimizer/model state is continuous.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Sequence

from library.datasets.buckets import ALLOWED_TARGET_RES

logger = logging.getLogger(__name__)

DEFAULT_RATIOS = (20.0, 30.0, 50.0)
DEFAULT_BASE_SIDES = (512, 768, 1024)


def _parse_csv_numbers(raw: Any, *, integer: bool) -> list[float] | list[int]:
    if raw is None:
        return []
    values = raw if isinstance(raw, (list, tuple)) else str(raw).split(",")
    try:
        if integer:
            return [int(str(value).strip()) for value in values]
        return [float(str(value).strip()) for value in values]
    except ValueError as exc:
        kind = "integer sides" if integer else "numeric ratios"
        raise ValueError(f"staged_resolution requires comma-separated {kind}") from exc


def build_stage_schedule(
    ratios: Sequence[float] = DEFAULT_RATIOS,
    base_sides: Sequence[int] = DEFAULT_BASE_SIDES,
) -> list[dict[str, Any]]:
    """Build contiguous stage dictionaries from side labels and percentages."""
    ratios = [float(value) for value in ratios]
    base_sides = [int(value) for value in base_sides]
    if not ratios or len(ratios) != len(base_sides):
        raise ValueError(
            "staged_resolution ratios and base_sides must have equal length"
        )
    if any(value <= 0 for value in ratios):
        raise ValueError("staged_resolution ratios must all be positive")
    if not math.isclose(sum(ratios), 100.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(
            f"staged_resolution ratios must sum to 100, got {sum(ratios):g}"
        )
    invalid = [side for side in base_sides if side not in ALLOWED_TARGET_RES]
    if invalid:
        raise ValueError(
            f"staged_resolution sides {invalid} are not in {list(ALLOWED_TARGET_RES)}"
        )

    stages: list[dict[str, Any]] = []
    cursor = 0.0
    for index, (ratio, side) in enumerate(zip(ratios, base_sides)):
        end = 1.0 if index == len(ratios) - 1 else cursor + ratio / 100.0
        stages.append(
            {
                "name": f"{side}px",
                "subset_index": index,
                "start_pct": cursor,
                "end_pct": end,
            }
        )
        cursor = end
    return stages


def configure_staged_resolution(args: Any) -> None:
    """Expand the deprecated three-row shorthand into the generic schedule."""
    if not bool(getattr(args, "staged_resolution", False)):
        return
    if bool(getattr(args, "stage_schedule_enabled", False)) and getattr(
        args, "stage_schedule", None
    ):
        raise ValueError(
            "staged_resolution and an explicit stage_schedule cannot both be enabled"
        )

    ratios = _parse_csv_numbers(
        getattr(args, "staged_resolution_ratios", None), integer=False
    ) or list(DEFAULT_RATIOS)
    sides = _parse_csv_numbers(
        getattr(args, "staged_resolution_base_sides", None), integer=True
    ) or list(DEFAULT_BASE_SIDES)
    args.stage_schedule_enabled = True
    args.stage_schedule = build_stage_schedule(ratios, sides)
    args._stage_expected_sides = tuple(int(side) for side in sides)
    logger.warning(
        "--staged_resolution is a deprecated shorthand; prefer stage_schedule. "
        "It now requires one preprocessed [[datasets]] row per stage: %s",
        " -> ".join(
            f"{stage['name']} ({ratio:g}%)"
            for stage, ratio in zip(args.stage_schedule, ratios)
        ),
    )
