"""Percent-based dataset curriculum for staged-resolution training.

Each stage covers a half-open interval of ``max_train_steps`` and selects one
dataset row (or one local subset for a single-row dataset group). All caches
must exist before training; a switch only rebuilds bucket membership and the
DataLoader.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StageSpec:
    """One interval in a dataset curriculum."""

    subset_index: int
    start_pct: float
    end_pct: float
    name: str = ""

    def contains_progress(self, progress: float) -> bool:
        progress = max(0.0, min(1.0, float(progress)))
        if self.end_pct >= 1.0 - 1e-12:
            return progress >= self.start_pct
        return self.start_pct <= progress < self.end_pct


@dataclass
class StageRuntimePlan:
    """Validated schedule plus the mutable dataset resources it selects."""

    stages: tuple[StageSpec, ...]
    dataset_group: Any
    target_batch_counts: tuple[int, ...]
    full_num_train_images: int
    full_num_reg_images: int
    full_dataset_counts: tuple[tuple[int, int], ...]
    initial_index: int = 0
    dataloader_kwargs: dict[str, Any] = field(default_factory=dict)
    loader_generator: Any = None

    def as_dicts(self) -> list[dict[str, Any]]:
        return [asdict(stage) for stage in self.stages]

    def index_for_step(self, global_step: int, max_train_steps: int) -> int:
        progress = progress_from_steps(global_step, max_train_steps)
        return resolve_stage_index(self.stages, progress)


def _parse_fraction(value: Any, *, field_name: str, default: float) -> float:
    if value is None:
        value = default
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric, got {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite, got {value!r}")
    if 1.0 < parsed <= 100.0:
        parsed /= 100.0
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{field_name} must be between 0..1 or 0..100")
    return parsed


def _parse_subset_index(value: Any, *, default: int) -> int:
    if value is None:
        value = default
    if isinstance(value, bool):
        raise ValueError("subset_index must be an integer, not a boolean")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"subset_index must be an integer, got {value!r}") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"subset_index must be an integer, got {value!r}")
    if parsed < 0:
        raise ValueError(f"subset_index must be non-negative, got {parsed}")
    return parsed


def normalize_stage_dicts(raw: Any) -> list[dict[str, Any]]:
    """Normalize field aliases and percent notation without hiding bad input."""
    if raw is None:
        return []
    if isinstance(raw, str):
        import json

        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("stage_schedule is not valid JSON") from exc
    if isinstance(raw, Mapping):
        raw = raw.get("stages", [raw])
    if not isinstance(raw, Sequence) or isinstance(raw, (bytes, bytearray, str)):
        raise ValueError("stage_schedule must be a list of stage tables")

    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"stage {index + 1} must be a table")
        start = _parse_fraction(
            item.get("start_pct", item.get("startPct")),
            field_name=f"stage {index + 1} start_pct",
            default=0.0,
        )
        end = _parse_fraction(
            item.get("end_pct", item.get("endPct")),
            field_name=f"stage {index + 1} end_pct",
            default=1.0,
        )
        subset_index = _parse_subset_index(
            item.get(
                "subset_index",
                item.get("subsetIndex", item.get("dataset_index", index)),
            ),
            default=index,
        )
        name = str(item.get("name") or f"stage{index + 1}").strip()
        normalized.append(
            {
                "name": name or f"stage{index + 1}",
                "subset_index": subset_index,
                "start_pct": start,
                "end_pct": end,
            }
        )
    return normalized


def parse_stage_specs(raw: Any) -> list[StageSpec]:
    return [
        StageSpec(
            subset_index=int(stage["subset_index"]),
            start_pct=float(stage["start_pct"]),
            end_pct=float(stage["end_pct"]),
            name=str(stage.get("name") or ""),
        )
        for stage in normalize_stage_dicts(raw)
    ]


def validate_stage_specs(
    stages: Sequence[StageSpec], *, subset_count: Optional[int] = None
) -> list[str]:
    """Return configuration problems; an empty list means the schedule is valid."""
    if not stages:
        return ["stage schedule is empty"]

    problems: list[str] = []
    if abs(stages[0].start_pct) > 1e-6:
        problems.append("the first stage must start at 0%")
    if abs(stages[-1].end_pct - 1.0) > 1e-6:
        problems.append("the last stage must end at 100%")

    for index, stage in enumerate(stages):
        label = index + 1
        if not math.isfinite(stage.start_pct) or not math.isfinite(stage.end_pct):
            problems.append(f"stage {label} boundaries must be finite")
        if not 0.0 <= stage.start_pct <= 1.0:
            problems.append(f"stage {label} start_pct is outside 0..1")
        if not 0.0 <= stage.end_pct <= 1.0:
            problems.append(f"stage {label} end_pct is outside 0..1")
        if stage.subset_index < 0:
            problems.append(f"stage {label} subset_index must be non-negative")
        if stage.end_pct <= stage.start_pct + 1e-9:
            problems.append(f"stage {label} has an empty interval")
        if subset_count is not None and not 0 <= stage.subset_index < subset_count:
            problems.append(
                f"stage {label} subset_index={stage.subset_index} is outside "
                f"0..{subset_count - 1}"
            )
        if index:
            seam = stages[index - 1].end_pct
            if abs(seam - stage.start_pct) > 1e-6:
                relation = "overlaps" if stage.start_pct < seam else "has a gap after"
                problems.append(f"stage {label} {relation} stage {label - 1}")
    return problems


def resolve_stage_index(stages: Sequence[StageSpec], progress: float) -> int:
    if not stages:
        return 0
    progress = max(0.0, min(1.0, float(progress)))
    for index, stage in enumerate(stages):
        if stage.contains_progress(progress):
            return index
    return len(stages) - 1


def progress_from_steps(global_step: int, max_train_steps: int) -> float:
    total = max(1, int(max_train_steps))
    return max(0.0, min(1.0, float(global_step) / total))


def stage_schedule_enabled(args: Any) -> bool:
    return bool(getattr(args, "stage_schedule_enabled", False))


def active_subset_indices_for_step(args: Any, global_step: int) -> Optional[set[int]]:
    if not stage_schedule_enabled(args):
        return None
    stages = parse_stage_specs(args.stage_schedule)
    if not stages:
        raise ValueError("stage_schedule_enabled requires at least one stage")
    progress = progress_from_steps(global_step, int(args.max_train_steps or 0))
    return {stages[resolve_stage_index(stages, progress)].subset_index}


def snapshot_full_image_data(dataset: Any, *, force: bool = False) -> None:
    """Snapshot every leaf before applying the first stage filter."""
    if dataset is None:
        return
    members = getattr(dataset, "datasets", None)
    if isinstance(members, (list, tuple)):
        for member in members:
            snapshot_full_image_data(member, force=force)
        return
    snapshot = getattr(dataset, "snapshot_full_image_data", None)
    if callable(snapshot):
        snapshot(force=force)


def count_stage_targets(dataset: Any) -> int:
    """Count selectable dataset rows, falling back to local subsets."""
    if dataset is None:
        return 0
    members = getattr(dataset, "datasets", None)
    if isinstance(members, (list, tuple)) and members:
        if len(members) > 1:
            return len(members)
        return count_stage_targets(members[0])
    subsets = getattr(dataset, "subsets", None)
    return len(subsets) if isinstance(subsets, (list, tuple)) else 0


def _empty_leaf_dataset(dataset: Any) -> None:
    has_snapshot = getattr(dataset, "has_full_image_data_snapshot", None)
    snapshot = getattr(dataset, "snapshot_full_image_data", None)
    if callable(snapshot) and (not callable(has_snapshot) or not has_snapshot()):
        snapshot(force=True)

    if hasattr(dataset, "image_data"):
        dataset.image_data = {}
    if hasattr(dataset, "image_to_subset"):
        dataset.image_to_subset = {}
    if hasattr(dataset, "num_train_images"):
        dataset.num_train_images = 0
    if hasattr(dataset, "num_reg_images"):
        dataset.num_reg_images = 0
    if hasattr(dataset, "buckets_indices"):
        dataset.buckets_indices = []
    if hasattr(dataset, "_length"):
        dataset._length = 0
    dataset._stage_active = False


def apply_active_subsets_to_dataset(
    dataset: Any, active_subset_indices: Optional[Iterable[int]]
) -> bool:
    """Apply stage membership to a DatasetGroup or a leaf dataset."""
    if dataset is None:
        return False

    members = getattr(dataset, "datasets", None)
    if isinstance(members, (list, tuple)) and members:
        if len(members) > 1:
            if active_subset_indices is None:
                restored = [
                    apply_active_subsets_to_dataset(member, None) for member in members
                ]
                ok = any(restored)
            else:
                active = {int(index) for index in active_subset_indices}
                if not any(0 <= index < len(members) for index in active):
                    return False
                ok = False
                for index, member in enumerate(members):
                    if index in active:
                        ok = apply_active_subsets_to_dataset(member, None) or ok
                    else:
                        _empty_leaf_dataset(member)
            if ok and hasattr(dataset, "refresh_concat_state"):
                dataset.refresh_concat_state()
            return ok

        ok = apply_active_subsets_to_dataset(members[0], active_subset_indices)
        if ok and hasattr(dataset, "refresh_concat_state"):
            dataset.refresh_concat_state()
        return ok

    rebuild = getattr(dataset, "rebuild_buckets_for_subsets", None)
    return bool(rebuild(active_subset_indices)) if callable(rebuild) else False


def prepare_stage_runtime(args: Any, dataset_group: Any) -> Optional[StageRuntimePlan]:
    """Validate all stage targets, snapshot full counts, and select stage zero."""
    if not stage_schedule_enabled(args):
        return None

    stages = tuple(parse_stage_specs(args.stage_schedule))
    target_count = count_stage_targets(dataset_group)
    problems = validate_stage_specs(stages, subset_count=target_count)
    problems.extend(
        validate_stage_target_sides(
            dataset_group,
            stages,
            getattr(args, "_stage_expected_sides", None),
        )
    )
    if getattr(args, "staged_resolution", False) and target_count != len(stages):
        problems.append(
            "--staged_resolution now requires one fully preprocessed [[datasets]] "
            "row per stage; see docs/guidelines/staged-resolution-training.md"
        )
    if problems:
        raise ValueError("invalid stage_schedule: " + "; ".join(problems))

    full_dataset_counts = tuple(
        (int(member.num_train_images), int(member.num_reg_images))
        for member in getattr(dataset_group, "datasets", ())
    )
    plan = StageRuntimePlan(
        stages=stages,
        dataset_group=dataset_group,
        target_batch_counts=(),
        full_num_train_images=int(dataset_group.num_train_images),
        full_num_reg_images=int(dataset_group.num_reg_images),
        full_dataset_counts=full_dataset_counts,
    )
    snapshot_full_image_data(dataset_group, force=True)

    members = getattr(dataset_group, "datasets", ())
    referenced = {stage.subset_index for stage in stages}
    if len(members) > 1:
        counts = tuple(len(member) for member in members)
        empty = sorted(index for index in referenced if counts[index] <= 0)
        if empty:
            raise ValueError(
                "stage_schedule dataset rows have no complete batches: "
                + ", ".join(str(index) for index in empty)
            )
    else:
        mutable_counts: list[int] = []
        for target_index in range(target_count):
            if not apply_active_subsets_to_dataset(dataset_group, {target_index}):
                mutable_counts.append(0)
                if target_index in referenced:
                    raise ValueError(
                        "stage_schedule dataset target is empty: "
                        f"subset_index={target_index}"
                    )
            else:
                mutable_counts.append(len(dataset_group))
        counts = tuple(mutable_counts)
        empty = sorted(index for index in referenced if counts[index] <= 0)
        if empty:
            raise ValueError(
                "stage_schedule subset targets have no complete batches: "
                + ", ".join(str(index) for index in empty)
            )

    plan.target_batch_counts = counts
    active = {stages[0].subset_index}
    if not apply_active_subsets_to_dataset(dataset_group, active):
        raise ValueError(
            "stage_schedule stage 1 produced an empty dataset: "
            f"subset_indices={sorted(active)}"
        )
    plan.initial_index = 0
    args.stage_schedule = plan.as_dicts()
    return plan


def validate_stage_target_sides(
    dataset: Any,
    stages: Sequence[StageSpec],
    expected_sides: Sequence[int] | None,
) -> list[str]:
    """Validate shorthand side labels against multi-row dataset bucket bands."""
    if not expected_sides:
        return []
    if len(expected_sides) != len(stages):
        return ["staged-resolution side count does not match the stage count"]

    members = getattr(dataset, "datasets", None)
    if not isinstance(members, (list, tuple)) or len(members) <= 1:
        return []

    from library.datasets.buckets import freefit_band_for_edge

    problems: list[str] = []
    for index, (stage, side) in enumerate(zip(stages, expected_sides)):
        if not 0 <= stage.subset_index < len(members):
            continue
        bucket_manager = getattr(members[stage.subset_index], "bucket_manager", None)
        resos = list(getattr(bucket_manager, "resos", ()) or ())
        if not resos:
            problems.append(
                f"stage {index + 1} dataset row {stage.subset_index} has no buckets"
            )
            continue
        lo, hi = freefit_band_for_edge(int(side))
        outside = [
            (width, height)
            for width, height in resos
            if not lo <= (width // 16) * (height // 16) <= hi
        ]
        if outside:
            preview = ", ".join(f"{w}x{h}" for w, h in outside[:3])
            problems.append(
                f"stage {index + 1} expects the {side}px tier, but dataset row "
                f"{stage.subset_index} contains out-of-band buckets: {preview}"
            )
    return problems


def stage_epoch_upper_bound(
    stages: Sequence[StageSpec],
    max_train_steps: int,
    target_batch_counts: Sequence[int],
    *,
    num_processes: int = 1,
    gradient_accumulation_steps: int = 1,
) -> int:
    """Conservative epoch budget when stage dataset lengths differ."""
    import math

    total_steps = max(1, int(max_train_steps))
    epochs = 0

    def boundary(fraction: float) -> int:
        # Stage selection happens before a batch at integer global_step. The
        # first step whose progress reaches a fractional seam is therefore the
        # ceiling of seam * total (with a tiny tolerance for float noise).
        return min(total_steps, max(0, math.ceil(total_steps * fraction - 1e-12)))

    for stage in stages:
        start = boundary(stage.start_pct)
        end = total_steps if stage.end_pct >= 1.0 - 1e-12 else boundary(stage.end_pct)
        stage_steps = max(0, end - start)
        batches = (
            target_batch_counts[stage.subset_index]
            if 0 <= stage.subset_index < len(target_batch_counts)
            else 0
        )
        updates = max(
            1,
            math.ceil(
                max(1, batches)
                / max(1, num_processes)
                / max(1, gradient_accumulation_steps)
            ),
        )
        epochs += max(1, math.ceil(stage_steps / updates))
    return max(1, epochs)


def log_stage_switch(
    stage: StageSpec, index: int, global_step: int, max_train_steps: int
) -> None:
    logger.info(
        "stage schedule -> #%s %s dataset=%s progress=%.1f%% (step %s/%s)",
        index + 1,
        stage.name or f"stage{index + 1}",
        stage.subset_index,
        100.0 * progress_from_steps(global_step, max_train_steps),
        global_step,
        max_train_steps,
    )
