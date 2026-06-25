"""Curriculum-learning scheduler for staged mixed-resolution training.

Trains at progressively higher resolutions (e.g. 512 → 768 → 1024) in a
single orchestrated run. Each phase launches a standalone train.py subprocess
with resolution-appropriate batch size and epoch count so the total training
budget is preserved across stages.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_TARGET_RES = 1024
DEFAULT_RATIOS = [20, 30, 50]
DEFAULT_BASE_SIDES = [512, 768, 1024]


def _lcm(a: int, b: int) -> int:
    return abs(a * b) // math.gcd(a, b)


def _lcm_of_list(values: list[int]) -> int:
    result = values[0]
    for v in values[1:]:
        result = _lcm(result, v)
    return result


@dataclass
class StagePlan:
    """Computed parameters for a single resolution stage."""

    side: int
    pixel_area: int
    batch_size: int
    epochs: int
    ratio_pct: int
    save_every_n_epochs: Optional[int]
    sample_every_n_epochs: Optional[int]


@dataclass
class MixedResolutionPlan:
    """Full staged-resolution training plan, SHA1-hashed for reproducibility."""

    stages: list[StagePlan]
    base_side: int
    base_batch_size: int
    total_epochs: int
    plan_hash: str

    def total_training_epochs(self) -> int:
        return sum(s.epochs for s in self.stages)


def build_staged_plan(
    target_res: int = DEFAULT_TARGET_RES,
    base_batch_size: int = 1,
    total_epochs: int = 100,
    save_every: Optional[int] = None,
    sample_every: Optional[int] = None,
    ratios: Optional[list[int]] = None,
    base_sides: Optional[list[int]] = None,
) -> MixedResolutionPlan:
    """Build a staged training plan with auto-scaled batch sizes and epochs.

    Batch size scales inversely with pixel area to keep GPU memory constant:
        new_batch = max(1, int(base_batch * base_area / stage_area))

    Epoch count compensates for the batch-size change so total samples seen
    stays proportional to the stage's ratio:
        raw_epochs = ceil(total_epochs * ratio/100 * stage_batch / base_batch)

    Save/sample intervals are LCM-aligned across all stages so every stage
    boundary coincides with at least one checkpoint/sample event.
    """
    if ratios is None:
        ratios = list(DEFAULT_RATIOS)
    if base_sides is None:
        base_sides = list(DEFAULT_BASE_SIDES)

    assert len(ratios) == len(base_sides), (
        f"ratios ({len(ratios)}) and base_sides ({len(base_sides)}) must have the same length"
    )
    assert abs(sum(ratios) - 100) < len(ratios), (
        f"ratios must sum to 100, got {sum(ratios)}"
    )

    base_area = target_res * target_res

    raw_epochs_per_stage = []
    stages: list[StagePlan] = []
    for side, ratio in zip(base_sides, ratios):
        area = side * side
        batch = max(1, int(base_batch_size * base_area / area))
        raw_epochs = math.ceil(total_epochs * ratio / 100.0 * batch / base_batch_size)
        raw_epochs_per_stage.append(raw_epochs)
        stages.append(
            StagePlan(
                side=side,
                pixel_area=area,
                batch_size=batch,
                epochs=raw_epochs,
                ratio_pct=ratio,
                save_every_n_epochs=None,
                sample_every_n_epochs=None,
            )
        )

    # LCM-align save/sample intervals across stages
    if save_every is not None and save_every > 0:
        epoch_counts = [s.epochs for s in stages]
        lcm_val = _lcm_of_list(epoch_counts)
        aligned_save = _lcm(save_every, lcm_val)
        for s in stages:
            s.save_every_n_epochs = aligned_save

    if sample_every is not None and sample_every > 0:
        epoch_counts = [s.epochs for s in stages]
        lcm_val = _lcm_of_list(epoch_counts)
        aligned_sample = _lcm(sample_every, lcm_val)
        for s in stages:
            s.sample_every_n_epochs = aligned_sample

    # SHA1 hash of the plan for reproducibility
    plan_dict = {
        "target_res": target_res,
        "base_batch_size": base_batch_size,
        "total_epochs": total_epochs,
        "ratios": ratios,
        "base_sides": base_sides,
        "stages": [
            {
                "side": s.side,
                "batch_size": s.batch_size,
                "epochs": s.epochs,
            }
            for s in stages
        ],
    }
    plan_json = json.dumps(plan_dict, sort_keys=True)
    plan_hash = hashlib.sha1(plan_json.encode()).hexdigest()[:12]

    plan = MixedResolutionPlan(
        stages=stages,
        base_side=target_res,
        base_batch_size=base_batch_size,
        total_epochs=total_epochs,
        plan_hash=plan_hash,
    )
    log_plan(plan)
    return plan


def log_plan(plan: MixedResolutionPlan) -> None:
    """Log the staged resolution plan details."""
    logger.info(
        "Staged resolution plan [%s]: %d stages, base=%d, base_batch=%d, total_epochs=%d",
        plan.plan_hash,
        len(plan.stages),
        plan.base_side,
        plan.base_batch_size,
        plan.total_epochs,
    )
    for i, s in enumerate(plan.stages):
        save_str = str(s.save_every_n_epochs) if s.save_every_n_epochs else "off"
        sample_str = str(s.sample_every_n_epochs) if s.sample_every_n_epochs else "off"
        logger.info(
            "  Stage %d: side=%d, batch=%d, epochs=%d (ratio=%d%%) | save=%s, sample=%s",
            i + 1,
            s.side,
            s.batch_size,
            s.epochs,
            s.ratio_pct,
            save_str,
            sample_str,
        )
    logger.info(
        "  Total training epochs across all stages: %d",
        plan.total_training_epochs(),
    )


def _build_phase_argv(
    base_argv: list[str],
    stage: StagePlan,
    stage_index: int,
    total_stages: int,
    plan: MixedResolutionPlan,
    output_suffix: str,
) -> list[str]:
    """Build the CLI argv for one staged-resolution phase.

    Copies the original argv and overrides resolution-dependent flags.
    """
    argv = list(base_argv)

    def _set_flag(flag: str, value: str) -> None:
        # Remove existing occurrences of the flag
        i = 0
        while i < len(argv):
            if argv[i] == flag and i + 1 < len(argv):
                argv.pop(i)
                argv.pop(i)
            elif argv[i].startswith(flag + "="):
                argv.pop(i)
            else:
                i += 1
        argv.extend([flag, value])

    _set_flag("--target_res", str(stage.side))
    _set_flag("--train_batch_size", str(stage.batch_size))
    _set_flag("--max_train_epochs", str(stage.epochs))

    # Append stage suffix to output name to keep checkpoints separate
    output_name = None
    for i, a in enumerate(argv):
        if a == "--output_name" and i + 1 < len(argv):
            output_name = argv[i + 1]
            break
    if output_name is not None:
        _set_flag("--output_name", f"{output_name}_{output_suffix}")

    if stage.save_every_n_epochs is not None:
        _set_flag("--save_every_n_epochs", str(stage.save_every_n_epochs))
    if stage.sample_every_n_epochs is not None:
        _set_flag("--sample_every_n_epochs", str(stage.sample_every_n_epochs))

    # Remove --max_train_steps if present (epochs override it)
    i = 0
    while i < len(argv):
        if argv[i] == "--max_train_steps" and i + 1 < len(argv):
            argv.pop(i)
            argv.pop(i)
        else:
            i += 1

    return argv


def run_staged_training(
    plan: MixedResolutionPlan,
    base_argv: list[str],
    *,
    python_exe: Optional[str] = None,
) -> int:
    """Launch one subprocess per stage sequentially.

    Returns 0 on success, or the first non-zero exit code.
    Uses ``accelerate launch`` when ANIMA_ACCELERATE_LAUNCH is set,
    otherwise invokes train.py directly (single-GPU fast path).
    """
    py = python_exe or sys.executable
    use_accelerate = bool(os.environ.get("ANIMA_ACCELERATE_LAUNCH"))

    for i, stage in enumerate(plan.stages):
        stage_suffix = f"stage{i + 1}_{stage.side}px"
        argv = _build_phase_argv(
            base_argv, stage, i, len(plan.stages), plan, stage_suffix
        )

        if use_accelerate:
            cmd = [
                py,
                "-m",
                "accelerate.commands.accelerate_cli",
                "launch",
                "--num_cpu_threads_per_process",
                "3",
                "--mixed_precision",
                "bf16",
                "train.py",
                *argv,
            ]
        else:
            cmd = [py, "train.py", *argv]

        logger.info(
            "Staged resolution: launching stage %d/%d (side=%d, batch=%d, epochs=%d)",
            i + 1,
            len(plan.stages),
            stage.side,
            stage.batch_size,
            stage.epochs,
        )
        logger.info("  cmd: %s", " ".join(cmd))

        result = subprocess.run(cmd)
        if result.returncode != 0:
            logger.error(
                "Staged resolution: stage %d exited with code %d",
                i + 1,
                result.returncode,
            )
            return result.returncode

    logger.info(
        "Staged resolution: all %d stages completed successfully", len(plan.stages)
    )
    return 0
