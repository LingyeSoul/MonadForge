"""Measure optimizer/scheduler continuity across saved-state boundaries.

Scenarios (each compares a resumed trajectory against the uninterrupted
reference, asserting bit-equal weights and identical LR sequences):

- ``extend``           constant_with_warmup, budget 10 → 16 with the horizon
                       pinned at (10, 4): warmup must not be recomputed.
- ``cosine_resume``    cosine at its original target: pinning the horizon keeps
                       the decay curve identical after a mid-run save/load.
- ``chained_extend``   10 → 16 → 24 in two hops: the second extension inherits
                       the horizon of the first, matching a single 10 → 24 hop.
"""

from __future__ import annotations

import copy
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from bench._common import make_run_dir, write_result
from library.training.schedulers import get_scheduler_fix

HORIZON_STEPS = 10
HORIZON_WARMUP = 4  # int(0.4 * 10), the pinned proportional warmup


def _run(
    total: int,
    device: str = "cpu",
    *,
    scheduler: str = "constant_with_warmup",
    transitions: dict[int, int] | None = None,
) -> tuple[torch.Tensor, list[float]]:
    """Train ``total`` steps, re-pinning the scheduler at ``transitions``
    ``{at_step: new_budget}`` by simulating a save → extend → load cycle."""
    transitions = transitions or {}
    args = SimpleNamespace(
        max_train_steps=HORIZON_STEPS,
        lr_warmup_steps=0.4,
        lr_scheduler=scheduler,
        lr_scheduler_args=[],
        lr_scheduler_type=None,
        optimizer_type="AdamW",
    )

    def build():
        weight = torch.nn.Parameter(torch.ones(8, device=device))
        optimizer = torch.optim.AdamW([weight], lr=0.01)
        scheduler = get_scheduler_fix(args, optimizer, 1)
        return weight, optimizer, scheduler

    weight, optimizer, scheduler = build()
    lrs = []
    for step in range(total):
        if step in transitions:
            saved = (
                weight.detach().clone(),
                copy.deepcopy(optimizer.state_dict()),
                copy.deepcopy(scheduler.state_dict()),
            )
            args.max_train_steps = transitions[step]
            args._continuation = dict(
                scheduler_steps=HORIZON_STEPS, warmup_steps=HORIZON_WARMUP
            )
            weight, optimizer, scheduler = build()
            weight.data.copy_(saved[0])
            optimizer.load_state_dict(saved[1])
            scheduler.load_state_dict(saved[2])
        lrs.append(optimizer.param_groups[0]["lr"])
        weight.grad = torch.full_like(weight, 0.1 + step / 100)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    return weight.detach(), lrs


def trajectory(cut: int | None = None, device: str = "cpu") -> tuple[torch.Tensor, list[float]]:
    """Backward-compatible entry: extend 10 → 16, restarting at ``cut``."""
    return _run(16, device, transitions={cut: 16} if cut is not None else None)


def cosine_resume(device: str = "cpu") -> tuple[tuple[torch.Tensor, list[float]], tuple[torch.Tensor, list[float]]]:
    reference = _run(HORIZON_STEPS, device, scheduler="cosine")
    resumed = _run(
        HORIZON_STEPS, device, scheduler="cosine", transitions={4: HORIZON_STEPS}
    )
    return reference, resumed


def chained_extend(device: str = "cpu") -> tuple[tuple[torch.Tensor, list[float]], tuple[torch.Tensor, list[float]]]:
    single_hop = _run(24, device, transitions={2: 24})
    two_hops = _run(24, device, transitions={2: 16, 6: 24})
    return single_hop, two_hops


def _compare(reference, resumed) -> dict:
    weight_ref, lrs_ref = reference
    weight_act, lrs_act = resumed
    metrics = dict(
        max_abs_error=(weight_ref - weight_act).abs().max().item(),
        identical_lrs=lrs_ref == lrs_act,
    )
    assert torch.equal(weight_ref, weight_act) and lrs_ref == lrs_act
    return metrics


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    started = time.perf_counter()
    reference, lrs = trajectory(device=device)
    extend = {
        str(cut): _compare((reference, lrs), trajectory(cut, device))
        for cut in (2, 6, 10)
    }
    cosine = {"4": _compare(*cosine_resume(device))}
    chained = {"2-6": _compare(*chained_extend(device))}
    run = make_run_dir("continuation", label="scheduler-state")
    write_result(
        run,
        script=__file__,
        args={},
        device=device,
        metrics=dict(
            extend=extend,
            cosine_resume=cosine,
            chained_extend=chained,
            elapsed_seconds=time.perf_counter() - started,
        ),
    )
    print(run)


if __name__ == "__main__":
    main()
