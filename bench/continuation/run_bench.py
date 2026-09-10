"""Measure optimizer/scheduler continuity across saved-state boundaries."""

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


def trajectory(
    cut: int | None = None, device: str = "cpu"
) -> tuple[torch.Tensor, list[float]]:
    args = SimpleNamespace(
        max_train_steps=10,
        lr_warmup_steps=0.4,
        lr_scheduler="constant_with_warmup",
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
    for step in range(16):
        if step == cut:
            saved = (
                weight.detach().clone(),
                copy.deepcopy(optimizer.state_dict()),
                copy.deepcopy(scheduler.state_dict()),
            )
            args.max_train_steps = 16
            args._continuation = dict(scheduler_steps=10, warmup_steps=4)
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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    started = time.perf_counter()
    reference, lrs = trajectory(device=device)
    comparisons = {}
    for cut in (2, 6, 10):
        resumed, resumed_lrs = trajectory(cut, device)
        comparisons[str(cut)] = dict(
            max_abs_error=(reference - resumed).abs().max().item(),
            identical_lrs=lrs == resumed_lrs,
        )
        assert torch.equal(reference, resumed) and lrs == resumed_lrs
    run = make_run_dir("continuation", label="scheduler-state")
    write_result(
        run,
        script=__file__,
        args={},
        device=device,
        metrics=dict(
            comparisons=comparisons, elapsed_seconds=time.perf_counter() - started
        ),
    )
    print(run)


if __name__ == "__main__":
    main()
