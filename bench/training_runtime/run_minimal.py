"""CPU regression bench for one-step schedules and three-stage boundaries.

Run with ``.venv/bin/python bench/training_runtime/run_minimal.py``. No model
weights, dataset, CUDA runtime or running daemon is needed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from bench._common import make_run_dir, write_result  # noqa: E402
from library.training.schedulers import make_warmup_cosine_scheduler  # noqa: E402
from library.training.stage_schedule import active_subset_indices_for_step  # noqa: E402
from library.training.staged_resolution import build_stage_schedule  # noqa: E402


def sample_order_replay_metrics():
    """Compare the former global-RNG replay with a saved loader permutation."""
    from accelerate import Accelerator

    accelerator = Accelerator(cpu=True)

    def loader(generator=None):
        return accelerator.prepare_data_loader(
            torch.utils.data.DataLoader(
                list(range(10)), batch_size=1, shuffle=True, generator=generator
            )
        )

    torch.manual_seed(7)
    old = iter(loader())
    consumed = [next(old).item() for _ in range(3)]
    saved_global = torch.get_rng_state()
    remaining = [item.item() for item in old]
    torch.set_rng_state(saved_global)
    old_replay = [item.item() for item in accelerator.skip_first_batches(loader(), 3)]

    generator = torch.Generator().manual_seed(7)
    epoch_start = generator.get_state()
    current = iter(loader(generator))
    new_consumed = [next(current).item() for _ in range(3)]
    expected = [item.item() for item in current]
    generator.set_state(epoch_start)
    new_replay = [
        item.item() for item in accelerator.skip_first_batches(loader(generator), 3)
    ]
    assert old_replay != remaining
    assert new_replay == expected
    return {
        "before_replay_matches": old_replay == remaining,
        "before_repeated_samples": sorted(set(consumed) & set(old_replay)),
        "after_replay_matches": new_replay == expected,
        "after_unique_samples": len(set(new_consumed + new_replay)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run_dir = make_run_dir("training_runtime", "minimal", root=args.output_root)
    schedules = []
    for total, warmup in ((1, 1), (2, 500), (1000, 20)):
        weight = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.AdamW([weight], lr=1e-3)
        scheduler = make_warmup_cosine_scheduler(
            optimizer, total, 1e-3, warmup_steps=warmup
        )
        for _ in range(total):
            weight.square().mean().backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        final_lr = scheduler.get_last_lr()[0]
        assert abs(final_lr - 1e-4) < 1e-12
        assert weight.item() < 1.0
        schedules.append(
            dict(total_steps=total, warmup_steps=warmup, final_lr=final_lr)
        )

    stage_args = SimpleNamespace(
        stage_schedule_enabled=True,
        stage_schedule=build_stage_schedule([100 / 3] * 3, [512, 768, 1024]),
        max_train_steps=3,
    )
    stages = [
        sorted(active_subset_indices_for_step(stage_args, step)) for step in range(3)
    ]
    assert stages == [[0], [1], [2]]
    output = write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics={
            "schedules": schedules,
            "active_stage_per_step": stages,
            "sample_replay": sample_order_replay_metrics(),
        },
        label="minimal",
        device="cpu",
    )
    print(output)


if __name__ == "__main__":
    main()
