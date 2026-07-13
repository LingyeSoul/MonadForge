"""CAME multi-resolution optimizer probe on a tiny convolutional fit.

The student alternates through three spatial resolutions while fitting a fixed
teacher convolution. The probe checks that CAME reduces the aggregate loss,
keeps all values finite, and retains resolution-independent optimizer state.

CPU invocation:
    uv run python bench/came/probe_multires.py --steps 48 --label verify
"""

from __future__ import annotations

import argparse
import csv

import torch

from bench._common import make_run_dir, write_result
from library.training.optimizers import get_optimizer

NAME = "probe_multires"
RESOLUTIONS = ((8, 8), (12, 16), (16, 12))


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label", default=None)


def _make_tasks(seed: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    teacher = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1)
    with torch.no_grad():
        teacher.weight.copy_(torch.randn(teacher.weight.shape, generator=generator))
        teacher.bias.copy_(torch.randn(teacher.bias.shape, generator=generator))

    tasks = []
    for height, width in RESOLUTIONS:
        x = torch.randn(2, 3, height, width, generator=generator)
        with torch.no_grad():
            target = teacher(x)
        tasks.append((x, target))
    return tasks


def _aggregate_loss(
    model: torch.nn.Module, tasks: list[tuple[torch.Tensor, torch.Tensor]]
) -> float:
    with torch.no_grad():
        losses = [torch.nn.functional.mse_loss(model(x), target) for x, target in tasks]
    return float(torch.stack(losses).mean())


def run(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed + 1)
    model = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1)
    tasks = _make_tasks(args.seed)
    optimizer_args = argparse.Namespace(
        optimizer_type="CAME",
        optimizer_args=["betas=0.9,0.999"],
        learning_rate=args.learning_rate,
        lr_scheduler="constant",
        max_grad_norm=0.0,
    )
    _, _, optimizer = get_optimizer(optimizer_args, model.parameters())

    initial_loss = _aggregate_loss(model, tasks)
    curve: list[float] = []
    first_state_shapes = None
    finite = True
    state_shapes_invariant = True

    for step in range(args.steps):
        x, target = tasks[step % len(tasks)]
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), target)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach())
        curve.append(loss_value)
        finite = finite and torch.isfinite(loss.detach()).item()

        state_shapes = {
            key: tuple(value.shape)
            for key, value in optimizer.state[model.weight].items()
            if torch.is_tensor(value)
        }
        if first_state_shapes is None:
            first_state_shapes = state_shapes
        elif state_shapes != first_state_shapes:
            state_shapes_invariant = False

    final_loss = _aggregate_loss(model, tasks)
    metrics = {
        "resolutions": [list(shape) for shape in RESOLUTIONS],
        "steps": args.steps,
        "finite": finite,
        "state_shapes_invariant": (
            first_state_shapes is not None and state_shapes_invariant
        ),
        "initial_aggregate_mse": initial_loss,
        "final_aggregate_mse": final_loss,
        "loss_reduction": initial_loss / max(final_loss, 1e-30),
        "came_descends": finite and final_loss < initial_loss,
    }

    label = f"{NAME}-{args.label}" if args.label else NAME
    run_dir = make_run_dir("came", label=label)
    curve_path = run_dir / "loss_curve.csv"
    with curve_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "height", "width", "mse"])
        for step, loss in enumerate(curve):
            height, width = RESOLUTIONS[step % len(RESOLUTIONS)]
            writer.writerow([step, height, width, f"{loss:.8e}"])

    write_result(
        run_dir,
        script=__file__,
        args=vars(args),
        metrics=metrics,
        artifacts=[curve_path],
        device="cpu",
        label=label,
    )
    print(
        f"CAME multires: init={initial_loss:.6e} final={final_loss:.6e} "
        f"reduction={metrics['loss_reduction']:.3f}x finite={finite}"
    )
    print(f"wrote {run_dir / 'result.json'}")

    if not metrics["came_descends"] or not metrics["state_shapes_invariant"]:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_args(parser)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
