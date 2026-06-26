"""Automagic optimizer convergence probe — AdamW vs Automagic on a toy fit.

Question: does the per-parameter adaptive-LR Automagic optimizer (Adafactor-style
factored second moment + element-wise LR-mask polarity adaptation) actually
descend on a real loss landscape, and how does its trajectory compare to a
well-tuned AdamW baseline?

Probe: a synthetic linear-regression fit. We freeze nothing — both optimizers
get the same fresh ``nn.Linear`` student (weight + bias) and fit a fixed random
target ``y = W*·x + b*`` under MSE. We record per-step train loss for each arm
and report init/final MSE plus the ratio of final losses.

Why linear regression: convex MSE ⇒ any correct preconditioned-gradient step is
a descent direction, so this is a clean *correctness* probe (Automagic must
reduce loss monotonically-ish) rather than a statement about LoRA-training
quality. A divergence here would flag a sign/preconditioner bug.

CPU, a few hundred steps — runs in seconds. Invocation:
    uv run python bench/automagic/probe_convergence.py [--steps 400] [--label verify]
"""

from __future__ import annotations

import argparse
import csv

import torch

from bench._common import make_run_dir, write_result
from library.training.automagic import Automagic

NAME = "probe_convergence"


def add_args(p):
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--in_dim", type=int, default=32)
    p.add_argument("--out_dim", type=int, default=16)
    p.add_argument("--n_samples", type=int, default=128)
    p.add_argument("--adamw_lr", type=float, default=1e-2)
    p.add_argument("--automagic_lr", type=float, default=1e-3,
                   help="Automagic starting LR (it adapts per-element from here)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--label", default=None)


def _make_task(args, device):
    """Fixed (x, y) regression target + a fresh student model factory."""
    g = torch.Generator(device=device).manual_seed(args.seed)
    # teacher weight/bias — the fit target
    W_star = torch.randn(args.out_dim, args.in_dim, generator=g, device=device)
    b_star = torch.randn(args.out_dim, generator=g, device=device)
    x = torch.randn(args.n_samples, args.in_dim, generator=g, device=device)
    y = x @ W_star.T + b_star  # (n_samples, out_dim)
    return x, y


def _new_student(args, device):
    """A fresh student Linear; seeded so both arms start identically."""
    torch.manual_seed(args.seed + 1)
    m = torch.nn.Linear(args.in_dim, args.out_dim, bias=True).to(device)
    return m


def _fit(name, model, x, y, build_opt, steps):
    """Run `steps` of MSE fitting; return (init_loss, final_loss, per_step_losses)."""
    opt = build_opt([p for p in model.parameters() if p.requires_grad])
    losses = []
    loss = torch.nn.functional.mse_loss(model(x), y)
    losses.append(loss.item())
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    print(
        f"{name:12s} init_mse={losses[0]:.4e} final_mse={losses[-1]:.4e} "
        f"reduction={losses[0] / max(losses[-1], 1e-30):.1f}x"
    )
    return losses


def run(args):
    device = torch.device("cpu")
    torch.manual_seed(args.seed)
    x, y = _make_task(args, device)

    arms = {
        "AdamW": lambda p: torch.optim.AdamW(p, lr=args.adamw_lr),
        "Automagic": lambda p: Automagic(
            p, lr=args.automagic_lr, min_lr=1e-7, max_lr=1e-2, lr_bump=1e-5
        ),
    }

    per_arm = {}
    for name, build_opt in arms.items():
        model = _new_student(args, device)
        losses = _fit(name, model, x, y, build_opt, args.steps)
        per_arm[name] = {
            "init_mse": losses[0],
            "final_mse": losses[-1],
            "min_mse": min(losses),
            "per_step": losses,
        }

    # Correctness gate: Automagic must reduce loss (descent-direction invariant).
    auto = per_arm["Automagic"]
    automagic_descends = auto["final_mse"] < auto["init_mse"]

    metrics = {
        "steps": args.steps,
        "automagic_descends": automagic_descends,
        "final_mse": {k: v["final_mse"] for k, v in per_arm.items()},
        "init_mse": {k: v["init_mse"] for k, v in per_arm.items()},
        "min_mse": {k: v["min_mse"] for k, v in per_arm.items()},
        "final_ratio_automagic_over_adamw": (
            auto["final_mse"] / max(per_arm["AdamW"]["final_mse"], 1e-30)
        ),
    }

    label = f"{NAME}-{args.label}" if args.label else NAME
    run_dir = make_run_dir("automagic", label=label)

    # per-step loss curves (both arms, aligned columns)
    curve = run_dir / "loss_curves.csv"
    with curve.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", *(f"{k}_mse" for k in per_arm)])
        n = max(len(v["per_step"]) for v in per_arm.values())
        for i in range(n):
            row = [i]
            for k, v in per_arm.items():
                row.append(f"{v['per_step'][i]:.6e}" if i < len(v["per_step"]) else "")
            w.writerow(row)

    write_result(
        run_dir, script=__file__, args=vars(args), metrics=metrics, device="cpu",
        artifacts=[curve], label=label,
    )

    print(f"\n  Automagic descends: {automagic_descends}")
    print(
        f"  final MSE ratio (Automagic/AdamW): "
        f"{metrics['final_ratio_automagic_over_adamw']:.3f}"
    )
    print(f"wrote {run_dir / 'result.json'}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_args(p)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
