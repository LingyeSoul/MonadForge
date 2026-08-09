#!/usr/bin/env python
"""Equivalence / adapter-grad probe for ConvRot W8A* base path.

Mirrors the int8 linear probe style without touching the training entrypoint
by default. Toy path is CI-friendly; optional multi-seed summary prints
output/loss/adapter-grad relative errors.

Examples::

    .venv/bin/python scripts/experiments/convrot_equivalence_probe.py
    .venv/bin/python scripts/experiments/convrot_equivalence_probe.py --mode w8a8 --seeds 0,1,2
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

# The standalone probe must add the repository root before importing project modules.
# ruff: noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from torch.nn import functional as F

from library.runtime.convrot.apply import apply_convrot_to_lora_network


@dataclass(frozen=True)
class SeedMetrics:
    seed: int
    output_rel_l2: float
    loss_rel: float
    adapter_grad_rel: float
    patched: int
    skipped: int


class _FakeLoRAModule(nn.Module):
    def __init__(self, name: str, linear: nn.Linear, rank: int = 4) -> None:
        super().__init__()
        self.lora_name = f"lora_unet_{name.replace('.', '_')}"
        self.original_name = name
        self.org_module_ref = [linear]
        self.org_forward = linear.forward
        self.lora_down = nn.Linear(linear.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, linear.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.org_forward(x) + self.lora_up(self.lora_down(x))


class _FakeLoRANetwork(nn.Module):
    def __init__(self, loras: list[_FakeLoRAModule]) -> None:
        super().__init__()
        self.unet_loras = nn.ModuleList(loras)


def _frozen_linear(in_f: int, out_f: int, seed: int) -> nn.Linear:
    g = torch.Generator().manual_seed(seed)
    linear = nn.Linear(in_f, out_f, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.randn(out_f, in_f, generator=g) * 0.05)
    linear.weight.requires_grad_(False)
    return linear


def _clone_network_state(network: _FakeLoRANetwork) -> dict[str, torch.Tensor]:
    return {
        k: v.detach().clone()
        for k, v in network.state_dict().items()
        if v.requires_grad or "lora_" in k
    }


def run_seed(
    *,
    seed: int,
    mode: str,
    group_size: int,
    dim: int,
    hidden: int,
) -> SeedMetrics:
    torch.manual_seed(seed)
    layer1 = _frozen_linear(dim, hidden, seed=seed + 11)
    layer2 = _frozen_linear(hidden, dim, seed=seed + 22)
    loras = [
        _FakeLoRAModule("blocks.0.mlp.layer1", layer1),
        _FakeLoRAModule("blocks.0.mlp.layer2", layer2),
    ]
    network = _FakeLoRANetwork(loras)
    # Shared adapter init for bf16 vs convrot comparison
    for lora in network.unet_loras:
        nn.init.normal_(lora.lora_down.weight, std=0.02)
        nn.init.zeros_(lora.lora_up.weight)

    x = torch.randn(4, dim, generator=torch.Generator().manual_seed(seed + 99))
    target = torch.randn(4, dim, generator=torch.Generator().manual_seed(seed + 77))

    def _forward() -> torch.Tensor:
        h = network.unet_loras[0](x)
        h = F.gelu(h)
        return network.unet_loras[1](h)

    # bf16 baseline
    y_ref = _forward()
    loss_ref = F.mse_loss(y_ref, target)
    network.zero_grad(set_to_none=True)
    loss_ref.backward()
    grad_ref = torch.cat(
        [p.grad.detach().flatten() for p in network.parameters() if p.grad is not None]
    )

    # Reset adapter grads; re-apply same adapter weights after convrot patch
    adapter_state = {
        k: v.detach().clone()
        for k, v in network.state_dict().items()
        if "lora_" in k
    }
    for p in network.parameters():
        if p.grad is not None:
            p.grad = None

    result = apply_convrot_to_lora_network(
        network,
        mode=mode,  # type: ignore[arg-type]
        scope="mlp",
        group_size=group_size,
    )
    network.load_state_dict(adapter_state, strict=False)

    y = _forward()
    loss = F.mse_loss(y, target)
    network.zero_grad(set_to_none=True)
    loss.backward()
    grad = torch.cat(
        [p.grad.detach().flatten() for p in network.parameters() if p.grad is not None]
    )

    out_rel = float(
        (y.detach() - y_ref.detach()).norm()
        / y_ref.detach().norm().clamp_min(1e-8)
    )
    loss_rel = float(
        abs(loss.detach() - loss_ref.detach())
        / loss_ref.detach().clamp_min(1e-8)
    )
    grad_rel = float(
        (grad - grad_ref).norm() / grad_ref.norm().clamp_min(1e-8)
    )
    return SeedMetrics(
        seed=seed,
        output_rel_l2=out_rel,
        loss_rel=loss_rel,
        adapter_grad_rel=grad_rel,
        patched=result.patched_count,
        skipped=result.skipped_count,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["w8a16", "w8a8"], default="w8a16")
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated seeds for multi-seed smoke.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON summary.")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    metrics = [
        run_seed(
            seed=seed,
            mode=args.mode,
            group_size=args.group_size,
            dim=args.dim,
            hidden=args.hidden,
        )
        for seed in seeds
    ]

    summary = {
        "mode": args.mode,
        "group_size": args.group_size,
        "seeds": seeds,
        "metrics": [asdict(m) for m in metrics],
        "max_output_rel_l2": max(m.output_rel_l2 for m in metrics),
        "max_adapter_grad_rel": max(m.adapter_grad_rel for m in metrics),
        "mean_adapter_grad_rel": sum(m.adapter_grad_rel for m in metrics) / len(metrics),
        "gates": {
            "output_rel_l2_le_3pct": all(m.output_rel_l2 <= 0.03 for m in metrics),
            "adapter_grad_rel_le_5pct": all(m.adapter_grad_rel <= 0.05 for m in metrics),
        },
    }

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(
            f"[convrot-probe] mode={args.mode} group={args.group_size} "
            f"seeds={seeds}"
        )
        for m in metrics:
            print(
                f"  seed={m.seed} out_rel={m.output_rel_l2:.4f} "
                f"loss_rel={m.loss_rel:.4f} grad_rel={m.adapter_grad_rel:.4f} "
                f"patched={m.patched} skipped={m.skipped}"
            )
        print(
            f"  max_out_rel={summary['max_output_rel_l2']:.4f} "
            f"max_grad_rel={summary['max_adapter_grad_rel']:.4f} "
            f"mean_grad_rel={summary['mean_adapter_grad_rel']:.4f}"
        )
        print(f"  gates={summary['gates']}")

    ok = summary["gates"]["output_rel_l2_le_3pct"] and summary["gates"][
        "adapter_grad_rel_le_5pct"
    ]
    # W8A8 smoke is looser; still report gates but only hard-fail W8A16.
    if args.mode == "w8a16" and not ok:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
