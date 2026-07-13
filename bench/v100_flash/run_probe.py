"""V100 FlashAttention stability probe.

This is a small, self-contained diagnostic for the experimental
``flash-attention-v100`` backend.  It runs a tiny Anima forward/backward under
fp16 autocast and reports whether outputs and trainable gradients stay finite.
Use it on a V100 environment before/after trying full training.

Examples::

    python -m bench.v100_flash.run_probe --attn_mode flash --stability hybrid --device cuda
    ANIMA_DEBUG_FINITE=1 python -m bench.v100_flash.run_probe --attn_mode flash --stability safe
"""

from __future__ import annotations

import argparse
import time

import torch

from bench._common import make_run_dir, write_result


def _build_tiny_anima(*, attn_mode: str, stability: str, debug_finite: bool, num_blocks: int):
    from library.anima.models import Anima

    model = Anima(
        max_img_h=16,
        max_img_w=16,
        max_frames=1,
        in_channels=16,
        out_channels=16,
        patch_spatial=2,
        patch_temporal=1,
        concat_padding_mask=False,
        model_channels=64,
        num_blocks=num_blocks,
        num_heads=4,
        mlp_ratio=2.0,
        crossattn_emb_channels=64,
        use_adaln_lora=True,
        adaln_lora_dim=16,
        use_llm_adapter=False,
        attn_mode=attn_mode,
        v100_flash_stability=stability,
        debug_finite_checks=debug_finite,
    ).train()
    model.enable_fp32_residual()
    return model


def _make_inputs(device: torch.device):
    torch.manual_seed(0)
    x = torch.randn(1, 16, 1, 4, 4, device=device)
    timesteps = torch.tensor([0.5], device=device)
    crossattn_emb = torch.randn(1, 8, 64, device=device)
    target = torch.randn_like(x)
    return x, timesteps, crossattn_emb, target


def _grads_finite(model) -> bool:
    for p in model.parameters():
        if p.grad is not None and not torch.isfinite(p.grad.detach()).all():
            return False
    return True


def _run(args):
    device = torch.device(args.device)
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
        props = torch.cuda.get_device_properties(device)
        gpu = {"name": props.name, "sm": f"{props.major}.{props.minor}"}
    else:
        gpu = {"name": "cpu", "sm": "n/a"}

    # Keep model initialization identical across separate backend runs so loss,
    # timing, and memory comparisons change only with the selected backend.
    torch.manual_seed(0)
    model = _build_tiny_anima(
        attn_mode=args.attn_mode,
        stability=args.stability,
        debug_finite=args.debug_finite,
        num_blocks=args.num_blocks,
    ).to(device)
    x, timesteps, crossattn_emb, target = _make_inputs(device)

    dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
    autocast_device = "cuda" if device.type == "cuda" else "cpu"

    results = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for step in range(args.steps):
        model.zero_grad(set_to_none=True)
        with torch.autocast(autocast_device, dtype=dtype):
            out = model.forward_mini_train_dit(x, timesteps, crossattn_emb)
            loss = (out - target).float().square().mean()
        out_finite = bool(torch.isfinite(out.detach()).all().item())
        loss_finite = bool(torch.isfinite(loss.detach()).all().item())
        if loss_finite:
            loss.backward()
        grad_finite = _grads_finite(model)
        results.append(
            {
                "step": step,
                "out_finite": out_finite,
                "loss_finite": loss_finite,
                "grad_finite": grad_finite,
                "loss": float(loss.detach().cpu()) if loss_finite else float("nan"),
            }
        )
        if not (out_finite and loss_finite and grad_finite):
            break
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0

    peak_allocated_mib = None
    peak_reserved_mib = None
    if device.type == "cuda":
        mib = 1024**2
        peak_allocated_mib = round(torch.cuda.max_memory_allocated(device) / mib, 3)
        peak_reserved_mib = round(torch.cuda.max_memory_reserved(device) / mib, 3)

    return {
        "device": args.device,
        "gpu": gpu,
        "attn_mode": args.attn_mode,
        "stability": args.stability,
        "mixed_precision": args.mixed_precision,
        "debug_finite": args.debug_finite,
        "num_blocks": args.num_blocks,
        "steps_requested": args.steps,
        "steps_run": len(results),
        "all_finite": all(
            r["out_finite"] and r["loss_finite"] and r["grad_finite"] for r in results
        ),
        "ms_per_step": round((elapsed / max(1, len(results))) * 1000.0, 3),
        "peak_allocated_mib": peak_allocated_mib,
        "peak_reserved_mib": peak_reserved_mib,
        "steps": results,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--attn_mode",
        choices=["torch", "mem_efficient", "flash", "flex"],
        default="flash",
    )
    p.add_argument("--stability", choices=["off", "hybrid", "safe"], default="hybrid")
    p.add_argument("--mixed_precision", choices=["fp16", "bf16"], default="fp16")
    p.add_argument("--debug_finite", action="store_true")
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--num_blocks", type=int, default=4)
    p.add_argument("--label", default=None)
    args = p.parse_args()

    metrics = _run(args)
    run_dir = make_run_dir("v100_flash", label=args.label)
    write_result(run_dir, script=__file__, args=args, metrics=metrics)

    print(f"=== v100_flash probe → {run_dir} ===")
    for k, v in metrics.items():
        if k == "steps":
            print("  steps:")
            for row in v:
                print(f"    {row}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
