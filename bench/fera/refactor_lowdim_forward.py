#!/usr/bin/env python
"""Bench: factored low-dim ortho FeRA forward vs naive Q_eff/P_eff expansion.

The ortho FeRA training forward used to materialize per-expert
``Q_eff (E, r, in)`` and ``P_eff (E, out, r)`` intermediates via
``_cayley_effective``, then einsum against ``x``. Since
``(P_basis, Q_basis)`` are frozen and shared across experts, the operation
factors:

    adapter = P_basis · [Σ_k w_k · R_p_k · diag(λ_k) · R_q_k · (Q_basis · x)]

Routing in r-dim instead of in/out-dim:

  * removes the (E, r, in) and (E, out, r) intermediates that autograd
    pinned for backward.
  * cuts dominant FLOPs from ``E·r·(in+out)`` to ``(r·in + r·out) + 4·E·r²``
    — roughly ``E×`` cheaper on the boundary matmuls.

This bench validates against the pre-refactor implementation (kept inline
in ``_naive_ortho_forward`` as the numerical oracle):

  1. **Numerical equivalence** — forward output + grads on ``S_q / S_p /
     λ`` between the two paths.
  2. **Activation memory** at backward time (``max_memory_allocated``).
  3. **Wall-clock** per fwd+bwd iteration.

Run::

    uv run python bench/fera/refactor_lowdim_forward.py

Writes ``result.json`` to ``bench/fera/results/<ts>-lowdim-forward/``.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

# Repo root on sys.path so ``bench._common`` and ``networks.methods.fera``
# both resolve when running this script directly (uv run python ...).
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench._common import make_run_dir, write_result  # noqa: E402
from networks.methods.fera import FeRALinear  # noqa: E402


# ---------------------------------------------------------------------------
# Reference (naive) ortho forward — frozen pre-refactor implementation.
# Materializes Q_eff / P_eff up front, exactly as fera.py used to do.
# ---------------------------------------------------------------------------


def _naive_ortho_forward(
    layer: FeRALinear, x: torch.Tensor, w: torch.Tensor
) -> torch.Tensor:
    base_out = layer.org_forward(x)
    Q_eff, P_eff = layer._cayley_effective()  # (E, r, in), (E, out, r)
    compute_dtype = layer.P_basis.dtype
    x_c = x if x.dtype == compute_dtype else x.to(compute_dtype)

    lx = torch.einsum("...i,eri->...er", x_c, Q_eff)
    lx = lx * layer.lambda_layer.to(compute_dtype)

    B = w.shape[0]
    E = w.shape[1]
    n_mid = lx.ndim - 3
    view_shape = (B,) + (1,) * n_mid + (E, 1)
    lx = lx * w.view(view_shape).to(compute_dtype)

    adapter = torch.einsum("...er,eor->...o", lx, P_eff)
    adapter = adapter * (layer.scale * layer._multiplier)
    return base_out + adapter.to(base_out.dtype)


# ---------------------------------------------------------------------------
# Layer construction
# ---------------------------------------------------------------------------


def _build_layer(
    in_features: int,
    out_features: int,
    rank: int,
    num_experts: int,
    device: torch.device,
    seed: int = 0,
):
    """Create a (frozen Linear + FeRALinear ortho adapter) pair on ``device``.

    Same seed → identical layer state across both bench paths.
    """
    torch.manual_seed(seed)
    base = nn.Linear(in_features, out_features, bias=False).to(device)
    base.weight.requires_grad_(False)

    layer = FeRALinear(
        base_layer=base,
        num_experts=num_experts,
        rank=rank,
        alpha=float(rank),
        lora_name="bench_lora",
        ortho=True,
        # Larger-than-default std → S_p/S_q are non-trivially non-zero so
        # the comparison exercises the full Cayley path (not just identity).
        ortho_init_std=0.1,
    )
    # Default λ init is zero (ΔW = 0 at init); push to non-zero so the
    # adapter contribution is non-trivial and gradients on S_q/S_p flow.
    with torch.no_grad():
        layer.lambda_layer.fill_(0.05)

    layer = layer.to(device)
    layer.apply_to()
    return layer


# ---------------------------------------------------------------------------
# Test 1: numerical equivalence (forward + grads)
# ---------------------------------------------------------------------------


def numerical_equivalence(
    in_features: int,
    out_features: int,
    rank: int,
    num_experts: int,
    batch: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, float]:
    layer = _build_layer(
        in_features, out_features, rank, num_experts, device
    )

    x = torch.randn(
        batch, seq_len, in_features, device=device, dtype=dtype, requires_grad=True
    )
    w = torch.softmax(torch.randn(batch, num_experts, device=device), dim=-1)
    layer.set_routing_weights(w)

    out_new = layer.forward(x)
    out_ref = _naive_ortho_forward(layer, x, w)

    fwd_max_abs = (out_new - out_ref).float().abs().max().item()
    fwd_scale = out_ref.float().abs().max().clamp_min(1e-12).item()
    fwd_rel = fwd_max_abs / fwd_scale

    # Grad equivalence: backprop a fixed random output-gradient through
    # each path independently, compare grads on the trainable params.
    g = torch.randn_like(out_new)

    def _grads(out):
        for p in (layer.S_q, layer.S_p, layer.lambda_layer):
            if p.grad is not None:
                p.grad = None
        out.backward(g, retain_graph=True)
        return {
            "S_q": layer.S_q.grad.detach().clone(),
            "S_p": layer.S_p.grad.detach().clone(),
            "lambda": layer.lambda_layer.grad.detach().clone(),
        }

    g_new = _grads(out_new)
    g_ref = _grads(out_ref)
    grad_diffs = {
        k: (g_new[k] - g_ref[k]).float().abs().max().item() for k in g_new
    }

    return {
        "forward_max_abs_diff": fwd_max_abs,
        "forward_rel_diff": fwd_rel,
        "grad_S_q_max_abs_diff": grad_diffs["S_q"],
        "grad_S_p_max_abs_diff": grad_diffs["S_p"],
        "grad_lambda_max_abs_diff": grad_diffs["lambda"],
    }


# ---------------------------------------------------------------------------
# Test 2: peak memory + step time
# ---------------------------------------------------------------------------


def memory_and_time(
    in_features: int,
    out_features: int,
    rank: int,
    num_experts: int,
    batch: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
    iters: int = 50,
    warmup: int = 5,
) -> Dict[str, Dict[str, float]]:
    """Measure ``peak_allocated`` + per-step wall time for each path.

    Each path gets a freshly-built layer (same seed → same params) and
    its own peak-memory window. Inputs are independently sampled per
    path; activation memory and step time are shape-determined so this
    doesn't bias the comparison.
    """
    out_metrics: Dict[str, Dict[str, float]] = {}

    for label, mode in (("new_lowdim", "new"), ("old_naive", "old")):
        layer = _build_layer(
            in_features, out_features, rank, num_experts, device
        )
        x = torch.randn(batch, seq_len, in_features, device=device, dtype=dtype)
        w = torch.softmax(torch.randn(batch, num_experts, device=device), dim=-1)
        layer.set_routing_weights(w)

        def _step() -> None:
            x_local = x.detach().clone().requires_grad_(True)
            if mode == "new":
                out = layer.forward(x_local)
            else:
                out = _naive_ortho_forward(layer, x_local, w)
            out.float().square().mean().backward()
            for p in (layer.S_q, layer.S_p, layer.lambda_layer):
                if p.grad is not None:
                    p.grad = None

        for _ in range(warmup):
            _step()
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _step()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        peak_mem = torch.cuda.max_memory_allocated()
        step_ms = (t1 - t0) / iters * 1e3

        out_metrics[label] = {
            "step_ms": step_ms,
            "peak_alloc_bytes": int(peak_mem),
            "peak_alloc_mib": peak_mem / (1024**2),
        }

        del layer, x, w
        gc.collect()
        torch.cuda.empty_cache()

    return out_metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_features", type=int, default=2048)
    ap.add_argument("--out_features", type=int, default=2048)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--num_experts", type=int, default=3)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq_len", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    ap.add_argument("--label", default="lowdim-forward")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for this bench.")
    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    print(
        f"Bench config: in={args.in_features} out={args.out_features} "
        f"E={args.num_experts} r={args.rank} "
        f"B={args.batch} T={args.seq_len} dtype={args.dtype}"
    )

    print("Step 1: numerical equivalence ...")
    eq = numerical_equivalence(
        args.in_features,
        args.out_features,
        args.rank,
        args.num_experts,
        args.batch,
        args.seq_len,
        device,
        dtype,
    )
    for k, v in eq.items():
        print(f"  {k}: {v:.3e}")

    print("Step 2: memory + step-time ...")
    mt = memory_and_time(
        args.in_features,
        args.out_features,
        args.rank,
        args.num_experts,
        args.batch,
        args.seq_len,
        device,
        dtype,
        iters=args.iters,
    )
    for path, m in mt.items():
        print(
            f"  {path:>10s}: {m['step_ms']:7.3f} ms/step, "
            f"peak {m['peak_alloc_mib']:7.2f} MiB"
        )

    delta_mib = mt["old_naive"]["peak_alloc_mib"] - mt["new_lowdim"]["peak_alloc_mib"]
    speedup = mt["old_naive"]["step_ms"] / mt["new_lowdim"]["step_ms"]
    print(
        f"  Δ peak memory: {delta_mib:+.2f} MiB (positive = new is smaller)"
    )
    print(f"  step-time speedup: {speedup:.2f}×")

    metrics = {
        "equivalence": eq,
        "memory_and_time": mt,
        "memory_delta_mib": delta_mib,
        "step_speedup": speedup,
    }

    run_dir = make_run_dir("fera", label=args.label)
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label,
        device=device,
    )
    print(f"Wrote results to {run_dir}")


if __name__ == "__main__":
    main()
