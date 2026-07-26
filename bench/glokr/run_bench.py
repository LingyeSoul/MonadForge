"""GLoKr (Kronecker + BoRA) invariant + magnitude-dynamics bench (Tier 2 gate).

Four surfaces, all CPU-analytical (no DiT load — ``bench/_anima.py`` is opt-in
per CONTRIBUTING.md and every number here is computable from synthetic
Linears at real Anima DiT shapes):

1. **Parameter accounting** — trainable params for GLoKr (kron factors +
   ``m_row``/``m_col`` magnitudes) vs plain LoRA and plain LoKr at the same
   ``(shape, lora_dim, factor)``. The BoRA overhead is exactly ``out + in``
   scalars per Linear.

2. **Invariant errors** (fp32, must be ~1e-6):
   * init identity: ``max |W' − W0|`` with ΔW = 0 and magnitudes = W0 norms;
   * merge parity: ``merge_to(state_dict)`` vs ``W0 + get_weight()``;
   * fuse round-trip: ``max |W0 − unfuse(fuse(W0))|``.

3. **Forward overhead** — wall time of the weight-decomposed forward
   (materialize W' + one GEMM) vs the plain base GEMM at a DiT-typical token
   count (interleaved min-of-N so scheduler noise can't invert the ratio).
   Indicative CPU timing; the point is the *ratio* (W' construction is
   elementwise over out×in and should be small next to the token GEMM).

4. **Row-rescale capability** (the falsifier): fit ``y = x @ (d ⊙ W0)^T`` — a
   pure per-output-row gain — with three arms sharing identical Kronecker
   direction params: GLoKr+BoRA, a DoRA-style column-only decomposition, and
   the plain additive delta. BoRA's ``m_row`` parameterizes the target
   transform directly; the baselines cannot represent it (no row magnitudes /
   rank-capped). **What would falsify the method**:
   ``bora_wins_row_rescale_task`` must be True (BoRA final loss < 0.5× both
   baselines); otherwise the bi-dimensional decomposition is not delivering
   the capability the paper claims and ``glokr_bora`` should not ship enabled
   by default. ΔM(t, t−1, d) per the paper's eq. (5) is logged as context.

Usage::

    python bench/glokr/run_bench.py
    python bench/glokr/run_bench.py --label <date> --lora_dim 4 16 32

Drops a ``result.json`` envelope (``bench/_common.py``) into
``bench/glokr/results/<ts>[-<label>]/``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench._common import make_run_dir, write_result  # noqa: E402
from networks.lora_modules.glokr import GLoKRModule  # noqa: E402

# Real Anima DiT Linear shapes (out, in): self-attn q/k/v/o + ffn projections.
DEFAULT_SHAPES: tuple[tuple[int, int], ...] = (
    (3072, 3072),
    (12288, 3072),
    (3072, 12288),
)


def _build(out_dim: int, in_dim: int, rank: int, factor: int, *, bora: bool):
    base = torch.nn.Linear(in_dim, out_dim, bias=False)
    base.weight.requires_grad_(False)
    module = GLoKRModule(
        "bench",
        base,
        lora_dim=rank,
        alpha=rank,
        glokr_factor=factor,
        bora=bora,
    )
    return base, module


def _param_count(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def bench_invariants(shapes, ranks, factor):
    rows = []
    for out_dim, in_dim in shapes:
        for rank in ranks:
            base, module = _build(out_dim, in_dim, rank, factor, bora=True)
            w0 = base.weight.detach().clone()

            init_err = module.get_weight().abs().max().item()

            with torch.no_grad():
                if module.use_w2:
                    module.glokr_w2.normal_(0, 0.02)
                else:
                    module.glokr_w2_b.normal_(0, 0.02)
                module.bora_m_row.mul_(1.03)

            expected = w0.float() + module.get_weight()
            sd = dict(module.state_dict())
            module.merge_to(sd, dtype=torch.float32, device="cpu")
            merge_err = (base.weight - expected).abs().max().item()

            with torch.no_grad():
                base.weight.copy_(w0)
            module.fuse_weight()
            module.unfuse_weight()
            fuse_err = (base.weight - w0).abs().max().item()

            lora_params = rank * (in_dim + out_dim)
            glokr_params = _param_count(module)
            kron_params = glokr_params - (out_dim + in_dim)
            rows.append(
                {
                    "shape": f"{out_dim}x{in_dim}",
                    "rank": rank,
                    "factor": factor,
                    "params_glokr": glokr_params,
                    "params_kron_only": kron_params,
                    "params_bora_magnitudes": out_dim + in_dim,
                    "params_plain_lora": lora_params,
                    "init_identity_max_err": init_err,
                    "merge_parity_max_err": merge_err,
                    "fuse_roundtrip_max_err": fuse_err,
                }
            )
    return rows


def bench_forward_overhead(rank: int, factor: int, tokens: int = 4096, iters: int = 9):
    """Interleaved min-of-N so scheduler noise can't invert the ratio (the
    adapter path is a strict superset of the plain GEMM's work)."""
    out_dim = in_dim = 3072
    base, module = _build(out_dim, in_dim, rank, factor, bora=True)
    module.apply_to()
    module.eval()
    x = torch.randn(1, tokens, in_dim)

    adapter_times: list[float] = []
    plain_times: list[float] = []
    with torch.no_grad():
        base(x)  # warmup adapter path
        module.enabled = False
        base(x)  # warmup plain path
        module.enabled = True
        for _ in range(iters):
            t0 = time.perf_counter()
            base(x)
            adapter_times.append(time.perf_counter() - t0)
            module.enabled = False
            t0 = time.perf_counter()
            base(x)
            plain_times.append(time.perf_counter() - t0)
            module.enabled = True

    adapter_s = min(adapter_times)
    plain_s = min(plain_times)
    return {
        "tokens": tokens,
        "shape": f"{out_dim}x{in_dim}",
        "plain_forward_s": plain_s,
        "glokr_forward_s": adapter_s,
        "overhead_ratio": adapter_s / plain_s if plain_s > 0 else float("inf"),
    }


def bench_magnitude_dynamics(
    rank: int, factor: int, steps: int = 200, lr: float = 1e-2
):
    """BoRA row-magnitude capability probe + §3.3 ΔM dynamics.

    Task: fit ``y = x @ (d_row ⊙ W0)^T`` — a pure ROW rescaling of the frozen
    base (per-output-neuron gains drawn from U[0.5, 2.0]). This is exactly the
    transform BoRA's ``m_row`` parameterizes directly; the two baselines can
    only approximate it:
      * ``dora_col``   — DoRA-style column-only decomposition (no row
        magnitudes; the asymmetry the paper argues against);
      * ``plain_kron`` — additive Kronecker delta ((d_row−1)⊙W0 is full-rank,
        far above the kron delta's effective rank).

    All arms share the identical Kronecker direction parameterization + LR.
    ΔM per eq. (5) (mean |per-step change of effective row/col norms|) is
    reported as descriptive data.
    """
    torch.manual_seed(0)
    out_dim = in_dim = 512
    x = torch.randn(256, in_dim)
    d_gain = torch.rand(out_dim, 1) * 1.5 + 0.5  # U[0.5, 2.0] per output row

    def run_arm(label: str):
        torch.manual_seed(1)
        base, module = _build(
            out_dim, in_dim, rank, factor, bora=(label == "glokr_bora")
        )
        w0 = base.weight.detach().float()
        target = x @ (d_gain * w0).t()  # pure row-rescale of the frozen base
        params = [p for p in module.parameters() if p.requires_grad]
        m_col = None
        if label == "dora_col":
            m_col = torch.nn.Parameter(w0.norm(dim=0, keepdim=True))
            params.append(m_col)
        # Adam matches the trainer's optimizer family; SGD stalls on the
        # magnitude params (their grads are ~1/out_dim of the loss scale).
        opt = torch.optim.Adam(params, lr=lr)

        def effective_weight() -> torch.Tensor:
            w = w0 + module._delta_weight(gate_rank=False)
            if label == "glokr_bora":
                return module._bora_compose(w)
            if label == "dora_col":
                col_norm = w.norm(dim=0, keepdim=True).clamp_min(1e-8).detach()
                return m_col * (w / col_norm)
            return w

        prev_row = prev_col = None
        dm_row, dm_col = [], []
        loss = None
        for _ in range(steps):
            opt.zero_grad()
            w_eff = effective_weight()
            loss = (x @ w_eff.t() - target).square().mean()
            loss.backward()
            opt.step()
            with torch.no_grad():
                w_now = effective_weight()
                row = w_now.norm(dim=1)
                col = w_now.norm(dim=0)
            if prev_row is not None:
                dm_row.append((row - prev_row).abs().mean().item())
                dm_col.append((col - prev_col).abs().mean().item())
            prev_row, prev_col = row, col
        mean_row = sum(dm_row) / len(dm_row)
        mean_col = sum(dm_col) / len(dm_col)
        return {
            "mean_dM_row": mean_row,
            "mean_dM_col": mean_col,
            "row_col_symmetry": mean_row / mean_col if mean_col else float("inf"),
            "final_loss": loss.item(),
        }

    results = {
        label: run_arm(label) for label in ("glokr_bora", "dora_col", "plain_kron")
    }
    bora_loss = results["glokr_bora"]["final_loss"]
    # Falsifier: on a pure row-rescale target, the bi-dimensional decomposition
    # must beat BOTH the column-only (DoRA-style) arm and the additive kron arm
    # by a wide margin — m_row parameterizes the target transform directly.
    results["bora_wins_row_rescale_task"] = bool(
        bora_loss < 0.5 * results["dora_col"]["final_loss"]
        and bora_loss < 0.5 * results["plain_kron"]["final_loss"]
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", default=None)
    parser.add_argument("--lora_dim", type=int, nargs="+", default=[4, 16, 32])
    parser.add_argument("--factor", type=int, default=8)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    invariants = bench_invariants(DEFAULT_SHAPES, args.lora_dim, args.factor)
    overhead = bench_forward_overhead(max(args.lora_dim), args.factor)
    dynamics = bench_magnitude_dynamics(
        max(args.lora_dim), args.factor, steps=args.steps
    )

    metrics = {
        "invariants": invariants,
        "forward_overhead": overhead,
        "magnitude_dynamics": dynamics,
    }
    run_dir = make_run_dir("glokr", label=args.label)
    write_result(run_dir, script=__file__, args=args, metrics=metrics)

    print(f"\nresult.json -> {run_dir}\n")
    print(
        f"{'shape':>12} {'r':>3} {'glokr':>10} {'kron':>10} {'lora':>10} "
        f"{'init_err':>9} {'merge_err':>9} {'fuse_err':>9}"
    )
    for row in invariants:
        print(
            f"{row['shape']:>12} {row['rank']:>3} {row['params_glokr']:>10} "
            f"{row['params_kron_only']:>10} {row['params_plain_lora']:>10} "
            f"{row['init_identity_max_err']:>9.1e} "
            f"{row['merge_parity_max_err']:>9.1e} "
            f"{row['fuse_roundtrip_max_err']:>9.1e}"
        )
    print(
        f"\nforward overhead @ {overhead['tokens']} tok, {overhead['shape']}: "
        f"{overhead['overhead_ratio']:.2f}x "
        f"({overhead['plain_forward_s'] * 1e3:.1f} ms -> "
        f"{overhead['glokr_forward_s'] * 1e3:.1f} ms, CPU-indicative)"
    )
    dy = dynamics
    print(f"\nrow-rescale task ({args.steps} steps; ΔM per eq. 5 as context):")
    for arm in ("glokr_bora", "dora_col", "plain_kron"):
        a = dy[arm]
        print(
            f"  {arm:>11}: loss={a['final_loss']:.4f} "
            f"dM(row)={a['mean_dM_row']:.2e} "
            f"dM(col)={a['mean_dM_col']:.2e} "
            f"row/col={a['row_col_symmetry']:.3f}"
        )
    print(
        f"bora_wins_row_rescale_task={dy['bora_wins_row_rescale_task']} "
        "(falsifier: must be True — m_row parameterizes the target directly)"
    )


if __name__ == "__main__":
    main()
