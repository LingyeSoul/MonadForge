"""LoHa numerics gate (Tier 2, LyCORIS-backend wrapper).

Three LoHa numerics surfaces are reported, on real Anima DiT Linear shapes:

1. **Wrapper-vs-official equivalence**: the MonadForge wrapper
   (``networks/lora_modules/loha.py``, official bypass ops) against the
   official LyCORIS (4.0.0) regular (rebuild) forward — max abs output error
   at fixed seed. Also quantifies the bypass-scale asymmetry: what the error
   would be if the pre-4.0 LoKr ``multiplier * self.scale`` compensator form
   were copied to LoHa (it double-scales — the bench prints the resulting
   blowup so the asymmetry stays visible).

2. **get_diff_weight double-scale bug**: official ``get_diff_weight``
   (double-scale still present in 4.0.0) vs the wrapper override — the
   official/raw ratio equals ``alpha/rank``.

3. **Capacity math vs plain LoRA**: LoHa spends 2× the params of a rank-r
   LoRA for an effective-rank ceiling of r²; the bench measures the numerical
   rank actually achieved by random-init deltas at each shape, plus the
   params-per-unit-rank comparison against a LoRA sized to the same budget.

Pure parameter math — no DiT load (``bench/_anima.py`` is opt-in per
CONTRIBUTING.md; this is an analytical simulator). CPU-friendly.

Usage::

    python bench/loha/run_bench.py
    python bench/loha/run_bench.py --label <date> --lora_dim 4 8 16 32

Drops a ``result.json`` envelope (``bench/_common.py``) into
``bench/loha/results/<ts>[-<label>]/``.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench._common import make_run_dir, write_result  # noqa: E402

# Real Anima DiT Linear shapes (out, in) — self-attn q/k/v/o + ffn projections.
DEFAULT_SHAPES: tuple[tuple[int, int], ...] = (
    (3072, 3072),  # self-attn q/k/v/o at a mid width
    (9216, 3072),  # ffn up (3x hidden) at the same width
    (12288, 3072),  # ffn up at a wider block
)
DEFAULT_LORA_DIMS: tuple[int, ...] = (4, 8, 16, 32)


def _numerical_rank(delta: torch.Tensor, rel_tol: float = 1e-5) -> int:
    s = torch.linalg.svdvals(delta)
    if s.numel() == 0:
        return 0
    return int((s > s[0] * rel_tol).sum().item())


def _equivalence_row(
    shape: tuple[int, int], rank: int, alpha: float, seed: int
) -> dict:
    from lycoris.modules.loha import LohaModule as LycorisLohaModule

    from networks.lora_modules.loha import LoHaModule

    out_dim, in_dim = shape
    torch.manual_seed(seed)
    base = torch.nn.Linear(in_dim, out_dim, bias=False)
    base.weight.requires_grad_(False)
    official_base = copy.deepcopy(base)

    wrapped = LoHaModule("bench", base, multiplier=0.75, lora_dim=rank, alpha=alpha)
    with torch.no_grad():
        wrapped.hada_w2_a.normal_(0, 0.1)
    official = LycorisLohaModule(
        "bench_official",
        official_base,
        multiplier=0.75,
        lora_dim=rank,
        alpha=alpha,
        bypass_mode=False,
    )
    official.load_state_dict(wrapped.state_dict())

    wrapped.apply_to()
    official.apply_to()
    x = torch.randn(2, 16, in_dim)
    with torch.no_grad():
        wrapped_out = base(x)
        official_out = official_base(x)
        base_out = wrapped.org_forward(x)
        # The pre-4.0 LoKr compensator form (scale=multiplier*self.scale)
        # double-applies the scale for LoHa — reproduce it to quantify the
        # blowup it would cause.
        double_scaled = base_out + wrapped.bypass_forward_diff(
            x, scale=wrapped.multiplier * wrapped.scale
        )

    err = float((wrapped_out - official_out).abs().max())
    ref_delta_mag = float((official_out - base_out).abs().max())
    double_err = float((double_scaled - official_out).abs().max())

    # get_diff_weight double-scale bug: probe the PURE official module (the
    # wrapper's virtual get_weight would muddle a super()-call probe) — the
    # official/fixed norm ratio equals ``scale``.
    with torch.no_grad():
        official_diff = official.get_diff_weight(multiplier=1.0)[0]
        fixed_diff = wrapped.get_diff_weight(multiplier=1.0)[0]
    ratio = float(
        official_diff.norm()
        / fixed_diff.norm().clamp_min(torch.finfo(torch.float32).tiny)
    )

    # Merge equivalence: merge_to over the canonical state-dict slice must
    # land exactly base_W + get_weight() (the fuse_weight delta).
    with torch.no_grad():
        w_before = base.weight.detach().clone()
        wrapped.merge_to(wrapped.state_dict(), dtype=torch.float32, device="cpu")
        merge_err = float((base.weight - (w_before + wrapped.get_weight())).abs().max())
        base.weight.data.copy_(w_before)

    return {
        "shape": f"{out_dim}x{in_dim}",
        "rank": rank,
        "alpha": alpha,
        "scale": float(wrapped.scale),
        "wrapper_vs_official_max_abs_err": err,
        "official_delta_max_abs": ref_delta_mag,
        "lokr_style_fix_max_abs_err": double_err,
        "get_diff_weight_official_over_fixed_ratio": round(ratio, 4),
        "merge_vs_get_weight_max_abs_err": merge_err,
    }


def _capacity_row(shape: tuple[int, int], rank: int, seed: int) -> dict:
    out_dim, in_dim = shape
    loha_params = 2 * rank * (out_dim + in_dim)
    lora_params_same_rank = rank * (out_dim + in_dim)
    # LoRA sized to LoHa's parameter budget:
    lora_rank_same_budget = 2 * rank

    g = torch.Generator().manual_seed(seed)
    w1a = torch.randn(out_dim, rank, generator=g) * 0.1
    w1b = torch.randn(rank, in_dim, generator=g)
    w2a = torch.randn(out_dim, rank, generator=g) * 0.1
    w2b = torch.randn(rank, in_dim, generator=g)
    delta = (w1a @ w1b) * (w2a @ w2b)
    measured = _numerical_rank(delta)

    return {
        "shape": f"{out_dim}x{in_dim}",
        "rank": rank,
        "loha_param_count": loha_params,
        "lora_param_count_same_rank": lora_params_same_rank,
        "lora_rank_at_same_budget": lora_rank_same_budget,
        "loha_rank_ceiling": min(rank * rank, out_dim, in_dim),
        "loha_measured_numerical_rank": measured,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--lora_dim",
        type=int,
        nargs="+",
        default=list(DEFAULT_LORA_DIMS),
        help="LoHa lora_dim values to sweep (default: %(default)s).",
    )
    p.add_argument(
        "--shape",
        type=int,
        nargs=2,
        action="append",
        default=None,
        metavar=("OUT", "IN"),
        help="Linear (out, in) shape to probe; repeatable. "
        "Defaults to a small set of real Anima DiT shapes.",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="alpha for the equivalence probe (default: 4*min(lora_dim), "
        "i.e. scale=4, so the double-scale probes stay visibly non-unit).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--label",
        type=str,
        default=None,
        help="Free-form label appended to the run directory name.",
    )
    args = p.parse_args()

    shapes = [tuple(s) for s in args.shape] if args.shape else list(DEFAULT_SHAPES)

    # Equivalence probe runs on a modest shape subset (full ΔW per module).
    eq_rank = min(args.lora_dim)
    eq_alpha = args.alpha if args.alpha is not None else 4.0 * eq_rank
    equivalence_rows = [
        _equivalence_row(shape, eq_rank, eq_alpha, args.seed) for shape in shapes[:1]
    ] + [
        _equivalence_row((512, 512), eq_rank, eq_alpha, args.seed),
    ]

    capacity_rows = [
        _capacity_row(shape, rank, args.seed)
        for shape in shapes
        for rank in args.lora_dim
    ]

    max_equiv_err = max(r["wrapper_vs_official_max_abs_err"] for r in equivalence_rows)
    min_double_err = min(r["lokr_style_fix_max_abs_err"] for r in equivalence_rows)
    max_merge_err = max(r["merge_vs_get_weight_max_abs_err"] for r in equivalence_rows)
    ceilings_hit = sum(
        1
        for r in capacity_rows
        if r["loha_measured_numerical_rank"] == r["loha_rank_ceiling"]
    )

    metrics = {
        "config": {
            "shapes": [f"{s[0]}x{s[1]}" for s in shapes],
            "lora_dims": list(args.lora_dim),
            "eq_rank": eq_rank,
            "eq_alpha": eq_alpha,
            "seed": args.seed,
        },
        "equivalence": equivalence_rows,
        "capacity": capacity_rows,
        # PR headline numbers:
        "headline_wrapper_vs_official_max_abs_err": max_equiv_err,
        "headline_lokr_style_fix_min_abs_err": min_double_err,
        "headline_merge_vs_get_weight_max_abs_err": max_merge_err,
        "headline_capacity_ceiling_hit": f"{ceilings_hit}/{len(capacity_rows)}",
    }

    run_dir = make_run_dir("loha", label=args.label)
    write_result(run_dir, script=__file__, args=args, metrics=metrics)

    print(f"=== loha bench → {run_dir} ===\n")
    print("-- wrapper vs official (fixed seed) --")
    print(
        f"{'shape':>12} {'rank':>5} {'scale':>6} "
        f"{'max_err':>10} {'delta_mag':>10} {'lokr_fix_err':>13} {'gdw_ratio':>10} "
        f"{'merge_err':>10}"
    )
    for r in equivalence_rows:
        print(
            f"{r['shape']:>12} {r['rank']:>5} {r['scale']:>6.1f} "
            f"{r['wrapper_vs_official_max_abs_err']:>10.2e} "
            f"{r['official_delta_max_abs']:>10.3f} "
            f"{r['lokr_style_fix_max_abs_err']:>13.3f} "
            f"{r['get_diff_weight_official_over_fixed_ratio']:>10.2f} "
            f"{r['merge_vs_get_weight_max_abs_err']:>10.2e}"
        )
    print(
        "\n  gdw_ratio = official get_diff_weight / wrapper override "
        "(equals scale => upstream double-applies it)."
    )
    print(
        "  lokr_fix_err = output error IF the pre-4.0 LoKr bypass "
        "compensator were copied (double-scale) -- must stay large, "
        "max_err must stay ~0."
    )

    print("\n-- capacity vs plain LoRA --")
    print(
        f"{'shape':>12} {'rank':>5} {'loha_prm':>10} {'lora_prm':>10} "
        f"{'lora_r@budget':>14} {'ceiling':>8} {'measured':>9}"
    )
    for r in capacity_rows:
        print(
            f"{r['shape']:>12} {r['rank']:>5} {r['loha_param_count']:>10} "
            f"{r['lora_param_count_same_rank']:>10} "
            f"{r['lora_rank_at_same_budget']:>14} "
            f"{r['loha_rank_ceiling']:>8} {r['loha_measured_numerical_rank']:>9}"
        )
    print(
        f"\n  wrapper-vs-official max abs err: {max_equiv_err:.2e}"
        f"\n  merge-vs-get_weight max abs err: {max_merge_err:.2e}"
        f"\n  rank-ceiling hit: {metrics['headline_capacity_ceiling_hit']}"
    )


if __name__ == "__main__":
    main()
