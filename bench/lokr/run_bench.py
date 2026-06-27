"""LoKR decomposition threshold + SVD rank-cap bench (Tier 1.5 numerics gate).

Two numerics changes landed in the same diff:

1. **Decomposition threshold** (``networks/lora_modules/lokr.py``): a LoKR
   factor decomposes into ``(Xa, Xb)`` pairs when ``lora_dim < <threshold>``.
   The old threshold was ``max(out, in) / 2``; the new one is ``min(out, in)``.
   The threshold governs the *effective rank* of ΔW = kron(w1, w2):
   ``eff_rank = rank(w1) * rank(w2)`` (full factor → ``min(out, in)``,
   decomposed → ``lora_dim``). Tightening ``max/2`` → ``min`` makes
   decomposition fire later, raising effective rank + parameter count.

2. **SVD rank cap** (``networks/lora_save.py::_convert_lokr_to_standard_lora``):
   the kron delta is SVD-split into ``lora_down``/``lora_up`` for ComfyUI.
   The old cap was the global ``lora_rank`` default (128); the new one is the
   per-module ``alpha`` (= ``lora_dim`` in shipped LoKR presets). Lower cap →
   smaller saved file, more SVD truncation error.

This bench reports **both before and after** for both changes, on real Anima
DiT Linear shapes (q/k/v/ffn proj). Pure parameter math — no DiT load (the
``bench/_anima.py`` harness is opt-in per CONTRIBUTING.md, and this is an
analytical simulator). CPU-friendly.

Headline numbers (PR-copy-pasteable):

* effective rank & param count: ``old_threshold`` vs ``new_threshold``
* saved param count & fixed-seed SVD drift: ``old_svd_cap`` vs ``new_svd_cap``

Usage::

    python bench/lokr/run_bench.py
    python bench/lokr/run_bench.py --label <date> --lora_dim 4 8 16 32

Drops a ``result.json`` envelope (``bench/_common.py``) into
``bench/lokr/results/<ts>[-<label>]/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench._common import make_run_dir, write_result  # noqa: E402

# Real Anima DiT Linear shapes (out, in) — self-attn q/k/v/o + ffn up/down
# projections, across the model widths actually shipped. ``lokr_factor=-1``
# (closest-to-square factorization) is the default in gui-methods/lokr.toml.
DEFAULT_SHAPES: tuple[tuple[int, int], ...] = (
    (3072, 3072),  # self-attn q/k/v/o at a mid width
    (9216, 3072),  # ffn up (3x hidden) at the same width
    (12288, 3072),  # ffn up at a wider block
)
DEFAULT_LORA_DIMS: tuple[int, ...] = (4, 8, 16, 32)
DEFAULT_LOKR_FACTOR = -1


def _factorization(dimension: int, factor: int = -1) -> tuple[int, int]:
    """Mirror of ``LoKRModule._factorization`` — split dim into (m, n), m<=n."""
    if factor > 0 and dimension % factor == 0:
        m, n = factor, dimension // factor
        if m > n:
            n, m = m, n
        return m, n
    if factor < 0:
        factor = dimension
    m, n = 1, dimension
    length = m + n
    while m < n:
        new_m = m + 1
        while dimension % new_m != 0:
            new_m += 1
        new_n = dimension // new_m
        if new_m + new_n > length or new_m > factor:
            break
        m, n = new_m, new_n
    if m > n:
        n, m = m, n
    return m, n


def _param_count(
    out_a: int,
    in_a: int,
    out_b: int,
    in_b: int,
    lora_dim: int,
    use_w1: bool,
    use_w2: bool,
) -> int:
    """Param count of the LoKR module's trainable factor params."""
    n = 0
    if use_w1:
        n += out_a * in_a
    else:
        n += out_a * lora_dim + lora_dim * in_a
    if use_w2:
        n += out_b * in_b
    else:
        n += out_b * lora_dim + lora_dim * in_b
    return n


def _effective_rank(
    out_a: int,
    in_a: int,
    out_b: int,
    in_b: int,
    lora_dim: int,
    use_w1: bool,
    use_w2: bool,
) -> int:
    """eff_rank = rank(w1) * rank(w2); full→min(shape), decomposed→lora_dim."""
    r1 = min(out_a, in_a) if use_w1 else lora_dim
    r2 = min(out_b, in_b) if use_w2 else lora_dim
    return r1 * r2


def _measure_threshold(
    shape: tuple[int, int], lora_dim: int, lokr_factor: int, threshold: str
) -> dict:
    """Compute effective rank + param count for one threshold policy.

    ``threshold`` ∈ {"old", "new"}:
      old: w2 decomposes iff ``lora_dim < max(out_b, in_b) / 2``
           w1 decomposes iff ``decompose_both and lora_dim < max(out_a, in_a) / 2``
      new: w2 decomposes iff ``lora_dim < min(out_b, in_b)``
           w1 decomposes iff ``decompose_both and lora_dim < min(out_a, in_a)``
    (decompose_both defaults to False in shipped presets → w1 stays full.)
    """
    out_dim, in_dim = shape
    out_a, out_b = _factorization(out_dim, lokr_factor)
    in_a, in_b = _factorization(in_dim, lokr_factor)
    decompose_both = False  # gui-methods/lokr.toml default

    if threshold == "old":
        w1_dec = decompose_both and lora_dim < max(out_a, in_a) / 2
        w2_dec = lora_dim < max(out_b, in_b) / 2
    else:  # new
        w1_dec = decompose_both and lora_dim < min(out_a, in_a)
        w2_dec = lora_dim < min(out_b, in_b)

    eff = _effective_rank(
        out_a, in_a, out_b, in_b, lora_dim, use_w1=not w1_dec, use_w2=not w2_dec
    )
    params = _param_count(
        out_a, in_a, out_b, in_b, lora_dim, use_w1=not w1_dec, use_w2=not w2_dec
    )
    return {
        "out": out_dim,
        "in": in_dim,
        "factor_a": [out_a, in_a],
        "factor_b": [out_b, in_b],
        "lora_dim": lora_dim,
        "w1_decomposed": w1_dec,
        "w2_decomposed": w2_dec,
        "effective_rank": eff,
        "param_count": params,
    }


def _svd_cap_delta(
    out_a: int, in_a: int, out_b: int, in_b: int, seed: int, target_rank: int
) -> dict:
    """Build a fixed-seed kron delta, SVD-truncate to target_rank, measure drift.

    Returns saved param count + relative Frobenius error of the rank-r
    reconstruction vs the full delta. Mirrors
    ``_convert_lokr_to_standard_lora``'s SVD path (sans inv_scale, which is
    a no-op without channel_scale).
    """
    g = torch.Generator().manual_seed(seed)
    w1 = torch.randn(out_a, in_a, generator=g)
    w2 = torch.randn(out_b, in_b, generator=g)
    delta = torch.kron(w1, w2)  # (out_a*out_b, in_a*in_b)
    # SVD path (full_matrices=False) — exact mirror of the save converter
    U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
    max_rank = min(target_rank, S.shape[0])
    recon = (U[:, :max_rank] * S[:max_rank]) @ Vh[:max_rank, :]
    rel_err = float(
        torch.linalg.norm(delta - recon) / torch.linalg.norm(delta).clamp_min(1e-30)
    )
    saved_params = max_rank * (delta.shape[0] + delta.shape[1])
    return {
        "target_rank_requested": target_rank,
        "svd_rank_used": max_rank,
        "saved_param_count": saved_params,
        "rel_frobenius_error": round(rel_err, 6),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--lora_dim",
        type=int,
        nargs="+",
        default=list(DEFAULT_LORA_DIMS),
        help="LoKR lora_dim values to sweep (default: %(default)s).",
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
        "--lokr_factor",
        type=int,
        default=DEFAULT_LOKR_FACTOR,
        help="LoKR factorization hint (default: %(default)s = closest-to-square).",
    )
    p.add_argument(
        "--svd_seed",
        type=int,
        default=0,
        help="Fixed seed for the SVD-drift delta (default: %(default)s).",
    )
    p.add_argument(
        "--label",
        type=str,
        default=None,
        help="Free-form label appended to the run directory name.",
    )
    args = p.parse_args()

    shapes = [tuple(s) for s in args.shape] if args.shape else list(DEFAULT_SHAPES)

    # --- Threshold sweep (effective rank + param count) ---
    threshold_rows: list[dict] = []
    for shape in shapes:
        for lora_dim in args.lora_dim:
            old = _measure_threshold(shape, lora_dim, args.lokr_factor, "old")
            new = _measure_threshold(shape, lora_dim, args.lokr_factor, "new")
            threshold_rows.append(
                {
                    "shape": f"{shape[0]}x{shape[1]}",
                    "lora_dim": lora_dim,
                    "old_threshold": old,
                    "new_threshold": new,
                    "eff_rank_delta": new["effective_rank"] - old["effective_rank"],
                    "param_count_delta": new["param_count"] - old["param_count"],
                }
            )

    # --- SVD-cap sweep (saved param count + fixed-seed drift) ---
    svd_rows: list[dict] = []
    for shape in shapes:
        out_dim, in_dim = shape
        out_a, out_b = _factorization(out_dim, args.lokr_factor)
        in_a, in_b = _factorization(in_dim, args.lokr_factor)
        # The smallest lora_dim in the sweep represents the strictest new cap;
        # use it so old (128) vs new (lora_dim) shows the widest contrast.
        alpha_cap = min(args.lora_dim)
        old_cap = _svd_cap_delta(out_a, in_a, out_b, in_b, args.svd_seed, 128)
        new_cap = _svd_cap_delta(out_a, in_a, out_b, in_b, args.svd_seed, alpha_cap)
        svd_rows.append(
            {
                "shape": f"{shape[0]}x{shape[1]}",
                "factor_a": [out_a, in_a],
                "factor_b": [out_b, in_b],
                "alpha_cap": alpha_cap,
                "old_svd_cap": old_cap,
                "new_svd_cap": new_cap,
                "saved_param_reduction_pct": round(
                    (1 - new_cap["saved_param_count"] / old_cap["saved_param_count"])
                    * 100.0,
                    1,
                ),
                "drift_increase_abs": round(
                    new_cap["rel_frobenius_error"] - old_cap["rel_frobenius_error"], 6
                ),
            }
        )

    # --- Derived aggregates (PR headline) ---
    avg_eff_delta = (
        sum(r["eff_rank_delta"] for r in threshold_rows) / len(threshold_rows)
        if threshold_rows
        else 0.0
    )
    avg_param_delta_pct = (
        sum(
            r["param_count_delta"] / r["old_threshold"]["param_count"] * 100.0
            for r in threshold_rows
            if r["old_threshold"]["param_count"] > 0
        )
        / len(threshold_rows)
        if threshold_rows
        else 0.0
    )
    max_drift_increase = max((r["drift_increase_abs"] for r in svd_rows), default=0.0)
    avg_save_reduction_pct = (
        sum(r["saved_param_reduction_pct"] for r in svd_rows) / len(svd_rows)
        if svd_rows
        else 0.0
    )

    metrics = {
        "config": {
            "shapes": [f"{s[0]}x{s[1]}" for s in shapes],
            "lora_dims": list(args.lora_dim),
            "lokr_factor": args.lokr_factor,
            "svd_seed": args.svd_seed,
        },
        "threshold_change": threshold_rows,
        "svd_cap_change": svd_rows,
        # PR headline numbers (before → after deltas, averaged across shapes):
        "headline_eff_rank_delta_avg": round(avg_eff_delta, 1),
        "headline_param_count_delta_avg_pct": round(avg_param_delta_pct, 1),
        "headline_save_param_reduction_avg_pct": round(avg_save_reduction_pct, 1),
        "headline_svd_drift_increase_max": round(max_drift_increase, 6),
    }

    run_dir = make_run_dir("lokr", label=args.label)
    write_result(run_dir, script=__file__, args=args, metrics=metrics)

    # --- Console summary (PR-copy-pasteable) ---
    print(f"=== lokr bench → {run_dir} ===\n")
    print("-- decomposition threshold (eff rank & param count) --")
    print(
        f"{'shape':>12} {'dim':>4} "
        f"{'eff_old':>8} {'eff_new':>8} {'d_eff':>7} "
        f"{'prm_old':>9} {'prm_new':>9} {'d_prm%':>8}"
    )
    for r in threshold_rows:
        o, n = r["old_threshold"], r["new_threshold"]
        d_prm_pct = (
            r["param_count_delta"] / o["param_count"] * 100.0
            if o["param_count"]
            else 0.0
        )
        print(
            f"{r['shape']:>12} {r['lora_dim']:>4} "
            f"{o['effective_rank']:>8} {n['effective_rank']:>8} "
            f"{r['eff_rank_delta']:>+7} "
            f"{o['param_count']:>9} {n['param_count']:>9} {d_prm_pct:>+7.1f}%"
        )
    print(
        f"\n  avg effective_rank delta: {metrics['headline_eff_rank_delta_avg']:+.1f}"
    )
    print(
        f"  avg param_count delta:    {metrics['headline_param_count_delta_avg_pct']:+.1f}%"
    )

    print("\n-- SVD rank cap (saved size & drift) --")
    print(
        f"{'shape':>12} "
        f"{'rk_old':>7} {'rk_new':>7} "
        f"{'sv_old':>10} {'sv_new':>10} {'save%':>7} "
        f"{'err_old':>9} {'err_new':>9} {'d_err':>9}"
    )
    for r in svd_rows:
        o, n = r["old_svd_cap"], r["new_svd_cap"]
        print(
            f"{r['shape']:>12} "
            f"{o['svd_rank_used']:>7} {n['svd_rank_used']:>7} "
            f"{o['saved_param_count']:>10} {n['saved_param_count']:>10} "
            f"{r['saved_param_reduction_pct']:>6.1f}% "
            f"{o['rel_frobenius_error']:>9.4f} {n['rel_frobenius_error']:>9.4f} "
            f"{r['drift_increase_abs']:>+9.4f}"
        )
    print(
        f"\n  avg saved-param reduction: {metrics['headline_save_param_reduction_avg_pct']:.1f}%"
    )
    print(
        f"  max SVD drift increase:    {metrics['headline_svd_drift_increase_max']:.4f}"
    )


if __name__ == "__main__":
    main()
