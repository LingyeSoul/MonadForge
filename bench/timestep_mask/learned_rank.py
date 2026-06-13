"""Effective rank of the *learned* ΔW — the empirical confirmation of the
column-utilization screen (bench/timestep_mask/column_utilization.py).

The activation-frequency screen is analytical and assumes gradient-when-active
is comparable across columns. This measures what training actually produced:
the singular-value spectrum of each adapted Linear's learned ΔW = up @ down.

Why ΔW, not per-column ‖up‖: the live default stack is OrthoLoRA
(use_ortho=true), where the Cayley rotation mixes the rank basis, so
"column j" isn't meaningful. The SVD of ΔW is parameterization-invariant —
identical handle for LoRA / OrthoLoRA / Hydra — and OrthoLoRA saves a thin SVD
of ΔW at rank=lora_dim, which is *lossless* for the spectrum (rank ≤ dim).
The α=alpha/dim scaling is a uniform scalar on ΔW, so stable rank and
participation ratio (both scale-invariant) ignore it entirely.

Metrics per Linear (singular values s_1≥…≥s_r):
  stable_rank        = (Σ s²) / s_max²        = ‖ΔW‖_F² / ‖ΔW‖_2²
  participation_ratio= (Σ s²)² / (Σ s⁴)        ("effective # of directions")
  rank@90 / rank@99  = #singular values to reach 90% / 99% of Σ s²
  nominal_rank       = r

Headline: median effective rank vs nominal — "of the N rank budget, training
used ~X directions". Compare T-LoRA on/off/σ-uniform.

Usage::

    python bench/timestep_mask/learned_rank.py \
        --ckpt output/ckpt/tlora_on.safetensors:on \
        --ckpt output/ckpt/tlora_off.safetensors:off \
        --ckpt output/ckpt/tlora_uniform.safetensors:uniform
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench._common import make_run_dir, write_result  # noqa: E402


def delta_singular_values(up: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
    """Singular values of ΔW = up @ down, computed via QR reduction.

    up: (out, r), down: (r, in). Nonzero singular values of (up @ down) equal
    those of (R_up @ R_downT) where up = Q_up R_up and down^T = Q_dn R_downT
    (both r×r). Avoids forming the (out × in) product; exact for rank ≤ r.
    """
    up = up.float()
    down = down.float()
    _, r_up = torch.linalg.qr(up, mode="reduced")  # (r, r)
    _, r_dn = torch.linalg.qr(down.T, mode="reduced")  # (r, r)
    return torch.linalg.svdvals(r_up @ r_dn.T)


def effrank_metrics(s: torch.Tensor) -> dict:
    s = s.detach().cpu().double()
    s2 = s * s
    energy = float(s2.sum())
    if energy <= 0:
        nominal = int(s.numel())
        return {
            "nominal_rank": nominal, "stable_rank": 0.0, "participation_ratio": 0.0,
            "rank90": 0, "rank99": 0, "energy": 0.0, "s_max": 0.0,
        }
    stable = energy / float(s2.max())
    pr = float(s2.sum() ** 2 / (s2 ** 2).sum())
    csum = torch.cumsum(s2, 0) / energy
    rank90 = int(torch.searchsorted(csum, torch.tensor(0.90, dtype=csum.dtype)).item()) + 1
    rank99 = int(torch.searchsorted(csum, torch.tensor(0.99, dtype=csum.dtype)).item()) + 1
    return {
        "nominal_rank": int(s.numel()),
        "stable_rank": stable,
        "participation_ratio": pr,
        "rank90": rank90,
        "rank99": rank99,
        "energy": energy,
        "s_max": float(s.max()),
    }


def find_pairs(keys) -> list[str]:
    return sorted(
        k.removesuffix(".lora_down.weight")
        for k in keys
        if k.endswith(".lora_down.weight")
    )


def analyze_checkpoint(path: Path, device: str) -> dict:
    per_module = []
    with safe_open(str(path), "pt", device=device) as h:
        keys = list(h.keys())
        prefixes = find_pairs(keys)
        if not prefixes:
            raise ValueError(f"no lora_down/up pairs in {path}")
        for pfx in prefixes:
            up_k, dn_k = f"{pfx}.lora_up.weight", f"{pfx}.lora_down.weight"
            if up_k not in keys:
                continue
            up = h.get_tensor(up_k)
            down = h.get_tensor(dn_k)
            # Skip non-2D (e.g. conv kernels reshaped) — rare for Anima Linears.
            if up.dim() != 2 or down.dim() != 2:
                continue
            s = delta_singular_values(up, down)
            m = effrank_metrics(s)
            m["module"] = pfx
            m["_spectrum"] = (s / (s.max() + 1e-12)).float().cpu().numpy().tolist()
            per_module.append(m)
    return {"path": str(path), "per_module": per_module}


def aggregate(per_module: list[dict]) -> dict:
    if not per_module:
        return {}
    w = np.array([m["energy"] for m in per_module], dtype=np.float64)
    w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
    def col(k):
        return np.array([m[k] for m in per_module], dtype=np.float64)
    nominal = int(np.median(col("nominal_rank")))
    return {
        "n_modules": len(per_module),
        "nominal_rank": nominal,
        "stable_rank_median": float(np.median(col("stable_rank"))),
        "stable_rank_wmean": float(np.sum(col("stable_rank") * w)),
        "participation_ratio_median": float(np.median(col("participation_ratio"))),
        "participation_ratio_wmean": float(np.sum(col("participation_ratio") * w)),
        "rank90_median": float(np.median(col("rank90"))),
        "rank99_median": float(np.median(col("rank99"))),
        "frac_budget_used_pr": float(np.median(col("participation_ratio")) / nominal),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ckpt", action="append", required=True,
        help="path[:label]; repeatable to compare conditions",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--label", default="learned-rank")
    p.add_argument("--no_plot", action="store_true")
    args = p.parse_args()

    conditions = []
    for spec in args.ckpt:
        if ":" in spec and not spec[1:3] == ":\\":
            path, _, lab = spec.rpartition(":")
        else:
            path, lab = spec, Path(spec).stem
        conditions.append((lab, Path(path)))

    run_dir = make_run_dir("timestep_mask", label=args.label)
    results = {}
    for lab, path in conditions:
        if not path.exists():
            print(f"[skip] {lab}: {path} not found")
            continue
        res = analyze_checkpoint(path, args.device)
        agg = aggregate(res["per_module"])
        results[lab] = {"agg": agg, "per_module": res["per_module"], "path": str(path)}

    # --- console summary ---
    hdr = (f"\n{'condition':>12} {'nominal':>8} {'stable_rk':>10} {'PR(med)':>9} "
           f"{'PR(wmean)':>10} {'rank@90':>8} {'rank@99':>8} {'budget_used':>12}")
    print(hdr)
    print("-" * len(hdr))
    for lab, r in results.items():
        a = r["agg"]
        print(f"{lab:>12} {a['nominal_rank']:>8} {a['stable_rank_median']:>10.2f} "
              f"{a['participation_ratio_median']:>9.2f} {a['participation_ratio_wmean']:>10.2f} "
              f"{a['rank90_median']:>8.0f} {a['rank99_median']:>8.0f} "
              f"{a['frac_budget_used_pr']:>11.0%}")
    print("\n(PR = participation ratio = effective # of rank directions used; "
          "budget_used = PR(med)/nominal)")

    # --- artifacts ---
    artifacts = []
    csv_path = run_dir / "per_module.csv"
    with csv_path.open("w") as fh:
        fh.write("condition,module,nominal_rank,stable_rank,participation_ratio,rank90,rank99\n")
        for lab, r in results.items():
            for m in r["per_module"]:
                fh.write(f"{lab},{m['module']},{m['nominal_rank']},{m['stable_rank']:.4f},"
                         f"{m['participation_ratio']:.4f},{m['rank90']},{m['rank99']}\n")
    artifacts.append("per_module.csv")

    if not args.no_plot and results:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
            # mean normalized singular-value spectrum per condition
            for lab, r in results.items():
                specs = [m["_spectrum"] for m in r["per_module"]]
                maxlen = max(len(s) for s in specs)
                arr = np.full((len(specs), maxlen), np.nan)
                for i, s in enumerate(specs):
                    arr[i, : len(s)] = s
                mean_spec = np.nanmean(arr, axis=0)
                ax1.plot(range(1, maxlen + 1), mean_spec, marker=".", ms=4, label=lab)
            ax1.set_yscale("log")
            ax1.set_xlabel("singular value index")
            ax1.set_ylabel("normalized σ_i / σ_1  (mean over modules)")
            ax1.set_title("Learned ΔW singular spectrum")
            ax1.legend(fontsize=9)
            ax1.grid(alpha=0.3, which="both")

            # participation-ratio distribution per condition
            labs = list(results.keys())
            data = [[m["participation_ratio"] for m in results[l]["per_module"]] for l in labs]
            ax2.boxplot(data, labels=labs, showmeans=True)
            nominal = results[labs[0]]["agg"]["nominal_rank"]
            ax2.axhline(nominal, color="r", ls="--", alpha=0.5, label=f"nominal rank={nominal}")
            ax2.set_ylabel("participation ratio (effective rank)")
            ax2.set_title("Effective rank per module")
            ax2.legend(fontsize=9)
            ax2.grid(alpha=0.3)
            fig.tight_layout()
            png = run_dir / "learned_rank.png"
            fig.savefig(png, dpi=110)
            plt.close(fig)
            artifacts.append("learned_rank.png")
        except Exception as e:  # noqa: BLE001
            print(f"[plot skipped: {e}]")

    metrics = {lab: r["agg"] for lab, r in results.items()}
    write_result(run_dir, script=__file__, args=args, metrics=metrics,
                 label=args.label, artifacts=artifacts, device=args.device)
    print(f"\nwrote {run_dir}")


if __name__ == "__main__":
    main()
