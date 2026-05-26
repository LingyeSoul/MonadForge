#!/usr/bin/env python
"""Procrustes / ridge bridge between PE features and postfix s-vectors.

See README.md. Inputs: glob of per-image `{stem}_s.safetensors` + a directory
of `{stem}_anima_pe.safetensors`. Output: standard bench envelope plus
singular-value curve, per-image predictions, and reusable bridge matrices.
"""

from __future__ import annotations

import argparse
import csv
import glob
from collections import Counter
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


from bench._common import make_run_dir, write_result  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--s_glob",
        required=True,
        help="Glob for per-image `_s.safetensors` (e.g. 'output/probes/postfix_tail/bridge_pool_v0/s/*_s.safetensors').",
    )
    p.add_argument(
        "--pe_dir",
        default="post_image_dataset/lora",
        help="Directory containing `{stem}_anima_pe.safetensors`.",
    )
    p.add_argument(
        "--pe_pool",
        choices=("mean", "cls", "max"),
        default="mean",
        help="Token-axis reduction over PE image_features.",
    )
    p.add_argument(
        "--ridge",
        type=float,
        default=1e-2,
        help="Ridge regularizer for unconstrained linear fit.",
    )
    p.add_argument("--test_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--label", type=str, default=None)
    return p.parse_args()


def discover_pairs(s_glob: str, pe_dir: str) -> tuple[list[str], list[str], list[str]]:
    """Return (stems, s_paths, pe_paths) where every triple is paired by stem."""
    s_paths = sorted(glob.glob(s_glob))
    if not s_paths:
        raise SystemExit(f"No files matched --s_glob={s_glob!r}")
    pe_root = Path(pe_dir)
    stems, sp, pp = [], [], []
    for s_path in s_paths:
        stem = Path(s_path).name.removesuffix("_s.safetensors")
        pe_path = pe_root / f"{stem}_anima_pe.safetensors"
        if pe_path.exists():
            stems.append(stem)
            sp.append(s_path)
            pp.append(str(pe_path))
    if not stems:
        raise SystemExit(
            f"Found {len(s_paths)} s-files but none paired with PE features in {pe_dir}"
        )
    return stems, sp, pp


def reduce_pe(tokens: torch.Tensor, mode: str) -> torch.Tensor:
    """(T, D) -> (D,)."""
    if mode == "mean":
        return tokens.mean(dim=0)
    if mode == "cls":
        return tokens[0]
    if mode == "max":
        return tokens.amax(dim=0)
    raise ValueError(mode)


def load_pool(
    stems: list[str], s_paths: list[str], pe_paths: list[str], pe_pool: str
) -> tuple[torch.Tensor, torch.Tensor]:
    s_list, pe_list = [], []
    ks = []
    for s_path, pe_path in zip(s_paths, pe_paths):
        s = load_file(s_path)["s"].float()
        pe = load_file(pe_path)["image_features"].float()
        ks.append(s.numel())
        s_list.append(s)
        pe_list.append(reduce_pe(pe, pe_pool))
    counter = Counter(ks)
    if len(counter) > 1:
        raise SystemExit(
            f"Mixed K in pool: {dict(counter)}. Filter --s_glob to a single K."
        )
    S = torch.stack(s_list)  # (N, K)
    F = torch.stack(pe_list)  # (N, D_pe)
    return S, F


def split(N: int, test_frac: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=g)
    n_test = max(1, int(round(N * test_frac)))
    return perm[n_test:], perm[:n_test]  # train, test


def fit_ridge(
    F_train: torch.Tensor, S_train: torch.Tensor, ridge: float
) -> torch.Tensor:
    """A ∈ (D_pe, K) such that S ≈ F @ A."""
    D = F_train.shape[1]
    gram = F_train.T @ F_train + ridge * torch.eye(D, dtype=F_train.dtype)
    return torch.linalg.solve(gram, F_train.T @ S_train)


def fit_rect_procrustes(
    F_train: torch.Tensor, S_train: torch.Tensor
) -> tuple[torch.Tensor, float]:
    """Rectangular Procrustes with global scale.

    Solve A ∈ R^{D×K}, A^T A = I_K; plus scalar c, minimizing
    ||c · F A - S||_F^2. Returns (A * c, c).
    """
    M = F_train.T @ S_train  # (D, K)
    U, sigma, Vt = torch.linalg.svd(M, full_matrices=False)
    A_orth = U @ Vt  # (D, K), columns are orthonormal
    # Scale c = trace(A^T F^T S) / ||F A||_F^2 = sum(sigma) / ||F A_orth||_F^2.
    num = sigma.sum()
    denom = (F_train @ A_orth).pow(2).sum().clamp_min(1e-12)
    c = float(num / denom)
    return A_orth * c, c


def eval_bridge(
    A: torch.Tensor,
    F_test: torch.Tensor,
    S_test: torch.Tensor,
    S_train_mean: torch.Tensor,
    label: str,
) -> dict:
    pred = F_test @ A  # (Nt, K), centered
    err = (pred - S_test).pow(2).sum(dim=1)
    # Mean-only baseline already accounts for the centering.
    baseline_err = S_test.pow(2).sum(dim=1)
    test_mse = float(err.mean())
    baseline_mse = float(baseline_err.mean())
    r2 = 1.0 - test_mse / max(baseline_mse, 1e-12)
    # Warm-start: distance from zero-init in the *uncentered* world.
    pred_unc = pred + S_train_mean
    true_unc = S_test + S_train_mean
    l2_err_pred = (pred_unc - true_unc).norm(dim=1)
    l2_err_zero = true_unc.norm(dim=1)
    warm_gain = (1.0 - l2_err_pred / l2_err_zero.clamp_min(1e-12)).mean()
    # Cosine in uncentered space, since that's how the inversion target lives.
    cos = torch.nn.functional.cosine_similarity(pred_unc, true_unc, dim=1).mean()
    return {
        f"{label}_test_mse": test_mse,
        f"{label}_baseline_mse": baseline_mse,
        f"{label}_test_r2": float(r2),
        f"{label}_mean_warm_gain": float(warm_gain),
        f"{label}_mean_cosine": float(cos),
    }


def svd_diagnostic(A: torch.Tensor) -> tuple[list[float], list[float]]:
    """Singular values + cumulative variance fraction."""
    sigma = torch.linalg.svdvals(A)
    var = sigma.pow(2)
    cum = var.cumsum(0) / var.sum().clamp_min(1e-12)
    return sigma.tolist(), cum.tolist()


def main():
    args = parse_args()
    run_dir = make_run_dir("postfix_inversion_bridge", label=args.label)

    stems, s_paths, pe_paths = discover_pairs(args.s_glob, args.pe_dir)
    S, F = load_pool(stems, s_paths, pe_paths, args.pe_pool)
    N, K = S.shape
    D_pe = F.shape[1]

    if N < 2:
        raise SystemExit(f"Need at least 2 paired samples (got {N}).")

    train_idx, test_idx = split(N, args.test_frac, args.seed)
    S_train, S_test = S[train_idx], S[test_idx]
    F_train, F_test = F[train_idx], F[test_idx]

    # Center.
    s_mean = S_train.mean(0)
    f_mean = F_train.mean(0)
    S_train_c = S_train - s_mean
    S_test_c = S_test - s_mean
    F_train_c = F_train - f_mean
    F_test_c = F_test - f_mean

    A_ridge = fit_ridge(F_train_c, S_train_c, args.ridge)
    A_proc, proc_scale = fit_rect_procrustes(F_train_c, S_train_c)

    metrics: dict = {
        "n_total": N,
        "n_train": int(train_idx.numel()),
        "n_test": int(test_idx.numel()),
        "K": K,
        "D_pe": D_pe,
        "pe_pool": args.pe_pool,
        "ridge": args.ridge,
        "proc_scale": proc_scale,
        "well_posed_warning": (
            f"N={N} < D_pe={D_pe}; ridge fit is regularization-dominated. "
            "Treat absolute test_r2 with caution; the SVD shape is still informative."
            if N < D_pe
            else None
        ),
    }
    metrics.update(eval_bridge(A_ridge, F_test_c, S_test_c, s_mean, "ridge"))
    metrics.update(eval_bridge(A_proc, F_test_c, S_test_c, s_mean, "procrustes"))

    sigma, cum_var = svd_diagnostic(A_ridge)
    metrics["ridge_top1_var"] = cum_var[0] if cum_var else None
    metrics["ridge_top5_var"] = cum_var[min(4, len(cum_var) - 1)] if cum_var else None
    metrics["ridge_top_k_var"] = cum_var[K - 1] if cum_var else None

    # Artifacts.
    sv_path = run_dir / "singular_values.csv"
    with sv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "sigma", "cumulative_variance"])
        for r, (sg, cv) in enumerate(zip(sigma, cum_var), start=1):
            w.writerow([r, f"{sg:.6e}", f"{cv:.6f}"])

    pred_path = run_dir / "predictions.csv"
    pred_unc = F_test_c @ A_ridge + s_mean
    true_unc = S_test
    cos_per = torch.nn.functional.cosine_similarity(pred_unc, true_unc, dim=1)
    l2_pred = (pred_unc - true_unc).norm(dim=1)
    l2_zero = true_unc.norm(dim=1)
    warm = 1.0 - l2_pred / l2_zero.clamp_min(1e-12)
    with pred_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["stem", "l2_err_pred", "l2_err_zero", "cosine", "warm_gain"])
        for i, idx in enumerate(test_idx.tolist()):
            w.writerow(
                [
                    stems[idx],
                    f"{l2_pred[i]:.6e}",
                    f"{l2_zero[i]:.6e}",
                    f"{cos_per[i]:.6f}",
                    f"{warm[i]:.6f}",
                ]
            )

    bridges_path = run_dir / "bridges.safetensors"
    save_file(
        {
            "A_ridge": A_ridge.contiguous(),
            "A_procrustes": A_proc.contiguous(),
            "s_mean": s_mean.contiguous(),
            "pe_mean": f_mean.contiguous(),
        },
        str(bridges_path),
    )

    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        artifacts=["singular_values.csv", "predictions.csv", "bridges.safetensors"],
        label=args.label,
    )

    print(f"[bench] wrote results to {run_dir}")
    print(f"[bench]   N={N} K={K} D_pe={D_pe}")
    print(
        f"[bench]   ridge test_r2={metrics['ridge_test_r2']:+.4f}  warm_gain={metrics['ridge_mean_warm_gain']:+.4f}  cos={metrics['ridge_mean_cosine']:+.4f}"
    )
    print(
        f"[bench]   procrustes test_r2={metrics['procrustes_test_r2']:+.4f}  warm_gain={metrics['procrustes_mean_warm_gain']:+.4f}  cos={metrics['procrustes_mean_cosine']:+.4f}"
    )
    print(
        f"[bench]   top1 var={metrics['ridge_top1_var']:.3f}  top5 var={metrics['ridge_top5_var']:.3f}"
    )


if __name__ == "__main__":
    main()
