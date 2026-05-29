#!/usr/bin/env python
"""Offline Spectrum-forecaster hyperparameter sweep over captured feature trajectories.

Live successor to `_archive/bench/spectrum/replay_forecaster.py`. Tunes the knobs
`analyze_drift.py` *can't*: the forecaster's own `cheby_degree` (m),
`ridge_lambda` (lam), `blend_w` (w), plus the cache-schedule knobs
`window_size` / `flex_window` / `warmup_steps`. It measures the **forecast
residual** — the cache error a drift sim only assumes.

How it's faithful
-----------------
* Reuses the *shipped* forecaster (`networks.spectrum.SpectrumPredictor`).
* Mirrors the *production* cache schedule (`spectrum_denoise`:
  `consec_cached % floor(curr_ws)`).
* The forecaster buffer only ever ingests *actual*-forward features, so replaying
  any schedule over an all-actual capture is exact: on actual steps we feed the
  captured feature; on cached steps we predict and compare to the captured
  feature the real forward *would* have produced.

CAVEAT — lower forecast-MSE is *not* a quality oracle on Anima (cf. project
memory `fm_val_loss_uninformative` / `cmmd_val_signal`). Treat this as a
*shortlist* tool; confirm the top few with the real-image bench.

`replay_at_combo()` / `load_capture()` / `production_actual_mask()` are importable
— `compare_samplers.py` uses them to score one fixed combo per sampler.

Usage
-----
    python -m bench.spectrum_pareto.replay_forecaster --captures_dir <run>/captures
    python -m bench.spectrum_pareto.replay_forecaster --captures_dir <dir> \
        --window_list 1.5 2.0 3.0 --flex_list 0.0 0.25 0.5 --warmup_list 5 7
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from networks.spectrum import SpectrumPredictor  # noqa: E402
from bench._common import make_run_dir, write_result  # noqa: E402

try:
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# Production-default forecaster + schedule (mirrors generation.py getattr defaults
# and the shipped node). compare_samplers.py scores at this combo.
PROD_COMBO = dict(m=3, lam=0.1, w=0.3, window=2.0, flex=0.25, warmup=7)


# ---------------------------------------------------------------------------
# Production cache schedule (mirrors networks/spectrum.py spectrum_denoise)
# ---------------------------------------------------------------------------


def production_actual_mask(
    num_steps: int,
    warmup_steps: int,
    window_size: float,
    flex_window: float,
    stop_caching_step: int = -1,
) -> np.ndarray:
    """Bool array length num_steps. True = actual forward; False = cached/forecast.

    Exact port of the decision in `spectrum_denoise`: actual while `i < warmup`
    or `i >= stop_at`, else `(consec_cached + 1) % floor(curr_ws) == 0`; `curr_ws`
    grows by `flex_window` after each post-warmup actual forward.
    """
    stop_at = num_steps - 3 if stop_caching_step < 0 else stop_caching_step
    curr_ws = float(window_size)
    consec_cached = 0
    mask = np.zeros(num_steps, dtype=bool)
    for i in range(num_steps):
        if i < warmup_steps or i >= stop_at:
            actual = True
        else:
            actual = (consec_cached + 1) % max(1, math.floor(curr_ws)) == 0
        mask[i] = actual
        if actual:
            if i >= warmup_steps:
                curr_ws = round(curr_ws + flex_window, 3)
            consec_cached = 0
        else:
            consec_cached += 1
    return mask


# ---------------------------------------------------------------------------
# Residual spectrum (HF-bias proxy) — radial power over the (H, W) patch grid
# ---------------------------------------------------------------------------


def _radial_power(arr2d_stack: np.ndarray, num_bins: int = 32) -> np.ndarray:
    """Mean radially-averaged 2D power spectrum of a (..., H, W) stack."""
    *lead, h, w = arr2d_stack.shape
    flat = arr2d_stack.reshape(-1, h, w).astype(np.float64)
    fy = np.fft.fftshift(np.fft.fftfreq(h))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(w))[None, :]
    r = np.sqrt(fy**2 + fx**2)
    r = r / max(r.max(), 1e-12)
    bin_idx = np.clip((r * num_bins).astype(int), 0, num_bins - 1).ravel()
    sums = np.zeros(num_bins)
    counts = np.zeros(num_bins)
    for sample in flat:
        f = np.fft.fftshift(np.fft.fft2(sample))
        power = (f.real**2 + f.imag**2) / (h * w)
        np.add.at(sums, bin_idx, power.ravel())
        np.add.at(counts, bin_idx, 1.0)
    return np.where(counts > 0, sums / np.maximum(counts, 1e-12), np.nan)


def _hf_slope(power: np.ndarray) -> float:
    """Power-law slope of P(ω) ∝ ω^{slope}. Positive ⇒ HF-heavy. Amplitude-domain."""
    n = len(power)
    freqs = (np.arange(n) + 0.5) / n
    valid = np.isfinite(power) & (power > 0) & (freqs > 1.0 / n)
    if valid.sum() < 3:
        return float("nan")
    lf, lp = np.log(freqs[valid]), np.log(power[valid])
    slope = np.polyfit(lf, lp, 1)[0]
    return float(slope / 2.0)


# ---------------------------------------------------------------------------
# Replay one capture × one combo
# ---------------------------------------------------------------------------


def replay_one(
    feats: list[torch.Tensor],
    spatial_hw: tuple[int, int] | None,
    m: int,
    lam: float,
    w: float,
    mask: np.ndarray,
    num_steps: int,
    spectrum_bins: int,
    device: torch.device,
    want_spec: bool = True,
) -> dict:
    """Replay the forecaster over one capture under `mask`. Returns per-cached-step
    relative-L2 residuals + (optionally) the pooled residual radial spectrum."""
    feat_shape = tuple(feats[0].shape)
    fc = SpectrumPredictor(
        m=m,
        lam=lam,
        w=w,
        device=device,
        feature_shape=feat_shape,
        total_steps=num_steps,
    )
    rel = np.full(num_steps, np.nan, dtype=np.float64)
    res_for_spec: list[np.ndarray] = []
    for i in range(num_steps):
        if mask[i]:
            fc.update(float(i), feats[i])
            continue
        if fc.cheb.t_buf.numel() < 2:
            continue
        pred = fc.predict(float(i)).float()
        actual = feats[i].float()
        res = actual - pred
        denom = float(actual.norm()) or 1e-8
        rel[i] = float(res.norm()) / denom
        if want_spec and spatial_hw is not None and len(res_for_spec) < 4:
            r = res.cpu().numpy()  # (T, H, W, D)
            h, wdim = spatial_hw
            r = np.moveaxis(r, -1, 1)  # (T, D, H, W)
            d_sub = r[:, :: max(1, r.shape[1] // 16)]  # ≤16 channels
            res_for_spec.append(d_sub.reshape(-1, h, wdim))
    spec = (
        _radial_power(np.concatenate(res_for_spec), spectrum_bins)
        if res_for_spec
        else None
    )
    return {"rel": rel, "spec": spec}


def _summarize_rel(rel: np.ndarray, mask: np.ndarray) -> dict | None:
    """mean / peak / late-third relative-L2 over the cached steps. None if empty."""
    cached_rel = rel[~np.isnan(rel)]
    if cached_rel.size == 0:
        return None
    late = rel[~mask]
    late = late[~np.isnan(late)]
    late_third = late[len(late) * 2 // 3 :] if late.size else cached_rel
    return {
        "mean_rel": float(np.mean(cached_rel)),
        "peak_rel": float(np.max(cached_rel)),
        "late_rel": float(np.mean(late_third)),
        "n_cached": int((~mask).sum()),
    }


def replay_at_combo(
    feats: list[torch.Tensor],
    hw: tuple[int, int] | None,
    num_steps: int,
    device: torch.device,
    combo: dict = PROD_COMBO,
    stop: int = -1,
) -> dict | None:
    """Score one capture at a single (m,lam,w,window,flex,warmup) combo.

    Returns mean/peak/late rel-L2 + n_cached, or None if no cached steps. Used by
    compare_samplers.py to compare samplers at the production-default forecaster.
    """
    mask = production_actual_mask(
        num_steps, combo["warmup"], combo["window"], combo["flex"], stop
    )
    out = replay_one(
        feats,
        hw,
        combo["m"],
        combo["lam"],
        combo["w"],
        mask,
        num_steps,
        32,
        device,
        want_spec=False,
    )
    return _summarize_rel(out["rel"], mask)


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------


def load_capture(npz_path: Path, device: torch.device):
    d = np.load(npz_path)
    cond = d["cond"]  # (num_steps, T, H, W, D) fp16
    num_steps = int(d["num_steps"])
    cond_t = torch.from_numpy(cond.astype(np.float32)).to(device)
    feats = [cond_t[i] for i in range(num_steps)]
    _t, h, w, _dd = (int(x) for x in d["feat_shape"])
    return feats, (h, w), num_steps


def sweep(args, out_dir: Path):
    cap_dir = Path(args.captures_dir)
    cap_files = sorted(cap_dir.glob("*.npz"))
    if not cap_files:
        raise SystemExit(f"no *.npz captures under {cap_dir}")
    print(f"{len(cap_files)} capture(s) under {cap_dir}")

    combos = list(
        itertools.product(
            args.m_list,
            args.lam_list,
            args.w_list,
            args.window_list,
            args.flex_list,
            args.warmup_list,
        )
    )
    print(f"{len(combos)} hyperparameter cells")

    spec_combo = (
        3,
        0.1,
        0.3,
        args.window_list[0],
        args.flex_list[0],
        args.warmup_list[0],
    )
    device = torch.device(args.device)
    print(f"device = {device}")

    acc: dict[tuple, list] = {c: [] for c in combos}
    for ci, cf in enumerate(cap_files):
        feats, hw, num_steps = load_capture(cf, device)
        print(f"[{ci + 1}/{len(cap_files)}] {cf.name}  steps={num_steps} hw={hw}")
        for combo in combos:
            m, lam, w, window, flex, warmup = combo
            mask = production_actual_mask(num_steps, warmup, window, flex, args.stop)
            out = replay_one(
                feats,
                hw,
                m,
                lam,
                w,
                mask,
                num_steps,
                args.spectrum_bins,
                device,
                want_spec=(combo == spec_combo),
            )
            summ = _summarize_rel(out["rel"], mask)
            if summ is None:
                continue
            summ["hf_slope"] = (
                _hf_slope(out["spec"]) if out["spec"] is not None else float("nan")
            )
            summ["rel_curve"] = out["rel"]
            acc[combo].append(summ)

    rows = []
    curves: dict[str, np.ndarray] = {}
    for combo in combos:
        recs = acc[combo]
        if not recs:
            continue
        m, lam, w, window, flex, warmup = combo
        rows.append(
            {
                "m": m,
                "lam": lam,
                "w": w,
                "window_size": window,
                "flex_window": flex,
                "warmup_steps": warmup,
                "n_cached": int(np.mean([r["n_cached"] for r in recs])),
                "rel_l2_mean": float(np.mean([r["mean_rel"] for r in recs])),
                "rel_l2_std": float(np.std([r["mean_rel"] for r in recs])),
                "peak_rel_l2": float(np.mean([r["peak_rel"] for r in recs])),
                "late_rel_l2": float(np.mean([r["late_rel"] for r in recs])),
                "hf_slope": float(np.nanmean([r["hf_slope"] for r in recs])),
            }
        )
        stacked = np.vstack([r["rel_curve"] for r in recs])
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            curves[f"m{m}_lam{lam}_w{w}_ws{window}_fx{flex}_wu{warmup}"] = np.nanmean(
                stacked, axis=0
            )

    rows.sort(key=lambda r: r["rel_l2_mean"])
    return rows, curves


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_outputs(rows, curves, args, out_dir: Path):
    artifacts = []
    csv_path = out_dir / "residual_summary.csv"
    with open(csv_path, "w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wri.writeheader()
        wri.writerows(rows)
    artifacts.append(csv_path.name)

    (out_dir / "residual_curves.json").write_text(
        json.dumps({k: v.tolist() for k, v in curves.items()}, indent=2)
    )
    artifacts.append("residual_curves.json")

    if HAS_MPL and not args.no_plot and len(args.m_list) > 1 and len(args.lam_list) > 1:
        best_w = rows[0]["w"]
        sub = [
            r
            for r in rows
            if r["w"] == best_w
            and r["window_size"] == rows[0]["window_size"]
            and r["flex_window"] == rows[0]["flex_window"]
            and r["warmup_steps"] == rows[0]["warmup_steps"]
        ]
        H = np.full((len(args.m_list), len(args.lam_list)), np.nan)
        look = {(r["m"], r["lam"]): r["rel_l2_mean"] for r in sub}
        for i, mm in enumerate(args.m_list):
            for j, ll in enumerate(args.lam_list):
                H[i, j] = look.get((mm, ll), np.nan)
        fig, ax = plt.subplots(
            figsize=(1.1 * len(args.lam_list) + 2, 0.6 * len(args.m_list) + 2)
        )
        im = ax.imshow(np.log10(np.maximum(H, 1e-6)), cmap="viridis_r", aspect="auto")
        ax.set_xticks(range(len(args.lam_list)))
        ax.set_xticklabels([str(x) for x in args.lam_list], rotation=35, ha="right")
        ax.set_yticks(range(len(args.m_list)))
        ax.set_yticklabels([str(x) for x in args.m_list])
        ax.set_xlabel("ridge_lambda")
        ax.set_ylabel("cheby_degree (m)")
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                if not np.isnan(H[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{H[i, j]:.3f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white" if H[i, j] > np.nanmedian(H) else "black",
                    )
        ax.set_title(f"forecast rel-L2 residual (w={best_w}) — lower = better")
        plt.colorbar(im, ax=ax, label=r"$\log_{10}$ rel-L2")
        plt.tight_layout()
        path = out_dir / "residual_heatmap.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        artifacts.append(path.name)

    print(
        f"\n{'m':>2} {'lam':>7} {'w':>5} {'ws':>4} {'fx':>5} {'wu':>3}  "
        f"{'cached':>7} {'rel-L2':>9} {'late':>9} {'hf_slope':>8}"
    )
    print("-" * 72)
    for r in rows[: args.top]:
        print(
            f"{r['m']:>2} {r['lam']:>7g} {r['w']:>5g} {r['window_size']:>4g} "
            f"{r['flex_window']:>5g} {r['warmup_steps']:>3} "
            f"{r['n_cached']:>7} {r['rel_l2_mean']:>9.4f} "
            f"{r['late_rel_l2']:>9.4f} {r['hf_slope']:>8.3f}"
        )
    return artifacts


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--captures_dir", required=True, help="Dir of *.npz from capture_features.py"
    )
    ap.add_argument("--m_list", type=int, nargs="+", default=[2, 3, 4, 5])
    ap.add_argument(
        "--lam_list", type=float, nargs="+", default=[1e-3, 1e-2, 1e-1, 1.0]
    )
    ap.add_argument("--w_list", type=float, nargs="+", default=[0.0, 0.3, 0.7, 1.0])
    ap.add_argument("--window_list", type=float, nargs="+", default=[2.0])
    ap.add_argument("--flex_list", type=float, nargs="+", default=[0.25])
    ap.add_argument("--warmup_list", type=int, nargs="+", default=[7])
    ap.add_argument(
        "--stop", type=int, default=-1, help="stop_caching_step (-1 = num_steps-3)"
    )
    ap.add_argument("--spectrum_bins", type=int, default=32)
    ap.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--no_plot", action="store_true")
    ap.add_argument("--label", type=str, default="replay")
    args = ap.parse_args()

    out_dir = make_run_dir("spectrum_pareto", label=args.label)
    print(f"out_dir = {out_dir}")
    rows, curves = sweep(args, out_dir)
    if not rows:
        raise SystemExit(
            "no rows produced — check captures / schedule (no cached steps?)"
        )
    artifacts = write_outputs(rows, curves, args, out_dir)

    def _find(m, lam, w):
        for r in rows:
            if r["m"] == m and abs(r["lam"] - lam) < 1e-12 and abs(r["w"] - w) < 1e-12:
                return r
        return None

    prod = _find(3, 0.1, 0.3)
    metrics = {
        "n_cells": len(rows),
        "rows": rows,
        "best_cell": rows[0],
        "calibration": (
            {
                "note": "feed to analyze_drift.py --error_mag/--err_hf_bias",
                "production_default": prod,
                "error_mag": prod["rel_l2_mean"] if prod else None,
                "err_hf_bias": prod["hf_slope"] if prod else None,
            }
            if prod
            else {"note": "production default (m3/lam0.1/w0.3) not in sweep grid"}
        ),
    }
    result_path = write_result(
        out_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics=metrics,
        artifacts=artifacts,
        device=args.device,
    )
    print(f"\nresult → {result_path}")


if __name__ == "__main__":
    main()
