#!/usr/bin/env python
"""Variance-reduction headroom probe for AsymFlow §5.2 on Anima latents.

See README.md. Loads cached (latent, T5 embedding) pairs, picks K samples
and T timesteps, samples N noise vectors per (sample, t), and measures the
correlation between the standard FM residual `(x_0 - x_hat_0)` and the
low-pass control-variate residual `(x_0^L - x_hat_0^L)`. The squared
correlation rho^2 bounds the variance reduction achievable by the
control-variate target swap; if it's near zero, the trick is dead.
"""

from __future__ import annotations

import argparse
import csv
import gc
import glob
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from bench._common import make_run_dir, write_result  # noqa: E402
from library.anima import weights as anima_utils  # noqa: E402
from library.runtime.fei import (  # noqa: E402
    compute_fei_2band,
    fei_sigma_low,
    gaussian_blur_2d,
)
from safetensors.torch import load_file  # noqa: E402


log = logging.getLogger("fm_vr_headroom")


def parse_args():
    p = argparse.ArgumentParser()
    # Legacy single-model shortcut. If set, both trainable and frozen default
    # to this path — that's the same-model regime, which produces a
    # ρ²≈1 measurement artifact and is kept only for back-compat with
    # the earlier smoke runs in `results/`.
    p.add_argument(
        "--dit",
        default=None,
        help="Legacy: use the same DiT for both Y and Z. Prefer --trainable_dit / --frozen_dit.",
    )
    p.add_argument(
        "--trainable_dit",
        default=None,
        help="DiT used to produce x̂_0 (the loss-side prediction). Defaults to --dit.",
    )
    p.add_argument(
        "--frozen_dit",
        default=None,
        help="DiT used to produce x̂_0^L (the control variate's prediction). Defaults to --dit.",
    )
    p.add_argument("--data_dir", default="post_image_dataset/lora")
    p.add_argument(
        "--bucket",
        default=None,
        help="Bucket filter WxH (latent dims). If omitted, the most common bucket in --data_dir is used.",
    )
    p.add_argument("--num_samples", type=int, default=8)
    p.add_argument("--num_timesteps", type=int, default=8)
    p.add_argument("--num_noise", type=int, default=64)
    p.add_argument(
        "--noise_batch_size",
        type=int,
        default=16,
        help="Chunk noise samples into batches of this size for the model forward.",
    )
    p.add_argument("--t_min", type=float, default=0.05)
    p.add_argument("--t_max", type=float, default=0.95)
    p.add_argument("--fei_sigma_low_div", type=float, default=4.0)
    p.add_argument(
        "--mid_t_lo",
        type=float,
        default=0.20,
        help="Lower edge of the 'mid-t' window used for the verdict (drops α_t→0 degeneracy).",
    )
    p.add_argument(
        "--mid_t_hi",
        type=float,
        default=0.80,
        help="Upper edge of the 'mid-t' window used for the verdict.",
    )
    p.add_argument(
        "--null_runs",
        action="store_true",
        help="Also run a decorrelated-ε null (independent ε for x_t and x_t^L). "
        "Pair construction requires shared ε, so this null tells you how much "
        "of the measured ρ² is artifact (model smoothness in input).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--attn_mode", default="flash")
    p.add_argument("--label", type=str, default=None)
    p.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile each DiT block (per-block, via DiT.compile_blocks). "
        "Compiles both trainable and frozen models. First (sample, t) pair pays "
        "the compile cost (~30–60s per model); subsequent pairs run faster.",
    )
    p.add_argument(
        "--compile_mode",
        default=None,
        help="Optional inductor mode for compile_blocks (e.g. 'reduce-overhead'). "
        "Leave unset for the default.",
    )
    return p.parse_args()


def discover_samples(data_dir: Path, bucket: str | None, num_samples: int, seed: int):
    """Return list of (stem, latent_key, npz_path, te_path) for the chosen bucket.

    npz filename pattern is `{stem}_{Wpix}x{Hpix}_anima.npz`; TE sidecar drops
    the resolution token. Strip both to recover `{stem}`.
    """
    import re

    res_re = re.compile(r"_(\d{3,5})x(\d{3,5})_anima\.npz$")
    npz_paths = sorted(glob.glob(str(data_dir / "*_anima.npz")))
    if not npz_paths:
        raise SystemExit(f"no `*_anima.npz` in {data_dir}")
    by_bucket: dict[str, list[tuple[str, str, str, str]]] = {}
    for p in npz_paths:
        name = Path(p).name
        m = res_re.search(name)
        if not m:
            continue
        stem = name[: m.start()]
        te = data_dir / f"{stem}_anima_te.safetensors"
        if not te.exists():
            continue
        with np.load(p) as z:
            for k in z.keys():
                if k.startswith("latents_"):
                    bk = k.removeprefix("latents_")
                    by_bucket.setdefault(bk, []).append((stem, k, p, str(te)))
                    break
    if not by_bucket:
        raise SystemExit("no paired (latent, TE) samples found")
    if bucket:
        chosen = bucket
    else:
        chosen = max(by_bucket, key=lambda k: len(by_bucket[k]))
    if chosen not in by_bucket:
        raise SystemExit(
            f"bucket {chosen!r} not found. Available: "
            f"{sorted((k, len(v)) for k, v in by_bucket.items())[:10]} (top 10)"
        )
    pool = by_bucket[chosen]
    if len(pool) < num_samples:
        raise SystemExit(
            f"bucket {chosen!r} has only {len(pool)} paired samples; "
            f"need at least {num_samples}"
        )
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), size=num_samples, replace=False)
    return chosen, [pool[i] for i in idx]


def sample_timesteps(num: int, t_min: float, t_max: float, seed: int) -> torch.Tensor:
    # Uniform grid; deterministic. Trims to avoid t=0 (σ_t in denominator).
    return torch.linspace(t_min, t_max, num)


def load_pair(npz_path: str, latent_key: str, te_path: str, device, dtype):
    z = np.load(npz_path)
    x0 = torch.from_numpy(z[latent_key]).to(device=device, dtype=dtype)  # (C, H, W)
    z.close()
    te_data = load_file(te_path)
    crossattn = te_data["crossattn_emb_v0"].to(device=device, dtype=dtype)  # (S, D)
    return x0, crossattn


@torch.inference_mode()
def predict_x0_batch(
    anima, x_t_batch: torch.Tensor, t_scalar: float, crossattn_B: torch.Tensor, sigma_t: float
) -> torch.Tensor:
    """Run model on `(B, C, H, W)` x_t, return x_0 prediction `(B, C, H, W)`.

    Anima predicts velocity `u = eps - x_0` in 5D; we convert.
    """
    B, C, H, W = x_t_batch.shape
    x_5d = x_t_batch.unsqueeze(2)  # (B, C, 1, H, W)
    timesteps = torch.full(
        (B,), t_scalar * 1000.0, dtype=x_t_batch.dtype, device=x_t_batch.device
    )
    padding_mask = torch.zeros(
        B, 1, H, W, dtype=x_t_batch.dtype, device=x_t_batch.device
    )
    u_pred = anima(
        x_5d,
        timesteps,
        crossattn_B,
        padding_mask=padding_mask,
    ).squeeze(2)  # (B, C, H, W)
    x0_hat = x_t_batch - sigma_t * u_pred
    return x0_hat


def variance_metrics(Y: torch.Tensor, Z: torch.Tensor) -> dict:
    """Y, Z: (N, ...) with N noise samples along dim 0.

    Returns rho_sq (global + per-element averaged), optimal lambdas,
    pre/post variance reductions.
    """
    Y_flat = Y.reshape(Y.shape[0], -1).double()
    Z_flat = Z.reshape(Z.shape[0], -1).double()
    Y_c = Y_flat - Y_flat.mean(dim=0, keepdim=True)
    Z_c = Z_flat - Z_flat.mean(dim=0, keepdim=True)

    cov_elem = (Y_c * Z_c).mean(dim=0)
    var_y_elem = (Y_c * Y_c).mean(dim=0)
    var_z_elem = (Z_c * Z_c).mean(dim=0)

    # Per-element rho^2 (cap any numerical >1 from finite N to 1.0).
    rho_sq_elem = (cov_elem.pow(2) / (var_y_elem * var_z_elem).clamp_min(1e-30)).clamp(0.0, 1.0)

    # Global lambda* minimizing Var(Y_total + lambda * Z_total) where total = sum over elements.
    sum_cov = cov_elem.sum()
    sum_var_z = var_z_elem.sum()
    lambda_global = -(sum_cov / sum_var_z.clamp_min(1e-30)).item()

    var_y_total = var_y_elem.sum()
    var_after_global = ((Y_c + lambda_global * Z_c).pow(2).mean(dim=0)).sum()
    reduction_global = (1.0 - var_after_global / var_y_total.clamp_min(1e-30)).item()

    # Per-element optimal lambda.
    lambda_per_elem = -cov_elem / var_z_elem.clamp_min(1e-30)
    var_after_per = ((Y_c + lambda_per_elem * Z_c).pow(2).mean(dim=0)).sum()
    reduction_per = (1.0 - var_after_per / var_y_total.clamp_min(1e-30)).item()

    # Global rho^2 (treating the full flattened vector as a single 'observation' per noise sample).
    Y_sum = Y_c.sum(dim=1)
    Z_sum = Z_c.sum(dim=1)
    rho_sq_global = (Y_sum * Z_sum).mean().pow(2) / (
        (Y_sum * Y_sum).mean() * (Z_sum * Z_sum).mean()
    ).clamp_min(1e-30)

    return {
        "rho_sq_global": float(rho_sq_global.clamp(0, 1).item()),
        "rho_sq_per_elem_mean": float(rho_sq_elem.mean().item()),
        "rho_sq_per_elem_median": float(rho_sq_elem.median().item()),
        "lambda_global": float(lambda_global),
        "reduction_global_lambda": float(reduction_global),
        "reduction_per_elem_lambda": float(reduction_per),
        "var_y_total": float(var_y_total.item()),
        "var_y_per_elem_mean": float(var_y_elem.mean().item()),
    }


def _resolve_dit_paths(args) -> tuple[str, str]:
    trainable = args.trainable_dit or args.dit
    frozen = args.frozen_dit or args.dit
    if trainable is None or frozen is None:
        raise SystemExit(
            "Specify --trainable_dit and --frozen_dit "
            "(or the legacy --dit shortcut for the same-model regime)."
        )
    return trainable, frozen


def _project_bands(x: torch.Tensor, sigma_low: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(low-pass, high-pass complement)`` of ``x`` with the FEI kernel."""
    low = gaussian_blur_2d(x, sigma_low)
    high = x - low
    return low, high


def _banded_metrics(Y: torch.Tensor, Z: torch.Tensor, sigma_low: float) -> dict:
    """ρ² / λ on the full signal AND on each FEI band separately.

    Returned keys are prefixed by band: ``global__``, ``low_band__``,
    ``high_band__``. The informative number for VR headroom is
    ``high_band__rho_sq_global`` — the low-band ρ² is ≈ 1 by
    construction (`x_0^L = LP(x_0)`) and reflects the part of the FM
    signal that VR will *cancel*, not the part it usefully de-noises.

    Also emits ``perband__reduction_combined`` — the v3 feasibility metric:
    full-signal variance reduction when λ_low and λ_high are applied to
    their respective bands of Z. Compare to ``global__reduction_global_lambda``
    via ``perband__delta_vs_global``.
    """
    Y_low, Y_high = _project_bands(Y, sigma_low)
    Z_low, Z_high = _project_bands(Z, sigma_low)
    out: dict[str, float] = {}
    for prefix, (a, b) in [
        ("global", (Y, Z)),
        ("low_band", (Y_low, Z_low)),
        ("high_band", (Y_high, Z_high)),
    ]:
        for k, v in variance_metrics(a, b).items():
            out[f"{prefix}__{k}"] = v

    lam_low = out["low_band__lambda_global"]
    lam_high = out["high_band__lambda_global"]
    Y_flat = Y.reshape(Y.shape[0], -1).double()
    Zl_flat = Z_low.reshape(Z_low.shape[0], -1).double()
    Zh_flat = Z_high.reshape(Z_high.shape[0], -1).double()
    Y_c = Y_flat - Y_flat.mean(dim=0, keepdim=True)
    Zl_c = Zl_flat - Zl_flat.mean(dim=0, keepdim=True)
    Zh_c = Zh_flat - Zh_flat.mean(dim=0, keepdim=True)
    residual = Y_c + lam_low * Zl_c + lam_high * Zh_c
    var_y_total = (Y_c * Y_c).mean(dim=0).sum()
    var_after_pb = (residual * residual).mean(dim=0).sum()
    reduction_pb = (1.0 - var_after_pb / var_y_total.clamp_min(1e-30)).item()
    out["perband__reduction_combined"] = float(reduction_pb)
    out["perband__delta_vs_global"] = float(reduction_pb - out["global__reduction_global_lambda"])
    out["perband__lambda_low"] = float(lam_low)
    out["perband__lambda_high"] = float(lam_high)
    return out


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_dir = make_run_dir("fm_vr_headroom", label=args.label)
    log.info(f"output → {run_dir}")

    data_dir = Path(args.data_dir)
    bucket, samples = discover_samples(data_dir, args.bucket, args.num_samples, args.seed)
    log.info(f"bucket={bucket} num_samples={len(samples)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    trainable_path, frozen_path = _resolve_dit_paths(args)
    same_model = trainable_path == frozen_path
    if same_model:
        log.info(
            "WARNING: trainable and frozen DiT are the same path — this is the "
            "smoke regime; ρ² will be inflated by the shared-model artifact. "
            "Verdict will be ARTIFACT."
        )

    log.info(f"loading trainable DiT: {trainable_path}")
    anima_trainable = anima_utils.load_anima_model(
        device=device,
        dit_path=trainable_path,
        attn_mode=args.attn_mode,
        loading_device=device,
        dit_weight_dtype=dtype,
    )
    anima_trainable.to(device, dtype=dtype).eval().requires_grad_(False)
    anima_trainable.reset_mod_guidance()
    if args.compile:
        log.info("compiling trainable DiT blocks (this can take ~30–60s)")
        anima_trainable.compile_blocks(mode=args.compile_mode)

    if same_model:
        anima_frozen = anima_trainable
    else:
        log.info(f"loading frozen DiT:    {frozen_path}")
        anima_frozen = anima_utils.load_anima_model(
            device=device,
            dit_path=frozen_path,
            attn_mode=args.attn_mode,
            loading_device=device,
            dit_weight_dtype=dtype,
        )
        anima_frozen.to(device, dtype=dtype).eval().requires_grad_(False)
        anima_frozen.reset_mod_guidance()
        if args.compile:
            log.info("compiling frozen DiT blocks (this can take ~30–60s)")
            anima_frozen.compile_blocks(mode=args.compile_mode)

    timesteps = sample_timesteps(args.num_timesteps, args.t_min, args.t_max, args.seed)
    log.info(f"timesteps: {[round(t.item(), 3) for t in timesteps]}")

    h_lat, w_lat = (int(x) for x in bucket.split("x"))
    sigma_low = fei_sigma_low(h_lat, w_lat, args.fei_sigma_low_div)
    log.info(f"sigma_low = {sigma_low:.3f}")
    if args.null_runs:
        log.info("running decorrelated-ε null alongside paired measurement")

    rng = torch.Generator(device=device).manual_seed(args.seed)

    rows: list[dict] = []
    total_pairs = len(samples) * len(timesteps)
    pbar = tqdm(total=total_pairs, desc="vr-headroom", dynamic_ncols=True)
    for si, (stem, latent_key, npz_path, te_path) in enumerate(samples):
        x0, crossattn = load_pair(npz_path, latent_key, te_path, device, dtype)
        # x0: (C, H_lat, W_lat). x0^L = low-pass via the FEI kernel.
        x0_4d = x0.unsqueeze(0)
        x0_L_4d = gaussian_blur_2d(x0_4d, sigma_low)
        fei = compute_fei_2band(x0_4d, sigma_low)[0].cpu().tolist()  # [e_low, e_high]

        for t_scalar in timesteps:
            t = float(t_scalar)
            alpha_t = 1.0 - t
            sigma_t = t

            N = args.num_noise
            eps_full = torch.randn(
                (N, *x0.shape), generator=rng, device=device, dtype=dtype
            )
            eps_null = (
                torch.randn(
                    (N, *x0.shape), generator=rng, device=device, dtype=dtype
                )
                if args.null_runs
                else None
            )

            x_t_full = alpha_t * x0_4d + sigma_t * eps_full
            x_t_L_full = alpha_t * x0_L_4d + sigma_t * eps_full
            x_t_L_null_full = (
                alpha_t * x0_L_4d + sigma_t * eps_null if eps_null is not None else None
            )

            crossattn_B = crossattn.unsqueeze(0)

            Y_chunks, Z_chunks, Zn_chunks = [], [], []
            for i in range(0, N, args.noise_batch_size):
                xb = x_t_full[i : i + args.noise_batch_size]
                xb_L = x_t_L_full[i : i + args.noise_batch_size]
                B_eff = xb.shape[0]
                ca_B = crossattn_B.expand(B_eff, -1, -1).contiguous()

                x0_hat = predict_x0_batch(anima_trainable, xb, t, ca_B, sigma_t)
                x0_L_hat = predict_x0_batch(anima_frozen, xb_L, t, ca_B, sigma_t)

                Y_chunks.append((x0_4d - x0_hat) / sigma_t)
                Z_chunks.append((x0_L_4d - x0_L_hat) / sigma_t)

                if x_t_L_null_full is not None:
                    xb_Ln = x_t_L_null_full[i : i + args.noise_batch_size]
                    x0_L_null_hat = predict_x0_batch(
                        anima_frozen, xb_Ln, t, ca_B, sigma_t
                    )
                    Zn_chunks.append((x0_L_4d - x0_L_null_hat) / sigma_t)

            Y = torch.cat(Y_chunks, dim=0)
            Z = torch.cat(Z_chunks, dim=0)

            metrics = _banded_metrics(Y, Z, sigma_low)
            if Zn_chunks:
                Z_null = torch.cat(Zn_chunks, dim=0)
                for k, v in _banded_metrics(Y, Z_null, sigma_low).items():
                    metrics[f"null__{k}"] = v

            metrics.update(
                {
                    "stem": stem,
                    "t": t,
                    "fei_low": fei[0],
                    "fei_high": fei[1],
                    "bucket": bucket,
                }
            )
            rows.append(metrics)
            postfix = {
                "t": f"{t:.2f}",
                "ρ²_g": f"{metrics['global__rho_sq_global']:.3f}",
                "ρ²_hi": f"{metrics['high_band__rho_sq_global']:.3f}",
                "ρ²_lo": f"{metrics['low_band__rho_sq_global']:.3f}",
                "λ_g": f"{metrics['global__lambda_global']:+.3f}",
                "Δpb": f"{metrics['perband__delta_vs_global']:+.4f}",
            }
            if Zn_chunks:
                postfix["ρ²_null_hi"] = f"{metrics['null__high_band__rho_sq_global']:.3f}"
            pbar.set_postfix(postfix)
            pbar.update(1)

        del x0, x0_4d, x0_L_4d, crossattn
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    pbar.close()

    # Per-sample-t CSV.
    csv_path = run_dir / "per_sample_t.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Aggregations — mid-t filter excludes the α_t → 0 degeneracy where
    # x_t and x_t^L become near-identical and ρ² is mechanically ≈ 1.
    def in_mid_t(r) -> bool:
        return args.mid_t_lo <= r["t"] <= args.mid_t_hi

    def agg(key: str, predicate=lambda r: True) -> dict:
        vals = np.array([r[key] for r in rows if predicate(r)], dtype=np.float64)
        if vals.size == 0:
            return {
                "mean": float("nan"),
                "median": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
                "n": 0,
            }
        return {
            "mean": float(vals.mean()),
            "median": float(np.median(vals)),
            "std": float(vals.std()),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "n": int(vals.size),
        }

    summary: dict = {
        "bucket": bucket,
        "n_pairs": len(rows),
        "trainable_dit": trainable_path,
        "frozen_dit": frozen_path,
        "same_model": same_model,
        "mid_t_window": [args.mid_t_lo, args.mid_t_hi],
        "global__rho_sq_global": agg("global__rho_sq_global"),
        "low_band__rho_sq_global": agg("low_band__rho_sq_global"),
        "high_band__rho_sq_global": agg("high_band__rho_sq_global"),
        "global__rho_sq_global__mid_t": agg("global__rho_sq_global", in_mid_t),
        "high_band__rho_sq_global__mid_t": agg("high_band__rho_sq_global", in_mid_t),
        "low_band__rho_sq_global__mid_t": agg("low_band__rho_sq_global", in_mid_t),
        "global__lambda_global": agg("global__lambda_global"),
        "high_band__lambda_global": agg("high_band__lambda_global"),
        "low_band__lambda_global": agg("low_band__lambda_global"),
        "global__reduction_global_lambda": agg("global__reduction_global_lambda"),
        "high_band__reduction_global_lambda": agg("high_band__reduction_global_lambda"),
        "perband__reduction_combined": agg("perband__reduction_combined"),
        "perband__delta_vs_global": agg("perband__delta_vs_global"),
        "perband__delta_vs_global__mid_t": agg("perband__delta_vs_global", in_mid_t),
        "perband__lambda_low": agg("perband__lambda_low"),
        "perband__lambda_high": agg("perband__lambda_high"),
    }
    if args.null_runs:
        summary["null__high_band__rho_sq_global__mid_t"] = agg(
            "null__high_band__rho_sq_global", in_mid_t
        )
        summary["null__global__rho_sq_global__mid_t"] = agg(
            "null__global__rho_sq_global", in_mid_t
        )

    # Per-t breakdown.
    def per_t(key: str) -> dict:
        bins: dict[float, list[float]] = {}
        for r in rows:
            bins.setdefault(round(r["t"], 3), []).append(r[key])
        return {
            str(k): {"median": float(np.median(v)), "n": len(v)}
            for k, v in sorted(bins.items())
        }

    summary["rho_sq_global_by_t"] = per_t("global__rho_sq_global")
    summary["rho_sq_high_band_by_t"] = per_t("high_band__rho_sq_global")

    # Verdict — driven by high-band ρ² in the mid-t window, and (if a null
    # run was requested) the gap between paired and decorrelated-ε ρ².
    hi_mid = summary["high_band__rho_sq_global__mid_t"]["median"]
    null_gap = None
    if args.null_runs:
        null_hi_mid = summary["null__high_band__rho_sq_global__mid_t"]["median"]
        null_gap = hi_mid - null_hi_mid
        summary["null_gap__high_band__mid_t"] = float(null_gap)

    if same_model:
        verdict = "ARTIFACT"
    elif hi_mid >= 0.30 and (null_gap is None or null_gap >= 0.20):
        verdict = "HEADROOM"
    elif hi_mid >= 0.10:
        verdict = "MARGINAL"
    else:
        verdict = "DEAD"
    summary["verdict"] = verdict

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=summary,
        artifacts=["per_sample_t.csv", "summary.json"],
        label=args.label,
    )
    log.info(f"[bench] verdict: {verdict}")
    log.info(
        f"[bench]   high-band ρ² mid-t median={hi_mid:.3f}"
        + (f"   null_gap={null_gap:+.3f}" if null_gap is not None else "")
    )
    log.info(
        f"[bench]   global    ρ² mid-t median="
        f"{summary['global__rho_sq_global__mid_t']['median']:.3f}"
    )


if __name__ == "__main__":
    main()
