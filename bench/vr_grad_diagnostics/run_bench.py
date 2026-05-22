#!/usr/bin/env python
"""Gradient-level diagnostic for VR-loss (AsymFlow §5.2) on Anima.

See README.md. Loads base DiT + a LoRA-family adapter at a frozen θ,
samples K minibatches of (latent, t, ε), and for each batch computes:

    y = u_pred(x_t; θ) − (ε − x_0)               # standard FM residual
    z = u_pred(x_t^L; θ_base) − (ε − x_0^L)      # VR control variate (no_grad)
    g_std = ∂‖y‖²        / ∂θ_adapter
    g_vr  = ∂‖y + λz‖²   / ∂θ_adapter            = g_std + 2λ·∂⟨z, y⟩/∂θ

Aggregates: gradient-variance ratio, expected-direction cosine, per-batch
vs reference cosines. These are what the *optimizer* sees, in contrast to
the loss-level ρ² measured by ``bench/fm_vr_headroom``.

Adapter parameters only carry gradient — the base DiT is frozen and the
control-variate forward runs under ``network.set_multiplier(0)`` (same
contract as ``library/training/vr_forward.py``).
"""

from __future__ import annotations

import argparse
import csv
import gc
import glob
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from bench._common import make_run_dir, write_result  # noqa: E402
from library.anima import weights as anima_utils  # noqa: E402
from library.runtime.fei import fei_sigma_low, gaussian_blur_2d  # noqa: E402
from networks.lora_anima.factory import create_network_from_weights  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

log = logging.getLogger("vr_grad_diag")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dit", required=True, help="Base DiT safetensors (frozen).")
    p.add_argument(
        "--adapter",
        required=True,
        help="LoRA-family adapter safetensors. Defines the trainable parameter "
        "surface for gradient computation.",
    )
    p.add_argument("--data_dir", default="post_image_dataset/lora")
    p.add_argument(
        "--bucket",
        default=None,
        help="Bucket filter WxH (latent dims). Default: most populous bucket.",
    )
    p.add_argument(
        "--num_batches",
        type=int,
        default=24,
        help="K — number of minibatches. Each contributes one (g_std, g_vr) pair.",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Per-batch image count. Default 1 matches training step granularity.",
    )
    p.add_argument(
        "--num_timesteps",
        type=int,
        default=4,
        help="Distinct t values cycled through the K batches.",
    )
    p.add_argument("--t_min", type=float, default=0.05)
    p.add_argument("--t_max", type=float, default=0.95)
    p.add_argument("--fei_sigma_low_div", type=float, default=4.0)
    p.add_argument(
        "--lambda_mode",
        choices=("online", "fixed"),
        default="online",
        help="`online`: build λ_ema across K batches (matches live trainer). "
        "`fixed`: use --lambda_value uniformly.",
    )
    p.add_argument(
        "--lambda_value",
        type=float,
        default=-1.0,
        help="λ when --lambda_mode=fixed.",
    )
    p.add_argument(
        "--lambda_beta",
        type=float,
        default=0.01,
        help="EMA β for --lambda_mode=online. Matches the trainer default.",
    )
    p.add_argument(
        "--multiplier",
        type=float,
        default=1.0,
        help="Adapter multiplier used at forward time (default 1.0). "
        "The control-variate forward always uses 0.0.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--attn_mode", default="flash")
    p.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable activation checkpointing on the DiT. Strongly recommended — "
        "this bench runs two backwards per batch with retain_graph=True, so "
        "activation memory is what OOMs first. Trades ~30%% compute for ~4-5× "
        "smaller activation footprint.",
    )
    p.add_argument(
        "--cpu_offload_checkpointing",
        action="store_true",
        help="With --gradient_checkpointing, also CPU-offload the checkpoint "
        "activations. Further VRAM savings at higher compute cost.",
    )
    p.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile each DiT block (via DiT.compile_blocks). First batch "
        "pays the compile cost (~30-60s); subsequent batches run faster. The "
        "adapter monkey-patches are part of the compiled graph, so compile "
        "MUST happen after `apply_to` + `load_weights`.",
    )
    p.add_argument(
        "--compile_mode",
        default=None,
        help="Optional inductor mode for compile_blocks (e.g. 'reduce-overhead'). "
        "Leave unset for the default.",
    )
    p.add_argument("--label", type=str, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset discovery (mirrors bench/fm_vr_headroom/run_bench.py — same cached
# latent / TE sidecar layout).
# ---------------------------------------------------------------------------

_RES_RE = re.compile(r"_(\d{3,5})x(\d{3,5})_anima\.npz$")


def discover_samples(
    data_dir: Path, bucket: str | None, min_count: int, seed: int
) -> tuple[str, list[tuple[str, str, str, str]]]:
    npz_paths = sorted(glob.glob(str(data_dir / "*_anima.npz")))
    if not npz_paths:
        raise SystemExit(f"no `*_anima.npz` in {data_dir}")
    by_bucket: dict[str, list[tuple[str, str, str, str]]] = {}
    for p in npz_paths:
        name = Path(p).name
        m = _RES_RE.search(name)
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
    chosen = bucket or max(by_bucket, key=lambda k: len(by_bucket[k]))
    if chosen not in by_bucket:
        raise SystemExit(
            f"bucket {chosen!r} not found. Top buckets: "
            f"{sorted(((k, len(v)) for k, v in by_bucket.items()), key=lambda x: -x[1])[:5]}"
        )
    pool = by_bucket[chosen]
    if len(pool) < min_count:
        log.warning(
            f"bucket {chosen!r} has {len(pool)} samples; will resample with "
            f"replacement to reach {min_count} batches."
        )
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), size=min_count, replace=(len(pool) < min_count))
    return chosen, [pool[i] for i in idx]


def load_pair(npz_path: str, latent_key: str, te_path: str, device, dtype):
    z = np.load(npz_path)
    x0 = torch.from_numpy(z[latent_key]).to(device=device, dtype=dtype)
    z.close()
    te = load_file(te_path)
    ca = te["crossattn_emb_v0"].to(device=device, dtype=dtype)
    return x0, ca


# ---------------------------------------------------------------------------
# Model + network setup.
# ---------------------------------------------------------------------------


def build_anima_and_network(args, device, dtype):
    log.info(f"loading base DiT: {args.dit}")
    anima = anima_utils.load_anima_model(
        device=device,
        dit_path=args.dit,
        attn_mode=args.attn_mode,
        loading_device=device,
        dit_weight_dtype=dtype,
    )
    anima.to(device, dtype=dtype).requires_grad_(False)
    anima.reset_mod_guidance()
    # NB: training mode is set AFTER apply_to (below). Default-eval here would
    # silently disable gradient checkpointing (models.py:1188 gates on
    # self.training) and force the LoRA inference path on adapters.

    log.info(f"loading adapter:  {args.adapter}")
    network, _sd = create_network_from_weights(
        args.multiplier,
        args.adapter,
        None,  # ae (unused)
        None,  # text_encoders (unused)
        anima,
        for_inference=False,
    )
    network.apply_to([], anima, apply_text_encoder=False, apply_unet=True)
    info = network.load_weights(args.adapter)
    log.info(f"adapter loaded — {info}")

    network.to(device=device, dtype=dtype)
    network.requires_grad_(True)
    anima.requires_grad_(False)

    trainable = [p for p in network.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    log.info(f"adapter trainable params: {n_train:,} ({len(trainable)} tensors)")
    if n_train == 0:
        raise SystemExit(
            "no trainable adapter parameters detected — check the checkpoint."
        )

    if args.gradient_checkpointing:
        suffix = " (cpu offload)" if args.cpu_offload_checkpointing else ""
        log.info(f"enabling gradient checkpointing{suffix}")
        anima.enable_gradient_checkpointing(cpu_offload=args.cpu_offload_checkpointing)

    # Training mode on both anima (gates checkpointing) and network (selects
    # LoRA training-path forward — the path live training actually sees, with
    # T-LoRA mask, fp32-bottleneck matmuls, etc.). Adapter dropout if present
    # will fire — for clean diagnostics, ensure the trained adapter used 0.
    anima.train()
    network.train()

    # Compile after apply_to + load_weights so the LoRA monkey-patches are
    # baked into the graph.
    if args.compile:
        log.info("compiling DiT blocks (this can take ~30-60s on the first batch)")
        anima.compile_blocks(mode=args.compile_mode)
    return anima, network, trainable


# ---------------------------------------------------------------------------
# Forward + residual construction.
# ---------------------------------------------------------------------------


def _make_inputs(x0: torch.Tensor, t: float, eps: torch.Tensor, sigma_low: float):
    """Build (x_t, x_t^L, target_y, target_z) for a single (x0, t, ε).

    x0:  (B, C, H, W). eps: same shape.
    """
    alpha_t, sigma_t = 1.0 - t, t
    x0_L = gaussian_blur_2d(x0.float(), sigma_low).to(x0.dtype)
    x_t = alpha_t * x0 + sigma_t * eps
    x_t_L = alpha_t * x0_L + sigma_t * eps
    target_y = eps - x0  # velocity target: u = ε − x_0
    target_z = eps - x0_L
    return x_t, x_t_L, target_y, target_z


def _forward_velocity(
    anima, x_t: torch.Tensor, t: float, crossattn: torch.Tensor
) -> torch.Tensor:
    """Return (B, C, H, W) velocity prediction. Anima expects 5D (B, C, 1, H, W)."""
    B, C, H, W = x_t.shape
    timesteps = torch.full((B,), t * 1000.0, dtype=x_t.dtype, device=x_t.device)
    padding_mask = torch.zeros(B, 1, H, W, dtype=x_t.dtype, device=x_t.device)
    u = anima(
        x_t.unsqueeze(2),
        timesteps,
        crossattn,
        padding_mask=padding_mask,
    )
    return u.squeeze(2)


def compute_y_z(
    anima,
    network,
    x_t,
    x_t_L,
    t,
    crossattn,
    target_y,
    target_z,
):
    """Return (y, z) where y carries gradient and z does not.

    z is computed under network.set_multiplier(0) — the base-DiT prediction.
    """
    u_pred = _forward_velocity(anima, x_t, t, crossattn)
    y = u_pred - target_y

    orig_mult = float(getattr(network, "multiplier", 1.0))
    network.set_multiplier(0.0)
    try:
        with torch.no_grad():
            u_pred_L = _forward_velocity(anima, x_t_L, t, crossattn)
    finally:
        network.set_multiplier(orig_mult)
    z = (u_pred_L - target_z).detach()
    return y, z


# ---------------------------------------------------------------------------
# Gradient bookkeeping.
# ---------------------------------------------------------------------------


def _flat_norm_sq(grads: list[torch.Tensor]) -> float:
    return float(sum((g.detach().double() ** 2).sum() for g in grads if g is not None))


def _flat_dot(a: list[torch.Tensor], b: list[torch.Tensor]) -> float:
    out = 0.0
    for ga, gb in zip(a, b):
        if ga is None or gb is None:
            continue
        out += float((ga.detach().double() * gb.detach().double()).sum())
    return out


def _running_mean_add(buf: list[torch.Tensor], grads: list[torch.Tensor], k: int):
    """Welford-style online mean: buf += (g - buf) / k."""
    for bi, gi in zip(buf, grads):
        if gi is None:
            continue
        bi.add_((gi.detach().to(bi.dtype) - bi) / float(k))


def _zero_like(params: list[torch.Tensor]) -> list[torch.Tensor]:
    return [torch.zeros_like(p, dtype=torch.float64, device="cpu") for p in params]


# ---------------------------------------------------------------------------
# Per-batch probe.
# ---------------------------------------------------------------------------


def probe_batch(
    anima,
    network,
    trainable_params: list[torch.Tensor],
    x0: torch.Tensor,
    t: float,
    crossattn: torch.Tensor,
    eps: torch.Tensor,
    sigma_low: float,
    lambda_value: float,
):
    """One (forward + 2 backward) pass. Returns (g_std, g_vr, λ_batch_local, ‖y‖², ‖z‖²)."""
    x_t, x_t_L, target_y, target_z = _make_inputs(x0, t, eps, sigma_low)
    y, z = compute_y_z(anima, network, x_t, x_t_L, t, crossattn, target_y, target_z)

    # λ_batch_local on detached residuals (same formula as the trainer).
    with torch.no_grad():
        y_d = y.detach().double()
        z_d = z.double()
        cov = (y_d * z_d).sum()
        var = (z_d * z_d).sum().clamp_min(1e-30)
        lambda_batch_local = float(-(cov / var).item())
        y_norm_sq = float((y_d * y_d).sum())
        z_norm_sq = float((z_d * z_d).sum())

    L_std = (y * y).sum()
    g_std = torch.autograd.grad(L_std, trainable_params, retain_graph=True)

    # Cross-term: d⟨z, y⟩/dθ. g_vr = g_std + 2λ · cross.
    L_cross = (z * y).sum()
    cross = torch.autograd.grad(L_cross, trainable_params, retain_graph=False)

    g_vr = [
        gs.detach() + (2.0 * lambda_value) * c.detach() for gs, c in zip(g_std, cross)
    ]
    g_std = [gs.detach() for gs in g_std]
    # Move to CPU immediately to keep GPU memory bounded.
    g_std_cpu = [g.to("cpu", dtype=torch.float64) for g in g_std]
    g_vr_cpu = [g.to("cpu", dtype=torch.float64) for g in g_vr]
    return g_std_cpu, g_vr_cpu, lambda_batch_local, y_norm_sq, z_norm_sq


# ---------------------------------------------------------------------------
# Pass orchestration.
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_dir = make_run_dir("vr_grad_diagnostics", label=args.label)
    log.info(f"output → {run_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    data_dir = Path(args.data_dir)
    bucket, picks = discover_samples(data_dir, args.bucket, args.num_batches, args.seed)
    log.info(f"bucket={bucket} num_batches={len(picks)} batch_size={args.batch_size}")

    anima, network, trainable = build_anima_and_network(args, device, dtype)

    h_lat, w_lat = (int(x) for x in bucket.split("x"))
    sigma_low = fei_sigma_low(h_lat, w_lat, args.fei_sigma_low_div)
    log.info(f"sigma_low = {sigma_low:.3f}")

    rng = torch.Generator(device=device).manual_seed(args.seed)
    t_grid = torch.linspace(args.t_min, args.t_max, args.num_timesteps).tolist()

    # ----- Pass 1: λ discovery (optional). Determines λ_value used in pass 2. -----
    if args.lambda_mode == "online":
        lambda_ema: float | None = None
        log.info("pass 0: warm-in for λ_ema across K batches (no_grad)")
        for k, (stem, lkey, npz, te) in enumerate(
            tqdm(picks, desc="λ-warmin", dynamic_ncols=True)
        ):
            x0, ca = load_pair(npz, lkey, te, device, dtype)
            x0 = x0.unsqueeze(0)
            ca = ca.unsqueeze(0)
            t = t_grid[k % args.num_timesteps]
            eps = torch.randn_like(x0)
            x_t, x_t_L, ty, tz = _make_inputs(x0, t, eps, sigma_low)
            with torch.no_grad():
                y, z = compute_y_z(anima, network, x_t, x_t_L, t, ca, ty, tz)
                y_d, z_d = y.double(), z.double()
                cov = (y_d * z_d).sum()
                var = (z_d * z_d).sum().clamp_min(1e-30)
                lb = float(-(cov / var).item())
            lambda_ema = (
                lb
                if lambda_ema is None
                else (1.0 - args.lambda_beta) * lambda_ema + args.lambda_beta * lb
            )
        lambda_value = float(lambda_ema)
        log.info(f"λ_ema (online, K={len(picks)}): {lambda_value:+.4f}")
    else:
        lambda_value = float(args.lambda_value)
        log.info(f"λ (fixed): {lambda_value:+.4f}")

    # ----- Pass 2: gradient probe. Two backwards per batch. -----
    log.info(f"pass 1: gradient probe (K={len(picks)} backwards × 2)")
    g_std_mean = _zero_like(trainable)
    g_vr_mean = _zero_like(trainable)
    g_std_norms: list[float] = []
    g_vr_norms: list[float] = []
    inner_std_vr: list[float] = []  # ⟨g_std_k, g_vr_k⟩ per batch
    rows: list[dict] = []

    rng = torch.Generator(device=device).manual_seed(args.seed + 1)
    pbar = tqdm(picks, desc="grad-probe", dynamic_ncols=True)
    for k, (stem, lkey, npz, te) in enumerate(pbar):
        x0, ca = load_pair(npz, lkey, te, device, dtype)
        x0 = x0.unsqueeze(0)
        ca = ca.unsqueeze(0)
        t = t_grid[k % args.num_timesteps]
        eps = torch.randn(x0.shape, generator=rng, device=device, dtype=dtype)

        g_std_k, g_vr_k, lam_local, y2, z2 = probe_batch(
            anima, network, trainable, x0, t, ca, eps, sigma_low, lambda_value
        )

        n_std = _flat_norm_sq(g_std_k) ** 0.5
        n_vr = _flat_norm_sq(g_vr_k) ** 0.5
        ip_sv = _flat_dot(g_std_k, g_vr_k)
        cos_sv = ip_sv / max(n_std * n_vr, 1e-30)
        g_std_norms.append(n_std)
        g_vr_norms.append(n_vr)
        inner_std_vr.append(ip_sv)

        _running_mean_add(g_std_mean, g_std_k, k + 1)
        _running_mean_add(g_vr_mean, g_vr_k, k + 1)

        rows.append(
            {
                "k": k,
                "stem": stem,
                "t": t,
                "lambda_batch_local": lam_local,
                "y_norm_sq": y2,
                "z_norm_sq": z2,
                "g_std_norm": n_std,
                "g_vr_norm": n_vr,
                "cos_g_vr_g_std": cos_sv,
            }
        )
        pbar.set_postfix(
            {
                "‖g_std‖": f"{n_std:.3e}",
                "‖g_vr‖": f"{n_vr:.3e}",
                "cos": f"{cos_sv:+.3f}",
                "λ_b": f"{lam_local:+.3f}",
            }
        )

        del g_std_k, g_vr_k, x0, ca, eps
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ----- Pass 3: per-batch cosines vs g_*_ref (re-probe). -----
    log.info("pass 2: per-batch alignment to g_*_ref (re-probe with same RNG)")
    norm_ref_std = _flat_norm_sq(g_std_mean) ** 0.5
    norm_ref_vr = _flat_norm_sq(g_vr_mean) ** 0.5
    ip_refref = _flat_dot(g_std_mean, g_vr_mean)
    cos_refref = ip_refref / max(norm_ref_std * norm_ref_vr, 1e-30)

    rng = torch.Generator(device=device).manual_seed(args.seed + 1)
    for k, (stem, lkey, npz, te) in enumerate(
        tqdm(picks, desc="align-probe", dynamic_ncols=True)
    ):
        x0, ca = load_pair(npz, lkey, te, device, dtype)
        x0 = x0.unsqueeze(0)
        ca = ca.unsqueeze(0)
        t = t_grid[k % args.num_timesteps]
        eps = torch.randn(x0.shape, generator=rng, device=device, dtype=dtype)
        g_std_k, g_vr_k, _, _, _ = probe_batch(
            anima, network, trainable, x0, t, ca, eps, sigma_low, lambda_value
        )
        cos_std_ref = _flat_dot(g_std_k, g_std_mean) / max(
            g_std_norms[k] * norm_ref_std, 1e-30
        )
        cos_vr_ref = _flat_dot(g_vr_k, g_vr_mean) / max(
            g_vr_norms[k] * norm_ref_vr, 1e-30
        )
        cos_vr_std_ref = _flat_dot(g_vr_k, g_std_mean) / max(
            g_vr_norms[k] * norm_ref_std, 1e-30
        )
        rows[k]["cos_g_std_g_std_ref"] = cos_std_ref
        rows[k]["cos_g_vr_g_vr_ref"] = cos_vr_ref
        rows[k]["cos_g_vr_g_std_ref"] = cos_vr_std_ref

        del g_std_k, g_vr_k, x0, ca, eps
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ----- Aggregation. -----
    arr_n_std = np.array(g_std_norms)
    arr_n_vr = np.array(g_vr_norms)
    # Absolute variance: Var[g] = E[‖g‖²] − ‖E[g]‖² (trace of Cov[g]).
    # Reported but NOT the headline — VR collapses gradient *magnitude*, so
    # raw variance shrinks even when the gradient is relatively noisier. The
    # ratio that the optimizer actually cares about is the coefficient of
    # variation: var / ‖mean‖² (= squared coefficient of variation in the
    # vector-valued sense). That normalizes out the magnitude collapse.
    var_std = float((arr_n_std**2).mean() - norm_ref_std**2)
    var_vr = float((arr_n_vr**2).mean() - norm_ref_vr**2)
    var_ratio = var_vr / max(var_std, 1e-30)
    cov_std = var_std / max(norm_ref_std**2, 1e-30)  # squared coefficient of variation
    cov_vr = var_vr / max(norm_ref_vr**2, 1e-30)
    cov_ratio = cov_vr / max(cov_std, 1e-30)
    gradient_magnitude_ratio = norm_ref_vr / max(norm_ref_std, 1e-30)

    cos_std_ref = float(np.mean([r["cos_g_std_g_std_ref"] for r in rows]))
    cos_vr_ref = float(np.mean([r["cos_g_vr_g_vr_ref"] for r in rows]))
    cos_vr_std_ref = float(np.mean([r["cos_g_vr_g_std_ref"] for r in rows]))
    direction_lift = cos_vr_ref - cos_std_ref  # the optimizer's question

    summary = {
        "bucket": bucket,
        "num_batches": len(rows),
        "lambda_mode": args.lambda_mode,
        "lambda_value": lambda_value,
        # Reference gradients (averaged over K batches).
        "norm_g_std_ref": norm_ref_std,
        "norm_g_vr_ref": norm_ref_vr,
        "gradient_magnitude_ratio_vr_over_std": gradient_magnitude_ratio,
        "cos_g_std_ref_g_vr_ref": cos_refref,
        # Raw variance — included for completeness, NOT the load-bearing metric.
        # Var has units of ‖g‖² and is dominated by the gradient magnitude.
        "var_g_std": var_std,
        "var_g_vr": var_vr,
        "var_ratio_vr_over_std_raw": var_ratio,
        # Coefficient of variation: var / ‖mean‖². Units-free. This is what
        # tells you whether VR is noisier per unit signal.
        "cov_sq_g_std": cov_std,
        "cov_sq_g_vr": cov_vr,
        "cov_sq_ratio_vr_over_std": cov_ratio,
        "mean_g_std_norm": float(arr_n_std.mean()),
        "mean_g_vr_norm": float(arr_n_vr.mean()),
        # Per-batch direction alignment with the long-horizon reference.
        # The cleanest "is VR good for the optimizer" answer.
        "mean_cos_g_std_g_std_ref": cos_std_ref,
        "mean_cos_g_vr_g_vr_ref": cos_vr_ref,
        "mean_cos_g_vr_g_std_ref": cos_vr_std_ref,
        "direction_lift_vr_minus_std": direction_lift,
    }

    csv_path = run_dir / "per_batch.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=summary,
        artifacts=["per_batch.csv", "summary.json"],
        label=args.label,
        device=device,
    )

    log.info("─" * 70)
    log.info(f"λ used:                              {lambda_value:+.4f}")
    log.info(
        f"‖g_std_ref‖ / ‖g_vr_ref‖:            {norm_ref_std:.3e} / {norm_ref_vr:.3e}"
    )
    log.info(
        f"magnitude_ratio (VR / std):          {gradient_magnitude_ratio:.4f}  "
        f"{'← VR collapsed' if gradient_magnitude_ratio < 0.5 else ''}"
    )
    log.info(f"cos(g_std_ref, g_vr_ref):            {cos_refref:+.4f}")
    log.info("─ raw variance (magnitude-dependent — read with caution) ─")
    log.info(f"  Var[g_vr] / Var[g_std]:            {var_ratio:.4f}")
    log.info("─ coefficient of variation² (var / ‖mean‖²) — units-free ─")
    log.info(f"  CoV²[g_std]:                       {cov_std:.4f}")
    log.info(f"  CoV²[g_vr]:                        {cov_vr:.4f}")
    log.info(
        f"  CoV² ratio (VR / std):             {cov_ratio:.4f}  "
        f"{'← VR is noisier per unit signal' if cov_ratio > 1.5 else ''}"
    )
    log.info("─ direction alignment to long-horizon reference ─")
    log.info(f"  mean_k cos(g_std_k, g_std_ref):    {cos_std_ref:+.4f}")
    log.info(f"  mean_k cos(g_vr_k,  g_vr_ref):     {cos_vr_ref:+.4f}")
    log.info(
        f"  direction_lift (VR − std):         {direction_lift:+.4f}  "
        f"{'← VR HURTS the optimizer' if direction_lift < -0.05 else ('← VR helps' if direction_lift > 0.05 else '')}"
    )
    log.info("─" * 70)


if __name__ == "__main__":
    main()
