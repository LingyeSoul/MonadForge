"""flash vs flex (vs torch SDPA) wall-clock on the static-padding self-attn path.

Companion to `probe_pad_leak.py`. That probe falsified the attention-sink
assumption (unmasked flash padding leaks into real-token velocity, up to 6.5%%
rel-L2 at large gaps). The fix is to migrate the production self-attention from
`attn_mode="flash"` (static 4096, no padding mask) to `attn_mode="flex"` (same
static 4096 with a `BlockMask` that excludes padded positions). flex is correct
— this script answers *what it costs*.

flex_attention is slow eager and fast compiled; the production static-token path
runs under `compile_core` (one CUDAGraph for the shape-invariant block stack).
So a fair comparison must include the compiled path — eager flex would unfairly
penalize the migration target. We time every (mode × compiled × workload) cell.

Matrix
------
  modes:    flash (production, leaks) · flex (migration target) · torch (SDPA ref)
  compiled: eager · compiled — matched to how each workload actually runs:
              fwd     → compile_core  (one CUDAGraph over the static block stack,
                                       grad-ckpt off — the inference path)
              fwd+bwd → compile_blocks (per-block _forward, grad-ckpt compatible —
                                       the training path)
  workload: fwd      — inference path, torch.no_grad, eval(), grad-ckpt off.
            fwd+bwd  — training path; gradient checkpointing ON (as real
                       constrained-GPU training runs, and so activations fit).
                       Params frozen so grad flows only to the input latent —
                       the attention *backward* (what differs flash↔flex) still
                       runs at every block, without materializing weight grads.

All cells run at static_token_count=4096 — flash does full 4096-token self-attn
(the wasted work), flex masks down to the bucket's real token count (and may skip
fully-padded query/key blocks via BlockMask sparsity, so flex can *win* at large
gaps). crossattn_seqlens is left None so cross-attention work is identical across
modes and only the self-attn padding handling differs.

Metric is median per-iter latency (CUDA events) + peak allocated VRAM.

Run (repo root):
    python -m bench.static_padding.bench_attn_speed
    python -m bench.static_padding.bench_attn_speed --modes flash flex --eager-only
    python -m bench.static_padding.bench_attn_speed --iters 50 --warmup 10
"""

from __future__ import annotations

import argparse
import csv
import logging
import statistics
import sys
from pathlib import Path

import torch

# Defensive: allow `python bench/static_padding/bench_attn_speed.py` as well as -m.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bench._common import make_run_dir, write_result  # noqa: E402
from library.anima.weights import load_anima_model  # noqa: E402

log = logging.getLogger("bench.static_padding.speed")
logging.basicConfig(level=logging.INFO, format="%(message)s")

DEFAULT_DIT = "models/diffusion_models/anima-base-v1.0.safetensors"
STATIC_TOKENS = 4096
TEXT_LEN = 226  # representative crossattn length; held fixed across all runs

# (label, H_lat, W_lat) — token count = (H_lat/2)*(W_lat/2). VAE-latent dims.
# Picked to span the gap spectrum: control (gap 0), a real worst bucket (gap 64),
# and a synthetic large gap where flex BlockMask sparsity should pull ahead.
CONFIGS: list[tuple[str, int, int]] = [
    ("bucket_4096_gap0", 128, 128),     # 1024x1024 — control, no padding
    ("bucket_4032_gap64", 112, 144),    # 896x1152  — worst real bucket
    ("synthetic_2048_gap2048", 64, 128),  # half-padded — sparsity regime
]


def _make_inputs(H, W, sigma, device, dtype, seed):
    """Identical, seeded inputs reused across every variant for this config."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    lat = torch.randn(1, 16, 1, H, W, generator=g, dtype=torch.float32)
    txt = torch.randn(1, TEXT_LEN, 1024, generator=g, dtype=torch.float32)
    return {
        "x_B_C_T_H_W": lat.to(device, dtype),
        "timesteps_B_T": torch.tensor([sigma], device=device, dtype=dtype),
        "crossattn_emb": txt.to(device, dtype),
        "padding_mask": torch.zeros(1, 1, H, W, device=device, dtype=dtype),
    }


def _load(device, dit_path, attn_mode, dtype):
    model = load_anima_model(
        device, dit_path, attn_mode=attn_mode,
        loading_device=device, dit_weight_dtype=dtype,
    )
    model.to(device)
    model.reset_mod_guidance()
    model.eval()
    # Freeze weights: the backward path then flows only to the input latent,
    # exercising attention backward without materializing weight grads.
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _configure(model, *, grad: bool):
    """Set train/eval + gradient checkpointing to match the workload.

    grad-ckpt only fires under ``model.training`` (see models.py forward), so
    fwd+bwd needs train() for the recompute path that keeps activations in VRAM.
    """
    if grad:
        model.train()
        model.enable_gradient_checkpointing()
    else:
        model.eval()
        model.disable_gradient_checkpointing()


def _run_once(model, inp, dtype, *, grad: bool):
    """One forward (grad=False) or forward+backward (grad=True)."""
    if grad:
        x = inp["x_B_C_T_H_W"].detach().requires_grad_(True)
        with torch.autocast("cuda", dtype=dtype):
            out = model.forward_mini_train_dit(
                x, inp["timesteps_B_T"], inp["crossattn_emb"],
                padding_mask=inp["padding_mask"], skip_pooled_text_proj=True,
            )
        loss = out.float().pow(2).mean()
        loss.backward()
        return
    with torch.no_grad():
        with torch.autocast("cuda", dtype=dtype):
            model.forward_mini_train_dit(
                inp["x_B_C_T_H_W"], inp["timesteps_B_T"], inp["crossattn_emb"],
                padding_mask=inp["padding_mask"], skip_pooled_text_proj=True,
            )


def _time_cell(model, inp, dtype, *, grad, warmup, iters):
    """Median per-iter latency (ms) + peak VRAM (MB) over `iters`, after warmup."""
    for _ in range(warmup):
        _run_once(model, inp, dtype, grad=grad)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    times_ms: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _run_once(model, inp, dtype, grad=grad)
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return statistics.median(times_ms), min(times_ms), peak_mb


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dit_path", default=DEFAULT_DIT)
    ap.add_argument("--modes", nargs="+", default=["flash", "flex", "torch"])
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--eager-only", action="store_true",
                    help="skip compile_core cells (fast smoke run)")
    ap.add_argument("--compiled-only", action="store_true",
                    help="skip eager cells (production path only)")
    ap.add_argument("--compile_mode", default=None,
                    help="inductor mode for compile_core (e.g. reduce-overhead)")
    ap.add_argument("--label", default="attn-speed")
    args = ap.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16

    compiled_opts = []
    if not args.compiled_only:
        compiled_opts.append(False)
    if not args.eager_only:
        compiled_opts.append(True)
    workloads = [("fwd", False), ("fwd+bwd", True)]

    rows: list[dict] = []
    # Reuse one eager model across eager cells; build fresh per compiled cell so
    # compile state never bleeds between modes/workloads.
    eager_model = None

    for mode in args.modes:
        for compiled in compiled_opts:
            for wl_name, grad in workloads:
                if compiled:
                    model = _load(device, args.dit_path, mode, dtype)
                    model.set_static_token_count(STATIC_TOKENS)
                    _configure(model, grad=grad)
                    try:
                        # fwd → compile_core (CUDAGraph, grad-ckpt off);
                        # fwd+bwd → compile_blocks (per-block, grad-ckpt on).
                        if grad:
                            model.compile_blocks(mode=args.compile_mode)
                        else:
                            model.compile_core(mode=args.compile_mode)
                    except Exception as e:
                        log.warning("compile failed (%s/%s): %s", mode, wl_name, e)
                        del model
                        torch.cuda.empty_cache()
                        continue
                else:
                    if eager_model is None:
                        eager_model = _load(device, args.dit_path, mode, dtype)
                    model = eager_model
                    model.attn_mode = mode
                    model.set_static_token_count(STATIC_TOKENS)
                    _configure(model, grad=grad)

                for label, H, W in CONFIGS:
                    seq_len = (H // 2) * (W // 2)
                    gap = STATIC_TOKENS - seq_len
                    inp = _make_inputs(H, W, args.sigma, device, dtype, args.seed)
                    try:
                        med, best, peak = _time_cell(
                            model, inp, dtype, grad=grad,
                            warmup=args.warmup, iters=args.iters,
                        )
                    except Exception as e:  # OOM / flex-eager pathologies
                        log.warning("cell failed %s/%s/%s/%s: %s",
                                    mode, "compiled" if compiled else "eager",
                                    wl_name, label, e)
                        torch.cuda.empty_cache()
                        continue
                    rows.append({
                        "mode": mode, "compiled": compiled, "workload": wl_name,
                        "config": label, "seq_len": seq_len, "gap": gap,
                        "median_ms": round(med, 4), "best_ms": round(best, 4),
                        "peak_mem_mb": round(peak, 1),
                    })
                    log.info(
                        "%-6s %-8s %-8s %-22s gap=%-4d  med=%8.3f ms  best=%8.3f ms  peak=%7.1f MB",
                        mode, "compiled" if compiled else "eager", wl_name,
                        label, gap, med, best, peak,
                    )

                if compiled:
                    del model
                    torch.cuda.empty_cache()

    # Summary: flex/flash latency ratio per (compiled, workload, config). <1 means
    # flex is faster (sparsity win); >1 is the cost of migrating off leaky flash.
    def _lookup(mode, compiled, wl, cfg):
        for r in rows:
            if (r["mode"] == mode and r["compiled"] == compiled
                    and r["workload"] == wl and r["config"] == cfg):
                return r["median_ms"]
        return None

    ratios: dict[str, float] = {}
    worst_ratio = 0.0
    for compiled in compiled_opts:
        for wl_name, _ in workloads:
            for label, _, _ in CONFIGS:
                fl = _lookup("flash", compiled, wl_name, label)
                fx = _lookup("flex", compiled, wl_name, label)
                if fl and fx:
                    ratio = fx / fl
                    key = f"flex_over_flash::{'compiled' if compiled else 'eager'}::{wl_name}::{label}"
                    ratios[key] = round(ratio, 3)
                    worst_ratio = max(worst_ratio, ratio)

    log.info("\n=== flex/flash median-latency ratios (>1 = flex slower) ===")
    for k, v in ratios.items():
        log.info("  %-58s %.3f", k, v)
    log.info("worst flex/flash ratio: %.3f", worst_ratio)

    run_dir = make_run_dir("static_padding", label=args.label)
    csv_path = run_dir / "per_cell.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)
    write_result(
        run_dir, script=__file__, args=args, label=args.label,
        metrics={
            "flex_over_flash_ratios": ratios,
            "worst_flex_over_flash": worst_ratio,
            "n_cells": len(rows),
            "static_tokens": STATIC_TOKENS,
        },
        artifacts=["per_cell.csv"], device=device,
    )
    log.info("wrote %s", run_dir)


if __name__ == "__main__":
    main()
