"""Bench: 2D-folded Qwen-Image VAE vs the stock 3D causal-Conv3d VAE.

Mirrors kohya-ss sd-scripts' ``--qwen_image_vae_2d``. For single images
(T=1) the causal temporal padding is zero-pad, so only the *last* temporal
tap of every ``QwenImageCausalConv3d`` sees real data. The exact 2D
equivalent is therefore ``weight[:, :, -1, :, :]`` (bias unchanged) run as a
plain ``nn.Conv2d`` on the squeezed (B,C,H,W) frame.

This bench:
  1. proves equivalence (fp32 algebra is exact; bf16 differs only by
     accumulation order) for both encode and decode,
  2. times encode + decode for 3D vs 2D,
  3. records peak CUDA memory for each.

Usage::
    uv run python bench/qwen_vae_2d/run_bench.py --res 1024 --batch 10
    uv run python bench/qwen_vae_2d/run_bench.py --res 1024 --image <path>
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
from library.models.qwen_vae import load_vae  # noqa: E402


def fold_to_2d(vae) -> int:
    """Fold the VAE's causal Conv3d stack to 2D (shipped in the library)."""
    return vae.convert_to_2d()


def _load_image(path: str, res: int, device, dtype) -> torch.Tensor:
    from PIL import Image
    import numpy as np

    img = Image.open(path).convert("RGB").resize((res, res), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0  # [-1, 1]
    t = torch.from_numpy(arr).permute(2, 0, 1)  # (3,H,W)
    return t.unsqueeze(0).to(device=device, dtype=dtype)


@torch.no_grad()
def _timed(fn, iters: int, device) -> tuple[float, int]:
    """Return (median ms per call, peak bytes)."""
    # warmup
    for _ in range(2):
        fn()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    times = []
    for _ in range(iters):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize(device)
        times.append((time.perf_counter() - t0) * 1e3)
    peak = torch.cuda.max_memory_allocated(device)
    times.sort()
    return times[len(times) // 2], peak


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=4, help="production default is 4")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--image", type=str, default=None, help="optional real image")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument(
        "--chunk_size",
        type=int,
        default=64,
        help="spatial_chunk_size for the 3D baseline (production default 64; "
        "0 disables). The 2D fold runs unchunked.",
    )
    ap.add_argument("--label", type=str, default=None)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "needs CUDA"
    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    chunk = args.chunk_size or None

    import anima_lora

    vae_path = str(anima_lora.ROOT / anima_lora.default_checkpoints().vae)

    # ---- build pixels (real image broadcast to batch, or random) ----
    if args.image:
        px1 = _load_image(args.image, args.res, device, dtype)
        pixels = px1.repeat(args.batch, 1, 1, 1).contiguous()
        src = args.image
    else:
        gen = torch.Generator(device=device).manual_seed(0)
        pixels = (
            torch.rand(
                args.batch,
                3,
                args.res,
                args.res,
                generator=gen,
                device=device,
                dtype=torch.float32,
            )
            * 2.0
            - 1.0
        ).to(dtype)
        src = "random[-1,1]"

    # ============ EQUIVALENCE (fp32, exact algebra; res-independent) ============
    # Run at a small res so the un-chunked fp32 3D path fits comfortably.
    res_eq = min(args.res, 256)
    if args.image:
        px_eq = _load_image(args.image, res_eq, device, torch.float32)
    else:
        geq = torch.Generator(device=device).manual_seed(1)
        px_eq = torch.rand(1, 3, res_eq, res_eq, generator=geq, device=device) * 2 - 1
    vae32 = load_vae(
        vae_path, device=device, dtype=torch.float32, eval=True, disable_cache=True
    )
    lat3d = vae32.encode_pixels_to_latents(px_eq)
    img3d = vae32.decode_to_pixels(lat3d)

    n_folded = fold_to_2d(vae32)
    lat2d = vae32.encode_pixels_to_latents(px_eq)
    img2d = vae32.decode_to_pixels(lat2d)

    enc_max = (lat3d - lat2d).abs().max().item()
    enc_rel = enc_max / (lat3d.abs().max().item() + 1e-12)
    dec_max = (img3d - img2d).abs().max().item()
    del vae32, px_eq, lat3d, img3d, lat2d, img2d
    torch.cuda.empty_cache()

    # ============ TIMING + VRAM (production dtype) ============
    def run_block(fold: bool):
        # 3D baseline gets the production chunk size; the 2D fold runs unchunked.
        vae = load_vae(
            vae_path,
            device=device,
            dtype=dtype,
            eval=True,
            spatial_chunk_size=(None if fold else chunk),
            disable_cache=True,
        )
        if fold:
            fold_to_2d(vae)
        try:
            lat = vae.encode_pixels_to_latents(pixels)
            enc_ms, enc_peak = _timed(
                lambda: vae.encode_pixels_to_latents(pixels), args.iters, device
            )
            dec_ms, dec_peak = _timed(
                lambda: vae.decode_to_pixels(lat), args.iters, device
            )
            out = {
                "encode_ms": enc_ms,
                "decode_ms": dec_ms,
                "encode_peak_mb": enc_peak / 1e6,
                "decode_peak_mb": dec_peak / 1e6,
                "oom": False,
            }
        except torch.OutOfMemoryError:
            out = {
                "encode_ms": None,
                "decode_ms": None,
                "encode_peak_mb": None,
                "decode_peak_mb": None,
                "oom": True,
            }
        # vae/lat are run_block locals — freed on return; just reclaim the cache.
        torch.cuda.empty_cache()
        return out

    m3d = run_block(fold=False)
    m2d = run_block(fold=True)

    # bf16/production-dtype numerical diff at small res (accumulation-order only)
    if args.image:
        px_p = _load_image(args.image, res_eq, device, dtype)
    else:
        gp = torch.Generator(device=device).manual_seed(1)
        px_p = (
            torch.rand(1, 3, res_eq, res_eq, generator=gp, device=device) * 2 - 1
        ).to(dtype)
    vaeP = load_vae(vae_path, device=device, dtype=dtype, eval=True, disable_cache=True)
    latP_3d = vaeP.encode_pixels_to_latents(px_p)
    fold_to_2d(vaeP)
    latP_2d = vaeP.encode_pixels_to_latents(px_p)
    enc_max_prod = (latP_3d.float() - latP_2d.float()).abs().max().item()
    del vaeP, latP_3d, latP_2d, px_p
    torch.cuda.empty_cache()

    def _ratio(a, b):
        return (a / b) if (a is not None and b not in (None, 0)) else None

    speedup = {
        "encode_x": _ratio(m3d["encode_ms"], m2d["encode_ms"]),
        "decode_x": _ratio(m3d["decode_ms"], m2d["decode_ms"]),
        "encode_peak_ratio_2d_over_3d": _ratio(
            m2d["encode_peak_mb"], m3d["encode_peak_mb"]
        ),
        "decode_peak_ratio_2d_over_3d": _ratio(
            m2d["decode_peak_mb"], m3d["decode_peak_mb"]
        ),
    }
    metrics = {
        "config": {
            "res": args.res,
            "batch": args.batch,
            "iters": args.iters,
            "dtype": args.dtype,
            "chunk_size_3d": chunk,
            "res_equivalence": res_eq,
            "source": src,
            "n_conv3d_folded": n_folded,
        },
        "equivalence_fp32": {
            "encode_max_abs_diff": enc_max,
            "encode_rel_diff": enc_rel,
            "decode_max_abs_diff": dec_max,
        },
        "equivalence_prod_dtype": {
            "encode_max_abs_diff": enc_max_prod,
        },
        "vae_3d": m3d,
        "vae_2d": m2d,
        "speedup": speedup,
    }

    run_dir = make_run_dir("qwen_vae_2d", label=args.label)
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label,
        device=device,
    )

    # ---- console summary ----
    def f(v, unit=""):
        return "  OOM  " if v is None else f"{v:7.1f}{unit}"

    def fx(v):
        return "n/a" if v is None else f"{v:.2f}x"

    print(
        f"\n=== Qwen VAE 2D-fold bench ({args.res}px x{args.batch}, {args.dtype}, "
        f"3D chunk={chunk}) ==="
    )
    print(f"folded {n_folded} Conv3d -> Conv2d   source={src}")
    print(f"\n[equivalence] (fp32 @ {res_eq}px, exact algebra)")
    print(f"  fp32  encode max|Δ| = {enc_max:.3e}  (rel {enc_rel:.2e})")
    print(f"  fp32  decode max|Δ| = {dec_max:.3e}")
    print(f"  {args.dtype}  encode max|Δ| = {enc_max_prod:.3e}  (accum-order only)")
    print("\n[encode]")
    print(f"  3D: {f(m3d['encode_ms'], ' ms')}  peak {f(m3d['encode_peak_mb'], ' MB')}")
    print(f"  2D: {f(m2d['encode_ms'], ' ms')}  peak {f(m2d['encode_peak_mb'], ' MB')}")
    print(
        f"  -> {fx(speedup['encode_x'])} faster, "
        f"{fx(speedup['encode_peak_ratio_2d_over_3d'])} peak mem"
    )
    print("\n[decode]")
    print(f"  3D: {f(m3d['decode_ms'], ' ms')}  peak {f(m3d['decode_peak_mb'], ' MB')}")
    print(f"  2D: {f(m2d['decode_ms'], ' ms')}  peak {f(m2d['decode_peak_mb'], ' MB')}")
    print(
        f"  -> {fx(speedup['decode_x'])} faster, "
        f"{fx(speedup['decode_peak_ratio_2d_over_3d'])} peak mem"
    )
    print(f"\nresult.json -> {run_dir}")


if __name__ == "__main__":
    main()
