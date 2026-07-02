"""Probe: bf16 vs fp32 VAE *encode* error on real training images.

Question: caching encodes in bf16 (2D-fold) then saves fp32. The fp32 save
recovers nothing -- the latent already carries bf16 encode error. Does
encoding in fp32 (`--no_half_vae`) give meaningfully better targets, or is
the bf16 error white dither below the training noise floor?

Reference  = fp32 encode (in fp32 the 2D fold is bit-exact, so this is the
             ground-truth latent for the image).
Production = bf16 2D-fold encode (the shipped caching path).

We report, per image and aggregated:
  * latent rel-error (RMS and max) bf16 vs fp32
  * decode PSNR: decode(bf16 latent) vs decode(fp32 latent), both decoded in
    fp32 to isolate the *encode* dtype effect.
  * structure probe: fraction of error energy surviving 8x avg-pool. High
    => low-freq / structured bias (worth fp32 encode). Low => white dither
    (fp32 encode is a no-op on data).

Usage::
    uv run python bench/qwen_vae_2d/encode_dtype_probe.py --n 12 --res 1024
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench._common import make_run_dir, write_result  # noqa: E402
from library.models.qwen_vae import load_vae  # noqa: E402


def _load_image(path: Path, res: int, device, dtype) -> torch.Tensor:
    from PIL import Image

    img = Image.open(path).convert("RGB")
    # keep native AR, longest edge -> res (mirrors free-fit-ish; res only sets scale)
    w, h = img.size
    s = res / max(w, h)
    img = img.resize((max(8, round(w * s)), max(8, round(h * s))), Image.BICUBIC)
    # snap to /16 so the VAE patch grid is clean
    W, H = (img.width // 16) * 16, (img.height // 16) * 16
    img = img.crop((0, 0, W, H))
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t.to(device=device, dtype=dtype)


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    # inputs in [-1,1]; report PSNR over the 2.0 dynamic range
    mse = (a.float() - b.float()).pow(2).mean().item()
    if mse <= 1e-20:
        return float("inf")
    return 10.0 * np.log10((2.0**2) / mse)


def _structure_frac(err: torch.Tensor, k: int = 8) -> float:
    """Fraction of error ENERGY that survives kxk average pooling.

    White noise -> pooling attenuates variance ~1/k^2 -> tiny fraction.
    Low-freq structured error -> survives -> large fraction.
    err: (1, C, H, W).
    """
    total = err.float().pow(2).mean().item() + 1e-20
    pooled = F.avg_pool2d(err.float(), kernel_size=k, stride=k)
    return pooled.pow(2).mean().item() / total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12, help="number of real images")
    ap.add_argument("--res", type=int, default=1024, help="longest-edge px")
    ap.add_argument("--root", type=str, default="post_image_dataset/resized")
    ap.add_argument(
        "--image",
        type=str,
        default=None,
        help="probe a single explicit image path instead of sampling --root",
    )
    ap.add_argument("--label", type=str, default=None)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "needs CUDA"
    device = torch.device("cuda:0")

    import anima_lora

    vae_path = str(anima_lora.ROOT / anima_lora.default_checkpoints().vae)

    if args.image:
        ip = Path(args.image)
        if not ip.is_absolute():
            ip = REPO_ROOT / ip
        assert ip.exists(), f"no such image {ip}"
        picks = [ip]
    else:
        # sample images deterministically (sorted, strided) for reproducibility
        root = REPO_ROOT / args.root
        paths = sorted(
            p
            for p in root.rglob("*")
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )
        assert paths, f"no images under {root}"
        stride = max(1, len(paths) // args.n)
        picks = paths[::stride][: args.n]

    # fp32 reference VAE (2D fold is bit-exact in fp32) and bf16 production VAE
    vae32 = load_vae(
        vae_path, device=device, dtype=torch.float32, eval=True, disable_cache=True
    )
    vae32.convert_to_2d()
    vae16 = load_vae(
        vae_path, device=device, dtype=torch.bfloat16, eval=True, disable_cache=True
    )
    vae16.convert_to_2d()

    rows = []
    with torch.no_grad():
        for p in picks:
            px32 = _load_image(p, args.res, device, torch.float32)
            px16 = px32.to(torch.bfloat16)

            lat32 = vae32.encode_pixels_to_latents(px32)  # reference
            lat16 = vae16.encode_pixels_to_latents(px16).float()  # production

            err = lat16 - lat32
            denom = lat32.float().pow(2).mean().sqrt().item() + 1e-12
            rms_rel = err.pow(2).mean().sqrt().item() / denom
            max_rel = err.abs().max().item() / (lat32.abs().max().item() + 1e-12)

            # decode both latents in fp32 -> isolates encode dtype
            rec32 = vae32.decode_to_pixels(lat32)
            rec16 = vae32.decode_to_pixels(lat16.to(torch.float32))
            # decode may return 5D (B,3,1,H,W) -- squeeze dim 2 if present
            if rec32.dim() == 5:
                rec32 = rec32.squeeze(2)
                rec16 = rec16.squeeze(2)
            psnr_decode = _psnr(rec16, rec32)
            # also: how good is the recon itself (sanity) vs source
            psnr_recon32 = _psnr(rec32, px32)

            sfrac = _structure_frac(err)

            rows.append(
                {
                    "image": str(p.relative_to(REPO_ROOT)),
                    "latent_hw": list(lat32.shape[-2:]),
                    "rms_rel_err": rms_rel,
                    "max_rel_err": max_rel,
                    "decode_psnr_bf16_vs_fp32_db": psnr_decode,
                    "recon_psnr_fp32_vs_src_db": psnr_recon32,
                    "err_structure_frac_pool8": sfrac,
                }
            )
            print(
                f"{p.name[:34]:34s} rms_rel={rms_rel:.2e} "
                f"decodePSNR={psnr_decode:6.2f}dB struct={sfrac:.3f} "
                f"(recon={psnr_recon32:5.1f}dB)"
            )

    def _agg(key):
        vals = [r[key] for r in rows if np.isfinite(r[key])]
        return {
            "mean": float(np.mean(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "median": float(np.median(vals)),
        }

    summary = {
        k: _agg(k)
        for k in (
            "rms_rel_err",
            "max_rel_err",
            "decode_psnr_bf16_vs_fp32_db",
            "recon_psnr_fp32_vs_src_db",
            "err_structure_frac_pool8",
        )
    }

    # white-noise baseline for the structure fraction: ~1/k^2 = 1/64 ~= 0.0156
    metrics = {
        "config": {"n": len(rows), "res": args.res, "root": args.root},
        "per_image": rows,
        "summary": summary,
        "white_noise_struct_frac_ref": 1.0 / 64.0,
    }

    run_dir = make_run_dir("qwen_vae_2d", label=args.label or "encode_dtype")
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label,
        device=device,
    )

    s = summary
    print("\n=== bf16 vs fp32 ENCODE on real images ===")
    print(f"images: {len(rows)} @ {args.res}px longest-edge")
    print(
        f"latent RMS rel-err : mean {s['rms_rel_err']['mean']:.2e} "
        f"(max {s['rms_rel_err']['max']:.2e})"
    )
    print(
        f"decode PSNR bf16vsfp32: mean {s['decode_psnr_bf16_vs_fp32_db']['mean']:.2f} dB "
        f"(min {s['decode_psnr_bf16_vs_fp32_db']['min']:.2f})"
    )
    print(
        f"err structure frac : mean {s['err_structure_frac_pool8']['mean']:.3f} "
        f"(white-noise ref {1 / 64:.3f}; >>ref => low-freq/structured)"
    )
    print(f"sanity recon PSNR  : mean {s['recon_psnr_fp32_vs_src_db']['mean']:.1f} dB")
    print(f"\nresult.json -> {run_dir}")


if __name__ == "__main__":
    main()
