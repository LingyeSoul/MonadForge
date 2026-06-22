"""Diagnostic: PiD whole-image vs tiled decode tone gap.

Decodes the same Anima latent three ways and measures the color/tone offset:
  ref   = native Qwen VAE       (the calib reference target)
  whole = PiD 4-step, no tiling (canonical: matches upstream + 2k->4k training)
  tiled = PiD 4-step, tile=64   (what the shipped calib was fit on)

The hypothesis (GroupNorm in LQProjection2D normalizes over spatial extent, so a
tile crop normalizes differently than the whole latent) predicts a *global* tone
offset between `whole` and `tiled`. This script quantifies it (per-channel RGB +
Oklab L/a/b means), so we know how big the mis-calibration is before changing code.

Run (from anima_lora/):
    uv run python bench/pid/tile_vs_whole_probe.py --num_images 4 --crop 96
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from anima_lora import default_checkpoints  # noqa: E402
from library.io.cache import load_cached_latents  # noqa: E402
from library.models.qwen_vae import load_vae  # noqa: E402

# Reuse the calib script's helpers (pid_core loader, oklab) so the decode path is
# bit-identical to how the shipped calib was produced.
_CALIB = REPO_ROOT / "scripts/calibration/fit_color_calib.py"
_spec = importlib.util.spec_from_file_location("_fit_color_calib", _CALIB)
calib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(calib)


def center_crop(lat: torch.Tensor, crop: int) -> torch.Tensor:
    if crop <= 0:
        return lat
    _, h, w = lat.shape
    ch, cw = min(crop, h), min(crop, w)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return lat[:, y0 : y0 + ch, x0 : x0 + cw]


def chan_means(img01: torch.Tensor) -> np.ndarray:
    return img01[0].mean(dim=(-2, -1)).double().cpu().numpy()


def oklab_means(img01: torch.Tensor) -> np.ndarray:
    ok = calib.rgb01_to_oklab(img01[0].permute(1, 2, 0)).reshape(-1, 3).double()
    return ok.mean(0).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_images", type=int, default=4)
    ap.add_argument(
        "--crop",
        type=int,
        default=96,
        help="center latent crop (grid px); must exceed tile to exercise tiling",
    )
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tile_latent", type=int, default=64)
    ap.add_argument("--tile_overlap", type=int, default=16)
    ap.add_argument(
        "--compile",
        action="store_true",
        help="off by default (each whole size is a fresh graph)",
    )
    ap.add_argument("--latent_glob", default="post_image_dataset/lora/**/*_anima.npz")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    pid_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    all_npz = sorted(REPO_ROOT.glob(args.latent_glob))
    # only latents big enough to actually tile after the crop
    pool = [
        p
        for p in all_npz
        if min(load_cached_latents(str(p))[0].shape[-2:]) >= args.crop
    ]
    if not pool:
        raise SystemExit(f"no latents with min grid >= crop {args.crop}")
    idx = (
        np.linspace(0, len(pool) - 1, min(args.num_images, len(pool)))
        .round()
        .astype(int)
    )
    npz_paths = [pool[i] for i in idx]
    print(
        f"[data] {len(all_npz)} latents, {len(pool)} >= crop {args.crop}; probing {len(npz_paths)}"
    )

    vae = load_vae(
        str(default_checkpoints().vae), device=dev, dtype=torch.float32, eval=True
    )
    vae.requires_grad_(False)

    net = calib.load_pid_core(REPO_ROOT.parent / "comfy/custom_nodes/comfyui-anima-pid")
    pid_core = net
    net = pid_core.build_pid_net(device=device, dtype=pid_dtype)
    missing, unexpected = pid_core.load_pid_weights(net, str(calib.DEFAULT_PID_CKPT))
    _, suspect, unexp = pid_core.categorize_load_keys(missing, unexpected)
    if suspect or unexp:
        raise SystemExit(f"ckpt/arch mismatch: {suspect[:3]} {unexp[:3]}")
    null_cap = pid_core.load_null_caption_embs(
        str(calib.DEFAULT_NULL_CAP), dev, pid_dtype
    )

    rows = []  # (name, dRGB whole-tiled, dOklab whole-tiled, whole-ref, tiled-ref oklab)
    for i, p in enumerate(npz_paths):
        lat = center_crop(load_cached_latents(str(p))[0], args.crop).to(dev)
        lat5 = lat.unsqueeze(0)
        lh, lw = lat.shape[-2:]
        common = dict(
            steps=args.steps,
            sigma=0.0,
            seed=args.seed,
            dtype=pid_dtype,
            compile=args.compile,
            caption_embs=null_cap,
        )
        with torch.no_grad():
            ref = ((vae.decode_to_pixels(lat5.float()) + 1) * 0.5).clamp(0, 1)
            whole = pid_core.pid_decode_latent(net, lat5.to(pid_dtype), **common)
            torch.cuda.empty_cache() if device == "cuda" else None
            tiled_fix = pid_core.pid_decode_latent_tiled(
                net,
                lat5.to(pid_dtype),
                tile=args.tile_latent,
                overlap=args.tile_overlap,
                global_lq=True,
                **common,
            )
            tiled_old = pid_core.pid_decode_latent_tiled(
                net,
                lat5.to(pid_dtype),
                tile=args.tile_latent,
                overlap=args.tile_overlap,
                global_lq=False,
                **common,
            )
        to01 = lambda px: F.interpolate(
            ((px.float() + 1) * 0.5).clamp(0, 1), size=ref.shape[-2:], mode="area"
        )
        whole01, fix01, old01 = to01(whole), to01(tiled_fix), to01(tiled_old)

        ok_w, ok_fix, ok_old = (
            oklab_means(whole01),
            oklab_means(fix01),
            oklab_means(old01),
        )
        name = f"{p.parent.name}/{p.stem[:18]}"
        # gaps vs the whole-image (canonical) decode
        rows.append((name, ok_fix - ok_w, ok_old - ok_w))
        print(
            f"[{i + 1}/{len(npz_paths)}] {name}  grid {lh}x{lw} -> whole {lh * 32}x{lw * 32}px"
        )
        print(
            f"    whole-tiled gap Oklab(L,a,b)  FIX(global_lq)={np.round(ok_fix - ok_w, 4)}  "
            f"OLD(per-tile)={np.round(ok_old - ok_w, 4)}"
        )
        del lat, lat5, ref, whole, tiled_fix, tiled_old, whole01, fix01, old01
        torch.cuda.empty_cache() if device == "cuda" else None

    dfix = np.array([r[1] for r in rows])
    dold = np.array([r[2] for r in rows])
    print("\n" + "=" * 60)
    print("SUMMARY: tiled-vs-WHOLE tone gap, Oklab(L,a,b), mean +/- std (lower=better)")
    print(
        f"  FIX (global_lq=True) : {dfix.mean(0).round(4)} +/- {dfix.std(0).round(4)}"
    )
    print(
        f"  OLD (per-tile  )     : {dold.mean(0).round(4)} +/- {dold.std(0).round(4)}"
    )
    print(
        f"  L-channel |gap|: fix={abs(dfix[:, 0]).mean():.4f}  old={abs(dold[:, 0]).mean():.4f}"
    )


if __name__ == "__main__":
    main()
