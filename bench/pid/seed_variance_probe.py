"""Is the tile-vs-whole tone gap systematic, or just SDE sampling variance?

Decodes the SAME latent multiple times changing only the seed, and measures the
Oklab tone spread. If the whole-image seed-to-seed spread is comparable to the
whole-vs-tiled gap (~0.003 L from tile_vs_whole_probe), the "tone differs at
tile=0" report is stochastic noise, not a systematic tiling artifact — and no
conditioning fix can close it.
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

_CALIB = REPO_ROOT / "scripts/calibration/fit_color_calib.py"
_spec = importlib.util.spec_from_file_location("_fit_color_calib", _CALIB)
calib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(calib)


def center_crop(lat, crop):
    if crop <= 0:
        return lat
    _, h, w = lat.shape
    ch, cw = min(crop, h), min(crop, w)
    return lat[
        :, (h - ch) // 2 : (h - ch) // 2 + ch, (w - cw) // 2 : (w - cw) // 2 + cw
    ]


def oklab_means(img01):
    ok = calib.rgb01_to_oklab(img01[0].permute(1, 2, 0)).reshape(-1, 3).double()
    return ok.mean(0).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", type=int, default=96)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--tile_latent", type=int, default=64)
    ap.add_argument("--tile_overlap", type=int, default=16)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pid_dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32

    pool = [
        p
        for p in sorted(REPO_ROOT.glob("post_image_dataset/lora/**/*_anima.npz"))
        if min(load_cached_latents(str(p))[0].shape[-2:]) >= args.crop
    ]
    p = pool[len(pool) // 2]

    vae = load_vae(
        str(default_checkpoints().vae), device=dev, dtype=torch.float32, eval=True
    )
    pid_core = calib.load_pid_core(
        REPO_ROOT.parent / "comfy/custom_nodes/comfyui-anima-pid"
    )
    net = pid_core.build_pid_net(device=str(dev), dtype=pid_dtype)
    pid_core.load_pid_weights(net, str(calib.DEFAULT_PID_CKPT))
    null_cap = pid_core.load_null_caption_embs(
        str(calib.DEFAULT_NULL_CAP), dev, pid_dtype
    )

    vae.requires_grad_(False)
    lat = center_crop(load_cached_latents(str(p))[0], args.crop).to(dev)
    lat5 = lat.unsqueeze(0).to(pid_dtype)
    with torch.no_grad():
        ref = ((vae.decode_to_pixels(lat.unsqueeze(0).float()) + 1) * 0.5).clamp(0, 1)
    to01 = lambda px: F.interpolate(
        ((px.float() + 1) * 0.5).clamp(0, 1), size=ref.shape[-2:], mode="area"
    )
    common = dict(
        steps=args.steps,
        sigma=0.0,
        dtype=pid_dtype,
        compile=False,
        caption_embs=null_cap,
    )

    print(
        f"latent {p.parent.name}/{p.stem}  crop {args.crop} -> {args.crop * 32}px; {args.seeds} seeds"
    )
    whole_ok, fix_ok = [], []
    for sd in range(args.seeds):
        with torch.no_grad():
            w = pid_core.pid_decode_latent(net, lat5, seed=sd, **common)
            f = pid_core.pid_decode_latent_tiled(
                net,
                lat5,
                seed=sd,
                tile=args.tile_latent,
                overlap=args.tile_overlap,
                global_lq=True,
                **common,
            )
        whole_ok.append(oklab_means(to01(w)))
        fix_ok.append(oklab_means(to01(f)))
        torch.cuda.empty_cache() if dev.type == "cuda" else None
    whole_ok, fix_ok = np.array(whole_ok), np.array(fix_ok)

    print("\nOklab(L,a,b) means per seed:")
    for sd in range(args.seeds):
        print(
            f"  seed {sd}: whole={whole_ok[sd].round(4)}  tiled_fix={fix_ok[sd].round(4)}"
        )
    print("\n" + "=" * 56)
    print(f"  whole  seed-to-seed std (NOISE FLOOR): {whole_ok.std(0).round(4)}")
    print(f"  tiled  seed-to-seed std             : {fix_ok.std(0).round(4)}")
    print(
        f"  mean |whole-tiled| per seed         : {np.abs(whole_ok - fix_ok).mean(0).round(4)}"
    )
    print(
        f"  whole-mean vs tiled-mean            : {(whole_ok.mean(0) - fix_ok.mean(0)).round(4)}"
    )
    print(
        "  -> if the gap ~ the noise floor, the tone difference is sampling variance."
    )


if __name__ == "__main__":
    main()
