#!/usr/bin/env python
"""Build a frozen Phase-0 eval set for the ResShift SR sidecar.

Samples N sharp art images from image_dataset/, makes an HR reference (long-edge
resized, even dims) and a synthetic LR (bicubic /scale). The LR feeds ResShift;
the HR is the ground truth for full-reference metrics. Deterministic sampling
(sorted stride across artists) so the eval set is stable across runs.

Run with the *anima* env (only needs PIL/numpy) — it just prepares data.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def laplacian_var(img: Image.Image) -> float:
    """Sharpness proxy — drop already-soft scans."""
    g = np.asarray(img.convert("L"), dtype=np.float32)
    # 3x3 laplacian via finite diff
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


def even(x: int) -> int:
    return x - (x % 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(REPO / "image_dataset"))
    ap.add_argument("--out", default=str(REPO / "sr" / "data"))
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument(
        "--hr_long_edge",
        type=int,
        default=2048,
        help="HR reference long edge (LR = this / scale). 2048 / x4 -> 512 LR.",
    )
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument(
        "--lap_floor",
        type=float,
        default=50.0,
        help="Min laplacian variance — skip soft scans.",
    )
    args = ap.parse_args()

    src = Path(args.src)
    hr_dir = Path(args.out) / "hr_eval"
    lr_dir = Path(args.out) / "lr_eval"
    hr_dir.mkdir(parents=True, exist_ok=True)
    lr_dir.mkdir(parents=True, exist_ok=True)
    # wipe any prior frozen set so a re-prep at a new target_res can't leave stale
    # mixed-resolution pairs behind (the eval scripts glob *.png blindly).
    for d in (hr_dir, lr_dir):
        for old in d.glob("*.png"):
            old.unlink()

    # rglob with -L semantics (Path.rglob follows symlinked dirs in 3.13+, but
    # image_dataset itself is a symlink to nested artist dirs — resolve it first)
    all_imgs = sorted(p for p in src.resolve().rglob("*") if p.suffix.lower() in EXTS)
    print(f"found {len(all_imgs)} candidate images under {src}")
    if not all_imgs:
        raise SystemExit("no images found")

    # stride sample for artist diversity, then filter on size+sharpness
    stride = max(1, len(all_imgs) // (args.n * 3))
    picked = []
    for p in all_imgs[::stride]:
        if len(picked) >= args.n:
            break
        try:
            im = Image.open(p).convert("RGB")
        except Exception as e:  # noqa: BLE001
            print(f"  skip (open) {p.name}: {e}")
            continue
        if max(im.size) < args.hr_long_edge:
            continue  # native long edge below target -> HR would be a fake upscale
        if laplacian_var(im) < args.lap_floor:
            print(f"  skip (soft) {p.name}")
            continue
        picked.append((p, im))

    print(f"picked {len(picked)} images (target {args.n})")
    for p, im in picked:
        w, h = im.size
        s = args.hr_long_edge / max(w, h)
        hw, hh = even(round(w * s)), even(round(h * s))
        hr = im.resize((hw, hh), Image.LANCZOS)
        lw, lh = even(hw // args.scale), even(hh // args.scale)
        lr = hr.resize((lw, lh), Image.BICUBIC)
        stem = f"{p.parent.name}__{p.stem}"
        hr.save(hr_dir / f"{stem}.png")
        lr.save(lr_dir / f"{stem}.png")
    print(f"wrote HR -> {hr_dir}\n      LR -> {lr_dir}")


if __name__ == "__main__":
    main()
