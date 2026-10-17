#!/usr/bin/env python
"""Phase-0 ResShift SR sanity run (domain-gap gate).

Runs the *released* ResShift x4 (v3, 4-step) on the frozen synthetic-LR eval set,
then scores it against (a) the HR ground truth [full-reference: PSNR/SSIM/LPIPS]
and (b) a plain bicubic x4 baseline [no-reference: CLIPIQA/MUSIQ via pyiqa], and
writes side-by-side montages (bicubic | ResShift | HR) for blind eyeballing.

Runs in the root venv (modern torch, no xformers). The ResShift inference is
invoked as a subprocess with cwd=ResShift so its cwd-relative config/weight
download logic just works.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[4]
SR_INFER = Path(__file__).resolve().parent / "sr_infer.py"


def load_rgb(p: Path) -> np.ndarray:
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0


def to_hr_size(arr_img: Image.Image, size) -> Image.Image:
    return arr_img.resize(size, Image.BICUBIC)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO / "sr" / "data"))
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--version", default="v3")
    ap.add_argument("--chop_size", type=int, default=256)
    ap.add_argument("--ckpt", default=None,
                    help="explicit checkpoint for a local --version (x2/x4ft/x4s4) — e.g. score "
                         "a make-sr-train checkpoint against the same frozen set the released "
                         "teachers are scored on. Ignored for the released v1/v2/v3.")
    ap.add_argument("--skip_infer", action="store_true",
                    help="reuse existing results/ (just rescore + montage)")
    args = ap.parse_args()

    data = Path(args.data)
    lr_dir = data / "lr_eval"
    hr_dir = data / "hr_eval"
    res_dir = data / "results"
    mont_dir = data / "montage"
    res_dir.mkdir(parents=True, exist_ok=True)
    mont_dir.mkdir(parents=True, exist_ok=True)

    # 1) ResShift inference (vendored, basicsr-free) --------------------------
    if not args.skip_infer:
        cmd = [
            sys.executable, str(SR_INFER),
            "-i", str(lr_dir), "-o", str(res_dir),
            "--version", args.version, "--chop_size", str(args.chop_size),
        ]
        if args.ckpt:
            cmd += ["--ckpt", args.ckpt]
        print("RUN:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    # 2) metrics ---------------------------------------------------------------
    try:
        from skimage.metrics import peak_signal_noise_ratio as psnr_fn
        from skimage.metrics import structural_similarity as ssim_fn
    except Exception as e:  # noqa: BLE001
        print(f"[warn] skimage unavailable ({e}); skipping PSNR/SSIM")
        psnr_fn = ssim_fn = None

    lpips_fn = None
    try:
        import torch
        import lpips
        _lp = lpips.LPIPS(net="alex").eval()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _lp = _lp.to(dev)

        def lpips_fn(a, b):  # a,b in [0,1] HWC
            ta = torch.from_numpy(a).permute(2, 0, 1)[None] * 2 - 1
            tb = torch.from_numpy(b).permute(2, 0, 1)[None] * 2 - 1
            with torch.no_grad():
                return float(_lp(ta.to(dev), tb.to(dev)).item())
    except Exception as e:  # noqa: BLE001
        print(f"[warn] lpips unavailable ({e}); skipping LPIPS")

    iqa = {}
    try:
        import torch
        import pyiqa
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        for name in ("clipiqa", "musiq"):
            try:
                iqa[name] = pyiqa.create_metric(name, device=dev)
            except Exception as e:  # noqa: BLE001
                print(f"[warn] pyiqa {name} unavailable ({e})")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] pyiqa unavailable ({e}); skipping no-ref IQA")

    def noref(path: Path):
        import torch
        out = {}
        for name, m in iqa.items():
            try:
                t = torch.from_numpy(load_rgb(path)).permute(2, 0, 1)[None]
                out[name] = float(m(t).item())
            except Exception as e:  # noqa: BLE001
                out[name] = None
                print(f"  [warn] {name} on {path.name}: {e}")
        return out

    rows = []
    for lr_path in sorted(lr_dir.glob("*.png")):
        stem = lr_path.stem
        hr_path = hr_dir / f"{stem}.png"
        sr_path = res_dir / f"{stem}.png"
        if not sr_path.exists():
            # ResShift may suffix; try to find it
            cands = list(res_dir.glob(f"{stem}*.png"))
            if cands:
                sr_path = cands[0]
            else:
                print(f"  [warn] no SR output for {stem}")
                continue
        hr_img = Image.open(hr_path).convert("RGB")
        hr = load_rgb(hr_path)
        sr = load_rgb(sr_path)
        # bicubic baseline at HR size
        bic_img = Image.open(lr_path).convert("RGB").resize(hr_img.size, Image.BICUBIC)
        bic = np.asarray(bic_img, dtype=np.float32) / 255.0
        # align SR to HR size if off by rounding
        if sr.shape[:2] != hr.shape[:2]:
            sr_img = Image.open(sr_path).convert("RGB").resize(hr_img.size, Image.BICUBIC)
            sr = np.asarray(sr_img, dtype=np.float32) / 255.0

        row = {"stem": stem}
        if psnr_fn:
            row["psnr_sr"] = round(psnr_fn(hr, sr, data_range=1.0), 3)
            row["psnr_bic"] = round(psnr_fn(hr, bic, data_range=1.0), 3)
            row["ssim_sr"] = round(ssim_fn(hr, sr, data_range=1.0, channel_axis=2), 4)
            row["ssim_bic"] = round(ssim_fn(hr, bic, data_range=1.0, channel_axis=2), 4)
        if lpips_fn:
            row["lpips_sr"] = round(lpips_fn(hr, sr), 4)
            row["lpips_bic"] = round(lpips_fn(hr, bic), 4)
        if iqa:
            for k, v in noref(sr_path).items():
                row[f"{k}_sr"] = None if v is None else round(v, 4)
            bic_tmp = mont_dir / f"_bic_{stem}.png"
            bic_img.save(bic_tmp)
            for k, v in noref(bic_tmp).items():
                row[f"{k}_bic"] = None if v is None else round(v, 4)
            bic_tmp.unlink(missing_ok=True)
        rows.append(row)

        # montage: bicubic | ResShift | HR
        h = hr_img.size[1]
        panel = Image.new("RGB", (hr_img.size[0] * 3 + 20, h), "white")
        panel.paste(bic_img, (0, 0))
        panel.paste(Image.open(sr_path).convert("RGB").resize(hr_img.size), (hr_img.size[0] + 10, 0))
        panel.paste(hr_img, (hr_img.size[0] * 2 + 20, 0))
        panel.save(mont_dir / f"{stem}.png")
        print(f"  scored {stem}: {row}")

    # 3) summary ---------------------------------------------------------------
    def mean(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    keys = [k for k in rows[0] if k != "stem"] if rows else []
    summary = {"n": len(rows), "version": args.version, "scale": args.scale,
               "means": {k: mean(k) for k in keys}}
    out_json = data / "phase0_summary.json"
    out_json.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, default=float))
    print("\n=== PHASE 0 SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nmontages: {mont_dir}\nsummary : {out_json}")


if __name__ == "__main__":
    main()
