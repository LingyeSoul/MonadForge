#!/usr/bin/env python3
"""Quantify + visualize the sigma-cutoff text-knockout sweep.

Capability complement to the CFG-delta probe. Each run generates the same seeded
prompts but knocks text out (continues unconditionally) below a cutoff sigma; the
baseline keeps text on throughout. If killing text below a cutoff barely moves the
image, late-sigma text is structurally inert (the model *can't* follow it there);
if it moves the image, there is real late-sigma steerability the base under-uses.

Layout expected (one subdir per condition, images named ``{timestamp}_{seed}.png``):

    <root>/baseline/   <root>/cut_0.95/   <root>/cut_0.80/   ...

Run:

    python bench/cross_attn_drive/knockout_compare.py output/tests/knockout

Drift vs baseline is reported as pixel RMSE (normalized 0..1) and, if scikit-image
is present, 1-SSIM. Both are *relative* signals across cutoffs, not absolute
quality. Writes drift_plot.png, contact_sheet.png, result.json under <root>.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _seed_of(p: Path) -> str:
    # filename: {timestamp}_{seed}_.png -> seed is the last all-digit token
    digits = [tok for tok in p.stem.split("_") if tok.isdigit()]
    return digits[-1] if digits else p.stem


def _index_dir(d: Path) -> dict[str, Path]:
    return {_seed_of(p): p for p in sorted(d.glob("*.png"))}


def _load(p: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    im = Image.open(p).convert("RGB")
    if size is not None:
        im = im.resize(size, Image.BICUBIC)
    return np.asarray(im, dtype=np.float32) / 255.0


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _ssim_drift(a: np.ndarray, b: np.ndarray):
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        return None
    s = ssim(a, b, channel_axis=2, data_range=1.0)
    return float(1.0 - s)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path, help="dir holding baseline/ + cut_*/ subdirs")
    args = ap.parse_args()

    baseline_dir = args.root / "baseline"
    if not baseline_dir.is_dir():
        raise SystemExit(f"missing {baseline_dir}")
    cut_dirs = sorted(
        (d for d in args.root.glob("cut_*") if d.is_dir()),
        key=lambda d: float(d.name.split("_")[1]),
        reverse=True,  # high cutoff (knock little) -> low cutoff (knock a lot)
    )
    if not cut_dirs:
        raise SystemExit(f"no cut_* subdirs under {args.root}")

    base_idx = _index_dir(baseline_dir)
    seeds = sorted(base_idx)
    print(f"baseline images: {len(seeds)} seeds {seeds}")

    # ---- drift table ----
    results: dict[str, dict] = {}
    print(f"\n{'cutoff':>8} {'rmse_mean':>10} {'1-ssim_mean':>12}  (drift vs baseline)")
    for d in cut_dirs:
        cutoff = float(d.name.split("_")[1])
        idx = _index_dir(d)
        rmses, ssims = [], []
        for s in seeds:
            if s not in idx:
                continue
            a = _load(base_idx[s])
            b = _load(idx[s], size=(a.shape[1], a.shape[0]))
            rmses.append(_rmse(a, b))
            sd = _ssim_drift(a, b)
            if sd is not None:
                ssims.append(sd)
        rmse_m = float(np.mean(rmses)) if rmses else float("nan")
        ssim_m = float(np.mean(ssims)) if ssims else None
        results[d.name] = {
            "cutoff": cutoff,
            "rmse_mean": rmse_m,
            "ssim_drift_mean": ssim_m,
            "n": len(rmses),
        }
        print(
            f"{cutoff:>8.2f} {rmse_m:>10.4f} "
            f"{(ssim_m if ssim_m is not None else float('nan')):>12.4f}"
        )

    print(
        "\nread: drift rising as cutoff drops => late text keeps steering (headroom);"
        "\n      drift ~flat/near-zero by some cutoff => text inert below it (structural)."
    )

    (args.root / "result.json").write_text(json.dumps(results, indent=2))

    # ---- contact sheet: rows = prompts(seeds), cols = baseline + cutoffs ----
    cols = [("baseline", baseline_dir)] + [
        (d.name.replace("cut_", "σ<"), d) for d in cut_dirs
    ]
    thumb = 288
    pad = 22
    col_idx = [_index_dir(d) for _, d in cols]
    n_rows, n_cols = len(seeds), len(cols)
    sheet = Image.new(
        "RGB",
        (n_cols * thumb, n_rows * thumb + pad),
        (255, 255, 255),
    )
    for r, s in enumerate(seeds):
        for c, idx in enumerate(col_idx):
            if s not in idx:
                continue
            im = Image.open(idx[s]).convert("RGB").resize((thumb, thumb), Image.BICUBIC)
            sheet.paste(im, (c * thumb, r * thumb + pad))
    try:
        from PIL import ImageDraw

        draw = ImageDraw.Draw(sheet)
        for c, (label, _) in enumerate(cols):
            draw.text((c * thumb + 4, 4), label, fill=(0, 0, 0))
    except Exception:
        pass
    sheet_path = args.root / "contact_sheet.png"
    sheet.save(sheet_path)
    print(f"\nwrote {sheet_path} and {args.root / 'result.json'}")

    # ---- drift plot ----
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    cuts = [results[d.name]["cutoff"] for d in cut_dirs]
    rmse = [results[d.name]["rmse_mean"] for d in cut_dirs]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(cuts, rmse, "o-", color="C3", label="pixel RMSE vs baseline")
    if all(results[d.name]["ssim_drift_mean"] is not None for d in cut_dirs):
        ssimd = [results[d.name]["ssim_drift_mean"] for d in cut_dirs]
        ax.plot(cuts, ssimd, "s--", color="C4", label="1 - SSIM vs baseline")
    ax.set_xlabel("knockout cutoff sigma (text off below this)")
    ax.set_ylabel("drift from full-prompt baseline")
    ax.invert_xaxis()  # high cutoff (knock little) on left
    ax.set_title("sigma-cutoff text-knockout: how much late text steers")
    ax.legend()
    fig.tight_layout()
    plot_path = args.root / "drift_plot.png"
    fig.savefig(plot_path, dpi=120)
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
