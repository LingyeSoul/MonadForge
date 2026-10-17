"""Build a long-edge-capped HR cache from image_dataset/ for RSD distillation.

Why a cap, not a downsize-to-1024: the practical task is 1024(LR)->4096(HR) ×4, so HR
patches must carry ~4096-scale detail (what the student learns to hallucinate) and
HR÷4 must look like a real 1024 input. Native masters (median ~2.6k, up to ~7.9k px)
already sit near 4096 scale, so we KEEP them — we only cap the oversized at 4096 long
edge (LANCZOS) to bound the per-sample DataLoader decode cost. Masters already <= cap
are copied verbatim (no recompress, no softening). Masters with min side < 256 (can't
yield a clean 256² crop) are skipped.

Run in the root venv:  python sr/scripts/prep_rsd_cache.py [--cap 4096 --workers 8]
"""
import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[4]
EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def even(x: int) -> int:
    return x - (x % 2)


def _one(args):
    p, out_dir, cap, min_side = args
    try:
        with Image.open(p) as probe:
            w, h = probe.size
    except Exception:  # noqa: BLE001
        return "open_fail"
    if min(w, h) < min_side:
        return "too_small"  # can't crop a clean 256² HR patch
    stem = f"{p.parent.name}__{p.stem}"
    if max(w, h) <= cap:
        shutil.copy2(p, out_dir / f"{stem}{p.suffix.lower()}")  # verbatim, no recompress
        return "copied"
    try:
        im = Image.open(p).convert("RGB")
    except Exception:  # noqa: BLE001
        return "open_fail"
    s = cap / max(w, h)
    im.resize((even(round(w * s)), even(round(h * s))), Image.LANCZOS).save(out_dir / f"{stem}.png")
    return "capped"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(REPO / "image_dataset"))
    ap.add_argument("--out", default=str(REPO / "sr" / "data" / "rsd_hr_cap4096"))
    ap.add_argument("--cap", type=int, default=4096, help="long-edge cap (== HR output scale)")
    ap.add_argument("--min_side", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    src = Path(args.src).resolve()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in EXTS)
    if not files:
        raise SystemExit(f"no images under {src}")
    print(f"capping {len(files)} masters at {args.cap}px long edge -> {out_dir}")

    counts = {"copied": 0, "capped": 0, "too_small": 0, "open_fail": 0}
    work = [(p, out_dir, args.cap, args.min_side) for p in files]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_one, w) for w in work]
        for i, f in enumerate(as_completed(futs), 1):
            counts[f.result()] += 1
            if i % 500 == 0:
                print(f"  {i}/{len(files)}  {counts}")
    kept = counts["copied"] + counts["capped"]
    print(f"done: {counts}  kept {kept} HR masters -> {out_dir}")


if __name__ == "__main__":
    main()
