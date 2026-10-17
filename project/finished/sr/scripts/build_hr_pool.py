"""Build a filtered HR pool for ResShift ×2 finetuning.

SR training needs genuinely SHARP, high-res masters — feeding soft/already-compressed
images teaches the model to reproduce that softness (garbage-in). This script sweeps
one or more source dirs (default: the gelcrawl/retrieved corpus + image_dataset/),
keeps only images that clear a short-edge floor and a Laplacian-variance sharpness
floor, and symlinks the survivors into a flat pool dir (sr/data/hr_pool/) with
artist__stem names. ArtSRDataset (train_x2/train.py) then rglobs that pool.

Symlinks, not copies — the gelcrawl webp masters are large; we don't duplicate ~30GB.
A manifest.json records counts + per-reason rejections so the pool is reproducible.

Runs under the MAIN anima env (PIL + numpy only) — no SR venv needed:
    python sr/scripts/build_hr_pool.py [--src DIR ...] [--min_edge 1024] [--lap_floor 18]
"""
import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

SR = Path(__file__).resolve().parents[1]          # project/finished/sr/
REPO = SR.parents[2]                              # repo root
DEFAULT_SRCS = [REPO.parent.parent / "gelcrawl" / "retrieved", REPO / "image_dataset"]
EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def lap_var(im: Image.Image, side: int = 512) -> float:
    """Variance of a 3x3 Laplacian on a downscaled grayscale image — a sharpness proxy.

    Downscale first so the score reflects real detail at a fixed scale (a huge soft
    scan shouldn't out-score a crisp smaller one just by pixel count).
    """
    g = im.convert("L")
    s = side / max(g.size)
    if s < 1.0:
        g = g.resize((max(1, int(g.width * s)), max(1, int(g.height * s))), Image.BILINEAR)
    a = np.asarray(g, dtype=np.float32)
    # 3x3 Laplacian via shifted sums (no scipy/cv2 dependency)
    lap = (-4.0 * a
           + np.roll(a, 1, 0) + np.roll(a, -1, 0)
           + np.roll(a, 1, 1) + np.roll(a, -1, 1))
    return float(lap[1:-1, 1:-1].var())


def list_images(srcs):
    out = []
    for src in srcs:
        src = Path(src)
        if not src.exists():
            print(f"  (skip missing src: {src})")
            continue
        # follow the image_dataset symlink-to-nested-artist-dirs; rglob is fine for files
        out += [p for p in sorted(src.rglob("*")) if p.suffix.lower() in EXTS and p.is_file()]
    return out


# module-level filter knobs so the Pool worker needs no per-call closure
_MIN_EDGE = 1024
_LAP_FLOOR = 18.0


def _judge(p: Path):
    """Worker: return (verdict, name, resolved_path). verdict in keep/too_small/too_soft/unreadable."""
    try:
        with Image.open(p) as im:
            im.draft("RGB", (1024, 1024))  # fast partial decode where the codec supports it
            w, h = im.size
            if min(w, h) < _MIN_EDGE:
                return ("too_small", None, None)
            if _LAP_FLOOR > 0 and lap_var(im) < _LAP_FLOOR:
                return ("too_soft", None, None)
    except Exception:  # noqa: BLE001
        return ("unreadable", None, None)
    name = f"{p.parent.name}__{p.stem}{p.suffix.lower()}"
    return ("keep", name, str(p.resolve()))


def _init_worker(min_edge, lap_floor):
    global _MIN_EDGE, _LAP_FLOOR
    _MIN_EDGE, _LAP_FLOOR = min_edge, lap_floor


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", action="append", default=None,
                    help="source dir (repeatable). Default: gelcrawl/retrieved + image_dataset/")
    ap.add_argument("--out", default=str(SR / "data" / "hr_pool"))
    ap.add_argument("--min_edge", type=int, default=1024,
                    help="reject if min(W,H) < this. 1024 guarantees real detail + clean "
                         "256-crops with 2x headroom for the x2 product.")
    ap.add_argument("--lap_floor", type=float, default=18.0,
                    help="reject if Laplacian-variance sharpness < this (soft/blurry scans). "
                         "Tune by eyeballing the rejected list; 0 disables.")
    ap.add_argument("--limit", type=int, default=0, help="cap pool size (0=all); for smoke runs")
    ap.add_argument("--workers", type=int, default=min(16, (os.cpu_count() or 4)),
                    help="parallel decode workers (webp decode is CPU-bound — the slow part)")
    ap.add_argument("--dry_run", action="store_true", help="count only, don't symlink")
    args = ap.parse_args()

    srcs = [Path(s) for s in (args.src or DEFAULT_SRCS)]
    out = Path(args.out)
    if not args.dry_run:
        out.mkdir(parents=True, exist_ok=True)

    print(f"sources: {[str(s) for s in srcs]}")
    print(f"filters: min_edge={args.min_edge}  lap_floor={args.lap_floor}  workers={args.workers}")
    files = list_images(srcs)
    print(f"listing: {len(files)} candidate images")

    stats = {"scanned": 0, "kept": 0, "too_small": 0, "too_soft": 0, "unreadable": 0, "dup_name": 0}
    seen_names = set()

    with Pool(args.workers, initializer=_init_worker, initargs=(args.min_edge, args.lap_floor)) as pool:
        for verdict, name, resolved in pool.imap_unordered(_judge, files, chunksize=16):
            stats["scanned"] += 1
            if stats["scanned"] % 2000 == 0:
                print(f"  scanned {stats['scanned']}/{len(files)}  kept {stats['kept']}")
            if verdict != "keep":
                stats[verdict] += 1
                continue
            if name in seen_names:
                stats["dup_name"] += 1
                continue
            seen_names.add(name)
            stats["kept"] += 1
            if not args.dry_run:
                link = out / name
                if link.is_symlink() or link.exists():
                    link.unlink()
                link.symlink_to(resolved)
            if args.limit and stats["kept"] >= args.limit:
                print(f"  hit --limit {args.limit}")
                break

    print("\n=== HR pool ===")
    for k, v in stats.items():
        print(f"  {k:12s} {v}")
    if not args.dry_run:
        manifest = {"out": str(out), "srcs": [str(s) for s in srcs],
                    "min_edge": args.min_edge, "lap_floor": args.lap_floor, "stats": stats}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"\nsymlinked {stats['kept']} masters -> {out}")
        print(f"manifest: {out / 'manifest.json'}")
        print(f"\nnext:  train_sr/train.py   (defaults --src to {out})")


if __name__ == "__main__":
    main()
