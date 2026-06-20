#!/usr/bin/env python3
"""Replace censored sincos webp with their matched uncensored counterparts.

Reads the manifest produced by match_decensored.py and, for each selected match,
overwrites image_dataset/sincos/<stem>.webp with the decensored image. The .txt
caption sidecar is left untouched (caption is unchanged). Image-dependent caches
for the swapped stem are invalidated so `make preprocess` regenerates them; the
text-only TE cache is kept.

DRY-RUN by default. Pass --apply to actually write.

  python scripts/curate/apply_decensored.py                 # dry-run, tier=auto
  python scripts/curate/apply_decensored.py --apply         # do it, tier=auto
  python scripts/curate/apply_decensored.py --include-review --apply
  python scripts/curate/apply_decensored.py --only 10800263,6806757 --apply
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SINCOS_DIR = ROOT / "image_dataset" / "sincos"
DECEN_DIR = ROOT / "sincos_decensored"
OUT_DIR = ROOT / "output" / "curate" / "sincos_decensored"
BACKUP_DIR = OUT_DIR / "backup_censored"
LORA_CACHE = ROOT / "post_image_dataset" / "lora" / "sincos"
RESIZED = ROOT / "post_image_dataset" / "resized" / "sincos"


def invalidate_caches(stem: str, apply: bool) -> list[str]:
    """Remove image-dependent caches for stem; keep the text-only TE cache."""
    removed = []
    if LORA_CACHE.exists():
        for p in LORA_CACHE.glob(f"{stem}_*"):
            if p.name.endswith("_anima_te.safetensors"):
                continue  # text-only, caption unchanged
            removed.append(str(p.relative_to(ROOT)))
            if apply:
                p.unlink()
    if RESIZED.exists():
        for p in RESIZED.glob(f"{stem}.*"):
            removed.append(str(p.relative_to(ROOT)))
            if apply:
                p.unlink()
    return removed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually write (default: dry-run)")
    ap.add_argument("--include-review", action="store_true", help="also apply tier=review")
    ap.add_argument("--only", default="", help="comma-separated sincos stems to restrict to")
    ap.add_argument("--exclude", default="", help="comma-separated sincos stems to skip")
    args = ap.parse_args()

    manifest = OUT_DIR / "matches.csv"
    if not manifest.exists():
        raise SystemExit(f"missing {manifest}; run match_decensored.py first")
    rows = list(csv.DictReader(manifest.open()))

    tiers = {"auto"} | ({"review"} if args.include_review else set())
    only = {s for s in args.only.split(",") if s}
    excl = {s for s in args.exclude.split(",") if s}

    selected = []
    for r in rows:
        stem = Path(r["sincos"]).stem
        if r["tier"] not in tiers:
            continue
        if only and stem not in only:
            continue
        if stem in excl:
            continue
        selected.append(r)

    print(f"{'APPLYING' if args.apply else 'DRY-RUN'}: {len(selected)} replacements "
          f"(tiers={sorted(tiers)})")
    if args.apply:
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    cache_hits = 0
    log = []
    for r in selected:
        src = DECEN_DIR / r["match"]
        dst = SINCOS_DIR / r["sincos"]
        stem = dst.stem
        if not src.exists():
            print(f"  ! missing decensored {src.name}, skipping")
            continue
        removed = invalidate_caches(stem, args.apply)
        cache_hits += len(removed)
        print(f"  {dst.name:>18}  <-  {src.name}   (ham={r['hamming']}, -{len(removed)} caches)")
        log.append({**r, "n_caches_removed": len(removed)})
        if args.apply:
            shutil.copy2(dst, BACKUP_DIR / dst.name)  # backup original censored
            shutil.copy2(src, dst)

    if args.apply:
        with (OUT_DIR / "applied.csv").open("w", newline="") as f:
            if log:
                w = csv.DictWriter(f, fieldnames=list(log[0].keys()))
                w.writeheader()
                w.writerows(log)
        print(f"\nbacked up originals -> {BACKUP_DIR.relative_to(ROOT)}")
        print(f"invalidated {cache_hits} image caches; run `make preprocess` to regenerate")
        print(f"wrote {(OUT_DIR / 'applied.csv').relative_to(ROOT)}")
    else:
        print(f"\nwould invalidate {cache_hits} image caches. re-run with --apply to write.")


if __name__ == "__main__":
    main()
