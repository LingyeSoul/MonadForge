#!/usr/bin/env python3
"""Match censored sincos training images to their uncensored ("decensored") counterparts.

Filenames don't correspond (sincos = gelbooru post IDs, decensored = pixiv IDs) and
resolutions differ, so matching is purely visual. Censored vs uncensored pairs differ
only in small mosaic/bar regions, so global perceptual descriptors match them reliably.

Descriptors (PIL+numpy only, no model download, scale-invariant via resize):
  - DCT perceptual hash (64-bit) on a 32x32 grayscale -> Hamming distance
  - 16x16 grayscale thumbnail, z-normalized -> cosine similarity
  - aspect-ratio gate (rejects candidates whose AR differs too much)

Outputs (non-destructive):
  output/curate/sincos_decensored/matches.csv     manifest, sorted by confidence
  output/curate/sincos_decensored/review.html      side-by-side contact sheet

Run replacement separately with apply_decensored.py after reviewing.
"""
from __future__ import annotations

import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

WORKERS = min(12, (os.cpu_count() or 4))

ROOT = Path(__file__).resolve().parents[2]
SINCOS_DIR = ROOT / "image_dataset" / "sincos"
DECEN_DIR = ROOT / "sincos_decensored"
OUT_DIR = ROOT / "output" / "curate" / "sincos_decensored"

CACHE_VER = 2  # bump to invalidate _desc_*.npz when descriptor schema changes
AR_TOL = 0.06  # |log(ar_a/ar_b)| gate; ~6% aspect-ratio tolerance
AUTO_HAM = 6  # hamming <= this AND cos >= AUTO_COS -> confident, safe to auto-replace
AUTO_COS = 0.95  # thumbnail-correlation floor for the auto tier
REVIEW_HAM = 12  # hamming in (AUTO_HAM, this] -> eyeball; beyond -> no counterpart


def _load_gray(path: Path, size: int) -> np.ndarray | None:
    try:
        im = Image.open(path).convert("L").resize((size, size), Image.LANCZOS)
    except Exception as e:  # noqa: BLE001
        print(f"  ! failed {path.name}: {e}", file=sys.stderr)
        return None
    return np.asarray(im, dtype=np.float32)


# tags that mean "this image is censored" (so replacing it is meaningful).
# substring 'censor' catches bar/heart/mosaic/blur/hair/steam/novelty/pointless censor,
# but 'uncensored'/'decensored' mean the opposite and must NOT trigger.
_NOT_CENSOR = {"uncensored", "decensored"}
_CENSOR_EXTRA = {"convenient hair"}  # hair censor without the word 'censor'


def is_censor_tag(tag: str) -> bool:
    """True if a single tag asserts the image is censored (not un/decensored)."""
    t = tag.strip().lower()
    if t in _NOT_CENSOR:
        return False
    return "censor" in t or t in _CENSOR_EXTRA


def has_censor_tag(stem: str) -> bool:
    txt = SINCOS_DIR / f"{stem}.txt"
    if not txt.exists():
        return False
    tags = txt.read_text(errors="ignore").replace("\n", ",").split(",")
    return any(is_censor_tag(t) for t in tags)


def _dct2(a: np.ndarray) -> np.ndarray:
    # separable DCT-II via matrix multiply (small N, no scipy dependency)
    n = a.shape[0]
    k = np.arange(n)
    basis = np.cos(np.pi * (2 * k[:, None] + 1) * k[None, :] / (2 * n))
    return basis @ a @ basis.T


def descriptors(path: Path) -> dict | None:
    """Return phash bits, normalized thumbnail vector, aspect ratio, frame count."""
    try:
        im = Image.open(path)
        w, h = im.size
        frames = int(getattr(im, "n_frames", 1))
    except Exception:  # noqa: BLE001
        return None
    g32 = _load_gray(path, 32)
    if g32 is None:
        return None
    d = _dct2(g32)[:8, :8]
    med = np.median(d[1:].flatten())  # skip DC term for the median
    phash = (d > med).flatten()

    thumb = _load_gray(path, 16).flatten()
    thumb = thumb - thumb.mean()
    norm = np.linalg.norm(thumb)
    thumb = thumb / norm if norm > 1e-6 else thumb

    return {"phash": phash, "thumb": thumb, "frames": frames,
            "logar": float(np.log((w + 1e-6) / (h + 1e-6)))}


def _worker(path_str: str):
    return path_str, descriptors(Path(path_str))


def build(dir_path: Path, label: str) -> dict[str, dict]:
    files = sorted(p for p in dir_path.iterdir() if p.suffix.lower() == ".webp")
    cache = OUT_DIR / f"_desc_{label}.npz"
    sig = max((p.stat().st_mtime for p in files), default=0.0)
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if (float(z["sig"]) == sig and int(z["n"]) == len(files)
                and int(z.get("ver", 0)) == CACHE_VER):
            print(f"[{label}] {len(files)} images -> loaded cached descriptors")
            return z["data"].item()
    print(f"[{label}] {len(files)} images -> computing descriptors on {WORKERS} workers")
    out = {}
    done = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for path_str, d in ex.map(_worker, [str(p) for p in files], chunksize=8):
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(files)}")
            if d is not None:
                out[Path(path_str).name] = d
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache, data=np.array(out, dtype=object), sig=sig, n=len(files), ver=CACHE_VER)
    return out


def main() -> None:
    if not DECEN_DIR.exists():
        sys.exit(f"missing {DECEN_DIR}")
    src = build(SINCOS_DIR, "sincos")
    cand = build(DECEN_DIR, "decensored")

    cand_names = list(cand)
    cand_phash = np.stack([cand[n]["phash"] for n in cand_names])  # (M,64) bool
    cand_thumb = np.stack([cand[n]["thumb"] for n in cand_names])  # (M,256)
    cand_logar = np.array([cand[n]["logar"] for n in cand_names])  # (M,)

    rows = []
    for name, d in src.items():
        ar_ok = np.abs(cand_logar - d["logar"]) <= AR_TOL
        ham = (cand_phash != d["phash"]).sum(axis=1).astype(np.float32)  # 0..64
        cos = cand_thumb @ d["thumb"]  # -1..1
        # combined score: low hamming + high cosine; penalize AR-gate failures
        score = (1.0 - ham / 64.0) * 0.5 + (cos + 1.0) / 2.0 * 0.5
        score = np.where(ar_ok, score, score - 1.0)
        order = np.argsort(-score)
        best, second = order[0], order[1] if len(order) > 1 else order[0]
        rows.append(
            {
                "sincos": name,
                "match": cand_names[best],
                "score": round(float(score[best]), 4),
                "hamming": int(ham[best]),
                "cos": round(float(cos[best]), 4),
                "margin": round(float(score[best] - score[second]), 4),
                "ar_ok": bool(ar_ok[best]),
                "match_frames": int(cand[cand_names[best]]["frames"]),
            }
        )

    # Tier by hamming (the reliable near-duplicate signal). Two hard gates force a
    # match to 'skip' regardless of visual score:
    #   - no censor tag in the sidecar  -> nothing to decensor
    #   - matched decensored is animated -> would inject animation into training
    # The decensored set also doesn't cover every sincos image, so a tail of genuine
    # non-matches is expected and skipped (left censored), not force-replaced.
    for r in rows:
        h = r["hamming"]
        stem = Path(r["sincos"]).stem
        if not has_censor_tag(stem):
            r["tier"], r["reason"] = "skip", "no_censor_tag"
        elif r["match_frames"] > 1:
            r["tier"], r["reason"] = "skip", "animated_match"
        elif h <= AUTO_HAM and r["cos"] >= AUTO_COS:
            r["tier"], r["reason"] = "auto", ""
        elif h <= REVIEW_HAM:
            r["tier"], r["reason"] = "review", ""
        else:
            r["tier"], r["reason"] = "skip", "no_match"

    rows.sort(key=lambda r: (r["hamming"], -r["score"]))  # best first
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "matches.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    rel_s = lambda n: str((SINCOS_DIR / n).resolve())  # noqa: E731
    rel_c = lambda n: str((DECEN_DIR / n).resolve())  # noqa: E731
    counts = {t: sum(1 for r in rows if r["tier"] == t) for t in ("auto", "review", "skip")}
    html = ["<html><head><meta charset='utf-8'><style>",
            "body{background:#222;color:#ddd;font-family:monospace}",
            "div.row{display:flex;align-items:center;gap:8px;border-bottom:1px solid #444;padding:6px}",
            "img{height:180px;object-fit:contain;background:#111}",
            "h2{position:sticky;top:0;background:#333;padding:8px;margin:0}",
            ".auto{color:#8f8}.review{color:#fd6}.skip{color:#f88}</style></head><body>"]
    for tier, blurb in (("review", "EYEBALL THESE — borderline"),
                        ("auto", "confident — will auto-replace"),
                        ("skip", "no counterpart in decensored set — left censored")):
        html.append(f"<h2 class='{tier}'>[{tier}] {counts[tier]} — {blurb}</h2>")
        for r in (x for x in rows if x["tier"] == tier):
            html.append(
                f"<div class='row'><span class='{tier}'>ham={r['hamming']} cos={r['cos']} "
                f"score={r['score']} margin={r['margin']}</span>"
                f"<img src='file://{rel_s(r['sincos'])}' title='{r['sincos']}'>"
                f"<span>&rarr;</span>"
                f"<img src='file://{rel_c(r['match'])}' title='{r['match']}'></div>"
            )
    html.append("</body></html>")
    (OUT_DIR / "review.html").write_text("\n".join(html))

    print(f"\nwrote {csv_path}")
    print(f"wrote {OUT_DIR / 'review.html'}  (open in a browser)")
    print(f"tiers: auto={counts['auto']}  review={counts['review']}  skip={counts['skip']}")


if __name__ == "__main__":
    main()
