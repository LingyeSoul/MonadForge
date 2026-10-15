#!/usr/bin/env python3
"""Lettering regions from hitocrawl edition twins (roadmap P1b).

The same page exists lettered in JA and in a twin edition (KO, or EN) with the
artwork untouched, so the lettering is whatever differs. Per work:

1. phash + monotonic Needleman-Wunsch to match ja pages to twin pages
   (``align.py`` only aligns ja↔en, so ko is matched here).
2. Register the twin onto the ja page: ORB + partial-affine RANSAC.
3. Diff. A region's extent comes from the pixel diff of the normalized grays;
   it is kept only when enough of it is *seed* — ink on one side with no ink
   within ``r`` px on the other — which a few-px misregistration of line art
   cannot produce.
4. Cut each region from both sides, each to its own ink extent (the twin box
   is mapped back to twin coordinates and cut from the original twin page).

    .venv/bin/python project/finished/ocr_reader/hito/build_crops.py --twin ko --workers 8
    .venv/bin/python project/finished/ocr_reader/hito/build_crops.py --works ie/1701561 --viz 6

Writes ``<out>/<anchor>/{crops/,viz/,pages.json,regions.jsonl}`` per work and
skips works whose ``regions.jsonl`` exists (``--force`` redoes them). No GPU.
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

HITO = Path.home() / "hitocrawl" / "retrieved"
OUT = Path("/media/sorryhyun/new/dataset/hito_ocr")


def phash(gray: np.ndarray) -> np.ndarray:
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(small)[:8, :8]
    return (d > np.median(d)).flatten()


def align_pages(a: list[np.ndarray], b: list[np.ndarray], gap=16.0, thr=14):
    """[(i, j, hamming)] — the align.py recipe on our own 64-bit phash."""
    n, m = len(a), len(b)
    dist = np.array([[int((x != y).sum()) for y in b] for x in a]).reshape(n, m)
    d = np.zeros((n + 1, m + 1))
    d[:, 0] = np.arange(n + 1) * gap
    d[0, :] = np.arange(m + 1) * gap
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(
                d[i - 1, j - 1] + dist[i - 1, j - 1],
                d[i - 1, j] + gap,
                d[i, j - 1] + gap,
            )
    out, i, j = [], n, m
    while i > 0 and j > 0:
        if d[i, j] == d[i - 1, j - 1] + dist[i - 1, j - 1]:
            if dist[i - 1, j - 1] <= thr:
                out.append((i - 1, j - 1, int(dist[i - 1, j - 1])))
            i, j = i - 1, j - 1
        elif d[i, j] == d[i - 1, j] + gap:
            i -= 1
        else:
            j -= 1
    return out[::-1]


def register(ja: np.ndarray, tw: np.ndarray, work_px=1200):
    """2×3 affine mapping twin → ja coordinates, and its RANSAC inlier count."""
    sa, sb = work_px / max(ja.shape), work_px / max(tw.shape)
    a = cv2.resize(ja, None, fx=sa, fy=sa, interpolation=cv2.INTER_AREA)
    b = cv2.resize(tw, None, fx=sb, fy=sb, interpolation=cv2.INTER_AREA)
    orb = cv2.ORB_create(5000)
    ka, da = orb.detectAndCompute(a, None)
    kb, db = orb.detectAndCompute(b, None)
    if da is None or db is None:
        return None, 0
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(db, da)
    if len(matches) < 20:
        return None, 0
    pb = np.float32([kb[mm.queryIdx].pt for mm in matches]) / sb
    pa = np.float32([ka[mm.trainIdx].pt for mm in matches]) / sa
    M, inl = cv2.estimateAffinePartial2D(
        pb, pa, method=cv2.RANSAC, ransacReprojThreshold=3.0
    )
    return M, int(inl.sum()) if inl is not None else 0


def norm_gray(g: np.ndarray, valid: np.ndarray) -> np.ndarray:
    b = cv2.GaussianBlur(g, (3, 3), 0)
    lo, hi = np.percentile(b[valid > 0], [1, 99])
    return np.clip((b.astype(np.float32) - lo) * 255.0 / max(hi - lo, 1), 0, 255)


def bbox(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def diff_regions(ja: np.ndarray, warped: np.ndarray, valid: np.ndarray, a):
    na, nb = norm_gray(ja, valid), norm_gray(warped, valid)
    ia, ib = (na < 110).astype(np.uint8), (nb < 110).astype(np.uint8)
    r = max(3, int(round(min(ja.shape) * a.slack_frac)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    only_a = ia & (1 - cv2.dilate(ib, ker)) & (valid > 0)
    only_b = ib & (1 - cv2.dilate(ia, ker)) & (valid > 0)
    seed = cv2.morphologyEx(only_a | only_b, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    pix = ((np.abs(na - nb) > 70) & (valid > 0)).astype(np.uint8)
    k = max(9, int(round(min(ja.shape) * a.close_frac)))
    mask = cv2.morphologyEx(
        pix | seed,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
    )
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    seeds_per = np.bincount(lab[seed > 0], minlength=n)
    page_area = ja.shape[0] * ja.shape[1]
    regions = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < a.min_area_frac * page_area or area > 0.35 * page_area:
            continue
        if seeds_per[i] < a.seed_frac * area:
            continue
        sl = (slice(y, y + h), slice(x, x + w))
        comp = lab[sl] == i
        # the side's own changed ink, plus its unchanged ink overlapping the
        # component, so a glyph half-shared with the other side stays whole
        side_a = comp & ((only_a[sl] > 0) | ((ia[sl] > 0) & (pix[sl] > 0)))
        side_b = comp & ((only_b[sl] > 0) | ((ib[sl] > 0) & (pix[sl] > 0)))
        ba, bb = bbox(side_a), bbox(side_b)
        off = np.array([x, y, x, y])
        # ring of the ja page around the component: bubble interiors are white
        pad = max(4, int(0.15 * max(w, h)))
        rx0, ry0 = max(0, x - pad), max(0, y - pad)
        ring = na[ry0 : y + h + pad, rx0 : x + w + pad]
        regions.append(
            {
                "box": [int(x), int(y), int(x + w), int(y + h)],
                "box_ja": (np.array(ba) + off).tolist() if ba else None,
                "box_twin_in_ja": (np.array(bb) + off).tolist() if bb else None,
                "area": int(area),
                "seed": int(seeds_per[i]),
                "ink_ja": int(only_a[sl][comp].sum()),
                "ink_twin": int(only_b[sl][comp].sum()),
                "white_ring": round(float((ring > 230).mean()), 3),
            }
        )
    return regions, float(seed[valid > 0].mean())


def padded(box, pad_frac, w, h):
    x0, y0, x1, y1 = box
    p = max(4, int(pad_frac * max(x1 - x0, y1 - y0)))
    return [max(0, x0 - p), max(0, y0 - p), min(w, x1 + p), min(h, y1 + p)]


def to_twin(box, Minv, w, h):
    x0, y0, x1, y1 = box
    c = np.float32([[x0, y0], [x1, y0], [x0, y1], [x1, y1]])
    t = cv2.transform(c[None], Minv)[0]
    return [
        int(max(0, t[:, 0].min())),
        int(max(0, t[:, 1].min())),
        int(min(w, t[:, 0].max())),
        int(min(h, t[:, 1].max())),
    ]


def build_work(job) -> str:
    rel, a = job
    work = HITO / rel
    out = Path(a.out) / work.name
    if (out / "regions.jsonl").exists() and not a.force:
        return f"{work.name}: done, skipped"
    (out / "crops").mkdir(parents=True, exist_ok=True)
    ja_files = sorted((work / "ja").glob("*.webp"))
    tw_files = sorted((work / a.twin).glob("*.webp"))
    if not ja_files or not tw_files:
        return f"{work.name}: no ja/{a.twin} pages"
    ja_g = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in ja_files]
    tw_g = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in tw_files]
    pairs = align_pages([phash(g) for g in ja_g], [phash(g) for g in tw_g])

    regs = [register(ja_g[i], tw_g[j]) for i, j, _ in pairs]
    scales = [
        float(np.hypot(M[0, 0], M[1, 0]))
        for M, inl in regs
        if M is not None and inl >= a.min_inliers
    ]
    work_scale = float(np.median(scales)) if scales else 0.0

    rows, pages, n_viz = [], [], 0
    for (i, j, hd), (M, inl) in zip(pairs, regs):
        ja, tw = ja_g[i], tw_g[j]
        page = {"ja": ja_files[i].name, "twin": tw_files[j].name, "phash": hd}
        if M is None or inl < a.min_inliers:
            pages.append({**page, "inliers": inl, "status": "noreg"})
            continue
        H, W = ja.shape
        scale = float(np.hypot(M[0, 0], M[1, 0]))
        # a page off the work's scan scale is a redrawn cover / different art
        if abs(scale / work_scale - 1) > a.scale_tol:
            pages.append(
                {
                    **page,
                    "inliers": inl,
                    "scale": round(scale, 4),
                    "status": "scale_outlier",
                }
            )
            continue
        warped = cv2.warpAffine(
            tw, M, (W, H), flags=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        )
        valid = cv2.warpAffine(np.full(tw.shape, 255, np.uint8), M, (W, H))
        valid = cv2.erode(valid, np.ones((15, 15), np.uint8))
        regions, changed = diff_regions(ja, warped, valid, a)
        pages.append(
            {
                **page,
                "inliers": inl,
                "scale": round(scale, 4),
                "changed_frac": round(changed, 4),
                "n_regions": len(regions),
                "status": "ok",
            }
        )
        Minv = cv2.invertAffineTransform(M)
        ja_rgb = cv2.imread(str(ja_files[i]))
        tw_rgb = cv2.imread(str(tw_files[j]))
        for r_i, r in enumerate(regions):
            rid = f"{work.name}_{ja_files[i].stem}_{r_i:02d}"
            row = {
                "id": rid,
                "work": rel,
                "anchor": work.name,
                "twin_lang": a.twin,
                "ja_page": ja_files[i].name,
                "twin_page": tw_files[j].name,
                **r,
            }
            if r["box_ja"]:
                bj = padded(r["box_ja"], a.pad, W, H)
                cv2.imwrite(
                    str(out / "crops" / f"{rid}_ja.png"),
                    ja_rgb[bj[1] : bj[3], bj[0] : bj[2]],
                )
                row["crop_ja"] = f"{work.name}/crops/{rid}_ja.png"
                row["cut_ja"] = bj
            if r["box_twin_in_ja"]:
                bt = to_twin(
                    padded(r["box_twin_in_ja"], a.pad, W, H), Minv, *tw.shape[::-1]
                )
                if bt[2] > bt[0] and bt[3] > bt[1]:
                    cv2.imwrite(
                        str(out / "crops" / f"{rid}_{a.twin}.png"),
                        tw_rgb[bt[1] : bt[3], bt[0] : bt[2]],
                    )
                    row["crop_twin"] = f"{work.name}/crops/{rid}_{a.twin}.png"
                    row["cut_twin"] = bt
            rows.append(row)
        if n_viz < a.viz:
            n_viz += 1
            (out / "viz").mkdir(exist_ok=True)
            vis = cv2.cvtColor(ja, cv2.COLOR_GRAY2BGR)
            for r in regions:
                for key, col in (
                    ("box_ja", (0, 0, 255)),
                    ("box_twin_in_ja", (255, 0, 0)),
                ):
                    if r[key]:
                        x0, y0, x1, y1 = r[key]
                        cv2.rectangle(vis, (x0, y0), (x1, y1), col, 3)
            side = np.concatenate(
                [vis, cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)], axis=1
            )
            s = 1400 / side.shape[1]
            cv2.imwrite(
                str(out / "viz" / f"{ja_files[i].stem}.jpg"),
                cv2.resize(side, None, fx=s, fy=s, interpolation=cv2.INTER_AREA),
            )

    (out / "pages.json").write_text(json.dumps(pages, indent=1, ensure_ascii=False))
    with open(out / "regions.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ok = [p for p in pages if p["status"] == "ok"]
    med = float(np.median([p["changed_frac"] for p in ok])) if ok else float("nan")
    return (
        f"{work.name}: ja {len(ja_files)} {a.twin} {len(tw_files)} matched {len(pairs)} "
        f"registered {len(ok)} regions {len(rows)} changed_frac p50 {med:.4f}"
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--twin", default="ko", help="twin edition dir (ko / en)")
    ap.add_argument(
        "--works",
        nargs="*",
        help="<artist>/<anchor> under retrieved/; default all with a twin",
    )
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--viz", type=int, default=0, help="overlay pages per work")
    ap.add_argument(
        "--slack_frac",
        type=float,
        default=0.004,
        help="seed misregistration slack r / short edge",
    )
    ap.add_argument("--close_frac", type=float, default=0.012)
    ap.add_argument("--seed_frac", type=float, default=0.08)
    ap.add_argument("--min_area_frac", type=float, default=0.0004)
    ap.add_argument("--min_inliers", type=int, default=50)
    ap.add_argument(
        "--scale_tol", type=float, default=0.02, help="max |scale / work median - 1|"
    )
    ap.add_argument("--pad", type=float, default=0.12, help="crop pad, the sincos cut")
    a = ap.parse_args()
    a.out = Path(a.out) / a.twin
    works = a.works or sorted(
        str(p.parent.relative_to(HITO))
        for p in HITO.glob(f"*/*/{a.twin}")
        if p.is_dir()
    )
    a.out.mkdir(parents=True, exist_ok=True)
    with Pool(a.workers) as pool:
        for msg in pool.imap_unordered(build_work, [(w, a) for w in works]):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
