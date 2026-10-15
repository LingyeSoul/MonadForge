#!/usr/bin/env python3
"""First-pass reads of the hitocrawl twin crops (``build_crops.py``) with our
own reader — v4 (``vl16_p2b_kozh_ep2/ep2``) by default.

    make daemon-run ARGS="--stall-timeout 900 project/finished/ocr_reader/hito/prelabel.py --twin ko"

Reads both sides of every region and appends to ``<out>/<twin>/prelabel_<name>.jsonl``
(``id, side, read, n_tokens, logprob``) in chunks, so a killed run resumes where
it stopped. ``logprob`` is the mean token logprob of the greedy read — the
triage key for a second labeler (K3: the top half by logprob is ~85 % exact,
the bottom half ~30 %; ``reports/k3_agreement.md``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ocr"))
import eval_manga109 as ev  # noqa: E402
import pseudo_label as pl  # noqa: E402

OUT = Path("/media/sorryhyun/new/dataset/hito_ocr")
V4 = ev.m109.REPO / "output/ocr/vl16_p2b_kozh_ep2/ep2"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--twin", default="ko")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--ckpt", default=str(V4))
    ap.add_argument("--name", default="v4lp")
    ap.add_argument("--limit", type=int, default=0, help="first N regions (smoke)")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    root = a.out / a.twin
    rows = [
        json.loads(line)
        for f in sorted(root.glob("*/regions.jsonl"))
        for line in f.open(encoding="utf-8")
    ]
    if a.limit:
        rows = rows[: a.limit]
    dst = root / f"prelabel_{a.name}.jsonl"
    done = set()
    if dst.exists():
        done = {
            (r["id"], r["side"]) for r in map(json.loads, dst.open(encoding="utf-8"))
        }
    todo = [
        (r["id"], side, r[key])
        for r in rows
        for side, key in (("ja", "crop_ja"), (a.twin, "crop_twin"))
        if r.get(key) and (r["id"], side) not in done
    ]
    print(
        f"{len(rows)} regions, {len(todo)} crops to read ({len(done)} done)", flush=True
    )
    if not todo:
        return

    reader = pl.Vl16ScoredSweeper(a.ckpt, a.device)
    t0, n = time.time(), 0
    with dst.open("a", encoding="utf-8") as f:
        for s in range(0, len(todo), a.chunk):
            part = todo[s : s + a.chunk]
            crops = [cv2.imread(str(root / p)) for _, _, p in part]
            # batch by area, as ``Vl16Reader.read`` does
            order = sorted(
                range(len(part)), key=lambda i: crops[i].shape[0] * crops[i].shape[1]
            )
            reads = [None] * len(part)
            for b in range(0, len(order), a.bs):
                idx = order[b : b + a.bs]
                for i, r in zip(idx, reader.read([crops[i] for i in idx])):
                    reads[i] = r
            for (rid, side, _), (t, nt, lp) in zip(part, reads):
                row = {
                    "id": rid,
                    "side": side,
                    "read": t,
                    "n_tokens": nt,
                    "logprob": round(lp, 5),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            n += len(part)
            print(f"{n}/{len(todo)} crops, {n / (time.time() - t0):.1f}/s", flush=True)


if __name__ == "__main__":
    main()
