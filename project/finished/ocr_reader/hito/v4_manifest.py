#!/usr/bin/env python3
"""The confident half of the hito v4 self-labels as a training manifest.

    ANIMA_MANGA109S_ROOT=… .venv/bin/python project/finished/ocr_reader/hito/v4_manifest.py \\
        [--quantile 0.5] [--drop_flagged] [--exclude ids.txt]

Reads ``prelabel_v4lp.jsonl`` (``hito/prelabel.py``) and keeps, per side, the reads
that pass ``pseudo_label.guard``, carry the side's script (JA: no Hangul; KO: at
least one Hangul syllable — v4 reads Korean hand-lettering as katakana, 0 / 345
right in the Opus relabel, so a KO-side read without Hangul is dropped), and sit
at or above the side's ``--quantile`` of mean token logprob (K3: the top half is
~85 % exact, the bottom half ~30 %; ``reports/k3_agreement.md``,
``reports/sfx_calibration_v4.md``). Crops that an Opus round hand-labelled
(``prelabel_opus*.jsonl``) are left to ``opus_manifest.py``.

Writes ``manifest_pseudo_hito_v4.parquet`` (``kind = "sfx"``, ``source = "v4"``).
Train with ``--extra_manifest pseudo_hito_v4``. ``--drop_flagged`` leaves out every
gallery in ``flagged_galleries_opus*.json``; ``--exclude`` takes one gallery id per
line (``#`` comments allowed).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ocr"))
import manga109 as m109  # noqa: E402
import pseudo_label as pl  # noqa: E402

KO = Path("/media/sorryhyun/new/dataset/hito_ocr/ko")


def has_hangul(s: str) -> bool:
    return any(0xAC00 <= ord(c) <= 0xD7A3 for c in s)


def read_jsonl(p: Path):
    return [json.loads(ln) for ln in p.open(encoding="utf-8") if ln.strip()]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ko", type=Path, default=KO)
    ap.add_argument("--quantile", type=float, default=0.5, help="keep logprob >= this quantile, per side")
    ap.add_argument("--exclude", type=Path)
    ap.add_argument("--drop_flagged", action="store_true")
    ap.add_argument("--out_name", default="hito_v4")
    a = ap.parse_args()

    drop = set()
    if a.exclude:
        for line in a.exclude.open(encoding="utf-8"):
            line = line.split("#")[0].strip()
            if line:
                drop.add(line)
    if a.drop_flagged:
        for p in a.ko.glob("flagged_galleries_opus*.json"):
            drop |= set(json.loads(p.read_text()))

    hand = set()
    for p in a.ko.glob("prelabel_opus*.jsonl"):
        hand |= {r["uid"].split(":", 1)[1] for r in read_jsonl(p)}  # "id:side"

    cuts = {}
    for f in a.ko.glob("*/regions.jsonl"):
        for r in read_jsonl(f):
            cuts[(r["id"], "ja")] = r.get("cut_ja")
            cuts[(r["id"], r["twin_lang"])] = r.get("cut_twin")

    rows = read_jsonl(a.ko / "prelabel_v4lp.jsonl")
    why = Counter()
    pool = {"ja": [], "ko": []}
    for r in rows:
        side, text = r["side"], r["read"]
        if f"{r['id']}:{side}" in hand:
            why["hand_labelled"] += 1
            continue
        g = pl.guard(text, r["n_tokens"])
        if g != "ok":
            why[g] += 1
            continue
        if (side == "ja") == has_hangul(text):
            why["script"] += 1
            continue
        if r["id"].split("_")[0] in drop:
            why["gallery_dropped"] += 1
            continue
        pool[side].append(r)

    out = []
    for side, rs in pool.items():
        lps = sorted(x["logprob"] for x in rs)
        thr = lps[int(a.quantile * len(lps))]
        kept = [x for x in rs if x["logprob"] >= thr]
        why[f"below_{side}_{thr}"] += len(rs) - len(kept)
        for x in kept:
            gal = x["id"].split("_")[0]
            cut = cuts[(x["id"], side)]
            w, h = cut[2] - cut[0], cut[3] - cut[1]
            out.append(
                {
                    "split": "train",
                    "kind": "sfx",
                    "id": f"{x['id']}_{side}",
                    "book": f"hi{gal}",
                    "page": int(x["id"].split("_")[1]),
                    "text": x["read"],
                    "joined": False,
                    "orient": "vertical" if h > w else "horizontal",
                    "w": w,
                    "h": h,
                    "poly": json.dumps([0.0, 0.0, float(w), 0.0, float(w), float(h), 0.0, float(h)]),
                    "path": str(a.ko / gal / "crops" / f"{x['id']}_{side}.png"),
                    "source": "v4",
                }
            )
    df = pd.DataFrame(out)
    path = m109.derived_root() / f"manifest_pseudo_{a.out_name}.parquet"
    df.to_parquet(path, index=False)
    print(f"wrote {path} ({len(df)} rows)")
    print("kept per side", Counter(r["id"].rsplit("_", 1)[1] for r in out))
    print("dropped", dict(why))


if __name__ == "__main__":
    main()
