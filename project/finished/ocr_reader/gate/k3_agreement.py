#!/usr/bin/env python3
"""Which teacher × voter to relabel the KO / ZH pools with, measured on K3.

``pseudo_label.py filter`` keeps a row when the teacher's read passes the
guards and agrees with the voter's (``exact_key``); the teacher's string is the
label. K3's hand labels are held out of every pseudo draw, so on K3 each
filter's **precision** (kept rows whose label is right) and **yield** (kept /
read) can be counted. Candidates:

- teacher ∈ {stock, v4}, voter ∈ {hayai25, the other VL}
- v4 alone, gated on its mean token logprob

``read`` (GPU, through the daemon) stores every reader's string + logprob; the
table is rebuilt from the stored reads on CPU with no flag.

    make daemon-run ARGS="project/finished/ocr_reader/gate/k3_agreement.py read"
    .venv/bin/python project/finished/ocr_reader/gate/k3_agreement.py

n = 48 per language: read only large gaps.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ocr"))
import eval_k3  # noqa: E402
import eval_manga109 as ev  # noqa: E402
import pseudo_label as pl  # noqa: E402
import textnorm  # noqa: E402

V4 = ev.m109.REPO / "output/ocr/vl16_p2b_kozh_ep2/ep2"
READS = ev.OUT / "k3_agreement_reads.jsonl"
REPORT = HERE.parent / "reports/k3_agreement.md"


def cmd_read(a):
    df = eval_k3.load_labels()
    crops = eval_k3.load_crops(df)
    rows = []
    for name, cls, ckpt in (
        ("stock", pl.Vl16ScoredSweeper, None),
        ("v4", pl.Vl16ScoredSweeper, str(V4)),
        ("hayai25", pl.Hayai25Sweeper, pl.CKPT["hayai25"]),
    ):
        r = cls(ckpt, a.device)
        reads = []
        for s in range(0, len(crops), a.bs):
            reads += r.read(crops[s : s + a.bs])
        for rid, (t, n, lp) in zip(df.id, reads):
            rows.append(dict(id=rid, reader=name, pred=t, n_tokens=n, logprob=lp))
        del r
        import torch

        torch.cuda.empty_cache()
        print(f"{name}: {len(reads)} reads", flush=True)
    READS.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_json(READS, orient="records", lines=True, force_ascii=False)
    print(f"wrote {READS}")


def wilson(k, n, z=1.0):
    if not n:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def cmd_table(a):
    labels = eval_k3.load_labels().set_index("id")
    reads = pd.read_json(READS, lines=True, dtype={"id": str})
    wide = reads.pivot(
        index="id", columns="reader", values=["pred", "n_tokens", "logprob"]
    )
    df = (
        labels.join(wide["pred"])
        .join(wide["n_tokens"].add_suffix("_n"))
        .join(wide["logprob"].add_suffix("_lp"))
    )
    key = textnorm.exact_key
    for r in ("stock", "v4", "hayai25"):
        df[f"{r}_ok"] = [key(p) == key(t) for p, t in zip(df[r].fillna(""), df.text)]
    for r in ("stock", "v4"):
        df[f"{r}_sp"] = [
            textnorm.normalize_target(p) == textnorm.normalize_target(t)
            for p, t in zip(df[r].fillna(""), df.text)
        ]

    md = [
        "# K3 — teacher × voter for the KO / ZH relabel\n",
        "Precision = kept rows whose label (the teacher's string) is exact-right; "
        "`spaced` = right *with* spacing (the KO 띄어쓰기 label). ±1σ Wilson.\n",
        "## Readers alone\n",
        "| lang | n | stock | v4 | hayai25 |",
        "|---|---|---|---|---|",
    ]
    for lang, g in df.groupby("lang"):
        md.append(
            f"| {lang} | {len(g)} | {int(g.stock_ok.sum())} | {int(g.v4_ok.sum())} | {int(g.hayai25_ok.sum())} |"
        )

    md += [
        "\n## Agreement filters\n",
        "| lang | teacher | voter | kept / n | precision exact | precision spaced | right rows lost |",
        "|---|---|---|---|---|---|---|",
    ]
    for lang, g in df.groupby("lang"):
        for t, v in (("stock", "hayai25"), ("v4", "hayai25"), ("v4", "stock")):
            ok = np.array(
                [pl.guard(p, n) == "ok" for p, n in zip(g[t].fillna(""), g[f"{t}_n"])]
            )
            agree = np.array(
                [key(p) == key(q) for p, q in zip(g[t].fillna(""), g[v].fillna(""))]
            )
            kept = ok & agree
            k, kp = int(kept.sum()), int((kept & g[f"{t}_ok"].values).sum())
            ks = int((kept & g[f"{t}_sp"].values).sum())
            lo, hi = wilson(kp, k)
            lost = int((~kept & g[f"{t}_ok"].values).sum())
            md.append(
                f"| {lang} | {t} | {v} | {k} / {len(g)} | {kp}/{k} ({lo:.2f}–{hi:.2f}) | {ks}/{k} | {lost} |"
            )

    md += [
        "\n## v4 alone, gated on mean token logprob\n",
        "| lang | logprob ≥ | kept / n | precision exact | precision spaced |",
        "|---|---|---|---|---|",
    ]
    for lang, g in df.groupby("lang"):
        ok = np.array([pl.guard(p, n) == "ok" for p, n in zip(g.v4.fillna(""), g.v4_n)])
        for q in (0.0, 0.25, 0.5, 0.75):
            thr = float(np.nanquantile(g.v4_lp[ok], q)) if ok.any() else float("nan")
            kept = ok & (g.v4_lp.values >= thr)
            k, kp = int(kept.sum()), int((kept & g.v4_ok.values).sum())
            ks = int((kept & g.v4_sp.values).sum())
            lo, hi = wilson(kp, k)
            md.append(
                f"| {lang} | {thr:.3f} (q{q:.2f}) | {k} / {len(g)} | {kp}/{k} ({lo:.2f}–{hi:.2f}) | {ks}/{k} |"
            )
    text = "\n".join(md) + "\n"
    REPORT.write_text(text, encoding="utf-8")
    print(text)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("cmd", nargs="?", default="table", choices=["read", "table"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bs", type=int, default=16)
    a = ap.parse_args()
    {"read": cmd_read, "table": cmd_table}[a.cmd](a)


if __name__ == "__main__":
    main()
