#!/usr/bin/env python3
"""JA logprob calibration: v4's mean token logprob against the sincos 597 SFX
hand labels — the JA counterpart of ``k3_agreement.py``'s logprob table, for
choosing the hitocrawl JA triage thresholds (``hito/prelabel.py``).

``read`` (GPU, through the daemon) stores v4's read + logprob per row; the
table is rebuilt from the stored reads on CPU with no flag. Scoring is
``eval_sfx``'s: ``ev.score`` exact (strict) and the heart-blind key.

    make daemon-run ARGS="project/ocr_reader/gate/sfx_calibration.py read"
    .venv/bin/python project/ocr_reader/gate/sfx_calibration.py

SFX only — the sincos speech rows carry PP-OCRv6 text, not hand labels.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ocr"))
import eval_manga109 as ev  # noqa: E402
import eval_sfx  # noqa: E402
import pseudo_label as pl  # noqa: E402

V4 = ev.m109.REPO / "output/ocr/vl16_p2b_kozh_ep2/ep2"
READS = ev.OUT / "sfx_calibration_v4_reads.jsonl"
REPORT = HERE.parent / "reports/sfx_calibration_v4.md"
HITO_LP = Path("/media/sorryhyun/new/dataset/hito_ocr/ko/prelabel_v4lp.jsonl")


def load():
    df = eval_sfx.load_labels(["sfx"], False)
    idx = pd.read_csv(eval_sfx.FROZEN / "index.tsv", sep="\t", dtype=str)
    df["orient"] = df.row.map(dict(zip(idx.row, idx.orient)))
    return df


def cmd_read(a):
    df = load()
    crops, _ = eval_sfx.crops_for(df, eval_sfx.FROZEN_PAD)
    r = pl.Vl16ScoredSweeper(str(V4), a.device)
    order = sorted(
        range(len(crops)), key=lambda i: crops[i].shape[0] * crops[i].shape[1]
    )
    reads = [None] * len(crops)
    for s in range(0, len(order), a.bs):
        idx = order[s : s + a.bs]
        for i, out in zip(idx, r.read([crops[i] for i in idx])):
            reads[i] = out
    READS.parent.mkdir(parents=True, exist_ok=True)
    with READS.open("w", encoding="utf-8") as f:
        for row, (t, n, lp) in zip(df.row, reads):
            f.write(
                json.dumps(
                    {"row": row, "pred": t, "n_tokens": n, "logprob": lp},
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"wrote {READS} ({len(reads)} rows)")


def wilson(k, n, z=1.0):
    if not n:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def cmd_table(a):
    df = load()
    reads = pd.read_json(READS, lines=True, dtype={"row": str}).set_index("row")
    df = df.join(reads, on="row")
    scored = ev.score(df.drop(columns=["pred"]), list(df.pred.fillna("")))
    strip = lambda t: ev.exact_key(t).replace("♡", "").replace("♥", "")  # noqa: E731
    scored["noheart"] = [
        strip(p) == strip(t) for p, t in zip(scored.pred_norm, scored.text)
    ]
    scored["logprob"] = df.logprob.values
    scored["guard_ok"] = [
        pl.guard(p, n) == "ok" for p, n in zip(df.pred.fillna(""), df.n_tokens)
    ]
    n = len(scored)
    lp = scored.logprob.values
    md = [
        "# JA logprob calibration — v4 on sincos 597 SFX\n",
        f"v4 = `{V4.relative_to(ev.m109.REPO)}`, bs 16 sorted by area. "
        f"exact **{int(scored.exact.sum())} / {n}** (heart-blind "
        f"{int(scored.noheart.sum())}); guard ok {int(scored.guard_ok.sum())}. "
        "Precision = exact among rows at or above the threshold; ±1σ Wilson.\n",
        "## By logprob threshold\n",
        "| logprob ≥ | rows | exact | heart-blind | below: exact |",
        "|---|---|---|---|---|",
    ]
    thr = [
        ("q0.90", np.quantile(lp, 0.10)),
        ("q0.75", np.quantile(lp, 0.25)),
        ("q0.50", np.quantile(lp, 0.50)),
        ("q0.25", np.quantile(lp, 0.75)),
    ]
    thr += [("K3 KO q75", -0.012), ("K3 KO q50", -0.077), ("K3 KO q25", -0.146)]
    for name, t in sorted(thr, key=lambda x: -x[1]):
        keep = lp >= t
        k = int(keep.sum())
        e = int((keep & scored.exact.values).sum())
        h = int((keep & scored.noheart.values).sum())
        lo, hi = wilson(e, k)
        below = ~keep
        eb = int((below & scored.exact.values).sum())
        md.append(
            f"| {t:.3f} ({name}) | {k} | {e}/{k} ({lo:.2f}–{hi:.2f}) | {h}/{k} | {eb}/{int(below.sum())} |"
        )

    md += [
        "\n## Exact rate by logprob decile\n",
        "| decile | logprob range | exact | heart-blind |",
        "|---|---|---|---|",
    ]
    q = np.quantile(lp, np.linspace(0, 1, 11))
    for i in range(10):
        m = (lp >= q[i]) & ((lp < q[i + 1]) if i < 9 else (lp <= q[i + 1]))
        md.append(
            f"| {i + 1} | {q[i]:.3f} … {q[i + 1]:.3f} | {int(scored.exact[m].sum())}/{int(m.sum())} "
            f"| {int(scored.noheart[m].sum())}/{int(m.sum())} |"
        )

    if HITO_LP.is_file():
        h = pd.read_json(HITO_LP, lines=True)
        h = h[(h.side == "ja") & (h.read.str.strip() != "")]
        hq = np.quantile(h.logprob, [0.1, 0.25, 0.5, 0.75, 0.9])
        sq = np.quantile(lp, [0.1, 0.25, 0.5, 0.75, 0.9])
        md += [
            "\n## Where hitocrawl JA sits\n",
            "| | p10 | p25 | p50 | p75 | p90 |",
            "|---|---|---|---|---|---|",
            "| sincos SFX | " + " | ".join(f"{x:.3f}" for x in sq) + " |",
            f"| hito JA (n={len(h)}) | " + " | ".join(f"{x:.3f}" for x in hq) + " |",
        ]
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
