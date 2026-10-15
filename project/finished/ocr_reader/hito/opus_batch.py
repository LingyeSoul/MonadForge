#!/usr/bin/env python3
"""Batches for an Opus hand-label round on the hito low tail, and the merge back
(``reports/opus_relabel.md``).

    P=project/finished/ocr_reader/hito/opus_batch.py; W=<work dir>
    .venv/bin/python $P select    $W --n 1000 --batches 15   # in/, zoom/, items_full.json
    .venv/bin/python $P verify-in $W --batches 6             # vin/ from out/ (status ok)
    .venv/bin/python $P merge     $W --name b2               # ko/prelabel_opus_<name>.jsonl

``select`` takes the lowest-logprob ``n_tokens >= 4`` reads of ``prelabel_v4lp.jsonl``
that no existing ``prelabel_opus*.jsonl`` already holds, so each round continues
where the last one stopped; ``--redo_excluded`` puts round 1's ``excluded`` crops back
in front of the queue (they are its lowest logprobs). Rows carry no v4 read (it would anchor the labeller).
Crops under ``--zoom_below`` px on the short side get an enlarged copy in
``zoom/batch_NN/``; ``view`` points at the file the agent should open.

``merge`` applies ``vout/`` verdicts when present (match → ok, fix → ok_fixed,
wrong_image / drop → drop_verify), attaches the v4 read, keeps the labeller's
``flag`` per row, and writes ``flagged_galleries_opus_<name>.json`` — the audit
list for ``FLAGGED.md``. Nothing is excluded at merge: labels stay as labelled.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

import cv2

KO = Path("/media/sorryhyun/new/dataset/hito_ocr/ko")
STATUSES = {"ok", "no_text", "multi", "cut", "illegible", "declined"}


def read_jsonl(p: Path):
    return [json.loads(ln) for ln in p.open(encoding="utf-8") if ln.strip()]


def write_jsonl(p: Path, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def uid_of(r):
    return f"hito:{r['id']}:{r['side']}"


def crop_of(r):
    return KO / r["id"].split("_")[0] / "crops" / f"{r['id']}_{r['side']}.png"


def select(a):
    done, redo = set(), set()
    for p in a.ko.glob("prelabel_opus*.jsonl"):
        for r in read_jsonl(p):
            done.add(r["uid"])
            if r["status"] == "excluded":
                redo.add(r["uid"])
    if a.redo_excluded:
        # round 1 zeroed whole galleries after a few refusals; label those crops now
        done -= redo
    v4 = [r for r in read_jsonl(a.ko / "prelabel_v4lp.jsonl") if r["n_tokens"] >= a.min_tokens]
    v4.sort(key=lambda r: r["logprob"])
    todo = [r for r in v4 if uid_of(r) not in done][: a.n]
    print(
        f"{len(done)} already labelled; taking {len(todo)} "
        f"({sum(uid_of(r) in redo for r in todo)} redo of excluded; "
        f"logprob {todo[0]['logprob']} .. {todo[-1]['logprob']})"
    )
    items = []
    for r in todo:
        c = crop_of(r)
        items.append(
            {
                "uid": uid_of(r),
                "src": "hito",
                "edition": r["side"],
                "crop": str(c),
                "view": str(c),
                "v4_read": r["read"],
                "v4_logprob": r["logprob"],
            }
        )
    random.Random(0).shuffle(items)
    (a.work / "in").mkdir(parents=True, exist_ok=True)
    zoomed = 0
    for b in range(a.batches):
        batch = items[b :: a.batches]
        for it in batch:
            im = cv2.imread(it["crop"])
            h, w = im.shape[:2]
            if min(h, w) < a.zoom_below:
                f = min(6, math.ceil(a.zoom_below / min(h, w)))
                out = a.work / "zoom" / f"batch_{b:02d}" / Path(it["crop"]).name
                out.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out), cv2.resize(im, (w * f, h * f), interpolation=cv2.INTER_CUBIC))
                it["view"] = str(out)
                zoomed += 1
        public = [{k: it[k] for k in ("uid", "src", "edition", "crop", "view")} for it in batch]
        write_jsonl(a.work / "in" / f"batch_{b:02d}.jsonl", public)
    (a.work / "items_full.json").write_text(json.dumps(items, ensure_ascii=False, indent=0))
    print(f"{a.batches} batches of ~{len(items) // a.batches}; {zoomed} crops zoomed")
    print("edition", Counter(it["edition"] for it in items))


def load_out(work: Path, sub: str):
    rows = {}
    for p in sorted((work / sub).glob("*.jsonl")):
        for r in read_jsonl(p):
            rows[r["uid"]] = r  # last write wins (agents may rewrite a row)
    return rows


def verify_in(a):
    out = load_out(a.work, "out")
    items = {it["uid"]: it for it in json.loads((a.work / "items_full.json").read_text())}
    ok = [
        {"uid": u, "crop": items[u]["crop"], "view": items[u]["view"], "label": r["label"], "note": r.get("note", "")}
        for u, r in out.items()
        if r["status"] == "ok"
    ]
    random.Random(1).shuffle(ok)
    for b in range(a.batches):
        write_jsonl(a.work / "vin" / f"v_{b}.jsonl", ok[b :: a.batches])
    print(f"{len(ok)} ok rows → {a.batches} verify batches")


def merge(a):
    items = json.loads((a.work / "items_full.json").read_text())
    out = load_out(a.work, "out")
    ver = load_out(a.work, "vout") if (a.work / "vout").exists() else {}
    rows, missing, bad = [], [], Counter()
    for it in items:
        r = out.get(it["uid"])
        if r is None:
            missing.append(it["uid"])
            continue
        status = r["status"]
        if status not in STATUSES:
            bad[status] += 1
        label, note = r.get("label", ""), r.get("note", "")
        v = ver.get(it["uid"])
        if v and status == "ok":
            if v["verdict"] == "fix":
                status, label = "ok_fixed", v["label"]
                note = v.get("note", "") or note
            elif v["verdict"] in ("wrong_image", "drop"):
                status, note = "drop_verify", f"{v['verdict']}: {v.get('note', '')}".strip(": ")
        rows.append(
            {
                "uid": it["uid"],
                "src": "hito",
                "edition": it["edition"],
                "crop": it["crop"],
                "status": status,
                "lang": r.get("lang", ""),
                "label": label,
                "note": note,
                "flag": r.get("flag", "") or (v or {}).get("flag", ""),
                "v4_read": it["v4_read"],
                "v4_logprob": it["v4_logprob"],
            }
        )
    dest = a.ko / f"prelabel_opus_{a.name}.jsonl"
    write_jsonl(dest, rows)
    flagged = {}
    for r in rows:
        if r["flag"]:
            flagged.setdefault(r["uid"].split(":")[1].split("_")[0], []).append(
                {"uid": r["uid"], "flag": r["flag"]}
            )
    (a.ko / f"flagged_galleries_opus_{a.name}.json").write_text(
        json.dumps(flagged, ensure_ascii=False, indent=1)
    )
    print(f"wrote {dest} ({len(rows)} rows); {len(missing)} missing; unknown statuses {dict(bad)}")
    print("status", Counter(r["status"] for r in rows))
    print("lang×status", Counter((r["lang"], r["status"]) for r in rows).most_common(12))
    print(f"flagged rows {sum(len(v) for v in flagged.values())} in {len(flagged)} galleries")
    if missing:
        print("missing:", missing[:20])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["select", "verify-in", "merge"])
    ap.add_argument("work", type=Path)
    ap.add_argument("--ko", type=Path, default=KO)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--min_tokens", type=int, default=4)
    ap.add_argument("--batches", type=int, default=15)
    ap.add_argument("--zoom_below", type=int, default=200)
    ap.add_argument(
        "--redo_excluded", action="store_true",
        help="select: also take crops an earlier round marked `excluded`",
    )
    ap.add_argument("--name", default="b2")
    a = ap.parse_args()
    {"select": select, "verify-in": verify_in, "merge": merge}[a.cmd](a)


if __name__ == "__main__":
    main()
