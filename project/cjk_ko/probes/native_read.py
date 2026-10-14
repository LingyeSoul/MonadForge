#!/usr/bin/env python
"""native_read — the micro arms off the bubble template (user 10-14): a KO
clause hung off the eight scene prompts (``NATIVE_PROMPTS``), scored against
the EN reference (``eval/enref.py``: same prompt and seed, ``English text
reads as "hi"``, delta off).

Items (``ITEMS``): 48 strings of 1–5 syllables, every syllable on the 192
(``assets/micro192.json``) — the ``held`` strings of ≤ 5 syllables and words
cut from the longer ones, 10 / 10 / 10 / 10 / 8 by length; item ``i`` rides
prompt ``i % 8``, seed 0, 512², the reseed ruler's sampler.

``render`` (GPU): the 8 EN references once (``render/enref/``), then every item
under every arm → ``output/cjk_ko/native_read/render/<arm>/p<pi>_<text>_s0.png``
(``NATIVE_CLAUSES["en"]`` relanged to Korean). A render on disk is kept.

``score`` (GPU): v4's free read of every render (``micro_read``'s crops) and
the EN-reference ruler (``en_cos`` / ``en_cos_out`` / ``box_iou``; the EN box
= the detector's largest) → ``results/<ts>-native-read/``: ``reads.jsonl``,
``result.json`` (per arm × length and per arm: ``exact``, ``f1_row``, the
ruler's means) and ``sheet_<i>.png`` (EN reference, then the arms).

    make daemon-run ARGS="--stall-timeout 900 project/cjk_ko/probes/native_read.py render"
    make daemon-run ARGS="--stall-timeout 900 project/cjk_ko/probes/native_read.py score"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from micro_read import ARMS, HOME, KO_OUT, SEED, SIZE  # noqa: E402  (bootstraps)

OUT_DIR = KO_OUT / "native_read"
RENDER = OUT_DIR / "render"
EN_WORD = "hi"
DEFAULT_ARMS = "floor,jamo_j1024_ko_cf,ko_mix2_cf_bp,ko_mix3_cf_bp,ko_x5_cf_bp"
ITEMS = (
    # 1 syllable
    "답", "너", "때", "나", "모", "될", "수", "안", "해", "좋",
    # 2
    "당신", "후회", "뭐야", "좋죠", "너무", "재미", "운동", "이별", "수면", "돼요",
    # 3
    "비오네", "지친다", "너 뭐니?", "잊을 때", "이상해", "아니면", "다음날", "친구가",
    "사이트", "아직도",
    # 4
    "썸이 뭐야", "이별후에", "뭐든 좋죠", "부러워요", "당신이요", "너무 더워",
    "재미 없어", "해보세요", "좋아하는", "좋겠어요",
    # 5
    "잊어야하나?", "이별 다음날", "습관이네요", "하는거같아", "힘들겠지만",
    "말해보세요", "모 아니면 도", "될 수 없어요",
)  # fmt: skip


def syllables(t: str) -> int:
    return len(re.findall(r"[가-힣]", t))


def prompts() -> list:
    from common.prompts import NATIVE_PROMPTS

    return [
        ln.strip()
        for ln in NATIVE_PROMPTS.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    ]


def plan() -> list:
    ps = prompts()
    return [
        {"i": i, "pi": i % len(ps), "prompt": ps[i % len(ps)], "text": t}
        for i, t in enumerate(ITEMS)
    ]


def caption(it: dict) -> str:
    from common.prompts import NATIVE_CLAUSES
    from reseed.pools import relang

    return relang(NATIVE_CLAUSES["en"].format(p=it["prompt"], k=it["text"]), "korean")


def render_file(arm: str, it: dict, seed: int = SEED) -> Path:
    return RENDER / arm / f"p{it['pi']:02d}_{it['text']}_s{seed}.png"


def check_items() -> None:
    s = json.loads((HOME / "assets" / "micro192.json").read_text(encoding="utf-8"))
    on = set(s["trained"])
    off = {t: c for t in ITEMS for c in re.findall(r"[가-힣]", t) if c not in on}
    assert not off, f"syllables off the 192: {off}"
    assert all(1 <= syllables(t) <= 5 for t in ITEMS)
    assert len(set(ITEMS)) == len(ITEMS) == 48


def render(arms: list) -> None:
    from eval.enref import enref_caption, enref_file
    from eval.ruler import VIEW
    from eval.ruler.arms import PACK_ARMS, tables
    from eval.ruler.render import Renderer, check

    VIEW.pack = "punct"
    PACK_ARMS["punct"].update({a: ARMS[a] for a in arms})
    _, tabs = tables(arms + ["retrain_kana"])
    r = Renderer()
    info = {"check_rk": check(r, tabs["retrain_kana"])}
    t0 = time.time()
    r.set_arm(None)
    for pi, p in enumerate(prompts()):
        r.render(
            enref_file(RENDER / "enref", pi, SEED),
            enref_caption(p, EN_WORD),
            SEED,
            SIZE,
        )
    its = plan()
    print(f"  caption: {caption(its[0])}", flush=True)
    for arm in arms:
        r.set_arm(tabs[arm])
        for n, it in enumerate(its):
            r.render(render_file(arm, it), caption(it), SEED, SIZE)
            if n % 12 == 0 or n == len(its) - 1:
                print(
                    f"  {arm}: {n + 1} / {len(its)} ({(time.time() - t0) / 60:.1f} min)",
                    flush=True,
                )
    info["minutes"] = round((time.time() - t0) / 60, 1)
    print(f"→ {RENDER}: {len(arms)} arms × {len(its)} items; {info}", flush=True)


def score(arms: list) -> None:
    import numpy as np
    import torch
    from anime_tools.ocr.animetext import AnimeTextDetector
    from common.readers import contact_sheet
    from eval.enref import EnRef, enref_file
    from jamo_vl import Reader, judge
    from jamo_vl_sheet import f1_row, whole
    from ko_reader_cal import crops
    from PIL import Image

    from bench._common import make_run_dir, write_result

    t0 = time.time()
    its = plan()
    recs, ims = [], []
    for arm in arms:
        for it in its:
            fn = render_file(arm, it)
            assert fn.is_file(), f"{fn}: render it first"
            recs.append({"arm": arm, **it, "seed": SEED, "file": str(fn)})
            ims.append(Image.open(fn).convert("RGB"))
    refs = sorted({it["pi"] for it in its})
    ref_ims = [
        Image.open(enref_file(RENDER / "enref", pi, SEED)).convert("RGB") for pi in refs
    ]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    det = AnimeTextDetector.load(device=device)
    bgr = lambda im: np.ascontiguousarray(np.array(im)[:, :, ::-1])  # noqa: E731
    boxes = [[[int(v) for v in b] for b in det.detect(bgr(im))] for im in ims]
    ref_boxes = {}
    for pi, im in zip(refs, ref_ims):
        bs = [[int(v) for v in b] for b in det.detect(bgr(im))]
        ref_boxes[enref_file(RENDER / "enref", pi, SEED).name] = (
            max(bs, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])) if bs else None
        )
    flat, owner = [], []
    for k, im in enumerate(ims):
        for c in crops(det, bgr(im)):
            flat.append(c)
            owner.append(k)
    del det
    print(f"{len(ims)} images, {len(flat)} crops", flush=True)
    rd = Reader("v4", device)
    free = []
    for i in range(0, len(flat), 64):
        free += rd.free(flat[i : i + 64])
        rd.vision.clear()
    del rd
    torch.cuda.empty_cache()
    per = [[] for _ in recs]
    for k, f in zip(owner, free):
        per[k].append(f)
    er = EnRef(device, RENDER / "enref", ref_boxes)
    for rec, f, bs in zip(recs, per, boxes):
        rec["v4"] = {"free": f, **judge(rec["text"], f, None)}
        # crops() order: one per detector box, then the whole image
        rec["reads"] = [{"box": b, "vl": r} for b, r in zip(bs, f)] + [
            {"whole": True, "vl": f[-1] if f else ""}
        ]
        cos, cos_out, iou = er.score(rec)
        rec.update(en_cos=cos, en_cos_out=cos_out, box_iou=iou)
    del er
    mean = lambda rs, k: round(sum(r[k] for r in rs) / len(rs), 3)  # noqa: E731

    def row(rs):
        return {
            "exact": f"{sum(r['v4']['free_hit'] for r in rs)}/{len(rs)}",
            **f1_row(rs),
            "en_cos": mean(rs, "en_cos"),
            "en_cos_out": mean(rs, "en_cos_out"),
            "box_iou": mean(rs, "box_iou"),
            "no_box": sum(not r["reads"][:-1] for r in rs),
        }

    tab = {}
    for arm in arms:
        rs = [r for r in recs if r["arm"] == arm]
        tab[f"{arm} all"] = row(rs)
        for n in range(1, 6):
            tab[f"{arm} {n}"] = row([r for r in rs if syllables(r["text"]) == n])
    out = make_run_dir("cjk_ko", label="native-read", root=HOME / "results")
    with (out / "reads.jsonl").open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    at = {(r["arm"], r["i"]): (r, im) for r, im in zip(recs, ims)}
    ref_at = dict(zip(refs, ref_ims))
    tiles = []
    for it in its:
        tiles.append((ref_at[it["pi"]], [f"p{it['pi']} EN {EN_WORD}", ""]))
        for a in arms:
            r, im = at[a, it["i"]]
            tiles.append(
                (
                    im,
                    [
                        f"{it['text']}  {a}",
                        f"{whole(r)}  cos_out {r['en_cos_out']:.2f}",
                    ],
                )
            )
    cols = len(arms) + 1
    sheets = []
    for k in range(0, len(tiles), 8 * cols):
        fn = out / f"sheet_{k // (8 * cols)}.png"
        contact_sheet(tiles[k : k + 8 * cols], fn, thumb=256, cols=cols)
        sheets.append(fn.name)
    write_result(
        out,
        script=__file__,
        args={"arms": arms, "items": list(ITEMS), "en_word": EN_WORD, "seed": SEED},
        label="native-read",
        metrics={"per_arm": tab, "minutes": round((time.time() - t0) / 60, 1)},
        artifacts=sheets,
    )
    print(json.dumps(tab, ensure_ascii=False, indent=1), flush=True)
    print(f"→ {out}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("render", "score", "check"))
    p.add_argument(
        "--arms", default=DEFAULT_ARMS, help="of micro_read.ARMS, comma-separated"
    )
    a = p.parse_args()
    check_items()
    arms = [x for x in a.arms.split(",") if x]
    assert set(arms) <= set(ARMS), sorted(set(arms) - set(ARMS))
    missing = [x for x in arms if not (ARMS[x] / "trained.pt").is_file()]
    assert not missing, f"no trained.pt for {missing}"
    if a.mode == "check":
        for it in plan()[:9]:
            print(it["pi"], syllables(it["text"]), caption(it))
    else:
        (render if a.mode == "render" else score)(arms)
