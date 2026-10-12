#!/usr/bin/env python
"""micro_read — the micro arms (A / B / B′, user 10-11) read by glyph.

``render`` (GPU): every read item under every arm, alone in a KO speech
bubble through the reseed ruler's ``Renderer`` (512², seed 0) →
``output/cjk_ko/micro_read/render/<arm>/bubble_<text>_s0.png``. A render on
disk is kept, so the floor renders once.

``score`` (GPU): v4's free read of every render (``jamo_vl``'s ``Reader``,
each image cut as the ruler cuts it) → ``results/<ts>-micro-read/``:
``reads.jsonl``, ``result.json`` — per arm × group: ``exact`` (a crop reads
the target), glyph / jamo P R F1 of the whole-image read and the jamo F1
against the group's other targets (chance; ``jamo_vl_sheet.f1_row``) — and
``sheet_<group>_<i>.png`` (the arms side by side, the whole-image read under
each).

Items (``assets/micro192.json``, ``corpus.py --rows 192``): ``trained``
every third of the 192 by frequency (64), ``next`` (ranks 193–256: no line of
the arms' data holds them; A leaves them at the floor, B / B′ move them
through the factors), ``held`` (the ruler's strings on the 192: 27).
``--seeds`` (default 0): one render per item and seed, each read; ``exact``
and the F1s count every render, the sheets show the first seed.
Arms (``ARMS``): ``floor`` = jamo_j256_ko (every arm's start), ko_micro_a /
_b / _bp, jamo_j1024_ko, jamo_j256_ko_g44, jamo_j256_ko_part; ``--arms`` picks some. ``--groups`` (default the
three above): also ``H`` (the jamo line's 32 held syllables) and ``jamo``
(the 51 compat jamo, by eye — v4 reads no lone jamo).

    make daemon-run ARGS="--stall-timeout 900 project/cjk_ko/probes/micro_read.py render"
    make daemon-run ARGS="--stall-timeout 900 project/cjk_ko/probes/micro_read.py score"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HOME = Path(__file__).resolve().parents[1]
RESEED = HOME.parent / "cjk_anima_reseed"
sys.path.insert(0, str(RESEED))
sys.path.insert(0, str(RESEED / "probes"))
os.environ["ANIMA_VOCAB_PACK"] = "models/vocab_packs/anima_cjk_vocab_pack_punct"
os.environ["ANIMA_VOCAB_GLYPH_ROUTE"] = "1"
os.environ["ANIMA_VOCAB_GLYPH_ROUTE_KO"] = "1"

from reseed import OUT, bootstrap  # noqa: E402

bootstrap()

SETS = HOME / "assets" / "micro192.json"
KO_OUT = OUT.parent / "cjk_ko"
RENDER = KO_OUT / "micro_read" / "render"
ARMS = {
    "floor": OUT / "jamo_j256_ko",
    "ko_micro_a": KO_OUT / "ko_micro_a",
    "ko_micro_b": KO_OUT / "ko_micro_b",
    "ko_micro_bp": KO_OUT / "ko_micro_bp",
    "jamo_j1024_ko": KO_OUT / "jamo_j1024_ko",
    "jamo_j256_ko_g44": KO_OUT / "jamo_j256_ko_g44",
    "jamo_j256_ko_part": KO_OUT / "jamo_j256_ko_part",
}
GROUPS = ("trained", "next", "held", "H", "jamo")
SIZE = (512, 512)
SEED = 0


def items(groups=GROUPS) -> dict:
    """{group: [text, …]}; ``H`` = the jamo line's 32 held syllables
    (``jamo_sets.json``), ``jamo`` = the 51 compat jamo (v4 reads none: by eye)."""
    from reseed.jamo import COMPAT

    s = json.loads(SETS.read_text(encoding="utf-8"))
    js = json.loads((RESEED / "assets" / "jamo_sets.json").read_text(encoding="utf-8"))
    out = {
        "trained": list(s["trained"])[::3],
        "next": list(s["next"]),
        "held": [t for t, _bin in s["held"]],
        "H": list(js["H"]),
        "jamo": list(COMPAT),
    }
    return {g: out[g] for g in groups}


def render_file(arm: str, t: str, seed: int = SEED) -> Path:
    return RENDER / arm / f"bubble_{t}_s{seed}.png"


def render(arms: list, groups: tuple, seeds: list) -> None:
    from common.prompts import TPL_BUBBLE
    from reseed.pools import relang

    from eval.ruler import VIEW
    from eval.ruler.arms import PACK_ARMS, tables
    from eval.ruler.render import Renderer, check

    VIEW.pack = "punct"
    PACK_ARMS["punct"].update({a: ARMS[a] for a in arms})
    _, tabs = tables(arms + ["retrain_kana"])
    r = Renderer()
    info = {"check_rk": check(r, tabs["retrain_kana"])}
    its = [t for ts in items(groups).values() for t in ts]
    t0 = time.time()
    jobs = [(t, sd) for sd in seeds for t in its]
    for arm in arms:
        r.set_arm(tabs[arm])
        for n, (t, sd) in enumerate(jobs):
            r.render(
                render_file(arm, t, sd),
                relang(TPL_BUBBLE.format(t), "korean"),
                sd,
                SIZE,
            )
            if n % 25 == 0 or n == len(jobs) - 1:
                print(
                    f"  {arm}: {n + 1} / {len(jobs)} ({(time.time() - t0) / 60:.1f} min)",
                    flush=True,
                )
    info["minutes"] = round((time.time() - t0) / 60, 1)
    print(
        f"→ {RENDER}: {len(arms)} arms × {len(its)} items × seeds {seeds}; {info}",
        flush=True,
    )


def score(arms: list, groups: tuple, seeds: list) -> None:
    import numpy as np
    import torch
    from anime_tools.ocr.animetext import AnimeTextDetector
    from common.readers import contact_sheet
    from jamo_vl import Reader, judge
    from jamo_vl_sheet import f1_row, whole
    from ko_reader_cal import crops
    from PIL import Image

    from bench._common import make_run_dir, write_result

    t0 = time.time()
    groups = items(groups)
    recs, ims = [], []
    for arm in arms:
        for g, ts in groups.items():
            for sd in seeds:
                for t in ts:
                    fn = render_file(arm, t, sd)
                    assert fn.is_file(), f"{fn}: render it first"
                    recs.append({"arm": arm, "group": g, "text": t, "seed": sd})
                    ims.append(Image.open(fn).convert("RGB"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    det = AnimeTextDetector.load(device=device)
    flat, owner = [], []
    for k, im in enumerate(ims):
        for c in crops(det, np.ascontiguousarray(np.array(im)[:, :, ::-1])):
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
    for rec, f in zip(recs, per):
        rec["v4"] = {"free": f, **judge(rec["text"], f, None)}
    tab = {}
    for arm in arms:
        for g in groups:
            rs = [r for r in recs if r["arm"] == arm and r["group"] == g]
            tab[f"{arm} {g}"] = {
                "exact": f"{sum(r['v4']['free_hit'] for r in rs)}/{len(rs)}",
                **f1_row(rs),
            }
    out = make_run_dir("cjk_ko", label="micro-read", root=HOME / "results")
    with (out / "reads.jsonl").open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    sheets = []
    at = {
        (r["arm"], r["text"]): (r, im)
        for r, im in zip(recs, ims)
        if r["seed"] == seeds[0]
    }
    for g, ts in groups.items():
        tiles = [
            (at[a, t][1], [f"{t}  {a.removeprefix('ko_micro_')}", whole(at[a, t][0])])
            for t in ts
            for a in arms
        ]
        cols, per_sheet = 2 * len(arms), 8 * len(arms)
        for k in range(0, len(tiles), per_sheet):
            fn = out / f"sheet_{g}_{k // per_sheet}.png"
            contact_sheet(tiles[k : k + per_sheet], fn, thumb=256, cols=cols)
            sheets.append(fn.name)
    write_result(
        out,
        script=__file__,
        args={"arms": arms, "groups": list(groups), "seeds": seeds},
        label="micro-read",
        metrics={"per_arm": tab, "minutes": round((time.time() - t0) / 60, 1)},
        artifacts=sheets,
    )
    print(json.dumps(tab, ensure_ascii=False, indent=1), flush=True)
    print(f"→ {out}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("render", "score"))
    p.add_argument("--arms", default=",".join(ARMS), help="of ARMS, comma-separated")
    p.add_argument("--groups", default="trained,next,held", help="of GROUPS")
    p.add_argument("--seeds", default=str(SEED), help="render seeds, comma-separated")
    a = p.parse_args()
    seeds = [int(x) for x in a.seeds.split(",") if x]
    groups = tuple(x for x in a.groups.split(",") if x)
    assert set(groups) <= set(GROUPS), sorted(set(groups) - set(GROUPS))
    arms = [x for x in a.arms.split(",") if x]
    assert set(arms) <= set(ARMS), sorted(set(arms) - set(ARMS))
    missing = [x for x in arms if not (ARMS[x] / "trained.pt").is_file()]
    assert not missing, f"no trained.pt for {missing}"
    (render if a.mode == "render" else score)(arms, groups, seeds)
