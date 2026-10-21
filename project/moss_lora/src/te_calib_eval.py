#!/usr/bin/env python3
"""Per-category TE error of fp8 files on ``te_calib.py``'s ``eval`` split.

Each entry is encoded by the bf16 TE (the reference) and by every ``--fp8 name=path``
file, all with the vision tower on the card (``vision_te.load_te``). Per arm and
category: rel L2 over all rows, over the image-token rows and over the text rows
(image entries), the mean row cosine and the share of rows under cosine 0.9.

    make daemon-run ARGS="project/moss_lora/src/te_calib_eval.py \
        --fp8 rtn=models/moss_image/fp8/qwen35_9b_fp8_e4m3fn.safetensors \
        --fp8 v2=models/moss_image/fp8_gptq_v2/qwen35_9b_fp8_e4m3fn_gptq.safetensors"
"""

import argparse
import json
import time

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from library.env import resolve_under_home
from library.moss import cache as moss_cache
from library.moss.loader import empty_cache, load_tokenizer
from vision_te import load_te, te_inputs

MANIFEST = "post_image_dataset/moss/_te_calib_v2/manifest.json"


@torch.no_grad()
def encode_all(tok, te, entries, device):
    out = {}
    for e in entries:
        if e["image"]:
            w, h = e["width"], e["height"]
            page = moss_cache.fit(moss_cache.load_rgb(e["image"]), w, h)
            inp = te_inputs(tok, page, e["text"], w * h, device)
        else:
            inp = te_inputs(tok, None, e["text"], 0, device)
        hidden = te(**inp, use_cache=False, return_dict=True).last_hidden_state[0]
        out[e["id"]] = hidden.to("cpu", torch.bfloat16)
        if "mm_token_type_ids" in inp:
            out[f"{e['id']}#img"] = inp["mm_token_type_ids"][0].cpu().bool()
    return out


def stats(x: torch.Tensor, ref: torch.Tensor) -> dict:
    x, ref = x.float(), ref.float()
    cos = F.cosine_similarity(x, ref, dim=-1)
    return {
        "d2": (x - ref).pow(2).sum().item(),
        "r2": ref.pow(2).sum().item(),
        "cos_sum": cos.sum().item(),
        "low": int((cos < 0.9).sum()),
        "rows": x.shape[0],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--fp8", action="append", default=[], help="name=path, repeatable")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = torch.device("cuda")
    out = resolve_under_home(
        args.out or f"output/tests/te_calib_eval/{time.strftime('%Y%m%d-%H%M')}"
    )
    out.mkdir(parents=True, exist_ok=True)
    entries = [
        e
        for e in json.loads(resolve_under_home(args.manifest).read_text())["entries"]
        if e["split"] == "eval"
    ]
    tok = load_tokenizer("")
    arms = {"bf16": None} | dict(f.split("=", 1) for f in args.fp8)
    embeds = {}
    for arm, path in arms.items():
        cached = out / f"embeds_{arm}.safetensors"
        if cached.exists():
            embeds[arm] = load_file(str(cached))
            continue
        if path is None:
            te = load_te(tok, device)
        else:
            te = load_te(
                tok, device, precision="fp8", fp8_path=resolve_under_home(path)
            )
        t0 = time.time()
        embeds[arm] = encode_all(tok, te, entries, device)
        print(f"[{arm}] {len(entries)} entries in {time.time() - t0:.0f}s", flush=True)
        save_file({k: v.contiguous() for k, v in embeds[arm].items()}, str(cached))
        del te
        empty_cache()

    ref = embeds["bf16"]
    cats = sorted({e["category"] for e in entries})
    table = {}
    for arm in arms:
        if arm == "bf16":
            continue
        for cat in cats + ["all"]:
            acc = {}
            for e in entries:
                if cat != "all" and e["category"] != cat:
                    continue
                x, r = embeds[arm][e["id"]], ref[e["id"]]
                parts = {"all": slice(None)}
                if f"{e['id']}#img" in ref:
                    img = ref[f"{e['id']}#img"]
                    parts |= {"image": img, "text": ~img}
                for part, sel in parts.items():
                    s = stats(x[sel], r[sel])
                    a = acc.setdefault(part, dict.fromkeys(s, 0.0))
                    for k, v in s.items():
                        a[k] += v
            table.setdefault(arm, {})[cat] = {
                part: {
                    "rel_l2": round((a["d2"] / a["r2"]) ** 0.5, 5),
                    "cos_mean": round(a["cos_sum"] / a["rows"], 5),
                    "low_share": round(a["low"] / a["rows"], 5),
                    "rows": int(a["rows"]),
                }
                for part, a in acc.items()
            }

    names = [a for a in arms if a != "bf16"]
    print(
        f"{'category':14s} "
        + " ".join(f"{n:>9s}" for n in names)
        + "   (rel L2, all rows)"
    )
    for cat in cats + ["all"]:
        row = " ".join(f"{table[n][cat]['all']['rel_l2']:9.4f}" for n in names)
        print(f"{cat:14s} {row}", flush=True)
        if "image" in table[names[0]][cat]:
            for part in ("image", "text"):
                row = " ".join(f"{table[n][cat][part]['rel_l2']:9.4f}" for n in names)
                print(f"  {part:12s} {row}", flush=True)
    print(
        "rows with cos < 0.9: "
        + ", ".join(f"{n} {table[n]['all']['all']['low_share']:.2%}" for n in names)
    )
    (out / "result.json").write_text(
        json.dumps(
            {"args": vars(args), "entries": len(entries), "table": table}, indent=1
        )
    )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
