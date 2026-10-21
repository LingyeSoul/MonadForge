#!/usr/bin/env python3
"""A colorize cache's holdout re-encoded with the fp8 TE, for ``colorize_eval.py``.

Each holdout TE file (``embeds``, ``embeds_plain``, the ``embeds_v{k}`` caption
variants) is encoded again from its own metadata (caption, variant captions,
condition page, size) through ``vision_te.load_te(precision="fp8")``: decoder
linears fp8, vision tower bf16 (``--te_file``: another fp8 file in the same format,
e.g. ``te_gptq.py``'s). Files land in ``<out>/holdout/`` with a
``holdout.json`` whose ``te`` points there (``ref`` still points at the source
cache's reference latents), so

    colorize_eval.py --lora <lora> --cache <out> --te_precision fp8

renders the fp8 arm. The empty prompt goes through the same TE into
``<out>/uncond.safetensors``, the uncond ``colorize_eval.py`` reads from the cache. Per key the log and ``<out>/te_error.json`` give rel L2 and
mean row cosine against the bf16 file, split into image-token rows and text rows.

    make daemon-run ARGS="project/moss_lora/src/colorize_te_fp8.py"
"""

import argparse
import json

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

from library.env import resolve_under_home
from library.moss import cache as moss_cache
from library.moss.cache import read_te
from library.moss.loader import empty_cache, load_tokenizer
from library.moss.scan import TE_SUFFIX
from vision_te import encode, load_te


def error(x: torch.Tensor, ref: torch.Tensor, rows: slice) -> dict:
    x, ref = x[rows].float(), ref[rows].float()
    return {
        "rel_l2": round(((x - ref).norm() / ref.norm()).item(), 5),
        "cos_mean": round(F.cosine_similarity(x, ref, dim=-1).mean().item(), 5),
        "cos_min": round(F.cosine_similarity(x, ref, dim=-1).min().item(), 5),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="post_image_dataset/moss/moss_colorize_3k")
    ap.add_argument("--out", default="")
    ap.add_argument("--te_file", default="", help="fp8 TE file; default the catalog's")
    args = ap.parse_args()
    src = resolve_under_home(args.cache)
    out = resolve_under_home(args.out or f"{args.cache}_fp8")
    (out / "holdout").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    items = json.loads((src / "holdout.json").read_text())

    tok = load_tokenizer("")
    te = load_te(tok, device, precision="fp8", fp8_path=args.te_file or None)
    report, records = {}, []
    for it in items:
        with safe_open(it["te"], "pt") as f:
            meta = f.metadata()
            keys = set(f.keys())
        w, h = it["width"], it["height"]
        page = moss_cache.fit(moss_cache.load_rgb(it["condition"]), w, h)
        embeds, n_img, img_end = encode(
            tok, te, page, meta["caption"], w * h, device, prefix=True
        )
        plain, _ = encode(tok, te, page, "", w * h, device)
        variants, prefix = [], 0
        for k in range(1, 1 + sum(key.startswith("embeds_v") for key in keys)):
            text = meta[f"caption_v{k}"]
            e, _, prefix = encode(tok, te, page, text, w * h, device, prefix=True)
            variants.append((text, e.to("cpu", torch.bfloat16)))
        tensors = {
            "embeds": embeds.to("cpu", torch.bfloat16).contiguous(),
            "embeds_plain": plain.to("cpu", torch.bfloat16).contiguous(),
        }
        var, var_meta = moss_cache.variant_entries(variants, prefix)
        path = out / "holdout" / f"{it['stem']}{TE_SUFFIX}"
        save_file(
            tensors | var,
            str(path),
            metadata={
                **{k: v for k, v in meta.items() if not k.startswith("caption_v")},
                "encoder": "Qwen3.5-9B fp8 + vision bf16, image then colour caption",
                **var_meta,
            },
        )

        # Image rows are one contiguous block ending at ``<|vision_end|>``.
        img = slice(img_end - 1 - n_img, img_end - 1)
        txt = slice(img_end, None)
        per = {}
        for key, x in [
            ("embeds", tensors["embeds"]),
            ("embeds_plain", tensors["embeds_plain"]),
        ] + [(f"embeds_v{k}", e) for k, (_, e) in enumerate(variants, 1)]:
            ref = read_te(it["te"], key)
            per[key] = {"all": error(x, ref, slice(None)), "image": error(x, ref, img)}
            if x.shape[0] > img_end:
                per[key]["text"] = error(x, ref, txt)
        report[it["stem"]] = {"n_image": n_img, "n_rows": embeds.shape[0], **per}
        e = per["embeds"]
        print(
            f"[fp8] {it['stem']}: {embeds.shape[0]} rows ({n_img} image) | embeds rel L2 "
            f"all {e['all']['rel_l2']:.4f} image {e['image']['rel_l2']:.4f} "
            f"text {e.get('text', {}).get('rel_l2', float('nan')):.4f} | plain "
            f"{per['embeds_plain']['all']['rel_l2']:.4f}",
            flush=True,
        )
        records.append({**it, "te": str(path)})
    uncond, _ = encode(tok, te, None, "", 0, device)
    save_file(
        {"embeds": uncond.to("cpu", torch.bfloat16).contiguous()},
        str(out / "uncond.safetensors"),
    )
    del te, tok
    empty_cache()

    (out / "holdout.json").write_text(json.dumps(records, indent=2))
    (out / "te_error.json").write_text(json.dumps(report, indent=2))
    for part in ("all", "image", "text"):
        vals = [
            r["embeds"][part]["rel_l2"] for r in report.values() if part in r["embeds"]
        ]
        print(
            f"[fp8] embeds {part}: mean rel L2 {sum(vals) / len(vals):.4f}", flush=True
        )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
