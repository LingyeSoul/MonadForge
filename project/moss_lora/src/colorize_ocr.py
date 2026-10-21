#!/usr/bin/env python3
"""The colorize cache with the B&W page's OCR in the caption (``colorize_ocr.md``).

Reads ``--src``'s train TE files and ``--holdout_from``'s ``holdout.json``, OCRs
each B&W condition (never the target) with the anime_tools path — the AnimeText
detector, every box read by the manga VL reader, ASCII lines kept — and keeps the
``usable_lines``, speech and SFX alike in reading order, an exact repeat dropped
(its first box kept). Every TE file is written with two captioned encodes:

    embeds       layout JSON: the colour caption through ``caption_to_layout``, one
                 ``{"label":"text","caption":"text","text":…,"bbox":…}`` item per line
    embeds_v1    {colour caption}. OCR result as follows "…" [y0, x0, y1, x1], …
    embeds_plain the page alone

Boxes are MOSS's ``abs_centralized`` ``yxyx`` in the generation canvas's pixels
(``centered_box``; the condition is read at the canvas size). A page with no
usable line takes the layout of its colour caption and the colour caption itself.
``train.py --caption_variant_rate`` picks between the two captions.

All three go through ``--te_precision`` (default fp8: the catalog's GPTQ-rounded
file, ``--te_file`` for another), and so does ``<out>/uncond.safetensors``, the
empty prompt ``colorize_eval.py --te_precision fp8`` reads from the cache.

``--out`` gets ``--src``'s latents and refs as symlinks, never its TE files, and
each holdout page's ref (``--holdout_from``'s, else ``--src``'s). The
reads land in ``<out>/ocr.json`` and a rerun reuses them; TE files already
written are kept.

    make daemon-run ARGS="--stall-timeout 900 project/moss_lora/src/colorize_ocr.py"
"""

import argparse
import json
import time
from pathlib import Path

import torch
from anime_tools.captions.layout_json import caption_to_layout, centered_box
from anime_tools.captions.ocr_sidecar import usable_lines
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file

from library.env import resolve_under_home
from library.moss import cache as moss_cache
from library.moss.cache import _tag_knowledge_base
from library.moss.loader import empty_cache, load_tokenizer
from library.moss.scan import LATENT_SUFFIX, REF_SUFFIX, TE_SUFFIX
from vision_te import encode, load_te


def dedupe(lines: list[list]) -> list[tuple[str, list[int]]]:
    """``[(text, (x0, y0, x1, y1))]``, ``"`` → ``”``, first of each text kept."""
    seen, keep = set(), []
    for text, box in lines:
        text = text.replace('"', "”").strip()
        if text and text not in seen:
            seen.add(text)
            keep.append((text, box))
    return keep


def layout_caption(caption, lines, size, kb) -> str:
    layout = caption_to_layout(caption, kb)
    items = [
        {"label": "text", "caption": "text", "text": t, "bbox": centered_box(b, size)}
        for t, b in lines
    ]
    if items:
        layout["items"] = layout.get("items", []) + items
        layout["bbox_coord_space"] = "abs_centralized"
        layout["bbox_format"] = "yxyx"
    return json.dumps(layout, ensure_ascii=False, separators=(",", ":"))


def text_caption(caption, lines, size) -> str:
    if not lines:
        return caption
    clause = "OCR result as follows " + ", ".join(
        f'"{t}" [{", ".join(map(str, centered_box(b, size)))}]' for t, b in lines
    )
    return f"{caption}. {clause}" if caption else clause


def read_pages(conditions: list[Path], device: str) -> list[list]:
    """Per condition, the usable lines as ``[text, [x0, y0, x1, y1]]`` in reading
    order, in the condition's pixels."""
    from anime_tools.ocr import load_ocr
    from anime_tools.ocr.reread import RereadEngine
    from anime_tools.ocr.sfx import SfxReader

    reader = SfxReader.load(device=device)
    engine = RereadEngine(
        engine=load_ocr(device=device),
        read_boxes=reader.read_boxes_scored,
        resized_dir=Path("/"),
        skip_en=False,
    )
    out, t0 = [], time.time()
    for i, lines in enumerate(engine.read_iter(conditions), 1):
        out.append([[ln.text, list(ln.box)] for ln in usable_lines(lines)])
        if i % 50 == 0 or i == len(conditions):
            print(f"  ocr {i}/{len(conditions)}, {time.time() - t0:.0f}s", flush=True)
    del engine, reader
    empty_cache()
    return out


def encode_ref(r: dict, path: Path, scale: int, device) -> None:
    """``colorize_cache.py``'s reference latent: the page fitted to the canvas, then
    to 1/``scale`` of its sides (multiples of 16), VAE-encoded."""
    from library.moss.loader import load_vae
    from library.moss.model import encode as vae_encode

    w, h = r["width"], r["height"]
    rw = max(16, round(w / scale / 16) * 16)
    rh = max(16, round(h / scale / 16) * 16)
    page = moss_cache.fit(moss_cache.load_rgb(Path(r["condition"])), w, h)
    vae = load_vae("", device=device)
    latent = vae_encode(moss_cache.to_pixels(moss_cache.fit(page, rw, rh)), vae)[0]
    save_file(
        {"latent": latent.cpu().contiguous()},
        str(path),
        metadata={
            "width": str(rw),
            "height": str(rh),
            "target": f"{w}x{h}",
            "condition": r["condition"],
        },
    )
    del vae
    empty_cache()
    print(f"  ref {r['stem']}: {w}x{h} -> {rw}x{rh}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="post_image_dataset/moss/colorize_ref")
    ap.add_argument("--holdout_from", default="post_image_dataset/moss/colorize_manga")
    ap.add_argument("--out", default="post_image_dataset/moss/colorize_ocr")
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--te_precision", default="fp8", choices=("bf16", "fp8"))
    ap.add_argument("--te_file", default="", help="fp8 TE file; default the catalog's")
    ap.add_argument(
        "--ref_scale", type=int, default=2, help="for a missing holdout ref"
    )
    args = ap.parse_args()
    src = resolve_under_home(args.src)
    out = resolve_under_home(args.out)
    (out / "holdout").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    for f in src.glob("*.safetensors"):
        if f.name.endswith((LATENT_SUFFIX, REF_SUFFIX)):
            dst = out / f.name
            if not dst.exists():
                dst.symlink_to(f.resolve())

    # (stem, source TE, condition, target, colour caption, output TE)
    jobs = []
    for te_path in sorted(src.glob(f"*{TE_SUFFIX}")):
        with safe_open(str(te_path), "pt") as f:
            meta = f.metadata()
        stem = te_path.name[: -len(TE_SUFFIX)]
        jobs.append(
            (
                stem,
                te_path,
                Path(meta["condition"]),
                Path(meta["source"]),
                meta["caption"],
                out / te_path.name,
            )
        )
    holdout = json.loads(
        (resolve_under_home(args.holdout_from) / "holdout.json").read_text()
    )
    for r in holdout:
        jobs.append(
            (
                r["stem"],
                Path(r["te"]),
                Path(r["condition"]),
                Path(r["target"]),
                r["caption"],
                out / "holdout" / f"{r['stem']}{TE_SUFFIX}",
            )
        )
    print(f"{len(jobs)} pages, {len(holdout)} held out", flush=True)

    ocr_path = out / "ocr.json"
    reads = json.loads(ocr_path.read_text()) if ocr_path.exists() else {}
    missing = [j for j in jobs if j[0] not in reads]
    if missing:
        for j, lines in zip(missing, read_pages([j[2] for j in missing], "cuda")):
            reads[j[0]] = lines
        ocr_path.write_text(json.dumps(reads, ensure_ascii=False, indent=1))
    print(
        f"{sum(bool(reads[j[0]]) for j in jobs)} of {len(jobs)} pages carry OCR lines",
        flush=True,
    )

    kb = _tag_knowledge_base()
    captions = {}
    for stem, _, cond, target, caption, _ in jobs:
        with Image.open(target) as im:
            w, h = moss_cache.target_size(*im.size, args.resolution)
        with Image.open(cond) as im:
            cw, ch = im.size
        lines = [
            (t, [b[0] * w / cw, b[1] * h / ch, b[2] * w / cw, b[3] * h / ch])
            for t, b in dedupe(reads[stem])
        ]
        captions[stem] = (
            layout_caption(caption, lines, (w, h), kb),
            text_caption(caption, lines, (w, h)),
            (w, h),
        )

    todo = [j for j in jobs if not j[-1].exists()]
    uncond_path = out / "uncond.safetensors"
    if todo or not uncond_path.exists():
        tok = load_tokenizer("")
        te = load_te(
            tok, device, precision=args.te_precision, fp8_path=args.te_file or None
        )
        encoder = f"Qwen3.5-9B {args.te_precision}" + (
            f" ({args.te_file})" if args.te_file else ""
        )
        t0 = time.time()
        for i, (stem, _, cond, target, caption, path) in enumerate(todo, 1):
            layout, text, (w, h) = captions[stem]
            page = moss_cache.fit(moss_cache.load_rgb(cond), w, h)
            embeds, n_img, prefix = encode(
                tok, te, page, layout, w * h, device, prefix=True
            )
            text_embeds, _, _ = encode(tok, te, page, text, w * h, device, prefix=True)
            plain, _ = encode(tok, te, page, "", w * h, device)
            var, var_meta = moss_cache.variant_entries([(text, text_embeds)], prefix)
            save_file(
                {
                    "embeds": embeds.to("cpu", torch.bfloat16).contiguous(),
                    "embeds_plain": plain.to("cpu", torch.bfloat16).contiguous(),
                    **var,
                },
                str(path),
                metadata={
                    "caption": layout,
                    "colour_caption": caption,
                    "n_text": str(embeds.shape[0]),
                    "n_text_plain": str(plain.shape[0]),
                    "n_image_tokens": str(n_img),
                    "condition": str(cond),
                    "source": str(target),
                    "encoder": f"{encoder} + vision bf16, image then layout / text + OCR",
                    **var_meta,
                },
            )
            print(
                f"  te {i}/{len(todo)} {stem}: {embeds.shape[0]} layout / "
                f"{text_embeds.shape[0]} text rows ({n_img} image), "
                f"{time.time() - t0:.0f}s",
                flush=True,
            )
        uncond, _ = encode(tok, te, None, "", 0, device)
        save_file(
            {"embeds": uncond.to("cpu", torch.bfloat16).contiguous()},
            str(uncond_path),
            metadata={"encoder": encoder},
        )
        del te, tok
        empty_cache()

    records = []
    for r in holdout:
        rec = {
            **r,
            "caption": captions[r["stem"]][0],
            "caption_v1": captions[r["stem"]][1],
            "te": str(out / "holdout" / f"{r['stem']}{TE_SUFFIX}"),
        }
        if "ref" in r:
            # ``--holdout_from`` may lack a page's ref; ``--src``'s is the same
            # encode, and a page neither has is encoded here.
            ref = Path(r["ref"])
            dst = out / "holdout" / ref.name
            if not dst.exists():
                if not ref.exists():
                    ref = src / "holdout" / ref.name
                if ref.exists():
                    dst.symlink_to(ref.resolve())
                else:
                    encode_ref(r, dst, args.ref_scale, device)
            rec["ref"] = str(dst)
        records.append(rec)
    (out / "holdout.json").write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
