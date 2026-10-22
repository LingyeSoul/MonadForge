#!/usr/bin/env python3
"""The colorize cache for a TE-only MOSS LoRA: the B&W page enters through
Qwen3.5's vision tower and as a VAE reference, the colour original is the target.

Pairs are EasyControl colorize's: the condition is the synthetic mangafied page in
``post_image_dataset/easycontrol/colorize/staging/<dir>/<stem>.png``, the target the
colour original at the same path under ``post_image_dataset/resized``; ``--manga N``
adds N pages of the ``manga_color`` slice (hitocrawl's full-colour works, JA / EN /
KO editions) as ``manga__{anchor}_{lang}_{page}``. Per pair, in a flat cache
``train.py`` reads unchanged:

    {stem}_{W}x{H}.moss_latent.safetensors   the colour target
    {stem}.moss_ref.safetensors              the page VAE-encoded at 1/``--ref_scale``
                                             of the target's sides (multiples of 16)
    {stem}.moss_te.safetensors               the page, then one of three captions:
        embeds        OCR caption: layout JSON, the colour caption through
                      ``caption_to_layout`` plus one ``{"label":"text",…,"bbox":…}``
                      item per OCR line
        embeds_v{k}   sparse caption: the colour caption shuffled with
                      ``--tag_dropout`` of its tags dropped, no OCR (k = 1 …
                      ``--variants`` - 1; ``--caption_variant_rate``)
        embeds_plain  the page alone (``--plain_caption_rate``)

The colour caption is ``color_caption.filter_to_colors_and_protected`` (copyright +
comic prefix, colour tags). The OCR reads the condition, never the target: the
AnimeText detector, every box read by the manga VL reader, ASCII lines kept,
speech and SFX alike in reading order, an exact repeat dropped (its first box
kept). Its floors are looser than the stage's (``LINE_MIN_DET`` 0.6, glyph 16):
the detector scores horizontal English balloons 0.3–0.55 while the reader reads
them right, so a box down to the detector's 0.25 is kept when the read scores
``LOW_DET_SCORE``, and an all-ASCII line's glyph floor is ``LATIN_MIN_GLYPH``.
Boxes are MOSS's ``abs_centralized`` ``yxyx`` in the canvas's pixels. A page with no
usable line keeps the layout of its colour caption.

Every encode goes through the fp8 TE (the catalog's file, ``--te_file`` for
another), as does ``<out>/uncond.safetensors``, the eval's empty prompt. Holdout:
``--holdout`` illustration pairs and one page each of ``--manga_holdout`` works (no
page of those works, in any edition, trains); their files land in ``<out>/holdout/``
with ``holdout.json``. OCR reads land in ``<out>/ocr.json``; a rerun reuses them and
every file already written.

    make daemon-run ARGS="--stall-timeout 1800 project/moss_lora/src/colorize_cache.py --out post_image_dataset/moss/<name>"
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from anime_tools.captions.layout_json import caption_to_layout, centered_box
from anime_tools.captions.ocr_sidecar import DEFAULT_MIN_GLYPH, OcrLine
from anime_tools.captions.variants import generate_caption_variants
from anime_tools.ocr.animetext import DEFAULT_CONF
from anime_tools.ocr.reread import LINE_MIN_DET
from PIL import Image
from safetensors.torch import save_file

from library.env import resolve_under_home
from library.moss import cache as moss_cache
from library.moss.cache import _tag_knowledge_base
from library.moss.loader import empty_cache, load_tokenizer
from library.moss.scan import REF_SUFFIX, TE_SUFFIX
from vision_te import encode, encode_variants, load_te

sys.path.insert(0, str(resolve_under_home("easycontrol_adapters/colorization")))
from color_caption import (  # noqa: E402
    filter_to_colors_and_protected,
    is_comic_tag,
    load_copyright_tags,
)

STAGING = "post_image_dataset/easycontrol/colorize/staging"
RESIZED = "post_image_dataset/resized"
MANGA = "post_image_dataset/easycontrol/colorize/manga_color"
LOW_DET_SCORE = 0.9  # reader score a box under LINE_MIN_DET needs
LATIN_MIN_GLYPH = 12.0


def find_pairs():
    staging, resized = resolve_under_home(STAGING), resolve_under_home(RESIZED)
    out = []
    for cond in sorted(staging.rglob("*.png")):
        rel = cond.relative_to(staging)
        target = resized / rel
        if target.exists() and target.with_suffix(".txt").exists():
            out.append((f"{rel.parent.name}__{rel.stem}", cond, target))
    return out


def find_manga_pairs():
    """``manga_color/{staging,resized}/{ja,en,ko}/{anchor}_{lang}_{page}.png``."""
    root = resolve_under_home(MANGA)
    out = []
    for cond in sorted((root / "staging").rglob("*.png")):
        target = root / "resized" / cond.relative_to(root / "staging")
        if target.exists() and target.with_suffix(".txt").exists():
            out.append((f"manga__{cond.stem}", cond, target))
    return out


def split_manga(pairs, n, n_holdout, rng):
    """Hold out whole works (every edition of them), one page each; draw ``n`` from
    the rest."""
    works = {}
    for p in pairs:
        works.setdefault(p[2].stem.split("_")[0], []).append(p)
    anchors = sorted(works)
    rng.shuffle(anchors)
    holdout = [rng.choice(works[a]) for a in anchors[:n_holdout]]
    rest = [p for a in anchors[n_holdout:] for p in works[a]]
    rng.shuffle(rest)
    return rest[:n], holdout


def readable(ln: OcrLine) -> bool:
    latin = all(ch.isascii() for ch in ln.text if ch.isalpha())
    floor = LATIN_MIN_GLYPH if latin else DEFAULT_MIN_GLYPH
    sure = ln.det >= LINE_MIN_DET or ln.score >= LOW_DET_SCORE
    return bool(ln.text) and sure and ln.glyph_px >= floor


def read_pages(conditions: list[Path], device: str) -> list[list]:
    """Per condition, the readable lines as ``[text, [x0, y0, x1, y1]]`` in reading
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
        min_det=DEFAULT_CONF,
    )
    out, t0 = [], time.time()
    for i, lines in enumerate(engine.read_iter(conditions), 1):
        out.append([[ln.text, list(ln.box)] for ln in lines if readable(ln)])
        if i % 50 == 0 or i == len(conditions):
            print(f"  ocr {i}/{len(conditions)}, {time.time() - t0:.0f}s", flush=True)
    del engine, reader
    empty_cache()
    return out


def dedupe(lines: list[list]) -> list[tuple[str, list]]:
    """``[(text, box)]``, ``"`` → ``”``, first of each text kept."""
    seen, keep = set(), []
    for text, box in lines:
        text = text.replace('"', "”").strip()
        if text and text not in seen:
            seen.add(text)
            keep.append((text, box))
    return keep


def ocr_caption(caption, lines, size, kb) -> str:
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1667, help="illustration pairs")
    ap.add_argument("--manga", type=int, default=1667, help="manga pages")
    ap.add_argument("--holdout", type=int, default=8, help="illustration pairs")
    ap.add_argument("--manga_holdout", type=int, default=4, help="manga works")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--ref_scale", type=int, default=2)
    ap.add_argument("--variants", type=int, default=4, help="sparse captions + 1")
    ap.add_argument("--tag_dropout", type=float, default=0.8)
    ap.add_argument("--te_file", default="", help="fp8 TE file; default the catalog's")
    args = ap.parse_args()
    if args.variants < 2:
        ap.error("--variants must be at least 2: the eval reads embeds_v1")
    out = resolve_under_home(args.out)
    (out / "holdout").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    rng = random.Random(args.seed)
    pairs = find_pairs()
    rng.shuffle(pairs)
    holdout, train = pairs[: args.holdout], pairs[args.holdout :][: args.n]
    m_train, m_holdout = split_manga(
        find_manga_pairs(), args.manga, args.manga_holdout, rng
    )
    train += m_train
    holdout += m_holdout
    print(f"{len(train)} train ({len(m_train)} manga), {len(holdout)} held out")

    def te_path(stem, held):
        return out / ("holdout" if held else "") / f"{stem}{TE_SUFFIX}"

    def ref_path(stem, held):
        return out / ("holdout" if held else "") / f"{stem}{REF_SUFFIX}"

    # (stem, condition, target, held out, (w, h))
    jobs = []
    for held, rows in ((False, train), (True, holdout)):
        for stem, cond, target in rows:
            with Image.open(target) as im:
                size = moss_cache.target_size(*im.size, args.resolution)
            jobs.append((stem, cond, target, held, size))

    moss_cache.cache_latents(
        [moss_cache.Pair(stem, target, "", "") for stem, _, target in train],
        out,
        "",
        device,
        args.resolution,
        "fp32",
        False,
        False,
    )

    todo = [j for j in jobs if not ref_path(j[0], j[3]).exists()]
    if todo:
        from library.moss.loader import load_vae
        from library.moss.model import encode as vae_encode

        vae = load_vae("", device=device)
        for i, (stem, cond, _, held, (w, h)) in enumerate(todo, 1):
            rw = max(16, round(w / args.ref_scale / 16) * 16)
            rh = max(16, round(h / args.ref_scale / 16) * 16)
            page = moss_cache.fit(moss_cache.load_rgb(cond), w, h)
            latent = vae_encode(moss_cache.to_pixels(moss_cache.fit(page, rw, rh)), vae)
            save_file(
                {"latent": latent[0].cpu().contiguous()},
                str(ref_path(stem, held)),
                metadata={
                    "width": str(rw),
                    "height": str(rh),
                    "target": f"{w}x{h}",
                    "condition": str(cond),
                },
            )
            print(f"  ref {i}/{len(todo)} {stem}: {w}x{h} -> {rw}x{rh}", flush=True)
        del vae
        empty_cache()

    ocr_path = out / "ocr.json"
    reads = json.loads(ocr_path.read_text()) if ocr_path.exists() else {}
    missing = [j for j in jobs if j[0] not in reads]
    if missing:
        for j, lines in zip(missing, read_pages([j[1] for j in missing], "cuda")):
            reads[j[0]] = lines
        ocr_path.write_text(json.dumps(reads, ensure_ascii=False, indent=1))
    print(
        f"{sum(bool(reads[j[0]]) for j in jobs)} of {len(jobs)} pages carry OCR lines",
        flush=True,
    )

    copyright_tags = load_copyright_tags()
    kb = _tag_knowledge_base()
    # stem: (colour caption, OCR caption, sparse captions)
    captions = {}
    for stem, cond, target, _, (w, h) in jobs:
        colour = filter_to_colors_and_protected(
            target.with_suffix(".txt").read_text(encoding="utf-8").strip(),
            copyright_tags,
            keep_copyright=True,
            keep_comic=True,
        )
        with Image.open(cond) as im:
            cw, ch = im.size
        lines = [
            (t, [b[0] * w / cw, b[1] * h / ch, b[2] * w / cw, b[3] * h / ch])
            for t, b in dedupe(reads[stem])
        ]
        random.seed(f"{args.seed}:{stem}")
        sparse = generate_caption_variants(
            colour, args.variants, args.tag_dropout, is_comic_tag
        )[1:]
        captions[stem] = (colour, ocr_caption(colour, lines, (w, h), kb), sparse)

    todo = [j for j in jobs if not te_path(j[0], j[3]).exists()]
    uncond_path = out / "uncond.safetensors"
    if todo or not uncond_path.exists():
        tok = load_tokenizer("")
        te = load_te(tok, device, precision="fp8", fp8_path=args.te_file or None)
        encoder = "Qwen3.5-9B fp8" + (f" ({args.te_file})" if args.te_file else "")
        t0 = time.time()
        for i, (stem, cond, _, held, (w, h)) in enumerate(todo, 1):
            colour, ocr, sparse = captions[stem]
            page = moss_cache.fit(moss_cache.load_rgb(cond), w, h)
            (embeds, n_img, prefix), (plain, _, _), *rest = encode_variants(
                tok, te, page, [ocr, "", *sparse], w * h, device
            )
            var, var_meta = moss_cache.variant_entries(
                [
                    (t, e.to("cpu", torch.bfloat16))
                    for t, (e, _, _) in zip(sparse, rest)
                ],
                prefix,
            )
            save_file(
                {
                    "embeds": embeds.to("cpu", torch.bfloat16).contiguous(),
                    "embeds_plain": plain.to("cpu", torch.bfloat16).contiguous(),
                    **var,
                },
                str(te_path(stem, held)),
                metadata={
                    "caption": ocr,
                    "colour_caption": colour,
                    "n_text": str(embeds.shape[0]),
                    "n_text_plain": str(plain.shape[0]),
                    "n_image_tokens": str(n_img),
                    "condition": str(cond),
                    "source": str(target),
                    "encoder": f"{encoder} + vision bf16, image then caption",
                    **var_meta,
                },
            )
            print(
                f"  te {i}/{len(todo)} {stem}: {embeds.shape[0]} rows ({n_img} image), "
                f"{len(sparse)} sparse, {time.time() - t0:.0f}s",
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

    records = [
        {
            "stem": stem,
            "condition": str(cond),
            "target": str(target),
            "caption": captions[stem][1],
            "caption_v1": captions[stem][2][0],
            "width": w,
            "height": h,
            "te": str(te_path(stem, True)),
            "ref": str(ref_path(stem, True)),
        }
        for stem, cond, target, held, (w, h) in jobs
        if held
    ]
    (out / "holdout.json").write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
