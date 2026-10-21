#!/usr/bin/env python3
"""Colorize pairs for a TE-only MOSS LoRA: the B&W condition enters only through
Qwen3.5's vision tower, the DiT is untouched.

Pairs are EasyControl colorize's: the condition is the synthetic mangafied page in
``post_image_dataset/easycontrol/colorize/staging/<dir>/<stem>.png``, the target the
colour original at the same path under ``post_image_dataset/resized``. Per pair, in a
flat cache ``train.py`` reads unchanged:

    {stem}_{W}x{H}.moss_latent.safetensors   the colour target (cache.cache_latents)
    {stem}.moss_te.safetensors               embeds       = TE(B&W page + colour caption)
                                             embeds_plain = TE(B&W page alone)

``embeds_plain`` rides ``--plain_caption_rate`` as the empty-prompt colorize path
(``--caption_dropout`` would drop the page too). The colour caption is
``color_caption.filter_to_colors_and_protected`` (copyright + comic prefix, colour
tags), the EasyControl colorize text policy. The page is fitted to the latent's
size and the vision tower sees it at that pixel count (~1k tokens at 1 MP).

``--holdout`` pairs (``PROBE_STEMS`` first: three of the four images in
``_archive/bench/moss_vision_reference/vision_recon.py``, the fourth has no colorize
pair; then random) are kept out of the cache; their TE files land in ``<out>/holdout/`` with ``holdout.json`` for the
eval.

``--ref_scale s`` (i2i probe) also writes ``{stem}.moss_ref.safetensors``: the B&W
page VAE-encoded at 1/s of the target's sides (multiples of 16), which ``train.py``
feeds as clean reference tokens on RoPE's ``r = 1`` axis. ``--link_from <cache>``
(repeatable) symlinks an existing cache's latents, TE files and holdout into ``--out``
first, so a reference variant reuses them and the source cache stays as it was (but for
``--variants``, which completes the linked TE files at their target).

``--variants N`` (Anima colorize's text policy: 4) adds ``embeds_v1 … embeds_v{N-1}``
to every TE file, train and holdout: the colour caption shuffled with
``--tag_dropout`` of its tags dropped (``generate_caption_variants``; comic tags
protected, copyright rides the dropout), each encoded after the page. The draw is
seeded per stem. A variant stores only its rows after the page (``variant_prefix``
in the metadata; ``cache.variant_entries`` / ``cache.add_variants`` write it,
``cache.read_te`` rebuilds it), and the log reports how far its
own page rows strayed from ``embeds``'. TE files already written without them are completed in place, so a
linked cache gains them too; ``train.py --caption_variant_rate`` reads them.

``--manga N`` adds N pages of the EasyControl ``manga_color`` slice (hitocrawl's
full-colour works, JA / EN / KO editions; condition under ``manga_color/staging``,
target under ``manga_color/resized``) as ``manga__{anchor}_{lang}_{page}`` stems. They
are drawn on their own, so the curated selection above does not move. Holdout is by
work: ``--manga_holdout`` works give one page each to the holdout, and none of their
pages, in any edition, train.

    make daemon-run ARGS="--stall-timeout 900 project/moss_lora/src/colorize_cache.py --n 1000"
    make daemon-run ARGS="--stall-timeout 900 project/moss_lora/src/colorize_cache.py --n 1000 \
        --link_from post_image_dataset/moss/colorize_te --out post_image_dataset/moss/colorize_ref --ref_scale 2"
    make daemon-run ARGS="--stall-timeout 900 project/moss_lora/src/colorize_cache.py --n 1000 \
        --manga 500 --link_from post_image_dataset/moss/colorize_te --out post_image_dataset/moss/colorize_manga --ref_scale 2"
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from anime_tools.captions.variants import generate_caption_variants
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file

from library.env import resolve_under_home
from library.moss import cache as moss_cache
from library.moss.loader import empty_cache, load_tokenizer
from library.moss.scan import REF_SUFFIX, TE_SUFFIX
from vision_te import encode_variants, load_te

sys.path.insert(0, str(resolve_under_home("easycontrol_adapters/colorization")))
from color_caption import (  # noqa: E402
    filter_to_colors_and_protected,
    is_comic_tag,
    load_copyright_tags,
)

STAGING = "post_image_dataset/easycontrol/colorize/staging"
RESIZED = "post_image_dataset/resized"
MANGA = "post_image_dataset/easycontrol/colorize/manga_color"
PROBE_STEMS = ("dan_6768037", "9859367", "11115380")


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
    """EasyControl's ``manga_color`` slice: ``{staging,resized}/{ja,en,ko}/{anchor}_{lang}_{page}.png``."""
    root = resolve_under_home(MANGA)
    out = []
    for cond in sorted((root / "staging").rglob("*.png")):
        target = root / "resized" / cond.relative_to(root / "staging")
        if target.exists() and target.with_suffix(".txt").exists():
            out.append((f"manga__{cond.stem}", cond, target))
    return out


def split_manga(pairs, n, n_holdout, seed):
    """Hold out whole works (every edition of them), one page each; draw ``n`` from the rest."""
    rng = random.Random(seed)
    works = {}
    for p in pairs:
        works.setdefault(p[2].stem.split("_")[0], []).append(p)
    anchors = sorted(works)
    rng.shuffle(anchors)
    held = anchors[:n_holdout]
    holdout = [rng.choice(works[a]) for a in held]
    rest = [p for a in anchors[n_holdout:] for p in works[a]]
    rng.shuffle(rest)
    return rest[:n], holdout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="training pairs")
    ap.add_argument("--holdout", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--out", default="post_image_dataset/moss/colorize_te")
    ap.add_argument("--ref_scale", type=int, default=0, help="0 = no reference")
    ap.add_argument("--link_from", action="append", default=[], help="repeatable")
    ap.add_argument("--manga", type=int, default=0, help="manga_color pairs added")
    ap.add_argument("--manga_holdout", type=int, default=4, help="held-out works")
    ap.add_argument("--variants", type=int, default=0, help="0 = none; v0 + N-1")
    ap.add_argument("--tag_dropout", type=float, default=0.8)
    args = ap.parse_args()
    out = resolve_under_home(args.out)
    (out / "holdout").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    for link in args.link_from:
        src = resolve_under_home(link)
        for f in [*src.glob("*.safetensors"), *(src / "holdout").glob("*.safetensors")]:
            if REF_SUFFIX in f.name:
                continue
            dst = out / f.relative_to(src)
            if not dst.exists():
                dst.symlink_to(f.resolve())

    pairs = find_pairs()
    rng = random.Random(args.seed)
    probe = [p for p in pairs if p[2].stem in PROBE_STEMS]
    rest = [p for p in pairs if p[2].stem not in PROBE_STEMS]
    rng.shuffle(rest)
    holdout = (probe + rest)[: args.holdout]
    train = rest[args.holdout - len(probe) :][: args.n]
    print(
        f"{len(pairs)} pairs: {len(train)} train, {len(holdout)} held out", flush=True
    )
    if args.manga:
        manga = find_manga_pairs()
        m_train, m_holdout = split_manga(
            manga, args.manga, args.manga_holdout, args.seed
        )
        train += m_train
        holdout += m_holdout
        print(
            f"{len(manga)} manga pairs: {len(m_train)} train, "
            f"{len(m_holdout)} held out (whole works)",
            flush=True,
        )

    copyright_tags = load_copyright_tags()

    def colour_caption(target: Path) -> str:
        return filter_to_colors_and_protected(
            target.with_suffix(".txt").read_text(encoding="utf-8").strip(),
            copyright_tags,
            keep_copyright=True,
            keep_comic=True,
        )

    # Latents first (targets only), so the TE pass knows each pair's size.
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

    jobs = [(s, c, t, out / f"{s}{TE_SUFFIX}") for s, c, t in train]
    jobs += [(s, c, t, out / "holdout" / f"{s}{TE_SUFFIX}") for s, c, t in holdout]
    todo = [j for j in jobs if not j[3].exists()]

    def has_variants(path):
        with safe_open(str(path), "pt") as f:
            return "embeds_v1" in f.keys()

    stale = []
    if args.variants > 1:
        stale = [j for j in jobs if j[3].exists() and not has_variants(j[3])]
    records = []
    if todo or stale:
        tok = load_tokenizer("")
        te = load_te(tok, device)

        def variant_texts(stem, caption):
            """The variant captions of ``caption`` (none below ``--variants 2``)."""
            if args.variants < 2:
                return []
            random.seed(f"{args.seed}:{stem}")
            return generate_caption_variants(
                caption, args.variants, args.tag_dropout, is_comic_tag
            )[1:]

        t0 = time.time()
        for i, (stem, cond, target, path) in enumerate(todo, 1):
            with Image.open(target) as im:
                w, h = moss_cache.target_size(*im.size, args.resolution)
            page = moss_cache.fit(moss_cache.load_rgb(cond), w, h)
            caption = colour_caption(target)
            texts = variant_texts(stem, caption)
            (embeds, n_img, prefix), (plain, _, _), *rest = encode_variants(
                tok, te, page, [caption, "", *texts], w * h, device
            )
            variants = [
                (t, e.to("cpu", torch.bfloat16)) for t, (e, _, _) in zip(texts, rest)
            ]
            var, var_meta = moss_cache.variant_entries(variants, prefix)
            save_file(
                {
                    "embeds": embeds.to("cpu", torch.bfloat16).contiguous(),
                    "embeds_plain": plain.to("cpu", torch.bfloat16).contiguous(),
                    **var,
                },
                str(path),
                metadata={
                    "caption": caption,
                    "n_text": str(embeds.shape[0]),
                    "n_text_plain": str(plain.shape[0]),
                    "n_image_tokens": str(n_img),
                    "condition": str(cond),
                    "source": str(target),
                    "encoder": "Qwen3.5-9B + vision, image then colour caption",
                    **var_meta,
                },
            )
            print(
                f"  te {i}/{len(todo)} {stem}: {embeds.shape[0]} rows "
                f"({n_img} image), {len(variants)} variants, "
                f"{time.time() - t0:.0f}s",
                flush=True,
            )
        t0 = time.time()
        for i, (stem, cond, target, path) in enumerate(stale, 1):
            with Image.open(target) as im:
                w, h = moss_cache.target_size(*im.size, args.resolution)
            page = moss_cache.fit(moss_cache.load_rgb(cond), w, h)
            with safe_open(str(path), "pt") as f:
                caption = f.metadata()["caption"]
                embeds = f.get_tensor("embeds")
            texts = variant_texts(stem, caption)
            got = encode_variants(tok, te, page, texts, w * h, device)
            prefix = got[0][2]
            variants = [
                (t, e.to("cpu", torch.bfloat16)) for t, (e, _, _) in zip(texts, got)
            ]
            # The stored embeds' page rows are an earlier forward's: the largest
            # deviation of this one's from them.
            dev = (
                (got[0][0][:prefix].to("cpu").float() - embeds[:prefix].float())
                .abs()
                .max()
                .item()
            )
            # Linked files are completed at their target.
            moss_cache.add_variants(path, variants, prefix)
            caps = " | ".join(text for text, _ in variants)
            print(
                f"  variants {i}/{len(stale)} {stem}: {caps}; page rows ±{dev:.3g}, "
                f"{time.time() - t0:.0f}s",
                flush=True,
            )
        del te, tok
        empty_cache()

    ref_paths = {}
    if args.ref_scale:
        ref_paths = {s: out / f"{s}{REF_SUFFIX}" for s, _, _ in train}
        ref_paths |= {s: out / "holdout" / f"{s}{REF_SUFFIX}" for s, _, _ in holdout}
        todo = [(s, c, t) for s, c, t in train + holdout if not ref_paths[s].exists()]
        if todo:
            from library.moss.loader import load_vae
            from library.moss.model import encode as vae_encode

            vae = load_vae("", device=device)
            for i, (stem, cond, target) in enumerate(todo, 1):
                with Image.open(target) as im:
                    w, h = moss_cache.target_size(*im.size, args.resolution)
                rw = max(16, round(w / args.ref_scale / 16) * 16)
                rh = max(16, round(h / args.ref_scale / 16) * 16)
                page = moss_cache.fit(moss_cache.load_rgb(cond), w, h)
                page = moss_cache.fit(page, rw, rh)
                latent = vae_encode(moss_cache.to_pixels(page), vae)[0]
                save_file(
                    {"latent": latent.cpu().contiguous()},
                    str(ref_paths[stem]),
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

    for stem, cond, target in holdout:
        with Image.open(target) as im:
            w, h = moss_cache.target_size(*im.size, args.resolution)
        records.append(
            {
                "stem": stem,
                "condition": str(cond),
                "target": str(target),
                "caption": colour_caption(target),
                "width": w,
                "height": h,
                "te": str(out / "holdout" / f"{stem}{TE_SUFFIX}"),
                **({"ref": str(ref_paths[stem])} if stem in ref_paths else {}),
            }
        )
    (out / "holdout.json").write_text(json.dumps(records, indent=2))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
