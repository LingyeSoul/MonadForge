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

``--holdout`` pairs (the vision_recon images first, then random) are kept out of
the cache; their TE files land in ``<out>/holdout/`` with ``holdout.json`` for the
eval.

``--ref_scale s`` (i2i probe) also writes ``{stem}.moss_ref.safetensors``: the B&W
page VAE-encoded at 1/s of the target's sides (multiples of 16), which ``train.py``
feeds as clean reference tokens on RoPE's ``r = 1`` axis. ``--link_from <cache>``
symlinks an existing cache's latents, TE files and holdout into ``--out`` first, so a
reference variant reuses them and the source cache stays as it was.

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
from PIL import Image
from safetensors.torch import save_file

from library.env import resolve_under_home
from library.moss import cache as moss_cache
from library.moss.loader import empty_cache, load_tokenizer
from library.moss.scan import REF_SUFFIX, TE_SUFFIX
from vision_te import encode, load_te

sys.path.insert(0, str(resolve_under_home("easycontrol_adapters/colorization")))
from color_caption import filter_to_colors_and_protected, load_copyright_tags  # noqa: E402

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
    ap.add_argument("--link_from", default="")
    ap.add_argument("--manga", type=int, default=0, help="manga_color pairs added")
    ap.add_argument("--manga_holdout", type=int, default=4, help="held-out works")
    args = ap.parse_args()
    out = resolve_under_home(args.out)
    (out / "holdout").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    if args.link_from:
        src = resolve_under_home(args.link_from)
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
    records = []
    if todo:
        tok = load_tokenizer("")
        te = load_te(tok, device)
        t0 = time.time()
        for i, (stem, cond, target, path) in enumerate(todo, 1):
            with Image.open(target) as im:
                w, h = moss_cache.target_size(*im.size, args.resolution)
            page = moss_cache.fit(moss_cache.load_rgb(cond), w, h)
            caption = colour_caption(target)
            embeds, n_img = encode(tok, te, page, caption, w * h, device)
            plain, _ = encode(tok, te, page, "", w * h, device)
            save_file(
                {
                    "embeds": embeds.to("cpu", torch.bfloat16).contiguous(),
                    "embeds_plain": plain.to("cpu", torch.bfloat16).contiguous(),
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
                },
            )
            print(
                f"  te {i}/{len(todo)} {stem}: {embeds.shape[0]} rows "
                f"({n_img} image), {time.time() - t0:.0f}s",
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
