"""Precache Qwen3.5-9B text embeddings and FLUX.2 VAE latents for a training folder.

Training never loads the TE (16.8 GB) or the VAE: this runs TE → free → VAE and
leaves two files per image in one flat folder.

    {stem}.moss_te.safetensors
        embeds  (S, 4096) bf16 — ``last_hidden_state`` at the caption's true length
        metadata: caption, caption_sha256, caption_format, source_sha256, n_text,
                  max_text_tokens, encoder
    {stem}_{W}x{H}.moss_latent.safetensors
        latent  (128, H/16, W/16) fp32 — packed 2×2 and BN-normalised, i.e. exactly
                what the DiT consumes (``model.encode``, the inverse of the decode)
        metadata: width, height, source, source_size, source_mtime, vae_dtype, resampled

``W×H`` is the pixel size the latent was encoded at. Text is never padded: the
DiT slices each context to its true length and the image's rope frame coordinate
equals that length, so a padded cache would train an offset inference never sees.

Invalidation: a TE file is current while its ``caption_sha256`` (and truncation
length) match the caption on disk, so a caption edit re-encodes that caption
only. A latent file is current while it is newer than its source image; a latent
at a different size for the same stem is deleted when the new one is written.
``--overwrite`` re-encodes everything.

Sizing (``target_size``): native aspect, both sides multiples of 16, area at most
``resolution²`` — **never upscaled**. An image whose sides are already multiples
of 16 and whose area is within ``AREA_TOLERANCE`` of the cap is encoded as-is (the
Anima resized tree is already on a 16-px grid at ~1.05 MP, so it is not resampled
twice). Otherwise it is LANCZOS-resized to cover the target and centre-cropped by
the < 16 px the rounding leaves. Transparent images are composited onto white.

Captions: the ``.txt`` beside each image, stripped, mapped onto MOSS's
layout-JSON prompt by ``anime_tools.captions.layout_json`` (the default
``--caption_format layout_json``; ``caption`` / ``caption_sha256`` are the JSON;
a ``{stem}.layout.txt`` published by anime_tools' Export is read instead) or verbatim with ``--caption_format text``. ``source_sha256`` is always the ``.txt``. Stems must be unique
in the flat cache; a stem that appears in several subfolders is prefixed with its
relative folder (``sub__dir__stem``).

    make daemon-run ARGS="scripts/moss/cache.py \\
        --src 'post_image_dataset/resized/channel_(caststation)' \\
        --out 'output/moss/cache/channel_(caststation)'"
"""

from __future__ import annotations

import json
import math
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from library.env import resolve_under_home
from library.moss.scan import (
    LATENT_SUFFIX,
    TE_SUFFIX,
    caption_sha,
    layout_sidecar,
    source_pairs,
)
from library.moss.scan import LATENT_RE as _LATENT_RE

STRIDE = 16
AREA_TOLERANCE = 1.10
ENCODER_ID = (
    "Qwen3.5-9B last_hidden_state, chat template add_generation_prompt+enable_thinking"
)


@dataclass
class CacheItem:
    stem: str
    latent_path: Path
    te_path: Path
    width: int
    height: int
    n_text: int


@dataclass
class Pair:
    stem: str
    image: Path
    caption: str  # what the TE encodes
    source_sha: str  # of the .txt it came from


# ------------------------------------------------------------------ reading


def list_items(cache_dir) -> list[CacheItem]:
    """Every stem with both a latent and a TE file, sorted by stem."""
    cache_dir = Path(cache_dir)
    latents: dict[str, tuple[Path, int, int]] = {}
    for p in cache_dir.glob(f"*{LATENT_SUFFIX}"):
        m = _LATENT_RE.match(p.name)
        if m is None:
            continue
        stem = m["stem"]
        prev = latents.get(stem)
        if prev is None or p.stat().st_mtime > prev[0].stat().st_mtime:
            latents[stem] = (p, int(m["w"]), int(m["h"]))
    items = []
    for stem, (path, w, h) in sorted(latents.items()):
        te = cache_dir / f"{stem}{TE_SUFFIX}"
        if not te.exists():
            continue
        with safe_open(te, "pt") as f:
            n_text = f.get_slice("embeds").get_shape()[0]
        items.append(CacheItem(stem, path, te, w, h, n_text))
    return items


def load_item(item: CacheItem) -> tuple[torch.Tensor, torch.Tensor]:
    """``latent (128, H/16, W/16)`` fp32, ``embeds (S, 4096)`` bf16, both on CPU."""
    latent = load_file(item.latent_path)["latent"].float()
    embeds = load_file(item.te_path)["embeds"].to(torch.bfloat16)
    return latent, embeds


# ------------------------------------------------------------------ sizing


def target_size(width: int, height: int, resolution: int) -> tuple[int, int]:
    """``(W, H)``: native aspect, multiples of 16, area ≤ ~``resolution²``, never
    larger than the source on either side."""
    cap = resolution * resolution
    if (
        width % STRIDE == 0
        and height % STRIDE == 0
        and width * height <= cap * AREA_TOLERANCE
    ):
        return width, height
    area = min(cap, width * height)
    aspect = width / height
    out_w = round(math.sqrt(area * aspect) / STRIDE) * STRIDE
    out_h = round(math.sqrt(area / aspect) / STRIDE) * STRIDE
    out_w = max(STRIDE, min(out_w, width // STRIDE * STRIDE))
    out_h = max(STRIDE, min(out_h, height // STRIDE * STRIDE))
    return out_w, out_h


def load_rgb(path: Path) -> Image.Image:
    image = Image.open(path)
    if image.mode in ("RGBA", "LA") or (
        image.mode == "P" and "transparency" in image.info
    ):
        rgba = image.convert("RGBA")
        canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(canvas, rgba).convert("RGB")
    return image.convert("RGB")


def fit(image: Image.Image, width: int, height: int) -> Image.Image:
    """Resize to cover ``(width, height)`` and centre-crop; identity at that size."""
    if image.size == (width, height):
        return image
    scale = max(width / image.width, height / image.height)
    rw = max(width, math.ceil(image.width * scale))
    rh = max(height, math.ceil(image.height * scale))
    image = image.resize((rw, rh), Image.LANCZOS)
    left, top = (rw - width) // 2, (rh - height) // 2
    return image.crop((left, top, left + width, top + height))


def to_pixels(image: Image.Image) -> torch.Tensor:
    """RGB PIL → ``(1, 3, H, W)`` fp32 in ``[-1, 1]``."""
    array = np.asarray(image, dtype=np.float32)
    return torch.from_numpy(array).permute(2, 0, 1)[None] / 127.5 - 1.0


# ------------------------------------------------------------------ source


def find_pairs(src: Path, caption_format: str = "text") -> list[Pair]:
    """Images with a ``.txt`` caption beside them, subfolders included. Stems that
    collide across subfolders get their relative folder as a prefix.

    Under ``layout_json`` a ``{stem}.layout.txt`` that anime_tools' Export
    published beside the caption is used as is, unless the ``.txt`` is newer
    (edited after the export); every other caption is converted here."""
    images, found = source_pairs(src)
    captioned = {image for _, image, _ in found}
    for p in images:
        if p not in captioned:
            print(f"skip {p.name}: no caption sidecar", flush=True)
    convert = None
    pairs = []
    from_sidecar = 0
    for stem, image, path in found:
        raw = path.read_text(encoding="utf-8").strip()
        caption = None
        if caption_format == "layout_json":
            sidecar = layout_sidecar(path)
            if sidecar.exists() and sidecar.stat().st_mtime >= path.stat().st_mtime:
                caption = sidecar.read_text(encoding="utf-8").strip()
                from_sidecar += 1
        if caption is None:
            convert = convert or caption_converter(caption_format)
            caption = convert(raw)
        pairs.append(Pair(stem, image, caption, caption_sha(raw)))
    if caption_format == "layout_json":
        print(
            f"layout_json: {from_sidecar} from Export's sidecars, "
            f"{len(pairs) - from_sidecar} converted",
            flush=True,
        )
    return pairs


def is_layout_json(raw: str) -> bool:
    """``raw`` is already a layout-JSON prompt (an object with ``summary``)."""
    if not raw.startswith("{"):
        return False
    try:
        obj = json.loads(raw)
    except ValueError:
        return False
    return isinstance(obj, dict) and "summary" in obj


def caption_converter(caption_format: str):
    """``raw caption -> what the TE encodes`` for one ``caption_format``.

    ``layout_json`` passes a caption that already is layout JSON through as
    written (converting it again would nest the whole object in
    ``summary.caption``), and loads the tag CSV only for the first caption that
    needs converting."""
    if caption_format == "text":
        return lambda raw: raw
    if caption_format != "layout_json":
        raise SystemExit(f"unknown caption_format {caption_format!r}")
    kb = None

    def convert(raw: str) -> str:
        nonlocal kb
        if is_layout_json(raw):
            return raw
        from anime_tools.captions.layout_json import layout_json

        if kb is None:
            kb = _tag_knowledge_base()
        return layout_json(raw, kb)

    return convert


def _tag_knowledge_base():
    from anime_tools.captions.correction import find_tag_csv, load_tag_knowledge_base

    from library.env import anima_home

    csv = find_tag_csv(anima_home())
    if csv is None:
        raise SystemExit(
            "layout_json needs the danbooru tag CSV: make download-model ARGS=danbooru_tags"
        )
    return load_tag_knowledge_base(csv)


def _te_current(path: Path, caption: str, max_tokens: int) -> bool:
    if not path.exists():
        return False
    try:
        with safe_open(path, "pt") as f:
            meta = f.metadata() or {}
    except Exception:
        return False
    return meta.get("caption_sha256") == caption_sha(caption) and meta.get(
        "max_text_tokens"
    ) == str(max_tokens)


def _latents_for(out_dir: Path, stem: str) -> list[Path]:
    found = []
    for p in out_dir.glob(f"{glob_escape(stem)}_*{LATENT_SUFFIX}"):
        m = _LATENT_RE.match(p.name)
        if m is not None and m["stem"] == stem:
            found.append(p)
    return found


def glob_escape(text: str) -> str:
    return re.sub(r"([\[\]*?])", r"[\1]", text)


# ------------------------------------------------------------------ passes


def cache_text(
    pairs, out_dir, model_dir, device, max_tokens, overwrite, caption_format="text"
) -> dict:
    from library.moss.loader import (
        empty_cache,
        encode_prompt,
        load_text_encoder,
        load_tokenizer,
        prompt_ids,
    )

    todo = [
        p
        for p in pairs
        if overwrite
        or not _te_current(out_dir / f"{p.stem}{TE_SUFFIX}", p.caption, max_tokens)
    ]
    if not todo:
        print("text: all cached", flush=True)
        return {"encoded": 0}
    t0 = time.time()
    tok = load_tokenizer(model_dir)
    longest = max(
        prompt_ids(tok, p.caption, max_tokens).input_ids.shape[1] for p in todo
    )
    tok, te = load_text_encoder(model_dir, device, longest, tok=tok)
    t_load = time.time() - t0
    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    lengths = []
    for i, p in enumerate(todo, 1):
        embeds = encode_prompt(tok, te, p.caption, device, max_tokens).to(
            "cpu", torch.bfloat16
        )
        n = embeds.shape[0]
        lengths.append(n)
        note = "  TRUNCATED" if n >= max_tokens else ""
        print(f"  text {i}/{len(todo)} {p.stem}: {n} tokens{note}", flush=True)
        save_file(
            {"embeds": embeds.contiguous()},
            out_dir / f"{p.stem}{TE_SUFFIX}",
            metadata={
                "caption": p.caption,
                "caption_sha256": caption_sha(p.caption),
                "caption_format": caption_format,
                "source_sha256": p.source_sha,
                "n_text": str(n),
                "max_text_tokens": str(max_tokens),
                "encoder": ENCODER_ID,
                "source": str(p.image),
            },
        )
    t_enc = time.time() - t1
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(
        f"text: {len(todo)} captions, load {t_load:.1f}s + encode {t_enc:.1f}s, "
        f"tokens {min(lengths)}-{max(lengths)}, peak {peak:.2f} GiB",
        flush=True,
    )
    del tok, te
    empty_cache()
    return {"encoded": len(todo), "load_s": t_load, "encode_s": t_enc, "peak_gib": peak}


def cache_latents(
    pairs, out_dir, model_dir, device, resolution, vae_dtype, overwrite, save_crops
) -> dict:
    from library.moss.loader import empty_cache, load_vae
    from library.moss.model import encode

    plan = []
    for p in pairs:
        with Image.open(p.image) as im:
            size = im.size
        w, h = target_size(*size, resolution)
        path = out_dir / f"{p.stem}_{w}x{h}{LATENT_SUFFIX}"
        current = path.exists() and path.stat().st_mtime >= p.image.stat().st_mtime
        if overwrite or not current:
            plan.append((p, size, w, h, path))
    if not plan:
        print("latents: all cached", flush=True)
        return {"encoded": 0}
    dtype = torch.float32 if vae_dtype == "fp32" else torch.bfloat16
    t0 = time.time()
    vae = load_vae(model_dir, dtype=dtype, device=device)
    t_load = time.time() - t0
    crops = out_dir / "crops"
    if save_crops:
        crops.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    resampled = 0
    for i, (p, size, w, h, path) in enumerate(plan, 1):
        image = load_rgb(p.image)
        fitted = fit(image, w, h)
        resampled += fitted is not image
        latent = encode(to_pixels(fitted), vae)[0].cpu().contiguous()
        for old in _latents_for(out_dir, p.stem):
            if old != path:
                old.unlink()
        save_file(
            {"latent": latent},
            path,
            metadata={
                "width": str(w),
                "height": str(h),
                "source": str(p.image),
                "source_size": f"{size[0]}x{size[1]}",
                "source_mtime": str(p.image.stat().st_mtime),
                "vae_dtype": vae_dtype,
                "resampled": str(fitted is not image),
            },
        )
        if save_crops:
            fitted.save(crops / f"{p.stem}.png")
        print(
            f"  latents {i}/{len(plan)} {p.stem}: {size[0]}x{size[1]} -> {w}x{h}",
            flush=True,
        )
    t_enc = time.time() - t1
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(
        f"latents: {len(plan)} images ({resampled} resampled), load {t_load:.1f}s + "
        f"encode {t_enc:.1f}s, peak {peak:.2f} GiB",
        flush=True,
    )
    del vae
    empty_cache()
    return {
        "encoded": len(plan),
        "resampled": resampled,
        "load_s": t_load,
        "encode_s": t_enc,
        "peak_gib": peak,
    }


def summarize(out_dir: Path) -> dict:
    """Token statistics over the complete pairs in the cache."""
    items = list_items(out_dir)
    if not items:
        return {"items": 0}
    text = [it.n_text for it in items]
    image = [(it.width // STRIDE) * (it.height // STRIDE) for it in items]
    joint = [a + b for a, b in zip(text, image)]
    size = sum(p.stat().st_size for p in out_dir.glob("*.moss_*.safetensors"))
    return {
        "items": len(items),
        "text_tokens": {
            "min": min(text),
            "median": statistics.median(text),
            "max": max(text),
        },
        "image_tokens": {
            "min": min(image),
            "median": statistics.median(image),
            "max": max(image),
        },
        "joint_tokens_max": max(joint),
        "shapes": len({(it.width, it.height) for it in items}),
        "bytes": size,
    }


def run_cache(req) -> Path:
    """Both passes over ``req.src``; returns the cache folder."""
    from library.moss.requests import resolve_model_dir

    src = resolve_under_home(req.src)
    out_dir = resolve_under_home(req.out)
    model_dir = resolve_model_dir(req.model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = find_pairs(src, req.caption_format)
    if not pairs:
        raise SystemExit(f"no image/caption pairs under {src}")
    print(f"{len(pairs)} pairs from {src}, model {model_dir}", flush=True)
    device = torch.device("cuda")
    report = {"src": str(src), "resolution": req.resolution}
    if not req.skip_text:
        report["text"] = cache_text(
            pairs,
            out_dir,
            model_dir,
            device,
            req.max_text_tokens,
            req.overwrite,
            req.caption_format,
        )
    if not req.skip_latents:
        report["latents"] = cache_latents(
            pairs,
            out_dir,
            model_dir,
            device,
            req.resolution,
            req.vae_dtype,
            req.overwrite,
            req.save_crops,
        )
    report["summary"] = summarize(out_dir)
    (out_dir / "cache_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2), flush=True)
    print(f"cache at {out_dir}", flush=True)
    return out_dir
