"""Precache Qwen3.5-9B text embeddings and FLUX.2 VAE latents for a training folder.

Training never loads the TE (16.8 GB) or the VAE: this runs TE → free → VAE and
leaves two files per image in one flat folder.

    {stem}.moss_te.safetensors
        embeds  (S, 4096) bf16: ``last_hidden_state`` at the caption's true length
        embeds_plain  (S', 4096) bf16: under ``layout_json`` only, the plain Anima
                caption the layout came from (training swaps it in at
                ``plain_caption_rate``)
        embeds_v1, embeds_v2, …  (S_k − P, 4096) bf16: optional tag-dropped
                caption variants (``variant_entries`` / ``add_variants``), swapped
                in at ``caption_variant_rate``. Only the rows after the first
                ``P = variant_prefix`` (metadata) are stored: the TE is causal and
                the variants share ``embeds``' leading rows
        metadata: caption, caption_sha256, caption_format, source_sha256, source,
                  n_text, max_text_tokens, encoder (+ plain_caption_sha256,
                  n_text_plain; + caption_v{k}, variant_prefix with variants)
    {stem}_{W}x{H}.moss_latent.safetensors
        latent  (128, H/16, W/16) fp32: packed 2×2 and BN-normalised, exactly what
                the DiT consumes (``model.encode``, the inverse of the decode)
        metadata: width, height, source, source_size, source_mtime, vae_dtype, resampled

    {stem}.moss_ref.safetensors  (``--pairs`` only)
        latent  (128, h, w) fp32: the pair's condition image, fitted to the
                target's ``W×H`` and then to 1/``ref_scale`` of its sides
        metadata: width, height, target, ref_scale, condition, condition_mtime

``W×H`` is the pixel size the latent was encoded at. Text is never padded: the
DiT slices each context to its true length and the image's rope frame coordinate
equals that length, so a padded cache would train an offset inference never sees.

Invalidation: a TE file is current while its ``caption_sha256``, truncation
length, ``encoder`` and (under ``layout_json``) ``plain_caption_sha256`` match, so
a caption edit re-encodes that caption only. A re-encode keeps the file's caption
variants while its caption is unchanged; after a caption edit they are stale and
dropped, with a warning. A latent file is current while it is newer than its
source image; a latent at a different size for the same stem is deleted when the
new one is written. ``--overwrite`` re-encodes everything.

Sizing (``target_size``): native aspect, both sides multiples of 16, area at most
``resolution²``, **never upscaled**. An image whose sides are already multiples
of 16 and whose area is within ``AREA_TOLERANCE`` of the cap is encoded as-is (the
Anima resized tree is already on a 16-px grid at ~1.05 MP, so it is not resampled
twice). Otherwise it is LANCZOS-resized to cover the target and centre-cropped by
the < 16 px the rounding leaves. Transparent images are composited onto white.

Captions: the ``.txt`` beside each image, stripped, mapped onto MOSS's
layout-JSON prompt by ``anime_tools.captions.layout_json`` (the default
``--caption_format layout_json``; ``caption`` / ``caption_sha256`` are the JSON;
a ``{stem}.layout.txt`` published by anime_tools' Export is read instead) or
verbatim with ``--caption_format text``. ``source_sha256`` is always the ``.txt``.
A stem that appears in several subfolders is prefixed with its relative folder
(``sub__dir__stem``). A caption in the Anima resized tree with no PNG beside it
pairs with its ``image_dataset/`` original, minus the excluded ones
(``scan.original_tree``).

    make daemon-run ARGS="scripts/moss/cache.py \\
        --src 'post_image_dataset/resized/channel_(caststation)' \\
        --out 'post_image_dataset/moss/channel_(caststation)'"
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
    REF_SUFFIX,
    TE_SUFFIX,
    caption_sha,
    latent_files,
    layout_sidecar,
    original_tree,
    source_pairs,
)

_VARIANT_RE = re.compile(r"embeds_v[1-9][0-9]*")
_VARIANT_META_RE = re.compile(r"caption_v[1-9][0-9]*|variant_prefix")

STRIDE = 16
AREA_TOLERANCE = 1.10
ENCODER_ID = (
    "Qwen3.5-9B last_hidden_state, chat template add_generation_prompt+enable_thinking"
)


def encoder_id(precision: str = "bf16") -> str:
    """The ``encoder`` stamp of a TE cache file; bf16 keeps the original string."""
    return ENCODER_ID if precision == "bf16" else f"{ENCODER_ID}, {precision} weights"


@dataclass
class CacheItem:
    stem: str
    latent_path: Path
    te_path: Path
    width: int
    height: int
    n_text: int
    n_text_plain: int = 0  # 0 = no plain-caption embedding in the TE file
    ref_path: Path | None = None  # clean reference latent (i2i probe)
    n_ref: int = 0
    n_variants: int = 0  # embeds_v1 … embeds_v{n} in the TE file
    n_text_variants: int = 0  # the longest variant, its shared prefix included


@dataclass
class Pair:
    stem: str
    image: Path
    caption: str  # what the TE encodes
    source_sha: str  # of the .txt it came from
    plain: str = ""  # the .txt itself, encoded beside a layout-JSON caption


# ------------------------------------------------------------------ reading


def list_items(cache_dir) -> list[CacheItem]:
    """Every stem with both a latent and a TE file, sorted by stem."""
    cache_dir = Path(cache_dir)
    latents: dict[str, tuple[Path, int, int]] = {}
    for stem, w, h, p in latent_files(cache_dir):
        prev = latents.get(stem)
        if prev is None or p.stat().st_mtime > prev[0].stat().st_mtime:
            latents[stem] = (p, w, h)
    items = []
    for stem, (path, w, h) in sorted(latents.items()):
        te = cache_dir / f"{stem}{TE_SUFFIX}"
        if not te.exists():
            continue
        with safe_open(te, "pt") as f:
            n_text = f.get_slice("embeds").get_shape()[0]
            n_plain = (
                f.get_slice("embeds_plain").get_shape()[0]
                if "embeds_plain" in f.keys()
                else 0
            )
            tails = [
                f.get_slice(k).get_shape()[0]
                for k in f.keys()
                if _VARIANT_RE.fullmatch(k)
            ]
            prefix = int((f.metadata() or {}).get("variant_prefix", 0))
        ref, n_ref = cache_dir / f"{stem}{REF_SUFFIX}", 0
        if ref.exists():
            with safe_open(ref, "pt") as f:
                _, rh, rw = f.get_slice("latent").get_shape()
            n_ref = rh * rw
        items.append(
            CacheItem(
                stem,
                path,
                te,
                w,
                h,
                n_text,
                n_plain,
                ref if n_ref else None,
                n_ref,
                len(tails),
                prefix + max(tails) if tails else 0,
            )
        )
    return items


def load_ref(item: CacheItem) -> torch.Tensor | None:
    """``ref (128, h, w)`` fp32 on CPU, or ``None``."""
    if item.ref_path is None:
        return None
    return load_file(item.ref_path)["latent"].float()


def load_item(item: CacheItem) -> tuple[torch.Tensor, torch.Tensor]:
    """``latent (128, H/16, W/16)`` fp32, ``embeds (S, 4096)`` bf16, both on CPU."""
    latent = load_file(item.latent_path)["latent"].float()
    embeds = read_te(item.te_path)
    return latent, embeds


def read_te(path, key: str = "embeds") -> torch.Tensor:
    """One embedding ``(S, 4096)`` bf16 from a TE file, only that key read; an
    ``embeds_v{k}`` tail comes back behind its shared ``embeds`` prefix."""
    with safe_open(str(path), "pt") as f:
        x = f.get_tensor(key)
        prefix = int((f.metadata() or {}).get("variant_prefix", 0))
        if _VARIANT_RE.fullmatch(key) and prefix:
            x = torch.cat((f.get_slice("embeds")[:prefix], x))
    return x.to(torch.bfloat16)


def load_variant(item: CacheItem, k: int) -> torch.Tensor:
    """``embeds_v{k} (S_k, 4096)`` bf16 on CPU, read from disk on each call."""
    return read_te(item.te_path, f"embeds_v{k}")


def load_plain(item: CacheItem) -> torch.Tensor | None:
    """``embeds_plain (S', 4096)`` bf16 on CPU, or ``None`` when the file has none."""
    if not item.n_text_plain:
        return None
    return read_te(item.te_path, "embeds_plain")


# ------------------------------------------------------------------ variants


def variant_entries(
    variants: list[tuple[str, torch.Tensor]], prefix: int
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """TE-file tensors + metadata for ``[(caption, embeds (S_k, 4096)), …]``, each a
    full embedding sharing its first ``prefix`` rows with ``embeds``: only the
    rows after them are stored (``read_te`` puts the prefix back)."""
    if not variants:
        return {}, {}
    tensors, meta = {}, {"variant_prefix": str(prefix)}
    for k, (caption, embeds) in enumerate(variants, 1):
        tensors[f"embeds_v{k}"] = embeds[prefix:].to("cpu", torch.bfloat16).contiguous()
        meta[f"caption_v{k}"] = caption
    return tensors, meta


def add_variants(path, variants: list[tuple[str, torch.Tensor]], prefix: int) -> None:
    """Replace the caption variants of the TE file at ``path`` with ``variants``
    (``variant_entries``), in place through a temp file; a symlinked file is
    rewritten at its target."""
    real = Path(path).resolve()
    tensors, meta = _without_variants(real)
    new, new_meta = variant_entries(variants, prefix)
    tmp = real.with_name(real.name + ".tmp")
    save_file(tensors | new, str(tmp), metadata=meta | new_meta)
    tmp.replace(real)


def _without_variants(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    with safe_open(str(path), "pt") as f:
        tensors = {k: f.get_tensor(k) for k in f.keys() if not _VARIANT_RE.fullmatch(k)}
        meta = f.metadata() or {}
    return tensors, {k: v for k, v in meta.items() if not _VARIANT_META_RE.fullmatch(k)}


def carried_variants(
    path: Path, caption: str
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """The caption variants (tensors, metadata) of the TE file a re-encode is about
    to replace, while its caption is still ``caption``. After a caption edit they
    are stale: dropped, with a warning."""
    if not path.exists():
        return {}, {}
    with safe_open(str(path), "pt") as f:
        keys = [k for k in f.keys() if _VARIANT_RE.fullmatch(k)]
        if not keys:
            return {}, {}
        meta = f.metadata() or {}
        if meta.get("caption_sha256") != caption_sha(caption):
            print(
                f"  warning: {path.name}: caption changed, dropping its "
                f"{len(keys)} stale caption variants",
                flush=True,
            )
            return {}, {}
        tensors = {k: f.get_tensor(k) for k in keys}
    return tensors, {k: v for k, v in meta.items() if _VARIANT_META_RE.fullmatch(k)}


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
    (edited after the export); every other caption is converted here. Either
    way the ``.txt`` rides along as ``plain``, unless it already is layout JSON
    (which has no plain form)."""
    originals, skip = original_tree(src)
    images, found = source_pairs(src, originals, skip)
    if originals is not None:
        n = sum(1 for _, image, _ in found if not image.is_relative_to(src))
        if n:
            print(
                f"{n} captions without a resized image: "
                f"using the originals under {originals}",
                flush=True,
            )
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
        plain = raw if caption_format == "layout_json" and caption != raw else ""
        pairs.append(Pair(stem, image, caption, caption_sha(raw), plain))
    if caption_format == "layout_json":
        print(
            f"layout_json: {from_sidecar} from Export's sidecars, "
            f"{len(pairs) - from_sidecar} converted",
            flush=True,
        )
    return pairs


def manifest_pairs(
    path: Path, caption_format: str = "text"
) -> tuple[list[Pair], dict[str, Path]]:
    """The targets of an i2i pair list as ``Pair``s, and each stem's condition
    image. ``path`` is a ``pairs.jsonl`` (``i2i_tasks`` staging), one
    ``{stem, target, cond, caption}`` object per line, paths relative to it."""
    root = path.parent
    pairs, refs = [], {}
    convert = caption_converter(caption_format) if caption_format != "text" else None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        stem = row["stem"]
        if stem in refs:
            raise SystemExit(f"{path}: stem {stem!r} appears twice")
        raw = (root / row["caption"]).read_text(encoding="utf-8").strip()
        caption = convert(raw) if convert else raw
        plain = raw if caption_format == "layout_json" and caption != raw else ""
        pairs.append(Pair(stem, root / row["target"], caption, caption_sha(raw), plain))
        refs[stem] = root / row["cond"]
    return pairs, refs


def ref_size(width: int, height: int, ref_scale: int) -> tuple[int, int]:
    """Reference ``(w, h)`` for a ``width × height`` target: 1/``ref_scale`` of
    each side, multiples of 16."""
    return (
        max(STRIDE, round(width / ref_scale / STRIDE) * STRIDE),
        max(STRIDE, round(height / ref_scale / STRIDE) * STRIDE),
    )


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


def _te_current(
    path: Path,
    caption: str,
    max_tokens: int,
    plain: str = "",
    encoder: str = ENCODER_ID,
) -> bool:
    if not path.exists():
        return False
    try:
        with safe_open(path, "pt") as f:
            meta = f.metadata() or {}
    except Exception:
        return False
    return (
        meta.get("caption_sha256") == caption_sha(caption)
        and meta.get("max_text_tokens") == str(max_tokens)
        and meta.get("encoder", ENCODER_ID) == encoder
        and meta.get("plain_caption_sha256", "")
        == (caption_sha(plain) if plain else "")
    )


# ------------------------------------------------------------------ passes


def cache_text(
    pairs,
    out_dir,
    model_dir,
    device,
    max_tokens,
    overwrite,
    caption_format="text",
    precision="bf16",
) -> dict:
    from library.moss.loader import (
        empty_cache,
        encode_prompt,
        load_text_encoder,
        load_tokenizer,
        prompt_ids,
    )

    encoder = encoder_id(precision)
    todo = [
        p
        for p in pairs
        if overwrite
        or not _te_current(
            out_dir / f"{p.stem}{TE_SUFFIX}", p.caption, max_tokens, p.plain, encoder
        )
    ]
    if not todo:
        print("text: all cached", flush=True)
        return {"encoded": 0}
    t0 = time.time()
    tok = load_tokenizer(model_dir)
    longest = max(
        prompt_ids(tok, text, max_tokens).input_ids.shape[1]
        for p in todo
        for text in (p.caption, p.plain)
        if text
    )
    tok, te = load_text_encoder(
        model_dir, device, longest, tok=tok, precision=precision
    )
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
        path = out_dir / f"{p.stem}{TE_SUFFIX}"
        tensors, meta = carried_variants(path, p.caption)
        tensors["embeds"] = embeds.contiguous()
        meta |= {
            "caption": p.caption,
            "caption_sha256": caption_sha(p.caption),
            "caption_format": caption_format,
            "source_sha256": p.source_sha,
            "n_text": str(n),
            "max_text_tokens": str(max_tokens),
            "encoder": encoder,
            "source": str(p.image),
        }
        plain_note = ""
        if p.plain:
            plain = encode_prompt(tok, te, p.plain, device, max_tokens).to(
                "cpu", torch.bfloat16
            )
            tensors["embeds_plain"] = plain.contiguous()
            meta["plain_caption_sha256"] = caption_sha(p.plain)
            meta["n_text_plain"] = str(plain.shape[0])
            plain_note = f" (plain {plain.shape[0]})"
        print(
            f"  text {i}/{len(todo)} {p.stem}: {n} tokens{plain_note}{note}",
            flush=True,
        )
        save_file(tensors, path, metadata=meta)
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
        for *_, old in latent_files(out_dir, p.stem):
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


def cache_refs(
    pairs, refs, out_dir, model_dir, device, resolution, ref_scale, vae_dtype, overwrite
) -> dict:
    """``{stem}.moss_ref`` per pair: the condition fitted to the target's cached
    size, then to :func:`ref_size`. Current while newer than the condition and
    stamped with the same target size and ``ref_scale``."""
    from library.moss.loader import empty_cache, load_vae
    from library.moss.model import encode

    plan = []
    for p in pairs:
        with Image.open(p.image) as im:
            w, h = target_size(*im.size, resolution)
        cond = refs[p.stem]
        path = out_dir / f"{p.stem}{REF_SUFFIX}"
        current = False
        if path.exists() and path.stat().st_mtime >= cond.stat().st_mtime:
            with safe_open(path, "pt") as f:
                meta = f.metadata() or {}
            current = meta.get("target") == f"{w}x{h}" and meta.get("ref_scale") == str(
                ref_scale
            )
        if overwrite or not current:
            plan.append((p.stem, cond, w, h, path))
    if not plan:
        print("refs: all cached", flush=True)
        return {"encoded": 0}
    dtype = torch.float32 if vae_dtype == "fp32" else torch.bfloat16
    vae = load_vae(model_dir, dtype=dtype, device=device)
    t0 = time.time()
    for i, (stem, cond, w, h, path) in enumerate(plan, 1):
        rw, rh = ref_size(w, h, ref_scale)
        image = fit(fit(load_rgb(cond), w, h), rw, rh)
        latent = encode(to_pixels(image), vae)[0].cpu().contiguous()
        save_file(
            {"latent": latent},
            path,
            metadata={
                "width": str(rw),
                "height": str(rh),
                "target": f"{w}x{h}",
                "ref_scale": str(ref_scale),
                "condition": str(cond),
                "condition_mtime": str(cond.stat().st_mtime),
            },
        )
        print(f"  refs {i}/{len(plan)} {stem}: {w}x{h} -> {rw}x{rh}", flush=True)
    seconds = time.time() - t0
    print(f"refs: {len(plan)} encoded in {seconds:.1f}s", flush=True)
    del vae
    empty_cache()
    return {"encoded": len(plan), "encode_s": seconds, "ref_scale": ref_scale}


def summarize(out_dir: Path) -> dict:
    """Token statistics over the complete pairs in the cache."""
    items = list_items(out_dir)
    if not items:
        return {"items": 0}
    text = [it.n_text for it in items]
    image = [(it.width // STRIDE) * (it.height // STRIDE) for it in items]
    joint = [a + b + it.n_ref for a, b, it in zip(text, image, items)]
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
    refs = None
    if req.pairs:
        src = resolve_under_home(req.pairs)
        if not src.is_file():
            raise SystemExit(f"no pair list at {src} (make i2i-stage TASK=<task>)")
        pairs, refs = manifest_pairs(src, req.caption_format)
    else:
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
            req.text_encoder,
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
    if refs is not None and not req.skip_latents:
        report["refs"] = cache_refs(
            pairs,
            refs,
            out_dir,
            model_dir,
            device,
            req.resolution,
            req.ref_scale,
            req.vae_dtype,
            req.overwrite,
        )
    report["summary"] = summarize(out_dir)
    (out_dir / "cache_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2), flush=True)
    print(f"cache at {out_dir}", flush=True)
    return out_dir
