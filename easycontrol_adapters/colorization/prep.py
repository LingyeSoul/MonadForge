#!/usr/bin/env python3
"""Mangafy + cache condition latents (and color-only text) for the colorization
EasyControl task.

Three idempotent stages:

1. **Mangafy** — walk every color image under ``--src`` (the existing resized
   training images), screen each to a synthetic B&W manga page, and write the
   result to ``--staging`` mirroring the source subpath. Two engines via
   ``--engine`` (both the XDoG + algorithmic-screentone synthesizer, no
   downloads): ``gpu`` (default) runs the screens on CUDA
   (:func:`mangafy_gpu.mangafy_array_gpu`, fast, serial); ``cv2`` runs the same
   math on a CPU ProcessPool (:func:`mangafy.mangafy_array`). Skips
   already-staged PNGs unless ``--overwrite``.

2. **Encode** — VAE-encode the staged manga images into ``--cond_cache_dir`` via
   the existing ``library.preprocess.cache_latents`` (same ``{stem}_{WxH}_anima.npz``
   format as the target latent cache), at the image's **native size** so each
   cond latent shape matches its target latent exactly. Skips cached resolutions.

3. **Text** — re-encode the *target* captions filtered to **color tags** (plus
   the **copyright/series tag** by default, ``--text_keep_copyright``, and
   optionally **comic/panel-format tags** via ``--text_keep_comic``;
   :func:`color_caption.filter_to_colors_and_protected`) into ``--text_cache_dir``, mirroring
   the source subpath. Reads ``.txt`` from ``--caption_src`` (the caption master,
   nested identically to the resized tree), so the TE caches key-match the
   colorize loader's ``image_dir=post_image_dataset/resized`` lookup. The colorize
   dataset points its subset ``text_cache_dir`` here, so training reads
   color-only captions while latents still come from the shared lora cache.
   Skipped without ``--qwen3``/``--dit`` (omit via ``--skip_text``).

The cond cache is paired with the color target at train time by the loader's
``cond_cache_dir`` subset knob (stem-matched); the color-only text by
``text_cache_dir``. Run from the repo root::

    python easycontrol_adapters/colorization/prep.py            # full dataset (all 3 stages)
    python easycontrol_adapters/colorization/prep.py --limit 8  # quick QA batch

Via the task runner the three stages are split across two targets (knobs come
from the ``[staging]`` / ``[preprocess]`` tables of ``configs/easycontrol/
colorize.toml``)::

    make easycontrol-staging    EASYADAPTER=colorize   # stage 1 (mangafy)
    make easycontrol-preprocess EASYADAPTER=colorize   # stages 2-3 (encode + text)
"""

from __future__ import annotations

import argparse
import os
import tempfile
import zlib
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

from typing import Callable

import numpy as np
import torch
from PIL import Image

from library.preprocess import cache_latents, tqdm_progress
from library.preprocess._dataset import walk_images

# Screener: color RGB uint8 (H,W,3) + per-stem seed → B&W RGB uint8 (H,W,3) same size.
Screener = Callable[[np.ndarray, int], np.ndarray]


def _stable_seed(name: str) -> int:
    """Process-stable per-stem seed (Python's hash() is salted; crc32 isn't)."""
    return zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF


def _save_png_atomic(arr: np.ndarray, out: Path) -> None:
    """Write ``arr`` to ``out`` atomically — temp file in the same dir + os.replace.

    A direct ``Image.save(out)`` that's interrupted (Ctrl-C, OOM, crash) mid-write
    leaves a *truncated* PNG at the final path; the mangafy skip-check
    (``out.exists()``) then keeps that corrupt file forever, and it only blows up
    later at VAE-encode decode time. Writing to a unique temp in the same directory
    and ``os.replace``-ing it in means the final name only ever appears fully
    written (replace is atomic on one filesystem). On any failure the temp is
    removed so no partial file or stray ``.tmp`` survives."""
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=out.parent, suffix=".tmp.png")
    os.close(fd)
    try:
        Image.fromarray(arr).save(tmp)
        os.replace(tmp, out)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _text_mask_path(mask_dir: Path | None, rel: Path) -> Path | None:
    """Locate the ``{stem}_mask.png`` text mask mirroring ``rel`` under ``mask_dir``.

    Returns ``None`` when masking is off or the page has no mask (most pages don't —
    only ~1886/2419 carry text), so the caller leaves the screened result untouched."""
    if mask_dir is None:
        return None
    p = mask_dir / rel.parent / f"{rel.stem}_mask.png"
    return p if p.exists() else None


def _apply_text_mask(
    manga: np.ndarray, img_rgb: np.ndarray, mask_path: Path | None, *, dilate: int
) -> np.ndarray:
    """Paste the source's own grayscale back into the text/speech-bubble regions.

    The MIT text masks (black = text) localize the overlaid text/logo regions;
    replacing them with a deterministic grayscale of the source sharpens the screened
    text so it stays pixel-exact. No-op when there's no mask for this stem or masking
    is disabled. ``dilate`` grows the mask a few px to catch anti-aliased glyph
    fringes."""
    if mask_path is None:
        return manga
    h, w = manga.shape[:2]
    mask = np.asarray(
        Image.open(mask_path).convert("L").resize((w, h), Image.Resampling.NEAREST)
    )
    text = mask < 128  # black = text (merge_masks union polarity)
    if dilate > 0:
        import cv2

        k = np.ones((dilate * 2 + 1, dilate * 2 + 1), np.uint8)
        text = cv2.dilate(text.astype(np.uint8), k).astype(bool)
    if not text.any():
        return manga
    gray = np.asarray(Image.fromarray(img_rgb).convert("L"))
    out = manga.copy()
    out[text] = np.stack([gray] * 3, axis=-1)[text]
    return out


def _make_screener(engine: str) -> Screener:
    """Resolve ``engine`` to a ``(img, seed) → manga`` callable.

    Imports are deferred so ``cv2`` stays import-light. Takes plain (picklable) args
    rather than the argparse Namespace so it can be rebuilt inside a worker process."""
    if engine == "cv2":
        import cv2

        cv2.setNumThreads(
            1
        )  # one cv2 thread per worker — the pool gives the parallelism
        from mangafy import mangafy_array

        return lambda img, seed: mangafy_array(img, seed=seed)

    # gpu: same algorithmic screentone as cv2, but the trig screens + XDoG blurs run on
    # CUDA — one serial GPU pass, no ProcessPool. Structurally identical per seed.
    from mangafy_gpu import mangafy_array_gpu

    return lambda img, seed: mangafy_array_gpu(img, seed=seed)


# ── Worker process state (cv2 engine only) ───────────────────────────────────
# The screener (a closure over a cv2/numpy module) isn't picklable, so each worker
# rebuilds it once in its initializer and stashes it — plus the constant paths — in
# a module global, then maps over image paths. The mangafy math is pure numpy/cv2,
# so this scales ~linearly with cores; the seeded per-stem jitter keeps every worker
# bit-identical to the serial path.
_WORKER: dict = {}


def _worker_init(
    engine: str,
    src: Path,
    staging: Path,
    overwrite: bool,
    mask_dir: Path | None,
    text_mask_dilate: int,
):
    _WORKER.update(
        screener=_make_screener(engine),
        src=src,
        staging=staging,
        overwrite=overwrite,
        mask_dir=mask_dir,
        text_mask_dilate=text_mask_dilate,
    )


def _worker_process(path_str: str) -> int:
    p = Path(path_str)
    rel = p.relative_to(_WORKER["src"])
    out = (_WORKER["staging"] / rel).with_suffix(".png")
    if out.exists() and not _WORKER["overwrite"]:
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    img = np.array(Image.open(p).convert("RGB"))
    manga = _WORKER["screener"](img, _stable_seed(rel.stem))
    manga = _apply_text_mask(
        manga,
        img,
        _text_mask_path(_WORKER["mask_dir"], rel),
        dilate=_WORKER["text_mask_dilate"],
    )
    _save_png_atomic(manga, out)
    return 1


def _mangafy_gpu_overlap(
    images: list[Path],
    src: Path,
    staging: Path,
    *,
    overwrite: bool,
    mask_dir: Path | None,
    text_mask_dilate: int,
    io_workers: int,
    progress: Callable,
) -> tuple[int, int]:
    """GPU mangafy with CPU/IO overlapped against the GPU.

    The screener holds a single GPU stream, so it stays serial on this (main) thread;
    the surrounding disk decode and the text-mask + PNG encode + write are pure CPU/IO,
    so they're farmed to thread pools and overlap the GPU work. ``mangafy_array_gpu``
    returns a host numpy array, so nothing GPU-bound crosses a thread boundary. The
    seed is per-stem, so decoding ahead of order is safe — output is identical."""
    screener = _make_screener("gpu")

    def _decode(p: Path):
        rel = p.relative_to(src)
        out = (staging / rel).with_suffix(".png")
        if out.exists() and not overwrite:
            return rel, out, None  # img=None → skip
        return rel, out, np.array(Image.open(p).convert("RGB"))

    def _finish(manga: np.ndarray, img: np.ndarray, out: Path, rel: Path) -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        manga = _apply_text_mask(
            manga, img, _text_mask_path(mask_dir, rel), dilate=text_mask_dilate
        )
        _save_png_atomic(manga, out)

    written = 0
    depth = max(4, io_workers)  # decode look-ahead window (bounds held-decoded memory)
    max_saves = 2 * io_workers  # cap in-flight saves (bounds held manga+img memory)
    it = iter(images)
    decode_q: deque = deque()
    save_q: deque = deque()
    with (
        ThreadPoolExecutor(max_workers=io_workers) as decode_ex,
        ThreadPoolExecutor(max_workers=io_workers) as save_ex,
    ):
        for _ in range(depth):  # prime the decode window
            p = next(it, None)
            if p is None:
                break
            decode_q.append(decode_ex.submit(_decode, p))
        while decode_q:
            rel, out, img = decode_q.popleft().result()
            p = next(it, None)  # refill so a decode is always in flight
            if p is not None:
                decode_q.append(decode_ex.submit(_decode, p))
            if img is None:
                progress(1, detail=f"skip {out.name}")
                continue
            manga = screener(img, _stable_seed(rel.stem))  # GPU, serial on this thread
            save_q.append(save_ex.submit(_finish, manga, img, out, rel))
            written += 1
            progress(1, detail=rel.name)
            while len(save_q) >= max_saves:  # backpressure on the save side
                save_q.popleft().result()
        for f in save_q:  # drain remaining writes
            f.result()
    return len(images), written


def stage_mangafy(
    src: Path,
    staging: Path,
    *,
    engine: str,
    recursive: bool,
    overwrite: bool,
    limit: int | None,
    workers: int,
    mask_dir: Path | None,
    text_mask_dilate: int,
    only_stems: frozenset[str] | None = None,
    exclude_stems: frozenset[str] = frozenset(),
) -> tuple[int, int]:
    images = walk_images(src, recursive=recursive)
    # Tag selection: keep only_stems (None = all), then drop exclude_stems.
    if only_stems is not None or exclude_stems:
        pre = len(images)
        images = [
            p
            for p in images
            if (only_stems is None or p.stem in only_stems)
            and p.stem not in exclude_stems
        ]
        if len(images) != pre:
            print(
                f"Tag filter: keeping {len(images)}/{pre} images for the colorize set"
            )
    if limit:
        images = images[:limit]
    progress = tqdm_progress("Mangafy")
    progress(0, total=len(images))

    # The cv2 engine fans out across processes (``--workers``) when there's more than
    # one to use; the gpu engine runs one serial GPU stream (handled below).
    if engine == "cv2" and workers > 1 and len(images) > 1:
        written = 0
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_init,
            initargs=(
                engine,
                src,
                staging,
                overwrite,
                mask_dir,
                text_mask_dilate,
            ),
        ) as ex:
            # small chunks: each image is ~seconds of cv2 work, so IPC overhead is
            # negligible and fine-grained chunks load-balance variable image sizes well.
            for w in ex.map(_worker_process, [str(p) for p in images], chunksize=2):
                written += w
                progress(1)
        return len(images), written

    # The gpu engine runs one serial GPU stream, but its disk decode + text-mask + PNG
    # write are CPU/IO — overlap them across thread pools so the GPU isn't starved
    # between images (the 0→100 utilization sawtooth).
    if engine == "gpu" and len(images) > 1:
        return _mangafy_gpu_overlap(
            images,
            src,
            staging,
            overwrite=overwrite,
            mask_dir=mask_dir,
            text_mask_dilate=text_mask_dilate,
            io_workers=max(2, workers),
            progress=progress,
        )

    screener = _make_screener(engine)
    written = 0
    for p in images:
        rel = p.relative_to(src)
        out = (staging / rel).with_suffix(".png")
        if out.exists() and not overwrite:
            progress(1, detail=f"skip {p.name}")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        img = np.array(Image.open(p).convert("RGB"))
        manga = screener(img, _stable_seed(rel.stem))
        manga = _apply_text_mask(
            manga, img, _text_mask_path(mask_dir, rel), dilate=text_mask_dilate
        )
        _save_png_atomic(manga, out)
        written += 1
        progress(1, detail=p.name)
    return len(images), written


def stage_encode(
    staging: Path,
    cond_cache_dir: Path,
    *,
    vae_path: str,
    batch_size: int,
    chunk_size: int,
    recursive: bool,
):
    from library.models import qwen_vae

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading VAE from {vae_path} ...")
    vae = qwen_vae.load_vae(
        vae_path,
        device="cpu",
        disable_mmap=True,
        spatial_chunk_size=chunk_size,
        disable_cache=True,
    )
    vae.to(device, dtype=torch.bfloat16)
    vae.requires_grad_(False)
    vae.eval()

    stats = cache_latents(
        staging,
        vae,
        cache_dir=cond_cache_dir,
        recursive=recursive,
        batch_size=batch_size,
        progress=tqdm_progress("Caching cond latents"),
    )

    vae.to("cpu")
    del vae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return stats


def stage_text(
    caption_src: Path,
    text_cache_dir: Path,
    *,
    qwen3_path: str,
    dit_path: str,
    t5_tokenizer_path: str | None,
    batch_size: int,
    recursive: bool,
    shuffle_variants: int,
    tag_dropout_rate: float,
    keep_copyright: bool = True,
    keep_comic: bool = False,
    caption_index: str | None = None,
    staging: Path | None = None,
):
    """Cache color-only TE embeddings for the color targets into ``text_cache_dir``.

    Loads Qwen3 + the DiT's LLM adapter (needed to produce ``crossattn_emb``),
    then runs ``library.preprocess.cache_text_embeddings`` over ``caption_src``
    with the color-only caption filter. ``caption_src`` (the caption master) is
    nested identically to ``post_image_dataset/resized``, so the resulting TE
    paths key-match the colorize loader's resized-rooted lookup.

    When ``staging`` is given (the synthesized cond tree), encoding is scoped to
    the stems present there — the matched colorize subset the mangafy stage already
    materialized — so the TE cache mirrors the cond cache instead of re-encoding
    the whole master. ``None`` / absent / empty staging = encode every caption.

    With ``shuffle_variants > 0`` each cache holds v0 (the full color set) plus
    shuffled variants with ``tag_dropout_rate`` of the color tags dropped. The
    color filter runs *before* variant generation, and the filtered caption has
    no ``@artist`` prefix, so every color tag is drop-eligible — this teaches the
    model to colorize from a *partial* color spec ("pink hair" alone) rather than
    expecting a complete palette every time.

    With ``keep_copyright`` (default) the copyright/series tag is kept too and
    placed *first*, then protected from tag-dropout via ``caption_protect_fn`` —
    so "genshin impact" survives in every partial-color variant while the colors
    around it still drop. The manga cond can't encode which series a page is from,
    so copyright is genuinely-ambiguous text worth binding. Copyright names are
    matched against ``caption_index`` (defaults to the corpus typed-tag index).

    ``keep_comic`` does the same for comic/panel-format tags (``comic``, ``4koma``,
    …; matched against the fixed :data:`color_caption.COMIC_TAGS` vocab) — kept in
    the protected prefix after copyright and immune to dropout, so the adapter
    learns the ``comic`` tag and a "comic" prompt steers toward panelled output.
    """
    from color_caption import (
        filter_to_colors,
        filter_to_colors_and_protected,
        is_comic_tag,
        load_copyright_tags,
    )

    from library.anima import weights as anima_utils
    from library.anima.strategy import AnimaTextEncodingStrategy, AnimaTokenizeStrategy
    from library.preprocess import cache_text_embeddings

    text_cache_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading Qwen3 text encoder from {qwen3_path} ...")
    text_encoder, qwen3_tokenizer = anima_utils.load_qwen3_text_encoder(
        qwen3_path, dtype=torch.bfloat16, device=str(device)
    )
    t5_tokenizer = anima_utils.load_t5_tokenizer(t5_tokenizer_path)
    print(f"Loading LLM adapter from {dit_path} ...")
    llm_adapter = anima_utils.load_llm_adapter(
        dit_path, dtype=torch.bfloat16, device=str(device)
    )

    tokenize_strategy = AnimaTokenizeStrategy(
        qwen3_tokenizer=qwen3_tokenizer, t5_tokenizer=t5_tokenizer
    )
    encoding_strategy = AnimaTextEncodingStrategy()

    # The colorize loader reuses the shared T5("") uncond sidecar for caption
    # dropout, staged by `make preprocess`; re-stage idempotently in case the
    # color run is the first to touch it.
    from library.preprocess.uncond import (
        DEFAULT_UNCOND_DIR,
        stage_uncond_sidecar_with_models,
    )

    stage_uncond_sidecar_with_models(
        DEFAULT_UNCOND_DIR,
        text_encoder,
        tokenize_strategy,
        encoding_strategy,
        llm_adapter,
        device=device,
        overwrite=False,
    )

    # Resolve the caption transform. Color tags are always kept; copyright and/or
    # comic tags optionally ride along as a leading prefix. Inclusion and dropout-
    # protection are decoupled: comic is the only family protected from dropout —
    # copyright is kept but droppable (it rides the same tag-dropout as the colors),
    # so the protect_fn below covers comic alone.
    caption_protect_fn = None
    copyright_tags: frozenset[str] = frozenset()
    if keep_copyright:
        copyright_tags = (
            load_copyright_tags(caption_index)
            if caption_index
            else load_copyright_tags()
        )
        if not copyright_tags:
            print(
                "Warning: no copyright tags found in caption index "
                f"({caption_index or 'default'}); copyright will not be kept. "
                "Run `make caption-index` or pass --caption_index."
            )

    if keep_copyright or keep_comic:

        def caption_transform(caption: str) -> str:
            return filter_to_colors_and_protected(
                caption,
                copyright_tags,
                keep_copyright=keep_copyright,
                keep_comic=keep_comic,
            )

        def caption_protect_fn(tag: str) -> bool:
            # Comic only — copyright is included by the filter but left droppable.
            return keep_comic and is_comic_tag(tag)
    else:
        caption_transform = filter_to_colors

    # Subset selection: the staged cond tree IS the matched colorize set (mangafy
    # already applied the only/exclude tag filter when it synthesized it), so
    # restrict TE encoding to those stems rather than re-encoding the whole
    # caption master. ``staging`` None (or absent/empty) = no filter — encode all.
    # Note: --limit applies to the mangafy/encode stages only; the text stage
    # walks caption_src itself and skips already-cached files, so re-runs are cheap.
    keep_stems: set[str] | None = None
    if staging is not None and staging.is_dir():
        keep_stems = {p.stem for p in walk_images(staging, recursive=recursive)}
        if not keep_stems:
            print(
                f"Warning: staging tree {staging} is empty; encoding the whole "
                "caption master (run the mangafy staging step first to scope this)."
            )
            keep_stems = None

    stats = cache_text_embeddings(
        caption_src,
        tokenize_strategy,
        encoding_strategy,
        text_encoder,
        llm_adapter=llm_adapter,
        device=device,
        cache_dir=text_cache_dir,
        recursive=recursive,
        keep_stems=keep_stems,
        batch_size=batch_size,
        caption_transform=caption_transform,
        caption_protect_fn=caption_protect_fn,
        caption_shuffle_variants=shuffle_variants,
        caption_tag_dropout_rate=tag_dropout_rate,
        progress=tqdm_progress("Caching color-only text"),
    )

    text_encoder.to("cpu")
    del text_encoder, llm_adapter
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default="post_image_dataset/resized")
    parser.add_argument(
        "--staging", default="post_image_dataset/easycontrol/colorize/staging"
    )
    parser.add_argument(
        "--cond_cache_dir", default="post_image_dataset/easycontrol/colorize/cond"
    )
    parser.add_argument("--vae", default="models/vae/qwen_image_vae.safetensors")
    parser.add_argument(
        "--engine",
        choices=["cv2", "gpu"],
        default="gpu",
        help="condition synthesizer (both XDoG+halftone, no downloads): "
        "gpu = on CUDA (fast, serial); cv2 = same math on a CPU ProcessPool",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="cv2 engine: parallel mangafy worker processes (1 = serial). The gpu "
        "engine ignores this (single GPU stream).",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument(
        "--recursive", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="re-mangafy staged PNGs that exist"
    )
    parser.add_argument("--limit", type=int, default=None, help="cap #images (QA)")
    parser.add_argument(
        "--only_data_includes",
        nargs="*",
        default=[],
        metavar="TAG",
        help="restrict the colorize set to pages whose caption (in --caption_src) "
        "carries one of these standalone tags (e.g. `genshin_impact`). Empty/omitted "
        "= all pages eligible. Applied before --exclude_data_includes.",
    )
    parser.add_argument(
        "--exclude_data_includes",
        nargs="*",
        default=[],
        metavar="TAG",
        help="drop any page whose caption (in --caption_src) carries one of these "
        "standalone tags (e.g. `comic monochrome`). Net set = only_data_includes − "
        "exclude_data_includes. Filtered pages get no condition synthesized, so the "
        "train-time loader pairs them out automatically.",
    )
    parser.add_argument(
        "--mask_dir",
        default="post_image_dataset/masks",
        help="text/speech-bubble masks (black=text); the source grayscale is pasted "
        "back over these regions so the screened text stays pixel-exact",
    )
    parser.add_argument(
        "--skip_text_mask",
        action="store_true",
        help="don't paste source text back (screen the whole page, garbled text and all)",
    )
    parser.add_argument(
        "--text_mask_dilate",
        type=int,
        default=0,
        help="px to grow the text mask before pasting (catches anti-aliased glyph fringes)",
    )
    parser.add_argument(
        "--skip_mangafy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="skip the mangafy condition-synthesis stage (it's the "
        "`easycontrol-staging EASYADAPTER=colorize` step); --no-skip_mangafy "
        "re-stages inline during preprocess",
    )
    parser.add_argument("--skip_encode", action="store_true")
    parser.add_argument(
        "--skip_text", action="store_true", help="skip color-only TE stage"
    )
    # ── color-only text stage ────────────────────────────────────────────────
    parser.add_argument(
        "--caption_src",
        default="image_dataset",
        help="caption master (.txt sidecars), nested like post_image_dataset/resized",
    )
    parser.add_argument(
        "--text_cache_dir",
        default="post_image_dataset/easycontrol/colorize/text",
        help="where color-only TE caches go; the colorize dataset's text_cache_dir",
    )
    parser.add_argument(
        "--qwen3", default="models/text_encoders/qwen_3_06b_base.safetensors"
    )
    parser.add_argument(
        "--dit",
        default="models/diffusion_models/anima-base-v1.0.safetensors",
        help="DiT for the LLM adapter (produces crossattn_emb)",
    )
    parser.add_argument("--t5_tokenizer_path", default=None)
    parser.add_argument(
        "--text_batch_size", type=int, default=16, help="text stage encode batch"
    )
    parser.add_argument(
        "--text_shuffle_variants",
        type=int,
        default=3,
        help="color-caption variants per image (v0=full colors, v1+=shuffled "
        "+ tag-dropped). 0 = single full-color caption.",
    )
    parser.add_argument(
        "--text_tag_dropout_rate",
        type=float,
        default=0.8,
        help="per-color-tag dropout in v1+ variants — trains robustness to "
        "partial color prompts. Ignored when --text_shuffle_variants <= 0.",
    )
    parser.add_argument(
        "--text_keep_copyright",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="carry the copyright/series tag alongside the color tags (placed "
        "first). Kept but NOT protected — it rides the same tag-dropout as the "
        "color tags (only comic is protected). --no-text_keep_copyright drops "
        "it entirely.",
    )
    parser.add_argument(
        "--text_keep_comic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="keep comic/panel-format tags (comic, 4koma, …) alongside the color "
        "tags (placed in the protected prefix, immune to tag-dropout) so the "
        "adapter learns the `comic` tag. On by default; "
        "--no-text_keep_comic drops them.",
    )
    parser.add_argument(
        "--caption_index",
        default=None,
        help="caption index JSON with a groups.copyright vocab, used to identify "
        "copyright tags. Defaults to post_image_dataset/captions/caption_index.json.",
    )
    args = parser.parse_args()

    src = Path(args.src)
    staging = Path(args.staging)
    cond_cache_dir = Path(args.cond_cache_dir)

    # Select which pages enter the colorize set, by caption tags (matched against the
    # caption master, --caption_src): net set = only_data_includes − exclude_data_includes.
    #   • --only_data_includes  — if given, restrict to pages carrying one of these
    #     tags (empty/omitted = all pages eligible);
    #   • --exclude_data_includes — then drop pages carrying one of these.
    # A page filtered out here gets no cond latent, and the train-time loader pairs only
    # against cached cond latents, so the selection carries through to training with no
    # second tag scan.
    from library.datasets.dreambooth import stems_with_any_tag

    only_stems = (
        stems_with_any_tag(args.caption_src, args.only_data_includes)
        if args.only_data_includes
        else None
    )
    exclude_stems = stems_with_any_tag(args.caption_src, args.exclude_data_includes)

    if not args.skip_mangafy:
        seen, written = stage_mangafy(
            src,
            staging,
            engine=args.engine,
            recursive=args.recursive,
            overwrite=args.overwrite,
            limit=args.limit,
            workers=args.workers,
            mask_dir=None if args.skip_text_mask else Path(args.mask_dir),
            text_mask_dilate=args.text_mask_dilate,
            only_stems=only_stems,
            exclude_stems=exclude_stems,
        )
        print(
            f"\nMangafy ({args.engine}): "
            f"{written} written, {seen - written} skipped/seen={seen}"
        )

    if not args.skip_encode:
        stats = stage_encode(
            staging,
            cond_cache_dir,
            vae_path=args.vae,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
            recursive=args.recursive,
        )
        print(
            f"\nCond latent caching complete: {stats.written} cached, "
            f"{stats.skipped} skipped (already existed)"
        )

    if not args.skip_text:
        tstats = stage_text(
            Path(args.caption_src),
            Path(args.text_cache_dir),
            qwen3_path=args.qwen3,
            dit_path=args.dit,
            t5_tokenizer_path=args.t5_tokenizer_path,
            batch_size=args.text_batch_size,
            recursive=args.recursive,
            shuffle_variants=args.text_shuffle_variants,
            tag_dropout_rate=args.text_tag_dropout_rate,
            keep_copyright=args.text_keep_copyright,
            keep_comic=args.text_keep_comic,
            caption_index=args.caption_index,
            staging=staging,
        )
        print(
            f"\nColor-only text caching complete: {tstats.written} cached, "
            f"{tstats.skipped} skipped (already existed)"
        )


if __name__ == "__main__":
    main()
