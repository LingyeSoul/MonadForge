"""What a source folder and a cache folder hold — torch-free, for the GUI.

The file layout constants live here so ``cache.py`` and the GUI name files the
same way. A TE file is current while its ``source_sha256`` (``caption_sha256`` on
files from before ``caption_format``) matches the caption on disk; the check reads
the safetensors JSON header directly, without torch. It does not see a
``caption_format`` switch — ``cache.py`` does, and re-encodes.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTS = (".webp", ".png", ".jpg", ".jpeg")
TE_SUFFIX = ".moss_te.safetensors"
LATENT_SUFFIX = ".moss_latent.safetensors"
LATENT_RE = re.compile(
    r"^(?P<stem>.+)_(?P<w>\d+)x(?P<h>\d+)\.moss_latent\.safetensors$"
)


def source_pairs(src: Path) -> tuple[list[Path], list[tuple[str, Path, Path]]]:
    """Every image under ``src`` and, for those with a ``.txt`` beside them,
    ``(cache stem, image, caption)``. A stem that appears in several subfolders
    is prefixed with its relative folder (``sub__dir__stem``)."""
    images = sorted(
        p for p in src.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file()
    )
    counts: dict[str, int] = {}
    for p in images:
        counts[p.stem] = counts.get(p.stem, 0) + 1
    pairs = []
    for p in images:
        caption = p.with_suffix(".txt")
        if not caption.exists():
            continue
        stem = p.stem
        if counts[stem] > 1:
            stem = "__".join((*p.relative_to(src).parent.parts, p.stem))
        pairs.append((stem, p, caption))
    return images, pairs


def caption_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def header_metadata(path: Path) -> dict:
    """The ``__metadata__`` of a safetensors file, or ``{}`` if unreadable."""
    try:
        with open(path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            return json.loads(f.read(n)).get("__metadata__") or {}
    except (OSError, ValueError, struct.error):
        return {}


@dataclass(frozen=True)
class CacheScan:
    images: int  # images in the source folder
    pairs: int  # of those, with a .txt caption
    text_cached: int  # current TE files
    latents_cached: int
    stale_text: int  # TE file whose caption changed — re-encoded on the next run
    duplicates: int  # stems prefixed with their folder because they collide


def scan(src: Path | None, out: Path | None) -> CacheScan:
    images: list[Path] = []
    pairs: list[tuple[str, Path, Path]] = []
    if src is not None and src.is_dir():
        images, pairs = source_pairs(src)
    dupes = sum(1 for stem, image, _ in pairs if stem != image.stem)
    text = latents = stale = 0
    if out is not None and out.is_dir():
        latent_stems = {
            m["stem"]
            for p in out.glob(f"*{LATENT_SUFFIX}")
            if (m := LATENT_RE.match(p.name))
        }
        for stem, _, caption in pairs:
            te = out / f"{stem}{TE_SUFFIX}"
            if te.exists():
                sha = caption_sha(caption.read_text(encoding="utf-8").strip())
                meta = header_metadata(te)
                if meta.get("source_sha256", meta.get("caption_sha256")) == sha:
                    text += 1
                else:
                    stale += 1
            latents += stem in latent_stems
    return CacheScan(len(images), len(pairs), text, latents, stale, dupes)


@dataclass(frozen=True)
class CacheCounts:
    text: int
    latents: int
    samples: int  # stems with both files — what training will actually see


def cache_counts(cache: Path) -> CacheCounts:
    if not cache.is_dir():
        return CacheCounts(0, 0, 0)
    text = {p.name[: -len(TE_SUFFIX)] for p in cache.glob(f"*{TE_SUFFIX}")}
    latents = {
        m["stem"]
        for p in cache.glob(f"*{LATENT_SUFFIX}")
        if (m := LATENT_RE.match(p.name))
    }
    return CacheCounts(len(text), len(latents), len(text & latents))
