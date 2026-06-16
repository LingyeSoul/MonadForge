#!/usr/bin/env python3
"""PE-Spatial image-feature extraction + per-image cache (library primitive).

Promoted out of the near-twin miner engine so any dataset-level tool — near-twin
pair mining, dataset grouping/clustering, dedup — can share one PE-Spatial
embedding path and one on-disk feature cache. Each image is encoded at PE's
native 512x512 bucket → a global CLS descriptor + a 32x32 patch grid pooled to
16x16 (both L2-normed), cached per-image as ``.npz`` under ``$NEAR_TWIN_CACHE``
(default ``~/.cache/near_twin/``), keyed by parent-dir hash + stem. The cache key
is encoder-agnostic in name but the features are PE-Spatial-B16-512 specific —
consumers that change the encoder must use a fresh cache root.

The encoder *bundle* is loaded by the caller via
``library.vision.load_pe_encoder(device, name="pe_spatial")`` and passed in, so
this module never owns the model lifetime. ``easycontrol_adapters.tools.near_twins.engine``
re-exports these names for backward compatibility.
"""

from __future__ import annotations

import hashlib
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

CACHE_ROOT = Path(
    os.environ.get("NEAR_TWIN_CACHE", Path.home() / ".cache" / "near_twin")
)
IMAGE_EXTS = (".png", ".webp", ".jpg", ".jpeg", ".jxl", ".avif")
PE_NATIVE = 512  # PE-Spatial-B16-512 square bucket → 32x32 patch grid
GRID_NATIVE = 32
GRID_CACHE = 16  # cached pooled grid edge; any pooled grid <= 16 pools down from here


def normalize_tag(tag: str) -> str:
    """Space-insensitive canonical form: lowercase, underscores→spaces, collapsed.

    ``speech_bubble`` and ``speech bubble`` map to the same key so either
    danbooru convention works.
    """
    return " ".join(tag.strip().lower().replace("_", " ").split())


def read_tags(txt_path: Path) -> set[str]:
    """Read a ``.txt`` caption sidecar → set of normalized tags ("" → empty)."""
    if not txt_path.is_file():
        return set()
    raw = txt_path.read_text(encoding="utf-8", errors="ignore")
    return {normalize_tag(t) for t in raw.split(",") if t.strip()}


def caption_text(txt_path: Path) -> str:
    return (
        txt_path.read_text(encoding="utf-8", errors="ignore").strip()
        if txt_path.is_file()
        else ""
    )


@dataclass
class Member:
    artist: str
    stem: str
    image_path: Path
    txt_path: Path
    wh: tuple[int, int] = (0, 0)  # native pixel (W, H); (0, 0) = unreadable header


def _image_size(path: Path) -> tuple[int, int]:
    """Native ``(W, H)`` from the image header (no pixel decode); (0,0) on error."""
    try:
        with Image.open(path) as im:
            return im.size  # PIL returns (width, height)
    except Exception:  # noqa: BLE001 — corrupt/unreadable image
        return (0, 0)


def iter_images(root: Path) -> list[Path]:
    """Every image file under ``root`` (recursive), sorted.

    Mirrors the GUI dataset browser's walk (``gui.discovery._imgs``) so a
    curation tool sees the same image pool the user browses. Follows symlinked
    subtrees (``image_dataset/`` is symlinks into nested artist dirs).
    """
    if not root.is_dir():
        return []
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def gather_members(
    image_dirs: list[Path], artists_filter: set[str] | None
) -> dict[str, list[Member]]:
    """Walk ``<dir>/<artist>/<stem>.<ext>`` trees → ``artist -> [Member]``.

    Scope is ``union`` across all ``image_dirs`` (a twin can straddle the
    curated cut). A ``(artist, stem)`` seen in more than one dir is kept once;
    the first dir listed wins (so put your preferred source — e.g. the curated
    ``selected`` PNGs — first if it matters for the export symlink target).
    """
    seen: dict[tuple[str, str], Member] = {}
    for d in image_dirs:
        if not d.is_dir():
            print(f"  [warn] image dir not found: {d}", file=sys.stderr)
            continue
        for artist_dir in sorted(p for p in d.iterdir() if p.is_dir()):
            artist = artist_dir.name
            if artists_filter and artist not in artists_filter:
                continue
            for img in sorted(artist_dir.iterdir()):
                if img.suffix.lower() not in IMAGE_EXTS:
                    continue
                key = (artist, img.stem)
                if key in seen:
                    continue
                seen[key] = Member(
                    artist, img.stem, img, img.with_suffix(".txt"), _image_size(img)
                )
    by_artist: dict[str, list[Member]] = {}
    for (artist, _), m in seen.items():
        by_artist.setdefault(artist, []).append(m)
    for artist in by_artist:
        by_artist[artist].sort(key=lambda m: m.stem)
    return by_artist


def keep_size_cohabiting(members: list[Member]) -> list[Member]:
    """Drop members with no exact same-size sibling — they can never form a pair.

    The same-size gate's pre-embedding half: a unique canvas size within an
    artist has nothing to pair against, so embedding it would be wasted work.
    Kept here as a generic ``Member`` filter; the miner's tag-pivot prune
    (``prune_for_pairing``) lives in the near-twin engine.
    """
    sizes = Counter(m.wh for m in members)
    return [m for m in members if m.wh != (0, 0) and sizes[m.wh] >= 2]


def _dir_hash(path: Path) -> str:
    return hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:16]


def _cache_path(member: Member) -> Path:
    return CACHE_ROOT / _dir_hash(member.image_path.parent) / f"{member.stem}.npz"


def _load_512(image_path: Path) -> torch.Tensor:
    """PIL → [3, 512, 512] in [-1, 1] (PE's Normalize(0.5, 0.5))."""
    with Image.open(image_path) as im:
        im = im.convert("RGB").resize((PE_NATIVE, PE_NATIVE), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0  # [H, W, 3] in [0, 1]
    t = torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]
    return t * 2.0 - 1.0


_BAD_TENSOR = torch.zeros(3, PE_NATIVE, PE_NATIVE)  # placeholder for a failed decode


class _ImageDataset(torch.utils.data.Dataset):
    """Decode+resize on DataLoader workers so CPU preprocessing overlaps the GPU
    forward. A corrupt image yields ``ok=False`` (skipped downstream) instead of
    crashing the whole pass."""

    def __init__(self, members: list[Member]):
        self.members = members

    def __len__(self) -> int:
        return len(self.members)

    def __getitem__(self, i: int):
        try:
            return i, _load_512(self.members[i].image_path), True
        except Exception:  # noqa: BLE001 — corrupt/unreadable image
            return i, _BAD_TENSOR, False


def _collate(batch):
    idxs = [b[0] for b in batch]
    tens = torch.stack([b[1] for b in batch])
    oks = [b[2] for b in batch]
    return idxs, tens, oks


@torch.no_grad()
def _forward_pe(bundle, batch: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Run PE-Spatial on a [B, 3, 512, 512] device batch → (cls, grid16) numpy."""
    out = bundle.encoder(batch)
    lhs = out.last_hidden_state.float()  # [B, 1+1024, 768]
    cls = F.normalize(lhs[:, 0], dim=-1)  # global descriptor
    grid = lhs[:, 1:].reshape(lhs.shape[0], GRID_NATIVE, GRID_NATIVE, -1)
    g = grid.permute(0, 3, 1, 2)  # [B, 768, 32, 32]
    g16 = F.adaptive_avg_pool2d(g, GRID_CACHE).permute(0, 2, 3, 1)  # [B, 16, 16, 768]
    return cls.cpu().numpy(), g16.cpu().numpy().astype(np.float16)


@dataclass
class Feature:
    cls: np.ndarray  # [768] L2-normed float32
    grid16: np.ndarray  # [16, 16, 768] float16


def _save_feature(cache_path: Path, f: Feature) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, cls=f.cls.astype(np.float32), grid16=f.grid16)


def embed_members(
    bundle, members: list[Member], batch_size: int, num_workers: int = 4
) -> dict[str, Feature]:
    """Load cached PE-Spatial features; embed + cache any misses once.

    Misses are streamed through a ``DataLoader``: worker processes decode +
    resize the next batches while the GPU runs the current forward, the batch is
    copied to the device with pinned-memory async H2D, and the ``.npz`` cache
    writes are handed to a thread pool — so CPU I/O, the host→device copy, and
    the GPU forward overlap instead of running serially.

    Returned dict is keyed by member ``stem`` (the pipeline enforces unique
    stems across the tree); a member whose image fails to decode is omitted.
    """
    feats: dict[str, Feature] = {}
    todo: list[Member] = []
    for m in members:
        cp = _cache_path(m)
        if cp.is_file():
            with np.load(cp) as z:
                feats[m.stem] = Feature(
                    cls=z["cls"].astype(np.float32), grid16=z["grid16"]
                )
        else:
            todo.append(m)
    if not todo:
        return feats

    pin = bundle.device.type == "cuda"
    loader = torch.utils.data.DataLoader(
        _ImageDataset(todo),
        batch_size=batch_size,
        num_workers=min(num_workers, len(todo)),
        pin_memory=pin,
        collate_fn=_collate,
        persistent_workers=False,
    )
    # tqdm to stderr so the daemon captures it and the GUI progress-bar tracker
    # (gui/progress.py TQDM_RE) can drive a determinate bar over the embed pass.
    from tqdm import tqdm

    pbar = tqdm(total=len(todo), desc="embedding", unit="img", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=2) as saver:
        for idxs, tens, oks in loader:
            batch = tens.to(bundle.device, bundle.dtype, non_blocking=pin)
            cls_b, grid_b = _forward_pe(bundle, batch)
            for k, i in enumerate(idxs):
                if not oks[k]:
                    print(
                        f"  [warn] skipped unreadable {todo[i].image_path}",
                        file=sys.stderr,
                    )
                    continue
                f = Feature(cls=cls_b[k], grid16=grid_b[k])
                feats[todo[i].stem] = f
                saver.submit(_save_feature, _cache_path(todo[i]), f)
            pbar.update(len(idxs))
    pbar.close()
    return feats
