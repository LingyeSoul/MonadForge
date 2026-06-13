#!/usr/bin/env python3
"""near_twin engine — discovery, embedding, Stage-B grid match, discriminators.

The algorithm core of the near-twin tag-gap miner: everything from member
discovery through ``run_artist`` (Stages A/B + the discriminators). Rendering and
export live in ``near_twin.outputs``; the CLI + config layering in
``near_twin.__main__`` (see that module's docstring for the full pipeline).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Run from the repo root; `library` is installed editable (`uv sync`).
from library.vision.encoder import encode_pe_from_imageminus1to1  # noqa: F401  (kept for API parity)

REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE_ROOT = Path(
    os.environ.get("NEAR_TWIN_CACHE", Path.home() / ".cache" / "near_twin")
)
IMAGE_EXTS = (".png", ".webp", ".jpg", ".jpeg", ".jxl", ".avif")
PE_NATIVE = 512  # PE-Spatial-B16-512 square bucket → 32x32 patch grid
GRID_NATIVE = 32
GRID_CACHE = 16  # cached pooled grid edge; any --grid <= 16 pools down from here

# ---------------------------------------------------------------------------- captions / tags


def normalize_tag(tag: str) -> str:
    """Space-insensitive canonical form: lowercase, underscores→spaces, collapsed.

    ``speech_bubble`` and ``speech bubble`` map to the same key so either
    danbooru convention works (see the tag-form note in the proposal).
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


# ---------------------------------------------------------------------------- member discovery


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


def keep_size_cohabiting(members: list[Member]) -> list[Member]:
    """Drop members with no exact same-size sibling — they can never form a pair.

    The same-size gate's pre-embedding half: a unique canvas size within an
    artist has nothing to pair against, so embedding it would be wasted work.
    """
    sizes = Counter(m.wh for m in members)
    return [m for m in members if m.wh != (0, 0) and sizes[m.wh] >= 2]


def prune_for_pairing(
    members: list[Member], mode: str, target: set[str]
) -> list[Member]:
    """Members that could still form an accepted pair → embed only these.

    Region/signal modes: just the same-size gate (``keep_size_cohabiting``).

    Tag mode adds the **tag pivot**: an accepted pair has the target tag in
    *exactly one* member, so a same-size group is only useful when it holds BOTH
    a tagged and an untagged member. An all-tagged or all-untagged size group can
    never produce an exactly-one gap, so none of it is worth embedding.
    """
    if mode != "tag":
        return keep_size_cohabiting(members)
    by_size: dict[tuple[int, int], list[Member]] = {}
    for m in members:
        if m.wh != (0, 0):
            by_size.setdefault(m.wh, []).append(m)
    kept: list[Member] = []
    for grp in by_size.values():
        flags = [bool(target & read_tags(m.txt_path)) for m in grp]
        if any(flags) and not all(flags):  # both a tagged and an untagged member
            kept.extend(grp)
    return kept


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


# ---------------------------------------------------------------------------- embedding + cache


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


def _save_feature(cache_path: Path, f: "Feature") -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, cls=f.cls.astype(np.float32), grid16=f.grid16)


@dataclass
class Feature:
    cls: np.ndarray  # [768] L2-normed float32
    grid16: np.ndarray  # [16, 16, 768] float16


def embed_members(
    bundle, members: list[Member], batch_size: int, num_workers: int = 4
) -> dict[str, Feature]:
    """Load cached PE-Spatial features; embed + cache any misses once.

    Misses are streamed through a ``DataLoader``: worker processes decode +
    resize the next batches while the GPU runs the current forward, the batch is
    copied to the device with pinned-memory async H2D, and the ``.npz`` cache
    writes are handed to a thread pool — so CPU I/O, the host→device copy, and
    the GPU forward overlap instead of running serially.
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
    done = 0
    with ThreadPoolExecutor(max_workers=2) as saver:
        for idxs, tens, oks in loader:
            batch = tens.to(bundle.device, bundle.dtype, non_blocking=pin)
            cls_b, grid_b = _forward_pe(bundle, batch)
            for k, i in enumerate(idxs):
                done += 1
                if not oks[k]:
                    print(
                        f"  [warn] skipped unreadable {todo[i].image_path}",
                        file=sys.stderr,
                    )
                    continue
                f = Feature(cls=cls_b[k], grid16=grid_b[k])
                feats[todo[i].stem] = f
                saver.submit(_save_feature, _cache_path(todo[i]), f)
            print(f"  embedded {done}/{len(todo)}", end="\r", file=sys.stderr)
    print(file=sys.stderr)
    return feats


# ---------------------------------------------------------------------------- Stage B match


@dataclass
class MatchResult:
    n_inliers: int
    match_frac: float
    diff_cells: set[int]  # union of unmatched a/b cells (grid index r*G + c)
    diff_a: set[int]
    diff_b: set[int]
    offset: tuple[float, float]  # estimated (drow, dcol) crop offset (geom-check)
    G: int


def _pool_cells(grid16: np.ndarray, G: int) -> np.ndarray:
    """[16, 16, 768] → [G*G, 768], L2-normed per cell."""
    t = torch.from_numpy(grid16.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    p = F.adaptive_avg_pool2d(t, G)[0].permute(1, 2, 0).reshape(G * G, -1).numpy()
    return p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-8)


def match_grids(
    fa: Feature, fb: Feature, G: int, cell_min: float, ratio: float, geom_check: bool
) -> MatchResult:
    """Mutual-NN + ratio-test dense cell match between two pooled grids.

    Raw "has a >0.9 neighbor" inflates badly on anime art's flat color fields,
    so we require **mutual** nearest neighbours that also pass a distinctiveness
    (ratio) test. Unmatched cells localize the difference region.
    """
    ca, cb = _pool_cells(fa.grid16, G), _pool_cells(fb.grid16, G)
    N = G * G
    sim = ca @ cb.T  # [N, N] cosine (cells are unit-norm)
    col_best = sim.argmax(axis=0)
    matched: list[tuple[int, int]] = []
    for i in range(N):
        row = sim[i]
        order = np.argsort(-row)
        nn1 = int(order[0])
        s1 = float(row[nn1])
        s2 = float(row[order[1]]) if N > 1 else -1.0
        if s1 < cell_min:
            continue
        if col_best[nn1] != i:  # mutual NN
            continue
        if (1.0 - s1) > ratio * (1.0 - s2):  # ratio test on cosine-distance
            continue
        matched.append((i, nn1))

    offset = (0.0, 0.0)
    if geom_check and matched:
        matched, offset = _geom_filter(matched, G)

    matched_a = {i for i, _ in matched}
    matched_b = {j for _, j in matched}
    diff_a = set(range(N)) - matched_a
    diff_b = set(range(N)) - matched_b
    return MatchResult(
        n_inliers=len(matched),
        match_frac=len(matched) / N,
        diff_cells=diff_a | diff_b,
        diff_a=diff_a,
        diff_b=diff_b,
        offset=offset,
        G=G,
    )


def _geom_filter(
    matched: list[tuple[int, int]], G: int
) -> tuple[list[tuple[int, int]], tuple[float, float]]:
    """RANSAC-lite translation consistency: keep matches near the median offset.

    Rejects "same character, different pose" (whose cell offsets scatter) and
    estimates the crop offset from the surviving translation. A full
    LoFTR/SIFT homography is the escape hatch if the coarse grid is too blunt.
    """
    a = np.array([[i // G, i % G] for i, _ in matched], dtype=np.float32)
    b = np.array([[j // G, j % G] for _, j in matched], dtype=np.float32)
    deltas = b - a
    med = np.median(deltas, axis=0)
    keep = (
        np.abs(deltas - med).max(axis=1) <= 1.0
    )  # within 1 cell of the consensus shift
    kept = [m for m, k in zip(matched, keep) if k]
    return kept, (float(med[0]), float(med[1]))


def _largest_blob(cells: set[int], G: int) -> tuple[int, set[int]]:
    """Largest 4-connected component of ``cells`` on the G×G grid → (size, blob)."""
    remaining = set(cells)
    best: set[int] = set()
    while remaining:
        seed = next(iter(remaining))
        comp: set[int] = set()
        q = deque([seed])
        remaining.discard(seed)
        while q:
            c = q.popleft()
            comp.add(c)
            r, col = divmod(c, G)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, col + dc
                if 0 <= nr < G and 0 <= nc < G:
                    n = nr * G + nc
                    if n in remaining:
                        remaining.discard(n)
                        q.append(n)
        if len(comp) > len(best):
            best = comp
    return len(best), best


def diff_bbox_norm(cells: set[int], G: int) -> tuple[float, float, float, float]:
    """Bounding box over ``cells`` (grid idx) → normalized (x0, y0, x1, y1) in [0,1]."""
    if not cells:
        return (0.0, 0.0, 0.0, 0.0)
    rs = [c // G for c in cells]
    cs = [c % G for c in cells]
    return (min(cs) / G, min(rs) / G, (max(cs) + 1) / G, (max(rs) + 1) / G)


# ---------------------------------------------------------------------------- discriminator


@dataclass
class PairVerdict:
    accept: bool
    gap_holder: str  # "a" | "b" | "?"
    n_extra_diff: int
    extra_diff_tags: list[str]
    reason: str = ""


def discriminate_tag(
    tags_a: set[str],
    tags_b: set[str],
    target: set[str],
    max_extra: int,
    rest_jaccard_min: float,
) -> PairVerdict:
    """Keep pairs where the target tag is present in **exactly one** member."""
    in_a = bool(target & tags_a)
    in_b = bool(target & tags_b)
    if in_a == in_b:  # both or neither → not a single-attribute gap on this tag
        return PairVerdict(False, "?", 0, [], "tag present in both/neither")
    gap_holder = "a" if in_a else "b"
    rest_a, rest_b = tags_a - target, tags_b - target
    extra = sorted(rest_a ^ rest_b)
    if max_extra >= 0 and len(extra) > max_extra:
        return PairVerdict(
            False, gap_holder, len(extra), extra, "too many other tag differences"
        )
    if rest_jaccard_min > 0:
        union = rest_a | rest_b
        jac = len(rest_a & rest_b) / len(union) if union else 1.0
        if jac < rest_jaccard_min:
            return PairVerdict(
                False, gap_holder, len(extra), extra, f"rest-jaccard {jac:.2f} < min"
            )
    return PairVerdict(True, gap_holder, len(extra), extra)


def discriminate_region(
    match: MatchResult, region_min_frac: float, region_max_frac: float, scatter_max: int
) -> PairVerdict:
    """Tagless: a single compact diff region of the right rough size IS the gap.

    ``gap_holder`` is a best-effort guess (the member with more unmatched cells
    in the diff region — the side carrying the added content); cross-check by
    eyeballing the HTML.
    """
    N = match.G * match.G
    size, blob = _largest_blob(match.diff_cells, match.G)
    frac = size / N
    scatter = len(match.diff_cells) - size  # diff cells outside the main blob
    if not (region_min_frac <= frac <= region_max_frac):
        return PairVerdict(
            False, "?", scatter, [], f"region frac {frac:.2f} out of range"
        )
    if scatter > scatter_max:
        return PairVerdict(False, "?", scatter, [], f"diff scatter {scatter} > max")
    gap_holder = "a" if len(match.diff_a & blob) >= len(match.diff_b & blob) else "b"
    return PairVerdict(True, gap_holder, scatter, [])


# ---------------------------------------------------------------------------- ranked pair record


@dataclass
class PairRecord:
    artist: str
    a: Member
    b: Member
    cosine: float
    match: MatchResult
    verdict: PairVerdict
    extra_fields: dict = field(default_factory=dict)

    @property
    def ids(self) -> tuple[str, str]:
        return tuple(sorted((self.a.stem, self.b.stem)))

    @property
    def pair_id(self) -> str:
        x, y = self.ids
        return f"{x}-{y}"

    def holder_member(self) -> Member:
        return self.a if self.verdict.gap_holder == "a" else self.b

    def clean_member(self) -> Member:
        return self.b if self.verdict.gap_holder == "a" else self.a


# ---------------------------------------------------------------------------- core run


def run_artist(
    artist: str,
    members: list[Member],
    feats: dict[str, Feature],
    args: argparse.Namespace,
    target_tags: set[str],
) -> list[PairRecord]:
    out: list[PairRecord] = []
    stems = [m.stem for m in members if m.stem in feats]
    cls = np.stack([feats[s].cls for s in stems]) if stems else np.empty((0, 768))
    idx = {m.stem: m for m in members}
    # Tag mode: read each sidecar once and cache the target-presence flag + full
    # tagset (reused by discriminate_tag below, so no per-pair re-read).
    tagsets = (
        {s: read_tags(idx[s].txt_path) for s in stems} if args.mode == "tag" else {}
    )
    n = len(stems)
    for i in range(n):
        for j in range(i + 1, n):
            si, sj = stems[i], stems[j]
            if idx[si].wh != idx[sj].wh:  # same-size gate: exact (W, H) only
                continue
            if args.mode == "tag":  # tag pivot: skip same-status pairs before the match
                if bool(target_tags & tagsets[si]) == bool(target_tags & tagsets[sj]):
                    continue
            if args.id_window and si.isdigit() and sj.isdigit():
                if abs(int(si) - int(sj)) > args.id_window:
                    continue
            cos = float(cls[i] @ cls[j])
            if cos < args.sim_min:  # Stage A prefilter
                continue
            match = match_grids(
                feats[si],
                feats[sj],
                args.grid,
                args.cell_match_min,
                args.ratio,
                args.geom_check,
            )
            if match.match_frac < args.match_frac_min:  # Stage B near-twin gate
                continue
            ma, mb = idx[si], idx[sj]
            if args.mode == "tag":
                verdict = discriminate_tag(
                    tagsets[si],
                    tagsets[sj],
                    target_tags,
                    args.max_extra_diff,
                    args.rest_jaccard_min,
                )
            elif args.mode == "region":
                verdict = discriminate_region(
                    match,
                    args.region_min_frac,
                    args.region_max_frac,
                    args.region_scatter_max,
                )
            else:  # signal
                verdict = discriminate_signal(ma, mb, args)
            if not verdict.accept:
                continue
            out.append(PairRecord(artist, ma, mb, cos, match, verdict))

    # Rank by edit-cleanliness: fewest other differences first, then tightest twin.
    out.sort(key=lambda p: (p.verdict.n_extra_diff, -p.cosine))
    if args.per_artist_topk:
        out = out[: args.per_artist_topk]
    return out


# ---------------------------------------------------------------------------- signal mode (MIT text)


_SIGNAL_STATE: dict = {}


def _mit_text_fraction(member: Member, device: str) -> float:
    """Per-image MIT text-area fraction — the detector behind post_image_dataset/masks/."""
    if "mit_model" not in _SIGNAL_STATE:
        sys.path.insert(0, str(REPO_ROOT / "scripts" / "preprocess"))
        from generate_masks_mit import _detect_mask, _load_model  # type: ignore

        _SIGNAL_STATE["mit_model"] = _load_model(None, device=device)
        _SIGNAL_STATE["mit_detect"] = _detect_mask
    cache = CACHE_ROOT / _dir_hash(member.image_path.parent) / f"{member.stem}.mit_text"
    if cache.is_file():
        return float(cache.read_text())
    with Image.open(member.image_path) as im:
        arr = np.asarray(im.convert("RGB"))
    mask = _SIGNAL_STATE["mit_detect"](
        _SIGNAL_STATE["mit_model"], arr, device=device, text_threshold=0.8
    )
    frac = float(np.count_nonzero(mask) / mask.size)
    cache.write_text(f"{frac:.6f}")
    return frac


def discriminate_signal(a: Member, b: Member, args: argparse.Namespace) -> PairVerdict:
    if args.signal != "mit_text":
        raise ValueError(f"unknown signal {args.signal!r}")
    fa = _mit_text_fraction(a, args.device)
    fb = _mit_text_fraction(b, args.device)
    hi, lo = (fa, fb) if fa >= fb else (fb, fa)
    if (
        hi - lo < args.signal_delta or lo > args.signal_delta
    ):  # gap present + low side ≈ 0
        return PairVerdict(False, "?", 0, [], f"signal gap {hi - lo:.3f} insufficient")
    gap_holder = "a" if fa >= fb else "b"
    return PairVerdict(True, gap_holder, 0, [])
