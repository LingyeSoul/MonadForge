#!/usr/bin/env python3
"""Dataset image grouping — connected-components clustering on PE-Spatial grids.

A *curation* primitive (distinct from ``library/preprocess``, which makes data
training-ready): group visually near-identical / same-concept images so the GUI
dataset browser can filter by group, dedup is easy to spot, and subsets are easy
to balance. The grouping edge is the **same near-twin gate the miner uses**
(:mod:`library.vision.pe_matching`): two images are connected when a large
*fraction* of their pooled PE-Spatial patch cells find a distinctive mutual
nearest neighbour. A cheap unit-norm CLS-cosine prefilter (``sim_min``) prunes
obviously-unrelated pairs before the dense grid match so the per-artist pass
stays ``O(candidate pairs)`` rather than ``O(n^2)`` grids.

Why grid match, not CLS cosine: a single global CLS descriptor collapses an
image to one vector, so a tight threshold misses near-duplicates that differ in
one region (an edit, a speech bubble) while a loose one merges unrelated images
that share a palette. The cell-level gate (``cell_match_min`` per cell,
``match_frac_min`` fraction of cells) tracks *spatial* overlap, which is what
"these are the same picture" actually means.

Scope is **per-artist**: components are computed independently within each
top-level folder under the source dir (the artist bucket), so two different
artists never merge into one group. The result is a JSON manifest the GUI reads
(plain JSON — no torch — so the GUI stays torch-free).

``connected_components`` is pure (union-find over a precomputed edge list).
``build_groups`` lazily imports the torch-backed embedding + matching
primitives, so importing this module for the pure clustering / manifest schema
stays torch-free.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MANIFEST_VERSION = 2
# Stage-B near-twin gate — same semantics as the miner's defaults, but looser on
# the inlier fraction so a partial overlap (one edited region) still groups.
DEFAULT_CELL_MATCH_MIN = 0.93
DEFAULT_MATCH_FRAC_MIN = 0.25
DEFAULT_SIM_MIN = (
    0.5  # Stage-A CLS-cosine prefilter (loose; the grid match is the gate)
)
DEFAULT_GRID = 7  # pooled grid edge (G×G cells), mirrors the miner
DEFAULT_RATIO = 0.8  # ratio-test distinctiveness (lower = stricter)


def connected_components(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    """Union-find over an explicit edge list → components as row-index lists.

    ``n`` rows, ``edges`` the unordered ``(i, j)`` pairs that share a component.
    Components are returned sorted (each member list ascending) and ordered by
    descending size.
    """
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path halving
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in edges:
        union(a, b)

    comps: dict[int, list[int]] = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    groups = [sorted(v) for v in comps.values()]
    groups.sort(key=lambda g: (-len(g), g[0]))
    return groups


def _mean_pairwise_cosine(sub: np.ndarray) -> float:
    """Mean off-diagonal cosine of a small (already-normed) embedding block."""
    if sub.shape[0] < 2:
        return 1.0
    sim = sub @ sub.T
    iu = np.triu_indices(sub.shape[0], k=1)
    return float(sim[iu].mean())


def _artist_of(rel: Path) -> str:
    """Top-level folder under the source dir = the artist bucket ("" if flat)."""
    return rel.parts[0] if len(rel.parts) > 1 else ""


def _grid_match_edges(
    bucket_feats: list,
    cls: np.ndarray,
    *,
    device,
    sim_min: float,
    grid: int,
    cell_match_min: float,
    match_frac_min: float,
    ratio: float,
    pool_chunk: int = 256,
    pair_chunk: int = 4096,
) -> list[tuple[int, int]]:
    """Near-twin edges within one artist bucket (Stage-A prefilter → Stage-B grid).

    Runs on ``device`` (the embedding GPU when available): pool every image's
    patch grid **once**, prefilter pairs by CLS cosine ``>= sim_min``, then score
    the surviving pairs with the batched grid gate, keeping an edge where the
    inlier fraction reaches ``match_frac_min``. Pooling and the pair match are
    chunked so peak memory is bounded regardless of bucket size.
    """
    import torch

    from library.vision.pe_matching import match_fracs, pool_cells_batch

    n = len(bucket_feats)
    # Pool each image's grid once (chunked H2D → [n, G*G, D] on device).
    cells_parts = []
    for s in range(0, n, pool_chunk):
        block = np.stack([f.grid16 for f in bucket_feats[s : s + pool_chunk]])
        t = torch.from_numpy(block.astype(np.float32)).to(device)
        cells_parts.append(pool_cells_batch(t, grid))
    cells = torch.cat(cells_parts, dim=0)  # [n, G*G, D]

    # Stage-A: CLS-cosine prefilter over the upper triangle, on device.
    cls_t = torch.from_numpy(cls).to(device)
    sims = cls_t @ cls_t.T
    iu = torch.triu_indices(n, n, offset=1, device=cls_t.device)
    cand = sims[iu[0], iu[1]] >= sim_min
    pi, pj = iu[0][cand], iu[1][cand]  # candidate pair endpoints

    # Stage-B: batched grid match over the candidate pairs, chunked.
    edges: list[tuple[int, int]] = []
    for s in range(0, pi.numel(), pair_chunk):
        ci, cj = pi[s : s + pair_chunk], pj[s : s + pair_chunk]
        frac = match_fracs(cells[ci], cells[cj], cell_match_min, ratio)
        keep = torch.nonzero(frac >= match_frac_min, as_tuple=False).squeeze(-1)
        for k in keep.tolist():
            edges.append((int(ci[k]), int(cj[k])))
    return edges


def build_groups(
    source_dir: Path,
    out_path: Path,
    *,
    cell_match_min: float = DEFAULT_CELL_MATCH_MIN,
    match_frac_min: float = DEFAULT_MATCH_FRAC_MIN,
    sim_min: float = DEFAULT_SIM_MIN,
    grid: int = DEFAULT_GRID,
    ratio: float = DEFAULT_RATIO,
    min_size: int = 2,
    encoder: str = "pe_spatial",
    device: str | None = None,
    batch_size: int = 16,
    num_workers: int = 4,
) -> dict:
    """Embed every image under ``source_dir``, cluster per-artist, write a manifest.

    Groups are connected components over the near-twin grid gate
    (:func:`library.vision.pe_matching.match_grids`): two images share a group
    when ``match_frac >= match_frac_min`` at per-cell floor ``cell_match_min``.
    Reuses the shared PE-Spatial feature cache (``library.vision.pe_features``),
    so a re-run after the near-twin miner — or a re-run with different
    thresholds — is just the (cached) matching pass. Returns the manifest dict
    (also written to ``out_path`` as JSON).
    """
    # Lazy torch-backed imports so the pure helpers above import without torch.
    import sys

    import torch
    from tqdm import tqdm

    from library.vision import load_pe_encoder
    from library.vision.pe_features import Member, embed_members, iter_images

    source_dir = Path(source_dir)
    paths = iter_images(source_dir)

    manifest: dict = {
        "version": MANIFEST_VERSION,
        "source_dir": str(source_dir),
        "encoder": encoder,
        "cell_match_min": cell_match_min,
        "match_frac_min": match_frac_min,
        "sim_min": sim_min,
        "grid": grid,
        "ratio": ratio,
        "min_size": min_size,
        "n_images": len(paths),
        "n_groups": 0,
        "n_grouped": 0,
        "n_singletons": len(paths),
        "groups": [],
    }
    if not paths:
        _write_manifest(out_path, manifest)
        return manifest

    # Bucket by artist (top-level dir) for per-artist scoping.
    by_artist: dict[str, list[Path]] = {}
    for p in paths:
        rel = p.relative_to(source_dir)
        by_artist.setdefault(_artist_of(rel), []).append(p)

    members = [
        Member(
            artist=_artist_of(p.relative_to(source_dir)),
            stem=p.stem,
            image_path=p,
            txt_path=p.with_suffix(".txt"),
        )
        for p in paths
    ]
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    bundle = load_pe_encoder(dev, name=encoder)
    feats = embed_members(bundle, members, batch_size, num_workers)

    groups: list[dict] = []
    gid = 0
    n_grouped = 0
    # tqdm to stderr (GUI bar) — covers the matching pass, the only progress
    # signal on a cached re-run where embedding is skipped.
    for artist in tqdm(
        sorted(by_artist), desc="grouping", unit="artist", file=sys.stderr
    ):
        bucket = [p for p in by_artist[artist] if p.stem in feats]
        if len(bucket) < 2:
            continue
        bucket_feats = [feats[p.stem] for p in bucket]
        cls = np.stack([f.cls for f in bucket_feats]).astype(np.float32)
        cls /= np.linalg.norm(cls, axis=1, keepdims=True) + 1e-8
        edges = _grid_match_edges(
            bucket_feats,
            cls,
            device=dev,
            sim_min=sim_min,
            grid=grid,
            cell_match_min=cell_match_min,
            match_frac_min=match_frac_min,
            ratio=ratio,
        )
        for comp in connected_components(len(bucket), edges):
            if len(comp) < min_size:
                continue
            members_rel = [bucket[i].relative_to(source_dir).as_posix() for i in comp]
            groups.append(
                {
                    "id": gid,
                    "artist": artist,
                    "size": len(comp),
                    "mean_cosine": round(_mean_pairwise_cosine(cls[comp]), 4),
                    "members": members_rel,
                }
            )
            gid += 1
            n_grouped += len(comp)

    manifest["n_groups"] = len(groups)
    manifest["n_grouped"] = n_grouped
    manifest["n_singletons"] = len(paths) - n_grouped
    manifest["groups"] = groups
    _write_manifest(out_path, manifest)
    return manifest


def _write_manifest(out_path: Path, manifest: dict) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
