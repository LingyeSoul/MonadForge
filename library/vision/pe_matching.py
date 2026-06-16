#!/usr/bin/env python3
"""PE-Spatial dense grid matching (library primitive).

Promoted out of the near-twin miner engine so any dataset-level tool — near-twin
pair mining, dataset grouping/clustering, dedup — shares one Stage-B match: a
mutual-NN + ratio-test dense cell match between two pooled PE-Spatial patch
grids (:class:`library.vision.pe_features.Feature`). Two images are near-twins
when a large *fraction* of their pooled cells find a distinctive mutual nearest
neighbour at or above a per-cell cosine floor; the unmatched cells localize the
difference region.

``easycontrol_adapters.tools.near_twins.engine`` re-exports these names for
backward compatibility. Pure numpy/torch — no model lifetime owned here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from library.vision.pe_features import Feature


@dataclass
class MatchResult:
    n_inliers: int
    match_frac: float
    diff_cells: set[int]  # union of unmatched a/b cells (grid index r*G + c)
    diff_a: set[int]
    diff_b: set[int]
    offset: tuple[float, float]  # estimated (drow, dcol) crop offset (geom-check)
    G: int


def pool_cells(grid16: np.ndarray, G: int) -> np.ndarray:
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
    ca, cb = pool_cells(fa.grid16, G), pool_cells(fb.grid16, G)
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


# Scalar ``match_grids`` is kept for the miner (needs the full MatchResult). The
# vectorized path below is for consumers needing only the inlier fraction over many
# pairs (dataset grouping); bit-comparable to the scalar gate (tests/test_grouping_grid_match.py).


@torch.no_grad()
def pool_cells_batch(grids: torch.Tensor, G: int) -> torch.Tensor:
    """[n, 16, 16, D] → [n, G*G, D], L2-normed per cell (on ``grids.device``).

    Batched form of :func:`pool_cells`: pool each image's patch grid to G×G
    once. Matches the scalar ``1e-8``-additive norm (not ``F.normalize``'s
    ``max(norm, eps)``) so the two paths agree to float.
    """
    t = grids.permute(0, 3, 1, 2)  # [n, D, 16, 16]
    p = F.adaptive_avg_pool2d(t, G)  # [n, D, G, G]
    p = p.permute(0, 2, 3, 1).reshape(grids.shape[0], G * G, -1)  # [n, G*G, D]
    return p / (p.norm(dim=-1, keepdim=True) + 1e-8)


@torch.no_grad()
def match_fracs(
    cells_a: torch.Tensor, cells_b: torch.Tensor, cell_min: float, ratio: float
) -> torch.Tensor:
    """Inlier fraction for a batch of grid pairs — vectorized ``match_grids``.

    ``cells_a`` / ``cells_b`` are ``[P, N, D]`` unit-norm pooled grids (P pairs,
    N=G*G cells). Returns ``[P]`` match fractions under the same mutual-NN +
    ratio + per-cell-floor gate as the scalar path (geom-check excluded — the
    grouping caller leaves it off). No diff-cell bookkeeping, so it stays a few
    fused kernels instead of a Python per-cell loop.
    """
    sim = cells_a @ cells_b.transpose(-1, -2)  # [P, N, N] cosine
    N = sim.shape[-1]
    top2 = sim.topk(2, dim=-1)  # forward NN per row
    s1, s2 = top2.values[..., 0], top2.values[..., 1]  # [P, N]
    nn1 = top2.indices[..., 0]  # [P, N]
    col_best = sim.argmax(dim=-2)  # [P, N] best row per column
    rows = torch.arange(N, device=sim.device)
    mutual = col_best.gather(-1, nn1) == rows  # NN is reciprocal
    ok = (s1 >= cell_min) & mutual & ((1.0 - s1) <= ratio * (1.0 - s2))
    return ok.sum(-1).to(torch.float32) / N


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
