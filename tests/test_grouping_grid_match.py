"""Batched grid-match parity + grouping edge-gate behavior.

The grouping path replaced the per-pair scalar ``match_grids`` with a batched
``match_fracs`` over once-pooled grids. This guards that the two agree on the
inlier fraction and that the connected-components gate behaves (twins connect,
unrelated images don't).
"""

from __future__ import annotations

import numpy as np
import torch

from library.datasets.grouping import _grid_match_edges, connected_components
from library.vision.pe_features import Feature
from library.vision.pe_matching import match_fracs, match_grids, pool_cells_batch


def _rand_grid(rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal((16, 16, 768)).astype(np.float16)


def _feat(grid: np.ndarray) -> Feature:
    return Feature(cls=grid.reshape(-1)[:768].astype(np.float32), grid16=grid)


def test_match_fracs_matches_scalar():
    """Batched match_fracs == scalar match_grids.match_frac (geom-check off)."""
    rng = np.random.default_rng(0)
    feats = [_feat(_rand_grid(rng)) for _ in range(8)]
    # Throw in an exact twin and a near-twin (one region perturbed).
    feats.append(_feat(feats[0].grid16.copy()))
    near = feats[1].grid16.copy()
    near[:4, :4] = _rand_grid(rng)[:4, :4]
    feats.append(_feat(near))

    G, cell_min, ratio = 7, 0.9, 0.8
    cells = pool_cells_batch(
        torch.from_numpy(np.stack([f.grid16 for f in feats]).astype(np.float32)), G
    )
    n = len(feats)
    pi, pj = np.triu_indices(n, k=1)
    batched = match_fracs(
        cells[torch.from_numpy(pi)], cells[torch.from_numpy(pj)], cell_min, ratio
    ).numpy()
    scalar = np.array(
        [
            match_grids(feats[i], feats[j], G, cell_min, ratio, False).match_frac
            for i, j in zip(pi, pj)
        ]
    )
    np.testing.assert_allclose(batched, scalar, atol=1e-6)


def test_grid_match_edges_gate():
    """Identical twins connect; an unrelated image stays a singleton."""
    rng = np.random.default_rng(1)
    base = _rand_grid(rng)
    feats = [_feat(base), _feat(base.copy()), _feat(_rand_grid(rng))]
    cls = np.stack([f.cls for f in feats]).astype(np.float32)
    cls /= np.linalg.norm(cls, axis=1, keepdims=True) + 1e-8

    edges = _grid_match_edges(
        feats,
        cls,
        device=torch.device("cpu"),
        sim_min=0.5,
        grid=7,
        cell_match_min=0.9,
        match_frac_min=0.4,
        ratio=0.8,
    )
    comps = connected_components(len(feats), edges)
    assert (0, 1) in edges
    assert all(2 not in {a, b} for a, b in edges)
    assert [0, 1] in comps and [2] in comps


def test_pool_chunk_invariant():
    """Chunked pooling gives the same edges as single-shot pooling."""
    rng = np.random.default_rng(2)
    feats = [_feat(_rand_grid(rng)) for _ in range(10)]
    feats[5] = _feat(feats[0].grid16.copy())  # a twin so an edge exists
    cls = np.stack([f.cls for f in feats]).astype(np.float32)
    cls /= np.linalg.norm(cls, axis=1, keepdims=True) + 1e-8
    kw = dict(
        device=torch.device("cpu"),
        sim_min=0.5,
        grid=7,
        cell_match_min=0.9,
        match_frac_min=0.4,
        ratio=0.8,
    )
    a = _grid_match_edges(feats, cls, pool_chunk=256, pair_chunk=4096, **kw)
    b = _grid_match_edges(feats, cls, pool_chunk=3, pair_chunk=7, **kw)
    assert a == b
