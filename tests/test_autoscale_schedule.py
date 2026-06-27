"""Resolution-curriculum (``autoscale_mode``) invariants.

Locks the schedule contract from
docs/proposal/autoscale_resolution_curriculum.md:
  * tier-edge round-trips: a cached latent's token count maps back to the tier
    that emitted it (``edge_for_token_count``),
  * the progress policy (``AutoscaleSchedule.active_rank``): warmup window and
    single-tier data both yield "no remap"; the bulk phase trains the cheap
    tier; the trailing ``finish`` fraction trains the top tier,
  * "step" == "stairs" for a 2-tier ladder,
  * the dataset ``__getitem__`` remap actually redirects out-of-phase batches to
    the active tier and never escapes the active tier mid-phase.
"""

import pytest

from library.datasets.autoscale import AutoscaleSchedule
from library.datasets.buckets import (
    EDGE_TOKEN_BANDS,
    edge_for_token_count,
    freefit_band_for_edge,
)


def test_edge_for_token_count_round_trips_every_tier():
    # The nominal (band midpoint) and both band endpoints of each tier map back
    # to that tier — the inverse of preprocess tier assignment.
    for edge, (lo, hi) in EDGE_TOKEN_BANDS.items():
        for n in (lo, (lo + hi) // 2, hi):
            assert edge_for_token_count(n) == edge
        # widened free-fit band endpoints also resolve to the tier
        wlo, whi = freefit_band_for_edge(edge)
        assert edge_for_token_count(wlo) == edge
        assert edge_for_token_count(whi) == edge


def test_active_rank_no_op_cases():
    sched = AutoscaleSchedule(finish=0.15, ramp="step", warmup_batches=8)
    # single tier -> nothing to schedule
    assert sched.active_rank(1000, 1000, n_ranks=1) is None
    # max_steps unknown -> no remap
    assert sched.active_rank(50, 0, n_ranks=2) is None
    # inside warmup window -> let every tier flow (warm compile graphs)
    assert sched.active_rank(0, 1000, n_ranks=2) is None
    assert sched.active_rank(7, 1000, n_ranks=2) is None


def test_step_ramp_two_tier_phases():
    sched = AutoscaleSchedule(finish=0.15, ramp="step", warmup_batches=8)
    T = 1000
    # bulk phase -> cheapest tier (rank 0)
    assert sched.active_rank(100, T, 2) == 0
    assert sched.active_rank(840, T, 2) == 0
    # finish phase (last 15%) -> top tier (rank 1)
    assert sched.active_rank(850, T, 2) == 1
    assert sched.active_rank(999, T, 2) == 1
    # past the end clamps to the top tier, never out of range
    assert sched.active_rank(1000, T, 2) == 1


def test_step_equals_stairs_for_two_tiers():
    step = AutoscaleSchedule(finish=0.2, ramp="step", warmup_batches=0)
    stairs = AutoscaleSchedule(finish=0.2, ramp="stairs", warmup_batches=0)
    T = 500
    for s in range(0, T, 7):
        assert step.active_rank(s, T, 2) == stairs.active_rank(s, T, 2)


def test_stairs_three_tier_progression():
    sched = AutoscaleSchedule(finish=0.2, ramp="stairs", warmup_batches=0)
    T = 1000
    # top tier (rank 2) owns the last 20%
    assert sched.active_rank(999, T, 3) == 2
    assert sched.active_rank(800, T, 3) == 2
    # the leading 80% splits equally across ranks 0 and 1 (40% each)
    assert sched.active_rank(0, T, 3) == 0
    assert sched.active_rank(399, T, 3) == 0
    assert sched.active_rank(400, T, 3) == 1
    assert sched.active_rank(799, T, 3) == 1


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        AutoscaleSchedule(finish=0.0)
    with pytest.raises(ValueError):
        AutoscaleSchedule(finish=1.0)
    with pytest.raises(ValueError):
        AutoscaleSchedule(ramp="ramp")


class _FakeBBI:
    def __init__(self, bucket_index):
        self.bucket_index = bucket_index


class _FakeBucketManager:
    def __init__(self, resos):
        self.resos = resos


class _FakeDataset:
    """Minimal stand-in exercising the real remap methods on BaseDataset."""

    from library.datasets.base import BaseDataset

    enable_autoscale = BaseDataset.enable_autoscale
    _rebuild_autoscale_positions = BaseDataset._rebuild_autoscale_positions
    _autoscale_remap = BaseDataset._autoscale_remap

    def __init__(self, resos, bucket_of_position):
        # resos: list[(W,H)] indexed by bucket_index
        # bucket_of_position: list[bucket_index] — one entry per buckets_indices slot
        self.bucket_manager = _FakeBucketManager(resos)
        self.buckets_indices = [_FakeBBI(b) for b in bucket_of_position]
        self._autoscale = None
        self._autoscale_bucket_rank = {}
        self._autoscale_n_ranks = 0
        self._autoscale_rank_positions = {}
        self.current_step = 0
        self.max_train_steps = 1000


def _two_tier_dataset():
    # bucket 0 = 896 tier (3024 tok = 56*54), bucket 1 = 1024 tier (4200 tok)
    resos = [(896, 864), (1120, 960)]  # (56*54)=3024 ; (70*60)=4200
    assert (resos[0][0] // 16) * (resos[0][1] // 16) == 3024
    assert (resos[1][0] // 16) * (resos[1][1] // 16) == 4200
    # 3 low-tier batch slots, 2 high-tier batch slots
    return _FakeDataset(resos, bucket_of_position=[0, 1, 0, 1, 0])


def test_dataset_remap_redirects_out_of_phase_batches():
    ds = _two_tier_dataset()
    ds.enable_autoscale(AutoscaleSchedule(finish=0.15, ramp="step", warmup_batches=2))
    assert ds._autoscale_n_ranks == 2
    # rank 0 == 896, rank 1 == 1024
    assert ds._autoscale_bucket_rank == {0: 0, 1: 1}

    # Bulk phase, past warmup: every fetched index must resolve to a rank-0 batch.
    ds.current_step = 100  # frac 0.1 < 0.85 -> rank 0 active
    for idx in range(len(ds.buckets_indices)):
        out = ds._autoscale_remap(idx)
        assert ds._autoscale_bucket_rank[ds.buckets_indices[out].bucket_index] == 0

    # Finish phase: every fetched index must resolve to a rank-1 (top) batch.
    ds.current_step = 900  # frac 0.9 >= 0.85 -> rank 1 active
    for idx in range(len(ds.buckets_indices)):
        out = ds._autoscale_remap(idx)
        assert ds._autoscale_bucket_rank[ds.buckets_indices[out].bucket_index] == 1


def test_dataset_remap_warmup_and_inphase_passthrough():
    ds = _two_tier_dataset()
    ds.enable_autoscale(AutoscaleSchedule(finish=0.15, ramp="step", warmup_batches=4))
    # Inside warmup -> no remap regardless of tier (warms both compile graphs).
    ds.current_step = 1
    for idx in range(len(ds.buckets_indices)):
        assert ds._autoscale_remap(idx) == idx
    # In-phase requests pass through untouched (rank-0 batch during bulk).
    ds.current_step = 100
    rank0_positions = [
        i
        for i, b in enumerate(ds.buckets_indices)
        if ds._autoscale_bucket_rank[b.bucket_index] == 0
    ]
    for idx in rank0_positions:
        assert ds._autoscale_remap(idx) == idx


def test_single_tier_dataset_is_no_op():
    ds = _FakeDataset([(1120, 960)], bucket_of_position=[0, 0, 0])
    ds.enable_autoscale(AutoscaleSchedule())
    assert ds._autoscale_n_ranks == 1
    ds.current_step = 900
    for idx in range(len(ds.buckets_indices)):
        assert ds._autoscale_remap(idx) == idx
