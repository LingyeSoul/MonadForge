"""compile_dit_blocks_for_pool — shape-derivation parity with the retired inline
distill_spd / distill_mod blocks.

The compile itself needs a GPU + a real DiT; the *derivation* (n_shapes,
n_token_families, seq_range) is pure and is what drifted across the three
scripts. `enabled=False` returns the derived `PoolCompileResult` without
touching torch.compile, so we pin the math here.
"""

import pytest

from library.runtime.harness import PoolCompileResult, compile_dit_blocks_for_pool


def _derive(token_counts, *, dynamic_seq):
    # enabled=False short-circuits before any compile/isolate call; anima unused.
    return compile_dit_blocks_for_pool(
        anima=None, token_counts=token_counts, enabled=False, dynamic_seq=dynamic_seq
    )


def _old_inline(token_counts, *, dynamic_seq):
    """The retired spd/mod derivation, verbatim, for parity."""
    n_shapes = max(1, len(set(token_counts)))
    if dynamic_seq and set(token_counts):
        n_token_families = n_shapes
        seq_range = (min(token_counts), max(token_counts))
    else:
        n_token_families = None
        seq_range = None
    return PoolCompileResult(n_shapes, n_token_families, seq_range)


@pytest.mark.parametrize(
    "counts",
    [
        {4032, 4200},
        {4032, 4200, 2160, 6300},
        {4032},
        set(),
        [4032, 4032, 4200, 4200],  # duplicates collapse
    ],
)
@pytest.mark.parametrize("dynamic_seq", [True, False])
def test_parity_with_old_inline_derivation(counts, dynamic_seq):
    got = _derive(counts, dynamic_seq=dynamic_seq)
    exp = _old_inline(counts, dynamic_seq=dynamic_seq)
    assert got == exp


def test_dynamic_seq_bounds():
    r = _derive({4032, 4200, 6300}, dynamic_seq=True)
    assert r.n_shapes == 3
    assert r.n_token_families == 3
    assert r.seq_range == (4032, 6300)


def test_static_path_drops_bounds():
    r = _derive({4032, 4200, 6300}, dynamic_seq=False)
    assert r.n_shapes == 3
    assert r.n_token_families is None
    assert r.seq_range is None


def test_empty_pool_is_one_shape_no_bounds():
    r = _derive(set(), dynamic_seq=True)
    assert r.n_shapes == 1  # max(1, 0)
    assert r.n_token_families is None and r.seq_range is None


def test_duplicates_collapse_to_distinct():
    r = _derive([4032, 4032, 4200], dynamic_seq=True)
    assert r.n_shapes == 2
    assert r.seq_range == (4032, 4200)
