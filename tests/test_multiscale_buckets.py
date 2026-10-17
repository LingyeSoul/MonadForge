"""Multi-scale tiers (``--target_res``) under free-fit.

The discrete constant-token bucket pool was removed; each tier now survives only
as a token-count band (``EDGE_TOKEN_BANDS``). These lock the invariants the
preprocess + compile paths still rely on:
  * each tier's band has the expected number of families (1 or 2),
  * tier assignment (``choose_edge``) never upscales when a closer tier exists
    (and the 2.25MP → 1536 case the feature was built for holds),
  * compile_blocks' dynamo budget scales with the active tier count.
"""

import math

import pytest

from library.datasets.buckets import (
    ALLOWED_TARGET_RES,
    EDGE_TOKEN_BANDS,
    choose_edge,
    token_count_families,
    token_count_range,
)


def test_band_family_counts():
    # 512 carries the 1024-tok square + a 1008-tok family; 1024 ships 4032/4200;
    # 896 ships 3000/3024; 768/1280/1536 are a single family each. The band's
    # endpoint count == the number of token-count families (== compiled graphs).
    expected_families = {512: 2, 768: 1, 896: 2, 1024: 2, 1280: 1, 1536: 1}
    for edge, (lo, hi) in EDGE_TOKEN_BANDS.items():
        n = 1 if lo == hi else 2
        assert n == expected_families[edge], (edge, lo, hi)


def test_1024_band_frozen():
    # The 1024 tier stays at its natural (4032, 4200) — the frozen top-5 aspect
    # set (DCW_ASPECT_BUCKETS, consumed by CNS/mod-distill) is drawn from it.
    assert EDGE_TOKEN_BANDS[1024] == (4032, 4200)


def test_token_count_families():
    assert token_count_families([1024]) == 2
    assert token_count_families([1024, 1536]) == 3
    assert token_count_families([512]) == 2  # 1024-tok square + 1008-tok family
    assert token_count_families([512, 1024]) == 4
    assert token_count_families(list(ALLOWED_TARGET_RES)) == 9


def test_token_count_range():
    assert token_count_range([1024]) == (4032, 4200)
    assert token_count_range([768, 1280]) == (2160, 6300)


def test_token_count_range_rejects_unknown_tier():
    with pytest.raises(ValueError):
        token_count_range([1024, 999])


@pytest.mark.parametrize(
    "w,h,target_res,expected",
    [
        (1500, 1500, [512, 768, 1024, 1280, 1536], 1536),  # 2.25MP — the ask
        (1440, 1536, [512, 768, 1024, 1280, 1536], 1536),  # exact 1536 bucket
        (1024, 1024, [768, 1024, 1536], 1024),
        (896, 1200, [512, 768, 1024, 1280, 1536], 1024),  # exact 1024 portrait
        # ~0.95MP near-square: closer to 1024 (tiny upscale) than 768 (big
        # downscale) — the case the nearest metric exists for.
        (1000, 950, [768, 1024], 1024),
        (800, 800, [512, 768, 1024], 768),  # 0.64MP closest to 768
        (300, 300, [512, 768, 1024], 512),  # tiny → least-bad (smallest) tier
        (4000, 4000, [512, 1024], 1024),  # huge → least downscale = largest tier
    ],
)
def test_choose_edge_nearest(w, h, target_res, expected):
    assert choose_edge(w, h, target_res) == expected


def test_choose_edge_minimizes_resize():
    """The chosen tier minimizes |log(nominal_tokens / native_tokens)| over tiers.

    Free-fit preserves aspect, so the only lever is total token budget (area);
    the nominal token count is the tier band's midpoint.
    """
    target_res = list(ALLOWED_TARGET_RES)

    def nominal(edge):
        lo, hi = EDGE_TOKEN_BANDS[edge]
        return (lo + hi) / 2.0

    for w, h in [(2000, 1200), (1100, 1100), (640, 900), (1500, 1500), (700, 1400)]:
        native = (w / 16.0) * (h / 16.0)
        chosen = choose_edge(w, h, target_res)

        def cost(edge):
            return abs(math.log(nominal(edge) / native))

        assert cost(chosen) == min(cost(e) for e in target_res)


def test_compile_blocks_budget_scales_with_tiers():
    import torch._dynamo as _dynamo

    from tests.test_native_flatten import _tiny_anima

    model = _tiny_anima()
    _dynamo.config.cache_size_limit = 1
    # full menu → 9 token-count families → 2*9 + 8 = 26.
    model.compile_blocks(
        backend="eager",
        n_token_families=token_count_families(list(ALLOWED_TARGET_RES)),
    )
    assert _dynamo.config.cache_size_limit >= 2 * 9 + 8
