"""Multi-scale constant-token tiers (``--target_res``).

Locks the invariants the preprocess + compile paths rely on:
  * every tier is constant-token (zero intra-bucket padding by construction),
  * the default single-1024 path is byte-identical to the canonical table,
  * tier assignment never upscales (and the 2.25MP → 1536 case the feature
    was built for holds),
  * compile_blocks' dynamo budget scales with the active tier count.
"""

import math

import pytest

from library.datasets.buckets import (
    ALLOWED_TARGET_RES,
    CONSTANT_TOKEN_BUCKETS,
    CONSTANT_TOKEN_BUCKETS_BY_EDGE,
    buckets_for_edges,
    choose_edge,
    token_count_families,
)

# Rope per-axis cap (max_img_h/w // patch_spatial = 512 // 2).
ROPE_PATCH_CAP = 256


def test_every_tier_is_constant_token():
    # 512 carries the 1024-tok square + a 1008-tok family; 1024 ships 4032/4200;
    # 768/1280/1536 are a single family each.
    expected_families = {512: 2, 768: 1, 896: 2, 1024: 2, 1280: 1, 1536: 1}
    for edge, table in CONSTANT_TOKEN_BUCKETS_BY_EDGE.items():
        counts = {(w // 16) * (h // 16) for w, h in table}
        assert len(counts) == expected_families[edge], (edge, counts)


def test_tiers_within_rope_caps():
    for edge, table in CONSTANT_TOKEN_BUCKETS_BY_EDGE.items():
        for w, h in table:
            assert max(w // 16, h // 16) <= ROPE_PATCH_CAP, (edge, w, h)


def test_landscape_mirrors_present():
    for edge, table in CONSTANT_TOKEN_BUCKETS_BY_EDGE.items():
        s = set(table)
        for w, h in table:
            assert (h, w) in s, f"tier {edge} missing mirror of {(w, h)}"


def test_default_path_is_canonical():
    # The unset / [1024] path must not drift from the frozen DCW-keyed table.
    assert buckets_for_edges([1024]) == CONSTANT_TOKEN_BUCKETS


def test_token_count_families():
    assert token_count_families([1024]) == 2
    assert token_count_families([1024, 1536]) == 3
    assert token_count_families([512]) == 2  # 1024-tok square + 1008-tok family
    assert token_count_families([512, 1024]) == 4
    assert token_count_families(list(ALLOWED_TARGET_RES)) == 9


def test_buckets_for_edges_rejects_unknown_tier():
    with pytest.raises(ValueError):
        buckets_for_edges([1024, 999])


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
    """The chosen tier must have the smallest |log cover-scale| of all tiers."""
    from library.datasets.buckets import (
        CONSTANT_TOKEN_BUCKETS_BY_EDGE,
        _cover_scale,
        _nearest_aspect_bucket,
    )

    target_res = list(ALLOWED_TARGET_RES)
    for w, h in [(2000, 1200), (1100, 1100), (640, 900), (1500, 1500), (700, 1400)]:
        chosen = choose_edge(w, h, target_res)

        def cost(edge):
            bw, bh = _nearest_aspect_bucket(w, h, CONSTANT_TOKEN_BUCKETS_BY_EDGE[edge])
            return abs(math.log(_cover_scale(w, h, bw, bh)))

        assert cost(chosen) == min(cost(e) for e in target_res)


def test_compile_blocks_budget_scales_with_tiers():
    import torch._dynamo as _dynamo

    from tests.test_native_flatten import _tiny_anima

    model = _tiny_anima()
    _dynamo.config.cache_size_limit = 1
    # full menu → 7 token-count families → 2*7 + 8 = 22.
    model.compile_blocks(
        backend="eager",
        n_token_families=token_count_families(list(ALLOWED_TARGET_RES)),
    )
    assert _dynamo.config.cache_size_limit >= 22
