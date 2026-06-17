"""Free-aspect token-band resize ("free-fit") solver invariants.

Locks the Phase 0 contract from
docs/proposal/free_aspect_token_band_resize.md:
  * every fitted grid lands inside the tier's token band,
  * the result respects the max_ratio clamp and the RoPE per-axis cap,
  * crop residual is sub-patch (< 16px) for in-range aspects on the 1024 tier,
  * the solver is deterministic,
  * the GUI/CLI preview (compute_resize_preview) agrees with the solver feeding
    process_image (select_resize_bucket).
"""

import pytest

from library.datasets.buckets import (
    ALLOWED_TARGET_RES,
    BucketManager,
    freefit_band_for_edge,
    freefit_bucket,
)
from library.preprocess.resize_preview import (
    compute_resize_preview,
    select_resize_bucket,
)

PATCH = 16
ROPE_CAP = 256

# A spread of native pixel sizes × aspect ratios. Aspects stay within the
# default 1:4 clamp so no crop should be forced.
_NATIVE_SIZES = [
    (1024, 1024),
    (1920, 1080),
    (1080, 1920),
    (1500, 1000),
    (1000, 1500),
    (2048, 768),
    (768, 2048),
    (1234, 987),
    (640, 2400),
    (3000, 2000),
]
_ASPECTS = [
    0.25,
    0.3,
    0.4,
    0.5,
    0.5625,
    0.667,
    0.75,
    0.8,
    1.0,
    1.25,
    1.5,
    1.78,
    2.0,
    3.0,
    4.0,
]


def _grids(edge):
    band = freefit_band_for_edge(edge)
    return band, [freefit_bucket(round(a * 1000), 1000, band) for a in _ASPECTS]


def test_band_matches_token_count_range():
    # 1024 → (4032, 4200); 512 → (1008, 1024); 768 single-family → (2160, 2160).
    assert freefit_band_for_edge(1024) == (4032, 4200)
    assert freefit_band_for_edge(512) == (1008, 1024)
    assert freefit_band_for_edge(768) == (2160, 2160)


@pytest.mark.parametrize("edge", ALLOWED_TARGET_RES)
def test_grid_lands_in_band(edge):
    band, grids = _grids(edge)
    lo, hi = band
    for w, h in grids:
        assert w % PATCH == 0 and h % PATCH == 0, (edge, w, h)
        tokens = (w // PATCH) * (h // PATCH)
        assert lo <= tokens <= hi, (edge, w, h, tokens, band)


@pytest.mark.parametrize("edge", ALLOWED_TARGET_RES)
def test_respects_rope_cap(edge):
    _, grids = _grids(edge)
    for w, h in grids:
        assert max(w // PATCH, h // PATCH) <= ROPE_CAP, (edge, w, h)


def test_max_ratio_clamp_on_degenerate_input():
    band = freefit_band_for_edge(1024)
    # 1:8 portrait — far outside the 1:4 clamp. The fitted aspect must not exceed
    # the clamp (within one patch of the integer grid).
    w, h = freefit_bucket(500, 4000, band, max_ratio=4.0)
    fitted_ar = w / h  # < 1 (portrait)
    assert fitted_ar >= 1.0 / 4.0 - (1.0 / (h // PATCH)), (w, h, fitted_ar)
    # And a custom tighter clamp is honored too.
    w2, h2 = freefit_bucket(4000, 500, band, max_ratio=2.0)
    assert (w2 / h2) <= 2.0 + (1.0 / (h2 // PATCH)), (w2, h2)


def test_crop_residual_subpatch_on_1024_tier():
    """In-range aspects crop < 1 patch (the <16px residual the proposal claims)."""
    band = freefit_band_for_edge(1024)
    for sw, sh in _NATIVE_SIZES:
        a = sw / sh
        if not (0.25 <= a <= 4.0):
            continue
        bw, bh = freefit_bucket(sw, sh, band)
        # Cover-resize native → bucket preserving aspect, then measure the crop.
        scale = max(bw / sw, bh / sh)
        covered_w, covered_h = sw * scale, sh * scale
        residual = max(covered_w - bw, covered_h - bh)
        assert residual < PATCH, (sw, sh, bw, bh, residual)


def test_deterministic():
    band = freefit_band_for_edge(1024)
    for sw, sh in _NATIVE_SIZES:
        first = freefit_bucket(sw, sh, band)
        for _ in range(3):
            assert freefit_bucket(sw, sh, band) == first


def test_near_square_is_exact_square():
    # 4096 = 64*64 ∈ [4032, 4200], so a square fits with zero aspect error.
    assert freefit_bucket(1000, 1000, freefit_band_for_edge(1024)) == (1024, 1024)


def test_aspect_error_is_minimal():
    """No neighboring in-band grid is closer to the (clamped) target aspect."""
    band = freefit_band_for_edge(1024)
    lo, hi = band
    for sw, sh in _NATIVE_SIZES:
        a = sw / sh
        bw, bh = freefit_bucket(sw, sh, band)
        wp, hp = bw // PATCH, bh // PATCH
        best_err = abs(wp / hp - a)
        for h2 in range(1, min(ROPE_CAP, hi) + 1):
            w_lo = max(1, -(-lo // h2))
            w_hi = min(ROPE_CAP, hi // h2)
            for w2 in range(w_lo, w_hi + 1):
                assert abs(w2 / h2 - a) >= best_err - 1e-12, (sw, sh, w2, h2)


@pytest.mark.parametrize("edge", ALLOWED_TARGET_RES)
def test_preview_agrees_with_solver(edge):
    """compute_resize_preview(freefit) == select_resize_bucket == freefit_bucket."""
    band = freefit_band_for_edge(edge)
    for sw, sh in _NATIVE_SIZES:
        solver = freefit_bucket(sw, sh, band)
        _, selected = select_resize_bucket(sw, sh, [edge], fit_mode="freefit")
        preview = compute_resize_preview(sw, sh, [edge], fit_mode="freefit")
        assert selected == solver, (edge, sw, sh)
        assert preview.bucket_size == solver, (edge, sw, sh)
        assert preview.target_edge == edge


def test_single_tier_choose_edge_is_noop():
    # Free-fit operates inside the choose_edge-picked tier; single tier is direct.
    _, b = select_resize_bucket(1500, 1000, [1024], fit_mode="freefit")
    assert b == freefit_bucket(1500, 1000, freefit_band_for_edge(1024))


# --- Phase 1: train-side bucket mode -----------------------------------------


def test_bucketmanager_freefit_keeps_native_shapes():
    """Free-fit bucket mode: each on-disk (W, H) exact-matches — never AR-snaps."""
    band = freefit_band_for_edge(1024)
    sizes = {
        freefit_bucket(1500, 1000, band),
        freefit_bucket(1000, 1500, band),
        (1024, 1024),
        freefit_bucket(2048, 700, band),
    }
    bm = BucketManager()
    bm.make_buckets(freefit_resos=sizes)
    assert set(bm.predefined_resos) == sizes
    for w, h in sizes:
        reso, resized, ar_error = bm.select_bucket(w, h)
        assert reso == (w, h), (w, h, reso)  # exact match, no snap
        assert resized == (w, h)
        assert ar_error == 0.0


def test_bucketmanager_freefit_token_counts_in_one_band():
    """All free-fit shapes for the 1024 tier share its [4032, 4200] band → one
    dynamic-seq graph (the compile-coupling invariant)."""
    lo, hi = freefit_band_for_edge(1024)
    sizes = {freefit_bucket(round(a * 1000), 1000, (lo, hi)) for a in _ASPECTS}
    counts = {(w // PATCH) * (h // PATCH) for (w, h) in sizes}
    assert all(lo <= c <= hi for c in counts)
