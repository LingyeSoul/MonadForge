"""SEA-schedule invariant tests (spectrum_sea_schedule.md P1).

Covers the load-bearing invariants of the SeaCache SEA-distance trigger that
grafts onto Spectrum's scheduler:

  - SEA filter physical limits: σ→0 is identity (passes all detail), σ→1 kills
    the signal; the gain is unit-mean (the relative-L1 distance stays
    scale-stable).
  - ``sea_filter`` broadcasts over leading dims (the bench's ``(C,H,W)`` and the
    runner's ``(B,C,H,W)`` both work, channel-wise identical).
  - ``count_refreshes`` is monotone non-increasing in δ (well-posed search).
  - ``solve_delta_for_refresh_ratio`` lands the refresh count on the target
    fraction.
  - The re-export shim keeps ``bench.spectrum_sea.sea`` importable.
  - ``schedule="window"`` is the default, so the SEA path is inert unless opted
    into — and the CLI delta parser maps 'auto' → None.
"""

from __future__ import annotations

import inspect

import pytest
import torch

from networks.spectrum_sea import (
    accumulate_distances,
    count_refreshes,
    l1rel,
    sea_filter,
    sea_gain,
    solve_delta_for_refresh_ratio,
)


# ── SEA filter physics ─────────────────────────────────────────────────────────


def test_sea_filter_sigma_zero_is_identity():
    x = torch.randn(4, 16, 16)
    y = sea_filter(x, sigma=0.0)
    assert torch.allclose(y, x, atol=1e-4)


def test_sea_filter_sigma_one_kills_signal():
    x = torch.randn(4, 16, 16)
    y = sea_filter(x, sigma=1.0)
    # b=1, a=0 → gain → 0 everywhere → output collapses to ~0.
    assert y.abs().mean() < 1e-3


def test_sea_gain_is_unit_mean():
    for sigma in (0.1, 0.3, 0.5, 0.75, 0.9):
        g = sea_gain(32, 32, sigma, beta=2.0, device="cpu", dtype=torch.float32)
        assert g.mean().item() == pytest.approx(1.0, abs=1e-5)


def test_sea_filter_broadcasts_over_leading_dims():
    # (C,H,W) bench shape vs (B,C,H,W) runner shape — same channels, identical.
    chw = torch.randn(8, 12, 12)
    bchw = chw.unsqueeze(0)  # (1, C, H, W)
    y3 = sea_filter(chw, sigma=0.5)
    y4 = sea_filter(bchw, sigma=0.5)
    assert y4.shape == bchw.shape
    assert torch.allclose(y4[0], y3, atol=1e-4)


def test_l1rel_matches_definition():
    a = torch.tensor([2.0, 0.0, -1.0])
    b = torch.tensor([1.0, 1.0, 1.0])
    # |Δ|₁ = 1 + 1 + 2 = 4; ‖b‖₁ = 3 → 4/3.
    assert l1rel(a, b) == pytest.approx(4.0 / 3.0, abs=1e-5)


# ── auto-δ refresh-ratio calibration ───────────────────────────────────────────


def test_count_refreshes_monotone_in_delta():
    dists = [0.1, 0.3, 0.05, 0.4, 0.2, 0.15, 0.25]
    prev = None
    for delta in [0.0, 0.1, 0.2, 0.3, 0.5, 1.0, 5.0]:
        c = count_refreshes(dists, delta)
        if prev is not None:
            assert c <= prev
        prev = c
    # δ above the total → no refresh fires.
    assert count_refreshes(dists, sum(dists) + 1.0) == 0


def test_solve_delta_hits_target_refresh_ratio():
    torch.manual_seed(0)
    dists = torch.rand(40).tolist()
    for ratio in (0.3, 0.5, 0.62, 0.8):
        delta = solve_delta_for_refresh_ratio(dists, ratio)
        target = max(1, round(ratio * len(dists)))
        # Step function in δ — landing within 1 of the target is exact convergence.
        assert abs(count_refreshes(dists, delta) - target) <= 1


def test_accumulate_distances_length():
    seas = [torch.randn(3, 8, 8) for _ in range(5)]
    d = accumulate_distances(seas)
    assert len(d) == 4
    assert d[0] == pytest.approx(l1rel(seas[1], seas[0]), abs=1e-5)


# ── re-export shim + defaults ───────────────────────────────────────────────────


def test_bench_sea_shim_reexports():
    from bench.spectrum_sea import sea as bench_sea

    assert bench_sea.sea_filter is sea_filter
    assert bench_sea.l1rel is l1rel
    assert bench_sea.solve_delta_for_refresh_ratio is solve_delta_for_refresh_ratio


def test_spectrum_denoise_defaults_to_window():
    from networks.spectrum import spectrum_denoise

    sig = inspect.signature(spectrum_denoise)
    assert sig.parameters["schedule"].default == "window"
    assert sig.parameters["delta"].default is None
    # <=0 sentinel → auto-match the window's own refresh fraction (not a fixed 0.62).
    assert sig.parameters["refresh_ratio"].default <= 0.0


def test_window_decision_fraction_matches_window_compute():
    """The auto-δ target must reproduce the window's own forward count.

    refresh_ratio·n_decision (rounded) == the window schedule's decision-region
    actual count under the *same* warmup/stop — so a SEA arm calibrated to it
    spends exactly as many forwards as the window (matched compute by
    construction). This is what the hard-coded 0.62 got wrong: 0.62 was the
    window's *total* fraction, applied to the decision region → over-compute.
    """
    import math

    from networks.spectrum import _window_decision_fraction

    def ref_window_decision_count(n, warmup, stop, ws, flex):
        curr, consec, dec_actual = ws, 0, 0
        for i in range(n):
            forced = i < warmup or i >= stop
            actual = forced or ((consec + 1) % max(1, math.floor(curr)) == 0)
            if not forced:
                dec_actual += int(actual)
            if actual:
                if i >= warmup:
                    curr = round(curr + flex, 3)
                consec = 0
            else:
                consec += 1
        return dec_actual

    for n, warmup, stop in [(28, 7, 27), (24, 6, 21), (50, 5, 47)]:
        n_dec = stop - warmup
        frac = _window_decision_fraction(n, warmup, stop, 2.0, 0.25)
        assert 0.0 < frac <= 1.0
        assert round(frac * n_dec) == ref_window_decision_count(
            n, warmup, stop, 2.0, 0.25
        )


def test_auto_delta_disk_roundtrip(tmp_path, monkeypatch):
    import networks.spectrum as spec

    store = tmp_path / "spectrum_sea_delta.json"
    monkeypatch.setattr(spec, "_auto_delta_store_path", lambda: store)
    monkeypatch.setattr(spec, "_AUTO_DELTA_CACHE", {})

    key = (28, 7, 27, 0.62)
    assert spec._auto_delta_lookup(key) is None
    spec._auto_delta_save(key, 0.1167)
    # In-memory hit.
    assert spec._auto_delta_lookup(key) == pytest.approx(0.1167)
    # Fresh-process simulation: clear in-memory, must reload from disk.
    monkeypatch.setattr(spec, "_AUTO_DELTA_CACHE", {})
    assert spec._auto_delta_lookup(key) == pytest.approx(0.1167)
    # A different schedule geometry is a separate entry.
    assert spec._auto_delta_lookup((24, 6, 21, 0.62)) is None


def test_cli_delta_parser_maps_auto_to_none():
    from library.inference.generation import _parse_spectrum_delta

    assert _parse_spectrum_delta("auto") is None
    assert _parse_spectrum_delta(None) is None
    assert _parse_spectrum_delta("0.5") == pytest.approx(0.5)
    assert _parse_spectrum_delta(0.5) == pytest.approx(0.5)
