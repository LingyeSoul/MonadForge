"""Invariants for Foresight Guidance (FSG) pre-step latent calibration.

The load-bearing safety property (CONTRIBUTING Tier-2): FSG with K=0 or an
out-of-band σ is a bit-exact no-op vs the baseline trajectory. Also checks the
forward-backward operator's fixed-point property (a constant velocity field is a
fixed point) and the documented extra-forward cost (3·K per scheduled step).

These run on a fake DiT (a plain callable) — the hydra/FEI setters are pure
``getattr`` no-ops on a non-adapter model, so no checkpoint is needed.
"""

from __future__ import annotations

import torch

from library.inference.corrections.fsg import FSGCalibrator


class _ConstField:
    """Fake DiT returning a fixed velocity regardless of input. A constant
    field makes F(x)=x exactly, so calibration must be a numerical no-op."""

    def __init__(self, v: torch.Tensor):
        self.v = v
        self.calls = 0

    def __call__(self, x, t, embed, padding_mask=None, **kw):
        self.calls += 1
        return self.v.expand_as(x).clone()


class _LinField:
    """Fake DiT with an input-dependent velocity (so F(x) != x)."""

    def __init__(self):
        self.calls = 0

    def __call__(self, x, t, embed, padding_mask=None, **kw):
        self.calls += 1
        return 0.1 * x


def _inputs():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, 4, 1, 8, 8, generator=g)
    embed = torch.randn(1, 16, 32, generator=g)
    nembed = torch.randn(1, 16, 32, generator=g)
    pad = torch.zeros(1, 1, 8, 8)
    return x, embed, nembed, pad


def test_scheduled_gate():
    fsg = FSGCalibrator(band=(0.45, 0.85), k=3)
    assert fsg.scheduled(0.6)
    assert fsg.scheduled(0.45) and fsg.scheduled(0.85)  # inclusive
    assert not fsg.scheduled(0.94)  # above band (the diverging dead zone)
    assert not fsg.scheduled(0.2)  # below band
    # K=0 is globally inert regardless of σ.
    assert not FSGCalibrator(band=(0.45, 0.85), k=0).scheduled(0.6)


def test_out_of_band_is_identical_object():
    """Out-of-band σ returns the exact same tensor object (bit-exact baseline)."""
    x, embed, nembed, pad = _inputs()
    anima = _ConstField(torch.zeros(1, 4, 1, 8, 8))
    fsg = FSGCalibrator(band=(0.45, 0.85), k=3)
    out = fsg.calibrate(anima, x, 0.94, 0, embed, nembed, pad, 4.0)
    assert out is x
    assert anima.calls == 0  # no forwards spent off-band


def test_k0_is_identical_object():
    x, embed, nembed, pad = _inputs()
    anima = _ConstField(torch.zeros(1, 4, 1, 8, 8))
    fsg = FSGCalibrator(band=(0.45, 0.85), k=0)
    out = fsg.calibrate(anima, x, 0.6, 0, embed, nembed, pad, 4.0)
    assert out is x
    assert anima.calls == 0


def test_constant_field_is_fixed_point():
    """If v^c == v^u == const, every iteration is an exact round-trip (F(x)=x),
    so the calibrated latent equals the input within fp tolerance."""
    x, embed, nembed, pad = _inputs()
    anima = _ConstField(torch.full((1, 4, 1, 8, 8), 0.3))
    fsg = FSGCalibrator(band=(0.45, 0.85), k=4, d_sigma=0.1)
    out = fsg.calibrate(anima, x, 0.6, 0, embed, nembed, pad, 4.0)
    assert torch.allclose(out, x, atol=1e-5)
    assert anima.calls == 3 * 4  # 3 forwards (v^c, v^u, v^u_lo) per iteration


def test_nonconstant_field_moves_latent():
    x, embed, nembed, pad = _inputs()
    anima = _LinField()
    fsg = FSGCalibrator(band=(0.45, 0.85), k=3, d_sigma=0.1)
    out = fsg.calibrate(anima, x, 0.6, 0, embed, nembed, pad, 4.0)
    assert not torch.allclose(out, x)
    assert out.shape == x.shape
    assert anima.calls == 3 * 3


# --- FSG × Spectrum/SEA composition --------------------------------------
# FSG-scheduled steps are forced to actual forwards and carved out of the
# spectrum cache scheduler's decision domain (treated like warmup/tail). These
# pin the SEA-accounting invariant: forcing a step neither inflates nor deflates
# the window/SEA refresh fraction it is matched against.


def test_forced_steps_excluded_from_decision_fraction():
    """A forced step is removed from the decision denominator, not counted as a
    refresh — so the fraction reflects only the steps SEA actually schedules."""
    from networks.spectrum import _window_decision_fraction

    base = _window_decision_fraction(20, 6, 17, 2.0, 0.25)
    # Force a single in-decision-region step (warmup=6 <= 10 < stop_at=17).
    forced = _window_decision_fraction(20, 6, 17, 2.0, 0.25, frozenset({10}))
    # Still a valid fraction, and the forced step did not get double-counted as
    # an extra refresh in the numerator (which would only ever raise it).
    assert 0.0 <= forced <= 1.0
    # Forcing a step outside the decision region (in warmup) is a no-op: those
    # indices are already forced actual and already excluded.
    assert _window_decision_fraction(20, 6, 17, 2.0, 0.25, frozenset({2})) == base


def test_forced_step_does_not_advance_window():
    """Forcing every decision step actual leaves no decision steps, so the
    fraction is the empty-domain sentinel (0/ max(1,0)) — i.e. SEA schedules
    nothing and FSG+warmup+tail carry the whole trajectory."""
    from networks.spectrum import _window_decision_fraction

    all_decision = frozenset(range(6, 17))
    assert _window_decision_fraction(20, 6, 17, 2.0, 0.25, all_decision) == 0.0


def test_fsg_step_set_matches_band():
    """The forced-step set the runner derives from a σ schedule is exactly the
    in-band steps — the contract spectrum.py relies on to force/exclude them."""
    sigmas = [0.97, 0.9, 0.82, 0.78, 0.6, 0.4, 0.2, 0.05]
    fsg = FSGCalibrator(band=(0.75, 0.85), k=3)
    steps = frozenset(i for i, s in enumerate(sigmas) if fsg.scheduled(s))
    assert steps == {2, 3}  # 0.82, 0.78 in [0.75, 0.85]; 0.9/0.6 out


# --- CFG++ substrate (the σ-scheduled guidance reweight) ------------------
# CFG++ is implemented as a pure reweight of the cond/uncond combine so it
# composes with any integrator (Euler, er_sde). These pin: (1) the reweight is
# algebraically identical to the "calibrate x̂ then Euler-step along v^u" form it
# replaced — so the validated Euler renders are unchanged; (2) the final-step
# boundary; (3) the weight is finite/positive across a real schedule.


def test_cfgpp_reweight_equals_calibrate_then_step():
    """w_eff form ≡ the old calibrate-then-step form, for the Euler integrator."""
    from library.inference.sampling import cfgpp_guidance_weight

    g = torch.Generator().manual_seed(1)
    x = torch.randn(1, 4, 1, 8, 8, generator=g)
    vc = torch.randn(1, 4, 1, 8, 8, generator=g)
    vu = torch.randn(1, 4, 1, 8, 8, generator=g)
    s_i, s_next, lam = 0.8, 0.72, 2.0
    step = s_i - s_next

    # Old form: calibrate x̂ = x − λ(1−σ')σ·Δv, then Euler-step along v^u.
    x_hat = x - lam * (1.0 - s_next) * s_i * (vc - vu)
    old = x_hat - step * vu

    # New form: reweight the combine, plain Euler step.
    w_eff = cfgpp_guidance_weight(s_i, s_next, lam)
    new = x - step * (vu + w_eff * (vc - vu))

    assert torch.allclose(old, new, atol=1e-6)


def test_cfgpp_weight_final_step_and_finite():
    from library.inference.sampling import (
        cfgpp_guidance_weight,
        get_timesteps_sigmas,
    )

    # Final step (σ_next → 0) collapses to λ.
    assert abs(cfgpp_guidance_weight(0.05, 0.0, 2.0) - 2.0) < 1e-9
    # Degenerate Δσ ≤ 0 falls back to λ rather than dividing by zero.
    assert cfgpp_guidance_weight(0.5, 0.5, 2.0) == 2.0
    # Across a real flow schedule: finite, positive, no blow-up.
    _, sigmas = get_timesteps_sigmas(20, 3.0, torch.device("cpu"))
    sig = [float(s) for s in sigmas]
    ws = [cfgpp_guidance_weight(sig[i], sig[i + 1], 2.0) for i in range(20)]
    assert all(0.0 < w < 50.0 for w in ws)
