"""Invariants of the SPD SNR-gated velocity loss (networks/spd.py).

The gate exists to remove demands-clairvoyance gradients from the SPD
fine-tune (supervising DCT bands whose clean coefficient is statistically
unrecoverable from x_t — the regression-to-the-mean blur mechanism). These
tests pin the math it rests on:

  * orthonormal units — white-noise latents measure P_ω ≈ 1, so
    SNR_ω(t) = ((1−t)/t)²·P_ω needs no unit conversion;
  * Parseval — with w ≡ 1 the gated loss IS the plain spatial velocity MSE
    it replaces;
  * monotonicity + hard-gate activation time (paper Prop. 1 / Eq. 9);
  * stage-grid consistency — dct_lowpass_init truncation preserves
    coefficients, so a low-res grid's weights are the top-left block of the
    full grid's.
"""

import torch
import torch.nn.functional as F

from networks.spd import SpdSnrGate, measure_dct_power_profile

N_BINS = 32


def test_white_noise_profile_is_unit_power():
    torch.manual_seed(0)
    lats = [torch.randn(16, 40, 48) for _ in range(24)]
    profile = measure_dct_power_profile(lats, n_bins=N_BINS)
    assert profile.shape == (N_BINS,)
    # Orthonormal DCT of N(0, I) is N(0, I) per coefficient → every bin ≈ 1.
    assert torch.allclose(profile, torch.ones(N_BINS), atol=0.15), profile


def test_saturated_soft_gate_reproduces_spatial_mse():
    torch.manual_seed(1)
    # P so large that w = SNR/(1+SNR) ≈ 1 at mid-t → Parseval makes the
    # DCT-domain weighted mean equal the spatial MSE.
    gate = SpdSnrGate(torch.full((N_BINS,), 1e9), mode="soft")
    pred = torch.randn(2, 16, 1, 12, 16)
    target = torch.randn(2, 16, 1, 12, 16)
    t = torch.full((2,), 0.4)
    loss, w_mean = gate.gated_mse(pred, target, t, (12, 16))
    ref = F.mse_loss(pred.float(), target.float())
    assert torch.allclose(loss, ref, rtol=1e-4)
    assert float(w_mean) > 0.999


def test_soft_weight_monotone_in_t_and_power():
    profile = torch.linspace(8.0, 0.01, N_BINS)  # decaying P_ω
    gate = SpdSnrGate(profile, mode="soft")
    w_early = gate.weights(torch.tensor([0.8]), 16, 16, 16, 16)
    w_late = gate.weights(torch.tensor([0.2]), 16, 16, 16, 16)
    assert (w_late >= w_early).all()  # signal emerges as t falls
    # Higher-P (lower-frequency) coefficients are never less supervised.
    w = w_early[0, 0]
    assert (w[0, 0] >= w).all()


def test_hard_gate_activation_time():
    p0 = 4.0
    delta = 0.01
    gate = SpdSnrGate(torch.full((N_BINS,), p0), mode="hard", delta=delta)
    # Eq. 9: t_ω = 1 / (1 + sqrt(δ / (P (1 + P − δ)))) ≈ 0.978 for P=4, δ=0.01.
    t_act = 1.0 / (1.0 + (delta / (p0 * (1.0 + p0 - delta))) ** 0.5)
    below = gate.weights(torch.tensor([t_act - 0.01]), 8, 8, 8, 8)
    above = gate.weights(torch.tensor([t_act + 0.01]), 8, 8, 8, 8)
    assert (below == 1.0).all()
    assert (above == 0.0).all()


def test_stage_grid_weights_match_fullres_topleft_block():
    torch.manual_seed(2)
    gate = SpdSnrGate(torch.rand(N_BINS) + 0.05, mode="soft")
    t = torch.tensor([0.55])
    w_full = gate.weights(t, 16, 16, 16, 16)
    w_low = gate.weights(t, 8, 8, 16, 16)
    assert torch.equal(w_low, w_full[..., :8, :8])
