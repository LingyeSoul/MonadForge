"""TeacherFeatureDiscriminator head granularities (pooled v0 vs LADD-style token).

The invariants that make the token head safe to wire:

* shape contract — pooled → (B, num_taps); token → (B, num_taps · N_tokens)
* layout agnosticism — identical logits on the eager (B, T, H, W, D) grid and the
  compiled native-flatten (B, 1, L, 1, D) layout (same tokens, same order)
* parameter compatibility — the two granularities share an identical state_dict
  (same MLP, applied pooled vs per-token), so resume bundles load across a switch
* downstream shape-agnosticism — the hinge losses and the f-distill ratio mean
  over the logit axis, so both heads feed them unchanged
"""

import torch

from networks.methods.turbo_dmd import (
    TeacherFeatureDiscriminator,
    gan_loss_discriminator,
    gan_loss_generator,
)


def _feats(B=2, T=1, H=4, W=6, D=16, num_taps=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(B, T, H, W, D, generator=g) for _ in range(num_taps)]


def test_pooled_shape_is_per_tap():
    disc = TeacherFeatureDiscriminator(inner_dim=16, num_taps=2, granularity="pooled")
    logits = disc(_feats())
    assert logits.shape == (2, 2)


def test_token_shape_is_per_token():
    disc = TeacherFeatureDiscriminator(inner_dim=16, num_taps=2, granularity="token")
    logits = disc(_feats())
    assert logits.shape == (2, 2 * 1 * 4 * 6)


def test_token_logits_layout_agnostic():
    # Eager 5D grid vs compiled native-flatten (B, 1, L, 1, D): same tokens in the
    # same flatten order must give identical logits.
    disc = TeacherFeatureDiscriminator(inner_dim=16, num_taps=1, granularity="token")
    (f,) = _feats(num_taps=1)
    B, T, H, W, D = f.shape
    flat = f.reshape(B, 1, T * H * W, 1, D)
    out_grid = disc([f])
    out_flat = disc([flat])
    assert torch.equal(out_grid, out_flat)


def test_pooled_logits_layout_agnostic():
    disc = TeacherFeatureDiscriminator(inner_dim=16, num_taps=1, granularity="pooled")
    (f,) = _feats(num_taps=1)
    B, T, H, W, D = f.shape
    flat = f.reshape(B, 1, T * H * W, 1, D)
    assert torch.allclose(disc([f]), disc([flat]), atol=1e-6)


def test_state_dict_interchangeable_across_granularities():
    pooled = TeacherFeatureDiscriminator(inner_dim=16, num_taps=2, granularity="pooled")
    token = TeacherFeatureDiscriminator(inner_dim=16, num_taps=2, granularity="token")
    token.load_state_dict(pooled.state_dict())  # strict=True by default
    pooled.load_state_dict(token.state_dict())


def test_hinge_losses_accept_both_shapes():
    for gran in ("pooled", "token"):
        disc = TeacherFeatureDiscriminator(inner_dim=16, num_taps=2, granularity=gran)
        real = disc(_feats(seed=1))
        fake = disc(_feats(seed=2))
        assert gan_loss_generator(fake).shape == ()
        assert gan_loss_discriminator(real, fake).shape == ()


def test_unknown_granularity_raises():
    try:
        TeacherFeatureDiscriminator(inner_dim=16, num_taps=1, granularity="conv")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown granularity")


# --- generator-side λ delay/warmup ramp (gan.delay_steps / gan.warmup_steps) ---


def _ramp_cfg(weight=0.03, delay=0, warmup=0):
    from types import SimpleNamespace

    return SimpleNamespace(
        gan_loss_weight_gen=weight, gan_delay_steps=delay, gan_warmup_steps=warmup
    )


def test_gan_weight_defaults_are_constant():
    from scripts.distill_turbo.primitives import gan_effective_weight

    cfg = _ramp_cfg()
    assert [gan_effective_weight(cfg, s) for s in (0, 1, 500)] == [0.03] * 3


def test_gan_weight_zero_during_delay_window():
    from scripts.distill_turbo.primitives import gan_effective_weight

    cfg = _ramp_cfg(delay=500)
    assert gan_effective_weight(cfg, 0) == 0.0
    assert gan_effective_weight(cfg, 499) == 0.0
    assert gan_effective_weight(cfg, 500) == 0.03  # instant-on with warmup=0


def test_gan_weight_ramps_linearly_after_delay():
    from scripts.distill_turbo.primitives import gan_effective_weight

    cfg = _ramp_cfg(delay=500, warmup=500)
    assert gan_effective_weight(cfg, 499) == 0.0
    w0 = gan_effective_weight(cfg, 500)
    assert 0.0 < w0 <= 0.03 / 500 + 1e-12  # first engaged step: 1/warmup of λ
    mid = gan_effective_weight(cfg, 749)
    assert abs(mid - 0.015) < 1e-9  # halfway up the ramp
    assert gan_effective_weight(cfg, 999) == 0.03  # ramp complete
    assert gan_effective_weight(cfg, 3000) == 0.03  # and stays there


def test_gan_weight_off_stays_off():
    from scripts.distill_turbo.primitives import gan_effective_weight

    cfg = _ramp_cfg(weight=0.0, delay=500, warmup=500)
    assert all(gan_effective_weight(cfg, s) == 0.0 for s in (0, 500, 1000, 3000))
