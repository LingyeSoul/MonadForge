"""Tests for the soft-tokens contrastive objective (Phase 1).

Covers the InfoNCE loss math, the warmup-gated weight contract, the
negative-mode validation (jaccard/hard are Phase 2), and the metadata stamp.
The dataset-sourcing and extra-forward wiring are integration-tested elsewhere;
here we exercise the pure-CPU network surface. See
docs/proposal/soft_tokens_contrastive.md.
"""

from __future__ import annotations

import math

import pytest
import torch

from networks.methods.soft_tokens import SoftTokensNetwork


def _net(**kw):
    base = dict(
        num_tokens=4,
        embed_dim=16,
        n_layers=2,
        n_t_buckets=4,
        init_std=0.02,
    )
    base.update(kw)
    return SoftTokensNetwork(**base)


def test_contrastive_disabled_by_default():
    net = _net()
    assert net._contrastive_target_weight == 0.0
    assert net._contrastive_weight == 0.0


def test_contrastive_loss_perfect_positive():
    """v_pos == target, v_neg far → low loss, accuracy 1, positive gap."""
    net = _net(contrastive_weight=0.1, contrastive_tau=0.5)
    target = torch.zeros(1, 4, 8, 8)
    v_pos = torch.zeros(1, 4, 8, 8)  # pos_err = 0  → logit_pos = 0
    v_neg = torch.ones(1, 1, 4, 8, 8)  # neg_err = 1  → logit_neg = -1/0.5 = -2
    loss, diag = net.contrastive_loss(v_pos, v_neg, target)

    expected = -0.0 + math.log(math.exp(0.0) + math.exp(-2.0))
    assert loss.item() == pytest.approx(expected, abs=1e-5)
    assert diag["contrastive_acc"] == 1.0
    assert diag["contrastive_logit_gap"] == pytest.approx(2.0, abs=1e-5)


def test_contrastive_loss_wrong_way_round():
    """v_pos far, v_neg matches target → accuracy 0, larger loss."""
    net = _net(contrastive_weight=0.1, contrastive_tau=0.5)
    target = torch.zeros(1, 4, 8, 8)
    v_pos = torch.ones(1, 4, 8, 8)
    v_neg = torch.zeros(1, 1, 4, 8, 8)
    loss, diag = net.contrastive_loss(v_pos, v_neg, target)
    assert diag["contrastive_acc"] == 0.0
    assert diag["contrastive_logit_gap"] < 0.0
    # logit_pos=-2, logit_neg=0 → loss = 2 + log(1+e^-2)
    assert loss.item() == pytest.approx(2.0 + math.log(1 + math.exp(-2.0)), abs=1e-5)


def test_contrastive_loss_carries_grad_to_tokens():
    net = _net(contrastive_weight=0.1)
    target = torch.zeros(1, 4, 8, 8)
    # v_pos must depend on net.tokens for grad to flow; emulate by adding the
    # bank's mean so autograd has a path.
    bias = net.tokens.mean()
    v_pos = torch.zeros(1, 4, 8, 8) + bias
    v_neg = torch.ones(1, 2, 4, 8, 8) + bias
    loss, _ = net.contrastive_loss(v_pos, v_neg, target)
    loss.backward()
    assert net.tokens.grad is not None
    assert torch.isfinite(net.tokens.grad).all()


def test_contrastive_warmup_gate():
    net = _net(contrastive_weight=0.3, contrastive_warmup_ratio=0.1)
    assert net._contrastive_weight == 0.0  # held during warmup
    net.step_contrastive_warmup(global_step=5, max_train_steps=100)
    assert net._contrastive_weight == 0.0
    net.step_contrastive_warmup(global_step=10, max_train_steps=100)
    assert net._contrastive_weight == 0.3
    net.step_contrastive_warmup(global_step=50, max_train_steps=100)
    assert net._contrastive_weight == 0.3


def test_contrastive_warmup_zero_ratio_active_immediately():
    net = _net(contrastive_weight=0.3, contrastive_warmup_ratio=0.0)
    assert net._contrastive_weight == 0.3
    net.step_contrastive_warmup(global_step=0, max_train_steps=100)
    assert net._contrastive_weight == 0.3


def test_contrastive_warmup_noop_when_disabled():
    net = _net(contrastive_weight=0.0)
    net.step_contrastive_warmup(global_step=0, max_train_steps=100)
    assert net._contrastive_weight == 0.0


def test_contrastive_every_n_default_fires_every_step():
    net = _net(contrastive_weight=0.3, contrastive_warmup_ratio=0.0)
    assert net._contrastive_every_n == 1
    for s in range(5):
        net.step_contrastive_warmup(global_step=s, max_train_steps=100)
        assert net._contrastive_fire_this_step is True


def test_contrastive_every_n_cadence_on_optimizer_step():
    """every_n strides over optimizer steps (global_step // accum), so an
    accumulation window fires uniformly across its micro-batches."""
    net = _net(contrastive_weight=0.3, contrastive_warmup_ratio=0.0, contrastive_every_n=3)
    # accum=1: fire on micro-batches 0, 3, 6 (== optimizer steps 0, 3, 6).
    fired = []
    for s in range(9):
        net.step_contrastive_warmup(global_step=s, max_train_steps=100, accum=1)
        fired.append(net._contrastive_fire_this_step)
    assert fired == [True, False, False, True, False, False, True, False, False]


def test_contrastive_every_n_uniform_within_accum_window():
    net = _net(contrastive_weight=0.3, contrastive_warmup_ratio=0.0, contrastive_every_n=2)
    # accum=2 → optimizer steps {0,0,1,1,2,2,3,3}; fire when opt_step even.
    fired = []
    for micro in range(8):
        net.step_contrastive_warmup(global_step=micro, max_train_steps=100, accum=2)
        fired.append(net._contrastive_fire_this_step)
    # Both micro-batches of each optimizer window agree.
    assert fired == [True, True, False, False, True, True, False, False]


def test_contrastive_every_n_clamped_and_stamped():
    net = _net(contrastive_weight=0.3, contrastive_every_n=0)
    assert net._contrastive_every_n == 1  # clamped to >= 1
    net2 = _net(contrastive_weight=0.3, contrastive_every_n=4)
    assert net2.metadata_fields()["ss_contrastive_every_n"] == "4"


@pytest.mark.parametrize("mode", ["shuffled", "jaccard", "hard"])
def test_all_modes_construct(mode):
    net = _net(contrastive_weight=0.1, contrastive_negative_mode=mode)
    assert net.contrastive_negative_mode == mode


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        _net(contrastive_negative_mode="bogus")


def test_jaccard_penalty_lowers_loss():
    """Down-weighting a negative's logit (jaccard mode) makes the positive win
    more easily → strictly lower InfoNCE loss than the unpenalized case."""
    net = _net(contrastive_weight=0.1, contrastive_tau=0.5)
    target = torch.zeros(1, 4, 8, 8)
    v_pos = torch.full((1, 4, 8, 8), 0.5)  # some pos error
    v_neg = torch.full((1, 2, 4, 8, 8), 0.5)  # negatives equally close
    base, _ = net.contrastive_loss(v_pos, v_neg, target)
    penalty = torch.full((1, 2), 1.0)  # α·s on every negative
    penalized, _ = net.contrastive_loss(v_pos, v_neg, target, neg_penalty=penalty)
    assert penalized.item() < base.item()


def test_zero_penalty_matches_plain():
    net = _net(contrastive_weight=0.1)
    target = torch.zeros(1, 4, 8, 8)
    v_pos = torch.full((1, 4, 8, 8), 0.3)
    v_neg = torch.full((1, 2, 4, 8, 8), 0.7)
    plain, _ = net.contrastive_loss(v_pos, v_neg, target)
    zero_pen, _ = net.contrastive_loss(
        v_pos, v_neg, target, neg_penalty=torch.zeros(1, 2)
    )
    assert zero_pen.item() == pytest.approx(plain.item(), abs=1e-6)


def test_metadata_stamps_contrastive_config():
    net = _net(
        contrastive_weight=0.2,
        contrastive_k=2,
        contrastive_tau=0.7,
        contrastive_warmup_ratio=0.15,
    )
    md = net.metadata_fields()
    assert md["ss_contrastive_weight"] == "0.2"
    assert md["ss_contrastive_k"] == "2"
    assert md["ss_contrastive_negative_mode"] == "shuffled"
    assert md["ss_contrastive_tau"] == "0.7"
    assert md["ss_contrastive_warmup_ratio"] == "0.15"
