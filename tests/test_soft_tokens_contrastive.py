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


# ── AGSM objective (Algorithm 1 / Eq. 17, single-bank Phase 2) ───────────────


def _agsm_net(**kw):
    base = dict(contrastive_weight=0.1, contrastive_objective="agsm")
    base.update(kw)
    return _net(**base)


def test_agsm_pl_weights_sum_to_one_and_favor_matched():
    """w = softmax over candidates of −FM-error; the closest-to-target candidate
    (here the matched at index 0) gets the largest weight, rows sum to 1."""
    net = _agsm_net(contrastive_tau=0.5)
    vt = torch.zeros(1, 4, 8, 8)
    v_pos_ema = torch.zeros(1, 4, 8, 8)  # matched: zero error
    v_neg_ema = torch.ones(1, 2, 4, 8, 8)  # negatives: error 1
    ema_all = torch.cat([v_pos_ema.unsqueeze(1), v_neg_ema], dim=1)
    w = net._agsm_pl_weights(ema_all, vt, net._contrastive_tau)
    assert w.shape == (1, 3)
    assert w.sum(dim=1).item() == pytest.approx(1.0, abs=1e-6)
    assert w[0, 0] > w[0, 1] and w[0, 0] > w[0, 2]  # matched dominates


def test_agsm_delta_self_anneals_when_matched_dominates():
    """The whole AGSM stability claim (§3.3): when the matched caption explains
    the latent and the negatives don't, w_matched → 1, the PL baseline → v̂⁺, so
    the positive target shift Δ⁺ → 0 and tgt_pos relaxes toward plain FM."""
    net = _agsm_net(contrastive_tau=0.05, agsm_gamma=1.0)
    vt = torch.zeros(1, 4, 8, 8)
    v_pos_ema = torch.zeros(1, 4, 8, 8)  # perfect match
    v_neg_ema = torch.full((1, 2, 4, 8, 8), 3.0)  # far-off negatives
    tgt_pos, tgt_neg, diag = net.agsm_targets(v_pos_ema, v_neg_ema, vt)
    assert diag["w_matched"] > 0.99
    # tgt_pos ≈ v_target (Δ⁺ annealed away).
    assert (tgt_pos - vt).abs().max().item() < 1e-2


def test_agsm_targets_per_candidate_and_signs():
    """tgt⁻ is per-candidate (B,k,…) and uses γ⁻; tgt⁺ uses γ⁺ with opposite
    sign. With distinct negatives the two negative targets differ."""
    net = _agsm_net(contrastive_tau=0.5, agsm_gamma=1.0, agsm_gamma_neg=0.1)
    assert net._agsm_gamma == 1.0 and net._agsm_gamma_neg == 0.1
    vt = torch.zeros(1, 4, 8, 8)
    v_pos_ema = torch.zeros(1, 4, 8, 8)
    v_neg_ema = torch.stack(
        [torch.full((1, 4, 8, 8), 1.0), torch.full((1, 4, 8, 8), -2.0)], dim=1
    )
    tgt_pos, tgt_neg, _ = net.agsm_targets(v_pos_ema, v_neg_ema, vt)
    assert tgt_pos.shape == (1, 4, 8, 8)
    assert tgt_neg.shape == (1, 2, 4, 8, 8)
    # Distinct EMA negatives ⇒ distinct per-candidate negative targets.
    assert not torch.allclose(tgt_neg[:, 0], tgt_neg[:, 1])


def test_agsm_gamma_neg_defaults_to_gamma():
    net = _agsm_net(agsm_gamma=0.7)
    assert net._agsm_gamma_neg == 0.7  # symmetric fallback when unset


def test_agsm_losses_zero_at_target_and_carry_grad():
    net = _agsm_net()
    vt = torch.zeros(1, 4, 8, 8)
    bias = net.tokens.mean()
    v_pos_ema = torch.zeros(1, 4, 8, 8)
    v_neg_ema = torch.ones(1, 2, 4, 8, 8)
    tgt_pos, tgt_neg, _ = net.agsm_targets(v_pos_ema, v_neg_ema, vt)
    # Live preds sit exactly on the (detached) targets → zero loss.
    l_pos, l_neg = net.agsm_losses(tgt_pos, tgt_neg, tgt_pos, tgt_neg)
    assert l_pos.item() == pytest.approx(0.0, abs=1e-6)
    assert l_neg.item() == pytest.approx(0.0, abs=1e-6)
    # Grad reaches the bank via the live velocity args.
    v_pos = torch.zeros(1, 4, 8, 8) + bias
    v_neg = torch.ones(1, 2, 4, 8, 8) + bias
    lp, ln = net.agsm_losses(v_pos, v_neg, tgt_pos, tgt_neg)
    (lp + ln).backward()
    assert net.tokens.grad is not None and torch.isfinite(net.tokens.grad).all()


def test_agsm_metadata_stamps_gammas():
    net = _agsm_net(agsm_gamma=1.0, agsm_gamma_neg=0.1)
    md = net.metadata_fields()
    assert md["ss_contrastive_objective"] == "agsm"
    assert md["ss_agsm_gamma"] == "1.0"
    assert md["ss_agsm_gamma_neg"] == "0.1"


# ── Dual bank ψ⁺/ψ⁻ (AGSM Phase 3a, §3.3) ────────────────────────────────────


def test_single_bank_is_default_and_3d():
    """Default (no flag) stays single-bank with the unchanged 3D token shape +
    legacy on-disk t_offsets width — Phase-2 checkpoints keep loading."""
    net = _net()
    assert net.n_banks == 1 and net.agsm_dual_bank is False
    assert net.tokens.shape == (2, 4, 16)  # (n_layers, K, D)
    assert net.t_offsets.weight.shape == (4, 2 * 16)  # (n_t_buckets, n_layers·D)
    md = net.metadata_fields()
    assert md["ss_n_banks"] == "1" and md["ss_agsm_dual_bank"] == "false"


def test_dual_bank_adds_branch_axis():
    """agsm_dual_bank=True prepends a branch axis to tokens and widens t_offsets
    by n_banks (bank-major column layout)."""
    net = _net(agsm_dual_bank=True)
    assert net.n_banks == 2 and net.agsm_dual_bank is True
    assert net.tokens.shape == (2, 2, 4, 16)  # (n_banks, n_layers, K, D)
    assert net.t_offsets.weight.shape == (4, 2 * 2 * 16)  # (Tb, n_banks·n_layers·D)
    md = net.metadata_fields()
    assert md["ss_n_banks"] == "2" and md["ss_agsm_dual_bank"] == "true"


def test_dual_bank_branches_are_independent():
    """ψ⁺ (branch 0) and ψ⁻ (branch 1) produce different step tokens — they are
    separately initialized, so the splice picks the right region per branch."""
    net = _net(agsm_dual_bank=True)
    t = torch.full((1,), 0.3)
    net._set_step_tokens(t, None, branch=0)
    psi_plus = net._step_layer_tokens.clone()
    net._set_step_tokens(t, None, branch=1)
    psi_minus = net._step_layer_tokens.clone()
    assert psi_plus.shape == psi_minus.shape
    assert not torch.allclose(psi_plus, psi_minus)
    # branch 0 must equal the explicit ψ⁺ lookup (append_postfix's branch).
    expected = net._layer_tokens_from(net.tokens, net.t_offsets.weight, t, branch=0)
    assert torch.allclose(psi_plus, expected)


def test_dual_bank_append_postfix_uses_psi_plus():
    """append_postfix (anchor + inference path) always splices ψ⁺ (branch 0)."""
    net = _net(agsm_dual_bank=True)
    t = torch.full((1,), 0.5)
    net.append_postfix(torch.zeros(1, 8, 16), torch.tensor([8]), timesteps=t)
    got = net._step_layer_tokens
    expected = net._layer_tokens_from(net.tokens, net.t_offsets.weight, t, branch=0)
    assert torch.allclose(got, expected)


def test_dual_checkpoint_inference_loads_psi_plus_slice():
    """A single-bank (inference) net loading a dual checkpoint keeps ONLY ψ⁺
    (Appendix H: ψ⁻ over-suppresses detail at inference) — branch-0 of tokens
    and the first n_layers·D columns of t_offsets."""
    dual = _net(agsm_dual_bank=True)
    with torch.no_grad():
        dual.tokens.copy_(torch.randn_like(dual.tokens))
        dual.t_offsets.weight.copy_(torch.randn_like(dual.t_offsets.weight))
    sd = dual.state_dict_for_save(torch.float32)
    assert sd["tokens"].shape == (2, 2, 4, 16)

    infer = _net()  # single bank
    tok, toff = infer._select_load_weights(sd["tokens"], sd["t_offsets.weight"])
    assert tok.shape == infer.tokens.shape  # (n_layers, K, D)
    assert toff.shape == infer.t_offsets.weight.shape
    assert torch.allclose(tok, sd["tokens"][0])  # ψ⁺ branch
    assert torch.allclose(toff, sd["t_offsets.weight"][:, : 2 * 16])  # first L·D cols


def test_single_checkpoint_into_dual_net_errors():
    """A 3D single-bank checkpoint can't unambiguously seed both ψ⁺/ψ⁻."""
    dual = _net(agsm_dual_bank=True)
    single = _net()
    sd = single.state_dict_for_save(torch.float32)
    with pytest.raises(ValueError, match="single"):
        dual._select_load_weights(sd["tokens"], sd["t_offsets.weight"])


def test_dual_bank_file_roundtrip_inference(tmp_path):
    """End-to-end: save a dual checkpoint, build an inference net via
    create_network_from_weights (single ψ⁺), load_weights from the file — the
    inference bank equals the saved ψ⁺ branch."""
    from safetensors.torch import save_file

    from networks.methods.soft_tokens import create_network_from_weights

    dual = _net(agsm_dual_bank=True)
    with torch.no_grad():
        dual.tokens.copy_(torch.randn_like(dual.tokens))
        dual.t_offsets.weight.copy_(torch.randn_like(dual.t_offsets.weight))
    sd = dual.state_dict_for_save(torch.float32)
    path = tmp_path / "dual.safetensors"
    save_file(dict(sd), str(path), metadata=dual.metadata_fields())

    net, _ = create_network_from_weights(
        1.0, str(path), None, None, None, for_inference=True
    )
    assert net.n_banks == 1  # inference is ψ⁺-only
    net.load_weights(str(path))
    assert torch.allclose(net.tokens, dual.tokens[0])
    assert torch.allclose(net.t_offsets.weight, dual.t_offsets.weight[:, : 2 * 16])


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
