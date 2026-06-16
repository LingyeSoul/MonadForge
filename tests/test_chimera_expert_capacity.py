"""Invariants for the per-expert capability levers on frozen-Cayley chimera.

Two training-only structural knobs expand each ortho expert WITHOUT freeing the
bases (the ``use_ortho_init`` collapse mode):

  * ``expert_basis_mult`` (m): over-complete ``(out, m·r)`` frozen pool per
    expert + an ``m·r`` Cayley rotation, with an r-dim Stiefel select at forward.
  * ``expert_diag``: per-expert ``(K, r)`` trainable diagonal σ.

Guards here:
  1. ΔW = 0 at init (centered gate) with both levers on.
  2. The levers distill faithfully — the training forward equals the free-form
     inference forward reconstructed from the distilled weights.
  3. Cross-expert column-space orthogonality survives (disjoint pools ⇒ no
     collapse, the whole point vs ortho_init).
  4. Defaults (m=1, diag off) keep the canonical r-slice parameterization.

CPU-only; SVD inits read ``torch.cuda.is_available()`` so we pin it False.
"""

import os
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from networks.lora_modules.chimera import (  # noqa: E402
    ChimeraHydraInferenceModule,
    ChimeraHydraLoRAModule,
)

K_C, K_F, R, M_MULT = 3, 2, 4, 2
IN_F, OUT_F = 64, 256
PREFIX = "lora_unet_block"


def _force_cpu():
    return mock.patch("torch.cuda.is_available", return_value=False)


def _base(seed=0):
    torch.manual_seed(seed)
    b = torch.nn.Linear(IN_F, OUT_F, bias=False).to(torch.bfloat16)
    b.weight.requires_grad_(False)
    return b


def _train_module(basis_mult=M_MULT, diag=True, seed=0):
    with _force_cpu():
        m = ChimeraHydraLoRAModule(
            "m",
            _base(seed),
            multiplier=1.0,
            lora_dim=R,
            alpha=R,
            num_experts_content=K_C,
            num_experts_freq=K_F,
            lambda_init=0.1,
            expert_basis_mult=basis_mult,
            expert_diag=diag,
        )
    m.apply_to()
    return m


def _set_routes(m):
    m.set_content_routing_weights(torch.tensor([[0.5, 0.3, 0.2]]))
    m.set_freq_routing_weights(torch.tensor([[0.6, 0.4]]))


def _x(seed=100):
    torch.manual_seed(seed)
    return torch.randn(2, 8, IN_F, dtype=torch.bfloat16)


def _perturb(m):
    """Move off the zero-init so the forward is a meaningful guard."""
    with torch.no_grad():
        for p in (m.S_q_c, m.S_q_f, m.S_p_c, m.S_p_f):
            p.copy_(torch.randn_like(p) * 0.05)
        m.lambda_c.copy_(torch.randn_like(m.lambda_c) * 0.3)
        m.lambda_f.copy_(torch.randn_like(m.lambda_f) * 0.3)
        if hasattr(m, "sigma_c"):
            m.sigma_c.copy_(1.0 + torch.randn_like(m.sigma_c) * 0.2)
            m.sigma_f.copy_(1.0 + torch.randn_like(m.sigma_f) * 0.2)


def test_delta_w_zero_at_init():
    """Both levers on, no training: the centered uniform gate gives ΔW=0, so
    the module reproduces the base Linear exactly regardless of σ/λ init."""
    m = _train_module()
    m.eval()
    x = _x()
    with torch.no_grad():
        assert torch.allclose(m(x), m.org_forward(x), atol=1e-5)


def test_overcomplete_shapes():
    """m>1 widens the P-side: M=m·r bases + M×M rotations; diag adds σ."""
    m = _train_module(basis_mult=M_MULT, diag=True)
    assert m._M == R * M_MULT
    assert m.P_bases_c.shape == (K_C, OUT_F, R * M_MULT)
    assert m.S_p_c.shape == (K_C, R * M_MULT, R * M_MULT)
    assert m.sigma_c.shape == (K_C, R)
    assert m.sigma_f.shape == (K_F, R)


def test_default_path_unchanged():
    """m=1, diag off ⇒ canonical r-slice parameterization, no σ params."""
    m = _train_module(basis_mult=1, diag=False)
    assert m._M == R
    assert m.P_bases_c.shape == (K_C, OUT_F, R)
    assert m.S_p_c.shape == (K_C, R, R)
    assert not hasattr(m, "sigma_c")
    assert not hasattr(m, "_eye_p")


def test_distill_roundtrip_faithful():
    """The trained forward == free-form inference forward rebuilt from the
    distilled weights — proves the Stiefel select + σ fold round-trip exactly."""
    m = _train_module(basis_mult=M_MULT, diag=True)
    _perturb(m)
    _set_routes(m)
    m.eval()
    x = _x()
    with torch.no_grad():
        y_train = m(x).float()

    # Distill into the on-disk free-form layout.
    sd = {f"{PREFIX}.{k}": v.detach().clone() for k, v in m.state_dict().items()}
    ChimeraHydraLoRAModule.distill_save_state_dict(sd, dtype=torch.float32)

    # The distilled per-pool keys map 1:1 onto the inference module params.
    with _force_cpu():
        inf = ChimeraHydraInferenceModule(
            "m",
            _base(),
            multiplier=1.0,
            lora_dim=R,
            alpha=R,
            num_experts_content=K_C,
            num_experts_freq=K_F,
        )
    inf.apply_to()
    with torch.no_grad():
        inf.lora_down_c.weight.data.copy_(sd[f"{PREFIX}.lora_down_c.weight"])
        inf.lora_down_f.weight.data.copy_(sd[f"{PREFIX}.lora_down_f.weight"])
        inf.lora_up_c_weight.data.copy_(sd[f"{PREFIX}.lora_up_c_weight"])
        inf.lora_up_f_weight.data.copy_(sd[f"{PREFIX}.lora_up_f_weight"])
    _set_routes(inf)
    inf.eval()
    with torch.no_grad():
        y_inf = inf(x).float()

    # bf16 training GEMMs vs fp32 distill ⇒ compare on relative L2; a Stiefel /
    # σ-fold bug would blow this to O(1).
    rel = (y_train - y_inf).norm() / y_inf.norm().clamp_min(1e-8)
    assert rel < 0.06, f"distill round-trip drifted: rel L2 = {rel:.4f}"


def test_cross_expert_orthogonality_preserved():
    """Disjoint per-expert pools ⇒ the distilled content experts keep nearly
    orthogonal column spaces even after rotation+σ — they cannot collapse into
    a shared subspace the way free (ortho_init) bases do."""
    m = _train_module(basis_mult=M_MULT, diag=True)
    _perturb(m)
    sd = {f"{PREFIX}.{k}": v.detach().clone() for k, v in m.state_dict().items()}
    ChimeraHydraLoRAModule.distill_save_state_dict(sd, dtype=torch.float32)
    ups = sd[f"{PREFIX}.lora_up_c_weight"]  # (K_c, out, r)

    # Unit-normalize every expert column, then look at the cross-expert Gram.
    cols = ups.permute(0, 2, 1).reshape(K_C * R, OUT_F)  # (K_c·r, out)
    cols = cols / cols.norm(dim=1, keepdim=True).clamp_min(1e-8)
    gram = cols @ cols.T  # (K_c·r, K_c·r)
    block = torch.zeros_like(gram, dtype=torch.bool)
    for k in range(K_C):
        block[k * R : (k + 1) * R, k * R : (k + 1) * R] = True
    cross_max = gram[~block].abs().max().item()
    assert cross_max < 0.2, f"experts not disjoint: max cross-cos = {cross_max:.3f}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
