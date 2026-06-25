"""VeRA (Vector-based Random Matrix Adaptation) module regression tests.

VeRA was the only registered LoRA-family variant
(``NETWORK_REGISTRY["vera"]``, ``configs/methods/vera.toml``) with **zero**
unit coverage — which is how a fatal bug slipped through: the frozen ``B``
matrix was initialized to ``torch.zeros`` (it's a non-trainable ``register_buffer``),
so ``ΔW = diag(b) @ B @ diag(d) @ A = 0`` and the whole module emitted a zero
delta. These tests pin the corrected Kaiming-initialized ``B`` and guard the
A/B sampling path that was deduplicated into ``_sample_AB``.

Unlike the golden-equivalence harness (``test_lora_module_equivalence.py``),
VeRA's only randomness is a seeded ``torch.Generator`` + ``kaiming_uniform_``
(no SVD/LAPACK), so its forwards are cross-platform bit-stable and we assert
with ``torch.allclose`` against an independently rebuilt delta — no Linux-only
``.pt`` regeneration required.
"""

from __future__ import annotations

import os

# Force CPU-only test runs (mirrors tests/conftest.py). Hide the GPU before
# torch picks a device so assertions are identical with or without a card.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from networks.lora_modules.vera import VeRAModule  # noqa: E402

# bf16 tolerance — matches test_lora_dtype_policy's _CS_ATOL/_CS_RTOL.
_ATOL = 5e-2
_RTOL = 1e-2


def _linear_base(in_f=32, out_f=24, seed=0):
    """Frozen bf16 Linear, matching the golden-harness base shape."""
    torch.manual_seed(seed)
    base = torch.nn.Linear(in_f, out_f, bias=False).to(torch.bfloat16)
    base.weight.requires_grad_(False)
    return base


def _conv_base(in_c=8, out_c=12, seed=0):
    """Frozen bf16 Conv2d."""
    torch.manual_seed(seed)
    base = torch.nn.Conv2d(in_c, out_c, 3, padding=1, bias=False).to(torch.bfloat16)
    base.weight.requires_grad_(False)
    return base


def _build_linear(seed=0):
    """Construct a VeRA module on a frozen Linear base and seed its A/B.

    ``set_shared_matrices`` MUST run before ``forward`` (it registers the A/B
    buffers); production wires it via ``_post_init_vera``, here we call it
    directly. VeRA is router-free — no sigma/fei/routing setup needed.
    """
    base = _linear_base()
    module = VeRAModule("m", base, multiplier=1.0, lora_dim=4, alpha=4)
    module.set_shared_matrices(seed=seed)
    module.apply_to()
    return base, module


def test_vera_delta_nonzero():
    """The bug this test exists for: the frozen B buffer must not be zero.

    A zero ``B`` (it's a ``register_buffer``, not trainable) zeroes the entire
    delta and makes VeRA a no-op. If anyone reverts B to ``torch.zeros`` this
    assertion fails immediately.
    """
    _, module = _build_linear(seed=0)

    # B is the frozen random matrix; it must carry signal.
    assert module.B.abs().sum().item() > 0, "VeRA B matrix is all zeros — delta would be 0"
    assert module.A.abs().sum().item() > 0, "VeRA A matrix is all zeros"

    # The forward must actually perturb the base output. If the delta were 0
    # the adapter would be inert. VeRA's eval path (``_eval_delta``) multiplies
    # the raw A/B buffers with the activation, so — like the LoRA / step-expert
    # raw-nn eval paths — the module must be loaded in the model dtype (bf16)
    # before eval; the buffer A/B follow the module cast.
    module.eval()
    module.to(torch.bfloat16)
    # VeRA's forward uses torch.mm (2D only), so the eval path is exercised
    # with a 2D batch — matching how the DiT feeds an adapted Linear.
    x = torch.randn(16, 32, dtype=torch.bfloat16)
    with torch.no_grad():
        org_y = module.org_forward(x)
        y = module.forward(x)
    assert y.shape == org_y.shape
    assert not torch.equal(y, org_y), (
        "VeRA forward equals org_forward — the delta is zero (B=zeros regression?)"
    )


def test_vera_forward_matches_reference():
    """Eval forward equals org_forward + x @ ΔW.T, ΔW from get_weight().

    ``get_weight`` rebuilds ΔW via ``_reconstruct_delta`` (re-sampling A/B from
    the seed); the eval forward uses the registered A/B buffers. Equality here
    is the core VeRA invariant: train-time A/B == reconstruction A/B. After the
    ``_sample_AB`` dedup both paths share one sampler, so this also guards the
    refactor.
    """
    _, module = _build_linear(seed=0)
    module.eval()
    module.to(torch.bfloat16)

    x = torch.randn(16, 32, dtype=torch.bfloat16)
    with torch.no_grad():
        org_y = module.org_forward(x)
        y = module.forward(x)
        # get_weight rebuilds ΔW in fp32 (``_reconstruct_delta`` casts to float),
        # independent of the module's current buffer dtype.
        delta_w = module.get_weight()  # (out_dim, in_dim), fp32

    # y = org_forward(x) + (x @ A.T * d) @ B.T * b * scale * multiplier.
    # In weight form: delta contribution = x @ ΔW.T (ΔW already carries scale).
    ref = org_y.float() + (x.float() @ delta_w.t())
    assert torch.allclose(y.float(), ref, atol=_ATOL, rtol=_RTOL), (
        f"VeRA forward diverged from x @ ΔW.T reference "
        f"(max abs diff {(y.float() - ref).abs().max().item():.3e})"
    )


def test_vera_seed_reproducibility():
    """Same seed → bit-identical A/B; different seed → different A/B.

    Guards the ``_sample_AB`` generator contract: the result is a pure function
    of (seed) and A consumes the stream before B.
    """
    _, m0 = _build_linear(seed=0)
    _, m0b = _build_linear(seed=0)
    _, m1 = _build_linear(seed=1)

    assert torch.equal(m0.A, m0b.A), "same seed produced different A"
    assert torch.equal(m0.B, m0b.B), "same seed produced different B"
    assert not torch.equal(m0.A, m1.A), "different seeds produced identical A"
    assert not torch.equal(m0.B, m1.B), "different seeds produced identical B"


def test_vera_conv2d():
    """VeRA declares supports_conv2d=True — the conv path must also be non-trivial."""
    base = _conv_base()
    module = VeRAModule("m", base, multiplier=1.0, lora_dim=4, alpha=4)
    module.set_shared_matrices(seed=0)
    module.apply_to()
    module.eval()
    module.to(torch.bfloat16)

    assert module.B.abs().sum().item() > 0

    x = torch.randn(2, 8, 12, 12, dtype=torch.bfloat16)
    with torch.no_grad():
        org_y = module.org_forward(x)
        y = module.forward(x)
    assert y.shape == org_y.shape
    assert not torch.equal(y, org_y), "VeRA conv2d forward equals org_forward — delta is zero"
