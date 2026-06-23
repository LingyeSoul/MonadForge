"""do_sample seeded sampling regression.

Locks in the fix for the generator/device-mismatch crash: the old code did
``generator = torch.manual_seed(seed)`` (the *CPU* default generator) then
``torch.randn(..., generator=generator, device=device)`` — on CUDA this raised
``RuntimeError: Expected a 'cuda' device type for generator but found 'cpu'``.
The fix builds a device-matched generator.

These tests run on CPU (conftest locks CUDA off) but exercise the same control
flow: generator construction, sigma reuse, Euler-step reuse, return dtype. A
seeded run must not crash and must be reproducible seed-to-seed; a None seed
must also work. Uses a stub DiT so no model weights are needed.
"""

from __future__ import annotations

import torch

from library.anima.training import do_sample


class _StubDiT:
    """Fake DiT: ignores all inputs, returns a constant fp32 velocity field.

    do_sample calls ``dit(x, t, crossattn_emb, padding_mask=padding_mask)``
    (positional x, t, emb + keyword padding_mask), so the call signature must
    match. Returning ``full_like(x)`` keeps the output dtype/device of x, which
    the caller then ``.float()``s.
    """

    dtype = torch.float32

    def __call__(self, x, t, emb, padding_mask=None):
        return torch.full_like(x, 0.0)


def _kwargs(**overrides):
    base = dict(
        height=64,
        width=64,
        dit=_StubDiT(),
        crossattn_emb=torch.zeros(1, 1, 1024),
        steps=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
        show_progress=False,
    )
    base.update(overrides)
    return base


def test_do_sample_seeded_does_not_crash():
    # Regression for the generator/device crash — the old CPU-generator path
    # raised on CUDA; on CPU this also exercises the device-matched generator
    # construction. Must not raise.
    out = do_sample(seed=42, **_kwargs())
    assert out.shape == (1, 16, 1, 8, 8)


def test_do_sample_seeded_is_reproducible():
    a = do_sample(seed=42, **_kwargs())
    b = do_sample(seed=42, **_kwargs())
    assert torch.allclose(a, b)


def test_do_sample_different_seeds_differ():
    a = do_sample(seed=42, **_kwargs())
    c = do_sample(seed=43, **_kwargs())
    assert not torch.allclose(a, c)


def test_do_sample_seed_none_does_not_crash():
    out = do_sample(seed=None, **_kwargs(steps=2))
    assert out.shape == (1, 16, 1, 8, 8)


def test_do_sample_returns_model_dtype_at_boundary():
    # Internal accumulation is fp32; the API edge normalizes back to ``dtype``.
    # A stub returning zero velocity leaves x == initial noise (fp32); the
    # return must be cast to the requested dtype.
    out = do_sample(seed=7, **_kwargs(dtype=torch.bfloat16))
    assert out.dtype == torch.bfloat16
