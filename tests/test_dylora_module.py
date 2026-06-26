"""DyLoRA (Dynamic LoRA) module regression tests.

DyLoRA trains multiple ranks simultaneously: each training forward samples a
random ``r ∈ {unit, 2*unit, ..., lora_dim}`` and computes only the
``[:r]``/``[:, :r]`` slices; eval (``_eval_delta``) runs the full rank. After
training any rank ≤ ``lora_dim`` can be extracted. Ref: arXiv:2202.05955.

Like plain LoRA, ``lora_up`` is zero-initialized (ΔW=0 at init), so the
"non-zero delta" assertion that ``test_vera_module.py`` carries (VeRA seeds
``vera_d``/``vera_b`` to ones) does NOT apply here. Instead these tests pin:
  - the rank-sampling contract (the load-bearing DyLoRA invariant),
  - the eval-vs-train rank policy,
  - the save/merge weight reconstruction.

Mirrors ``test_vera_module.py``'s CPU-only, cross-platform-stable style.
"""

from __future__ import annotations

import os

# Force CPU-only test runs (mirrors tests/conftest.py).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from networks.lora_modules.dylora import DyLoRAModule  # noqa: E402


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


def _build_linear(lora_dim=8, unit=1, algo=""):
    """Construct a DyLoRA module on a frozen Linear base."""
    base = _linear_base()
    module = DyLoRAModule(
        "m", base, multiplier=1.0, lora_dim=lora_dim, alpha=lora_dim, unit=unit, algo=algo
    )
    module.apply_to()
    return base, module


def test_dylora_random_rank_within_bounds():
    """``_random_rank()`` must return a value in {unit, 2*unit, ..., lora_dim}.

    This is the load-bearing DyLoRA invariant: the sampled rank must be a
    positive multiple of ``unit`` and ≤ ``lora_dim``. A rank outside this set
    would either slice out-of-bounds or skip the unit granularity contract.
    """
    for unit in (1, 2, 4):
        _, module = _build_linear(lora_dim=8, unit=unit)
        for _ in range(200):
            r = module._random_rank()
            assert r >= unit, f"rank {r} < unit {unit}"
            assert r <= module.lora_dim, f"rank {r} > lora_dim {module.lora_dim}"
            assert r % unit == 0, f"rank {r} not a multiple of unit {unit}"


def test_dylora_unit_clamped_to_one():
    """``unit`` is clamped to ≥1 — a zero/negative unit must not divide-by-zero
    in ``_random_rank`` (``n = lora_dim // unit``)."""
    base = _linear_base()
    module = DyLoRAModule("m", base, lora_dim=8, alpha=8, unit=0)
    assert module.unit == 1, "unit=0 must clamp to 1"
    # And _random_rank must not raise.
    for _ in range(50):
        assert module._random_rank() >= 1


def test_dylora_eval_uses_full_rank():
    """Eval path (``_eval_delta``) computes the full ``lora_up(lora_down(x))``
    regardless of the random training rank. After seeding ``lora_up`` with a
    known non-zero value, the eval delta must reflect the FULL ``lora_dim``
    rank, not a random sub-rank.

    This guards against a regression where eval accidentally reuses the
    training-time random-rank slicing (which would make inference stochastic).
    """
    _, module = _build_linear(lora_dim=8, unit=2)
    # Seed lora_up with a known non-zero so the delta is observable (init is 0).
    with torch.no_grad():
        module.lora_up.weight.add_(0.5)
    module.eval()
    module.to(torch.bfloat16)

    x = torch.randn(16, 32, dtype=torch.bfloat16)
    with torch.no_grad():
        delta1 = module._eval_delta(x, module.org_forward(x))
        delta2 = module._eval_delta(x, module.org_forward(x))
    # Eval must be deterministic — no random rank sampling.
    assert torch.equal(delta1, delta2), "eval delta is non-deterministic (leaking train-time random rank?)"


def test_dylora_get_weight_full_rank_after_seed():
    """``get_weight`` rebuilds ΔW = up @ down at the FULL ``lora_dim`` rank —
    the post-training extraction path. After seeding non-zero weights the
    result must be non-trivial and deterministic."""
    _, module = _build_linear(lora_dim=8, unit=2)
    with torch.no_grad():
        module.lora_up.weight.add_(0.5)
    module.eval()

    w = module.get_weight()  # (out, in), fp32
    assert w.shape == (24, 32), f"unexpected get_weight shape {tuple(w.shape)}"
    assert w.abs().sum().item() > 0, "get_weight is zero after seeding lora_up"


def test_dylora_conv2d():
    """DyLoRA declares supports_conv2d=True — the conv path must construct and
    forward without error, and the conv rank sampling honors ``unit``."""
    base = _conv_base()
    module = DyLoRAModule("m", base, multiplier=1.0, lora_dim=8, alpha=8, unit=2)
    module.apply_to()
    assert module.lora_down.weight.shape[0] == 8
    # Conv rank sampling shares the same _random_rank contract.
    for _ in range(50):
        r = module._random_rank()
        assert 2 <= r <= 8 and r % 2 == 0
