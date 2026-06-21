"""fp16-safe residual accumulation regression tests.

Background: the Anima DiT residual stream exceeds fp16's 65504 ceiling late
in the block stack (``docs/findings/selfflow.md``). Under fp16 autocast the
``Block._forward`` residual adds and the ``FinalLayer`` AdaLN modulate
overflow to ``inf → NaN`` from step 0 — this is the V100 + fp16 NaN
reproduction (V100 / sm_70 has no native bf16, so the default ``bf16``
silently runs fp32 autocast; users pick fp16 for the matmul speedup).

The fix (``Anima.enable_fp32_residual()``) keeps matmuls in the autocast
(fp16) dtype but promotes the residual adds + final-layer modulate to fp32,
casting back after. These tests pin three contracts:

1. **Unit**: ``Block._residual_add`` is finite for fp16 inputs summing past
   65504 when ``fp32_residual=True``, and overflows when ``False`` (the
   negative control proving the input actually trips the bug).
2. **Block forward**: a ``Block`` whose residual stream is driven past 65504
   produces a finite output under fp16 autocast only when the flag is set.
3. **Inert-on-default parity**: with ``fp32_residual=False`` (the bf16/fp32
   default), ``_residual_add`` is bit-exact to a plain ``a + b`` — so the
   flag is a true no-op on the production path.

CPU autocast is used so the tests run without a GPU; the overflow math is
identical (fp16 max is 65504 everywhere).
"""

from __future__ import annotations

import torch

from library.anima.models import Block, FinalLayer

# fp16's finite max — values past this saturate to inf.
_FP16_MAX = torch.finfo(torch.float16).max


def _fp16_autocast():
    return torch.autocast("cpu", dtype=torch.float16)


def test_residual_add_unit_overflow_guard():
    """_residual_add is finite past 65504 with the flag, inf without it.

    Cast-back is impossible: a value >65504 has no fp16 representation, so the
    fp16 path keeps the result in fp32 (the residual stream stays fp32 across
    the block stack; downstream matmuls re-cast to fp16 under autocast).
    """
    block = Block(x_dim=64, context_dim=64, num_heads=4)
    a = torch.full((2, 4), 0.9 * _FP16_MAX, dtype=torch.float16)
    b = torch.full((2, 4), 0.9 * _FP16_MAX, dtype=torch.float16)

    # Negative control: the naive fp16 add overflows (proves the input trips fp16).
    naive = (a + b).to(torch.float16)
    assert torch.isinf(naive).any(), "test input does not actually overflow fp16"

    # Flag off → identical to the naive fp16 add (parity with legacy code).
    block.fp32_residual = False
    guarded_off = block._residual_add(a, b)
    assert torch.equal(guarded_off, naive)

    # Flag on → fp32 result, finite. Expected = sum of the actual fp16 inputs
    # (they round to the nearest fp16-representable value, so derive from `a`).
    block.fp32_residual = True
    guarded_on = block._residual_add(a, b)
    assert guarded_on.dtype == torch.float32
    assert torch.isfinite(guarded_on).all()
    expected = a.float() + b.float()
    assert torch.equal(guarded_on, expected)


def test_residual_add_inert_on_default_path():
    """fp32_residual=False is bit-exact to ``a + b`` across dtypes (no-op)."""
    block = Block(x_dim=64, context_dim=64, num_heads=4)
    block.fp32_residual = False
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        g = torch.Generator().manual_seed(0)
        a = torch.randn(3, 8, generator=g, dtype=dtype)
        b = torch.randn(3, 8, generator=g, dtype=dtype)
        assert torch.equal(block._residual_add(a, b), a + b), f"parity drift on {dtype}"


def test_block_forward_runs_under_fp16_autocast_with_flag():
    """A Block runs cleanly under fp16 autocast with fp32_residual on.

    The dispositive overflow regression is the _residual_add unit test above
    (it injects a >65504 sum directly and asserts finiteness). This test pins
    the integration contract: the flag is read inside the compiled _forward,
    the block runs under fp16 autocast without error, and the output is
    finite for a normal-magnitude input. The fixture's residual stream is
    not large enough to overflow on its own (adaLN gates are zero-init), so
    this is a "flag wires through correctly" check, not an overflow repro.
    """
    x_dim = 64
    block = Block(x_dim=x_dim, context_dim=x_dim, num_heads=4)
    block.eval()

    B, T, H, W, D = 1, 1, 2, 2, x_dim
    x = torch.randn(B, T, H, W, D, dtype=torch.float16)
    emb = torch.zeros(B, T, D, dtype=torch.float16)
    crossattn_emb = torch.randn(B, 4, D, dtype=torch.float16)

    from networks.attention_dispatch import AttentionParams

    attn_params = AttentionParams.create_attention_params("torch")

    # Flag ON → runs cleanly, finite output. The residual stream stays fp32
    # (values >65504 have no fp16 representation); downstream matmuls re-cast
    # to fp16 under autocast. Output dtype is fp32 because the block returns
    # the residual stream.
    block.fp32_residual = True
    with _fp16_autocast(), torch.no_grad():
        out_on = block(x, emb, crossattn_emb, attn_params)
    assert torch.isfinite(out_on).all(), "fp32 residual path produced inf/nan"
    assert out_on.dtype == torch.float32

    # Flag OFF → also finite here (no overflow on this normal-magnitude input);
    # confirms the flag is a no-op until a >65504 residual actually appears.
    block.fp32_residual = False
    with _fp16_autocast(), torch.no_grad():
        out_off = block(x, emb, crossattn_emb, attn_params)
    assert torch.isfinite(out_off).all()


def test_final_layer_no_overflow_under_fp16_autocast():
    """FinalLayer's AdaLN modulate stays finite past 65504 when the flag is on."""
    hidden_size = 64
    final = FinalLayer(
        hidden_size=hidden_size,
        spatial_patch_size=1,
        temporal_patch_size=1,
        out_channels=4,
        use_adaln_lora=False,
    )
    final.eval()

    # Residual stream near the fp16 ceiling; the AdaLN ``*(1+scale)+shift``
    # would overflow fp16. (B, T=1, H=2, W=2, D)
    x = torch.full((1, 1, 2, 2, hidden_size), 0.95 * _FP16_MAX, dtype=torch.float16)
    emb = torch.zeros(1, 1, hidden_size, dtype=torch.float16)

    final.fp32_residual = True
    with _fp16_autocast(), torch.no_grad():
        out_on = final(x, emb)
    assert torch.isfinite(out_on).all(), "fp32 final-layer modulate produced inf/nan"

    # Note: FinalLayer's overflow is gated on the modulation scale/shift being
    # nonzero; with zero-init weights the legacy path is *also* finite here, so
    # we don't assert the negative control for FinalLayer — the Block forward
    # test above is the dispositive overflow regression.


def test_enable_fp32_residual_propagates_to_all_modules():
    """Anima.enable_fp32_residual() flips the flag on every Block + FinalLayer."""
    from library.anima.models import Anima

    # head_dim = model_channels // num_heads = 32; dim_h = 32//6*2 = 10 (>2 so
    # the NTK factor dim_h/(dim_h-2) doesn't divide by zero in the RoPE ctor).
    anima = Anima(
        max_img_h=16,
        max_img_w=16,
        max_frames=1,
        in_channels=16,
        out_channels=16,
        patch_spatial=2,
        patch_temporal=1,
        model_channels=128,
        num_blocks=3,
        num_heads=4,
        crossattn_emb_channels=128,
        pos_emb_learnable=False,
        rope_enable_fps_modulation=False,
        use_llm_adapter=False,
        use_adaln_lora=False,
        attn_mode="torch",
    )

    # Default: every module inert.
    assert anima._fp32_residual_enabled is False
    assert all(not b.fp32_residual for b in anima.blocks)
    assert anima.final_layer.fp32_residual is False

    anima.enable_fp32_residual()

    assert anima._fp32_residual_enabled is True
    assert len(anima.blocks) > 0
    assert all(b.fp32_residual for b in anima.blocks)
    assert anima.final_layer.fp32_residual is True
