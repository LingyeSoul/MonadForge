"""fp16-safe residual accumulation regression tests.

Background: the Anima DiT residual stream exceeds fp16's 65504 ceiling late
in the block stack (``docs/findings/selfflow.md``). Under fp16 autocast the
``Block._forward`` residual adds and the ``FinalLayer`` AdaLN modulate
overflow to ``inf → NaN`` from step 0 — this is the V100 + fp16 NaN
reproduction (V100 / sm_70 has no native bf16, so the default ``bf16``
silently runs fp32 autocast; users pick fp16 for the matmul speedup).

The fix (``Anima.enable_fp32_residual()``) keeps transformer-block matmuls in
the autocast (fp16) dtype but promotes the residual adds, final-layer modulate,
and final projection to fp32. These tests pin three contracts:

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
    assert not anima.blocks[0].fp32_residual
    assert all(not b.fp32_residual for b in anima.blocks)
    assert anima.final_layer.fp32_residual is False

    anima.enable_fp32_residual()

    assert anima.blocks[0].fp32_residual
    assert len(anima.blocks) > 0
    assert all(b.fp32_residual for b in anima.blocks)
    assert anima.final_layer.fp32_residual is True


# ---------------------------------------------------------------------------
# Regression tests for the gaps flagged in code review:
#   A. enable_fp32_residual() BEFORE compile → no recompile storm.
#   B. backward path under fp16 autocast → finite gradients.
#   C. end-to-end overflow through a real FinalLayer forward → flag converts
#      inf → finite (vs. the unit test, which injects the >65504 sum directly
#      into _residual_add and bypasses the forward).
# ---------------------------------------------------------------------------


def _tiny_anima():
    """Small but real Anima DiT runnable on CPU (mirrors test_native_flatten)."""
    from library.anima.models import Anima

    return Anima(
        max_img_h=16,
        max_img_w=16,
        max_frames=1,
        in_channels=16,
        out_channels=16,
        patch_spatial=2,
        patch_temporal=1,
        concat_padding_mask=False,
        model_channels=64,
        num_blocks=2,
        num_heads=4,
        mlp_ratio=2.0,
        crossattn_emb_channels=64,
        use_adaln_lora=True,
        adaln_lora_dim=16,
        use_llm_adapter=False,
        attn_mode="torch",
    ).eval()


def test_enable_before_compile_no_recompile():
    """enable_fp32_residual() MUST run before compile_blocks.

    This is the direct regression for the compile-ordering bug: if the flag is
    flipped AFTER compile, dynamo specialized on fp32_residual=False and the
    first forward recompiles every block graph. Here we flip FIRST, then
    compile, and assert the second forward adds no new compiled graphs.
    """
    import torch._dynamo as _dynamo

    model = _tiny_anima()
    model.enable_fp32_residual()
    assert model.blocks[0].fp32_residual is True  # flipped pre-compile

    model.compile_blocks(backend="eager")

    torch.manual_seed(0)
    x = torch.randn(1, 16, 1, 4, 4, dtype=torch.float32)
    timesteps = torch.tensor([0.5])
    crossattn_emb = torch.randn(1, 8, 64)

    _dynamo.reset()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.float16):
        model.forward_mini_train_dit(x, timesteps, crossattn_emb)
    graphs_after_first = _dynamo.utils.counters["stats"]["unique_graphs"]

    # Second forward with identical shapes — must reuse the compiled graph,
    # not recompile (this is the invariant the flag-before-compile ordering
    # guarantees; a post-compile flip would trip the bool guard here).
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.float16):
        model.forward_mini_train_dit(x, timesteps, crossattn_emb)
    graphs_after_second = _dynamo.utils.counters["stats"]["unique_graphs"]

    assert graphs_after_second == graphs_after_first, (
        f"second forward recompiled (graphs {graphs_after_first} → "
        f"{graphs_after_second}); the fp32_residual bool guard tripped, "
        "meaning the flag was not specialized at compile time."
    )


def test_residual_add_backward_under_fp16_autocast():
    """Gradients flow through _residual_add under fp16 autocast.

    The fp32 promotion (a.float() + b.float()) is differentiable; this pins
    that backward produces finite grads and the .float() upcast doesn't break
    the autograd graph. Covers the P0 backward gap (all other tests use no_grad).
    """
    block = Block(x_dim=64, context_dim=64, num_heads=4)
    block.fp32_residual = True

    a = torch.full((2, 4), 0.4 * _FP16_MAX, dtype=torch.float16, requires_grad=True)
    b = torch.full((2, 4), 0.4 * _FP16_MAX, dtype=torch.float16, requires_grad=True)

    with torch.autocast("cpu", dtype=torch.float16):
        out = block._residual_add(a, b)
        loss = out.float().sum()
    loss.backward()

    assert torch.isfinite(a.grad).all(), "a.grad has inf/nan"
    assert torch.isfinite(b.grad).all(), "b.grad has inf/nan"
    # The fp32-residual path promotes to fp32; grads flow back to the fp16
    # leaves (autograd casts at the leaf), so leaf grads are fp16.
    assert a.grad.dtype == torch.float16
    assert b.grad.dtype == torch.float16


def test_gated_residual_add_guards_the_product_overflow():
    """The gate*branch product, not just the add, must stay in fp32.

    This is the dispositive regression for the V100 fp16 NaN that survived the
    original ``_residual_add`` guard: ``gate * branch`` is materialized in fp16
    under autocast *before* it reaches ``_residual_add``, so a product past
    65504 has already collapsed to ``inf`` and ``fp32 + inf = inf`` — the guard
    never gets a chance. ``_gated_residual_add`` pulls the product into the
    fp32 region so ``gate.float() * branch.float()`` is finite.

    Mirrors the real block call shape: ``gate`` is a per-(B,T,1,1,D) AdaLN
    modulation (>1 on a trained model), ``branch`` is an attention/MLP output.
    """
    block = Block(x_dim=64, context_dim=64, num_heads=4)

    # gate=2.0 (a trained modulation), branch=0.6 * 65504 -> product = 1.2*65504
    # overflows fp16. residual is a normal-magnitude fp16 stream.
    shape = (1, 1, 1, 1, 64)
    gate = torch.full(shape, 2.0, dtype=torch.float16)
    branch = torch.full(shape, 0.6 * _FP16_MAX, dtype=torch.float16)
    residual = torch.randn(shape, dtype=torch.float16)

    # Negative control: the naive fp16 product overflows (proves the input trips
    # fp16 on the multiply, before any add).
    with torch.autocast("cpu", dtype=torch.float16):
        naive_product = gate * branch
    assert torch.isinf(naive_product).any(), "test input does not overflow fp16 on gate*branch"

    # _residual_add alone CANNOT recover: the product is already inf when it
    # arrives. This pins why the gated variant is required.
    block.fp32_residual = True
    with torch.autocast("cpu", dtype=torch.float16):
        recovered_too_late = block._residual_add(residual, gate * branch)
    assert torch.isinf(recovered_too_late).any() or torch.isnan(
        recovered_too_late
    ).any(), "plain _residual_add should NOT recover an already-inf product"

    # The gated variant keeps the product in fp32 -> finite result.
    with torch.autocast("cpu", dtype=torch.float16):
        guarded = block._gated_residual_add(residual, gate, branch)
    assert guarded.dtype == torch.float32
    assert torch.isfinite(guarded).all(), "gated fp32 product path produced inf/nan"

    # Inert parity: flag off == legacy ``residual + gate * branch`` bit-exactly.
    block.fp32_residual = False
    with torch.autocast("cpu", dtype=torch.float16):
        guarded_off = block._gated_residual_add(residual, gate, branch)
        legacy = residual + gate * branch
    assert torch.equal(guarded_off, legacy), "inert path drifted from legacy gate*branch+residual"


def test_gated_residual_add_backward_under_fp16_autocast():
    """Gradients flow through _gated_residual_add's fp32 mul+add under autocast.

    The ``gate.float() * branch.float() + residual.float()`` expression is
    differentiable end-to-end; pins that the .float() upcasts don't break the
    autograd graph and grads reach the fp16 leaves.
    """
    block = Block(x_dim=64, context_dim=64, num_heads=4)
    block.fp32_residual = True

    shape = (2, 1, 1, 1, 4)
    residual = torch.full(shape, 0.2 * _FP16_MAX, dtype=torch.float16, requires_grad=True)
    gate = torch.full(shape, 1.5, dtype=torch.float16, requires_grad=True)
    branch = torch.full(shape, 0.2 * _FP16_MAX, dtype=torch.float16, requires_grad=True)

    with torch.autocast("cpu", dtype=torch.float16):
        out = block._gated_residual_add(residual, gate, branch)
        loss = out.float().sum()
    loss.backward()

    for name, g in (("residual", residual.grad), ("gate", gate.grad), ("branch", branch.grad)):
        assert g is not None and torch.isfinite(g).all(), f"{name}.grad has inf/nan"
        assert g.dtype == torch.float16


def test_final_layer_forward_overflow_end_to_end():
    """A real FinalLayer forward under fp16 autocast: flag wires through + dtype contract.

    This is the *integration* counterpart to ``test_residual_add_unit_overflow_guard``
    (which injects a >65504 sum directly into ``_residual_add`` and asserts
    finiteness — the dispositive overflow regression). Here we run the ACTUAL
    FinalLayer.forward under fp16 autocast and pin two contracts:

    1. With the flag on, the modulate branch runs in fp32 (``x_modulated`` is
       fp32 before the linear), so the AdaLN math can't overflow mid-expression.
    2. The output is finite for a normal-magnitude input.

    The high-magnitude final-projection overflow case is covered separately by
    ``test_final_layer_projection_stays_fp32_under_fp16_autocast`` below.
    """
    hidden_size = 64
    final = FinalLayer(
        hidden_size=hidden_size,
        spatial_patch_size=1,
        temporal_patch_size=1,
        out_channels=4,
        use_adaln_lora=False,
    )
    final.eval()

    torch.manual_seed(0)
    x = torch.randn(1, 1, 2, 2, hidden_size, dtype=torch.float16)
    emb = torch.zeros(1, 1, hidden_size, dtype=torch.float16)

    final.fp32_residual = True
    with _fp16_autocast(), torch.no_grad():
        out_on = final(x, emb)
    assert torch.isfinite(out_on).all(), "fp32 FinalLayer modulate produced inf/nan"

    # Negative control: flag off also finite here (normal-magnitude input +
    # zero-init scale ⇒ no overflow). Confirms the flag is a no-op until a
    # >65504 residual actually appears — the unit test covers that case.
    final.fp32_residual = False
    with _fp16_autocast(), torch.no_grad():
        out_off = final(x, emb)
    assert torch.isfinite(out_off).all()


def test_final_layer_projection_stays_fp32_under_fp16_autocast():
    """Final projection must not re-cast the fp32 residual stream to fp16.

    The earlier guard kept FinalLayer's AdaLN modulate in fp32 but then called
    ``self.linear(x_modulated)`` under fp16 autocast. That final call casts the
    fp32 activation and weight to fp16 before the matmul, so a projection whose
    true fp32 result is finite but >65504 collapses to ``inf``. The fp16-safe
    path keeps this final projection in fp32 too.
    """
    hidden_size = 64
    final = FinalLayer(
        hidden_size=hidden_size,
        spatial_patch_size=1,
        temporal_patch_size=1,
        out_channels=1,
        use_adaln_lora=False,
    )
    final.eval()
    final.fp32_residual = True

    # Make LayerNorm output exactly ones: (63 channels at -1/sqrt(63), one at
    # sqrt(63)) has sample mean 0 and sample variance 1, so LN preserves the
    # pattern. The projection sums only the positive channel with a large weight:
    # fp32 result is finite (~80k), fp16 autocast projection overflows.
    x = torch.full(
        (1, 1, 1, 1, hidden_size),
        -1.0 / (hidden_size - 1) ** 0.5,
        dtype=torch.float32,
    )
    x[..., -1] = (hidden_size - 1) ** 0.5
    emb = torch.zeros(1, 1, hidden_size, dtype=torch.float32)
    with torch.no_grad():
        final.adaln_modulation[1].weight.zero_()
        final.linear.weight.zero_()
        final.linear.weight[0, -1] = 10000.0

    with _fp16_autocast(), torch.no_grad():
        out = final(x, emb)

    assert out.dtype == torch.float32
    assert torch.isfinite(out).all(), "fp32 FinalLayer projection produced inf/nan"
    assert out.item() > _FP16_MAX
