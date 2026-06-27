"""Native-shape flatten invariants (replaces the retired static-pad path).

`compile_blocks()` is the single switch that turns on native-shape flattening:
the forward reshapes each bucket's patch grid to a fake-5D `(B, 1, seq_len, 1, D)`
so dynamo keys the block graph on token count alone. The reshape must be
*bit-exact* to the eager 5D path (the gap=0 control the old pad-leak probe
verified), and eager (uncompiled) forwards must skip it entirely.

The two token counts exercised correspond to the shipped CONSTANT_TOKEN_BUCKETS
families: 4032 (= 63·64) and 4200 (= 60·70).
"""

from __future__ import annotations

import pytest
import torch

from library.anima.models import Anima


def _tiny_anima(*, num_blocks: int = 2) -> Anima:
    """A small but real Anima DiT runnable on CPU."""
    model = Anima(
        max_img_h=256,
        max_img_w=256,
        max_frames=4,
        in_channels=16,
        out_channels=16,
        patch_spatial=2,
        patch_temporal=1,
        concat_padding_mask=False,
        model_channels=64,
        num_blocks=num_blocks,
        num_heads=4,
        mlp_ratio=2.0,
        crossattn_emb_channels=64,
        use_adaln_lora=True,
        adaln_lora_dim=16,
        use_llm_adapter=False,
        attn_mode="torch",
    )
    return model.eval()


def _inputs(latent_h: int, latent_w: int):
    """Inputs whose patchified token count is (h/2)*(w/2)."""
    torch.manual_seed(0)
    x = torch.randn(1, 16, 1, latent_h, latent_w)
    timesteps = torch.tensor([0.5])
    crossattn_emb = torch.randn(1, 8, 64)
    return x, timesteps, crossattn_emb


def test_compile_blocks_sets_native_flatten_and_budget():
    import torch._dynamo as _dynamo

    model = _tiny_anima()
    assert model._native_flatten is False  # off until compile_blocks

    _dynamo.config.cache_size_limit = 1  # force the max() to raise it
    model.compile_blocks(backend="eager")

    assert model._native_flatten is True
    # 2 token-count families → 2*2 + 8 = 12, and never lowered below current.
    assert _dynamo.config.cache_size_limit >= 12


def test_compile_blocks_does_not_lower_a_higher_budget():
    """The max() lets a multi-resolution caller (e.g. SPD) pre-raise the limit."""
    import torch._dynamo as _dynamo

    model = _tiny_anima()
    _dynamo.config.cache_size_limit = 64  # a caller asked for more headroom
    model.compile_blocks(backend="eager")
    assert _dynamo.config.cache_size_limit == 64


def _callable_identity(fn):
    """Stable identity for bound methods and instance-assigned callables."""
    return getattr(fn, "__func__", fn)


def test_compile_blocks_compiles_all_blocks_without_swap(capsys):
    model = _tiny_anima()
    before = [_callable_identity(block._forward) for block in model.blocks]

    model.blocks_to_swap = 0
    model.compile_blocks(backend="eager")

    after = [_callable_identity(block._forward) for block in model.blocks]
    assert all(new is not old for new, old in zip(after, before))
    assert "compiled 2 block._forward" in capsys.readouterr().out


def test_compile_blocks_with_block_swap_keeps_tail_eager(capsys):
    model = _tiny_anima(num_blocks=3)
    before = [_callable_identity(block._forward) for block in model.blocks]

    model.blocks_to_swap = 1
    model.compile_blocks(backend="eager")

    after = [_callable_identity(block._forward) for block in model.blocks]
    assert after[0] is not before[0]
    assert after[1] is not before[1]
    assert after[2] is before[2]
    assert "compiled 2 resident compiled / 1 swapped (eager)" in capsys.readouterr().out


def test_compile_blocks_rejects_invalid_blocks_to_swap():
    model = _tiny_anima()
    model.blocks_to_swap = model.num_blocks

    with pytest.raises(ValueError, match="Invalid blocks_to_swap=2"):
        model.compile_blocks(backend="eager")


@torch.no_grad()
def _run(model: Anima, inp, *, native_flatten: bool) -> torch.Tensor:
    model._native_flatten = native_flatten
    x, timesteps, crossattn_emb = inp
    return model.forward_mini_train_dit(x, timesteps, crossattn_emb)


@torch.no_grad()
def test_flatten_is_bit_exact_4032_family():
    # latent 126x128 → (63)*(64) = 4032 tokens at patch_spatial=2
    model = _tiny_anima()
    inp = _inputs(126, 128)
    out_eager = _run(model, inp, native_flatten=False)
    out_flat = _run(model, inp, native_flatten=True)
    assert torch.equal(out_eager, out_flat)
    assert out_eager.shape == out_flat.shape


@torch.no_grad()
def test_flatten_is_bit_exact_4200_family():
    # latent 120x140 → (60)*(70) = 4200 tokens at patch_spatial=2
    model = _tiny_anima()
    inp = _inputs(120, 140)
    out_eager = _run(model, inp, native_flatten=False)
    out_flat = _run(model, inp, native_flatten=True)
    assert torch.equal(out_eager, out_flat)
