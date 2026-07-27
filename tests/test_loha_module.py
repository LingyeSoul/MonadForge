"""Official LyCORIS LoHa backend and compatibility-boundary regressions.

Mirrors the LoKr wrapper suite (``test_lokr_channel_scale.py``) — same
wrapper-vs-official reference style, no Linux-only goldens. The scale
assertions are load-bearing: LyCORIS 3.4.0's LoHa bypass already includes
``self.scale`` (unlike LoKr's, which omits it), and ``get_diff_weight``
double-applies it (like LoKr's) — see ``networks/lora_modules/loha.py``.
"""

from __future__ import annotations

import copy

import pytest
import torch
import networks.lora_modules.loha as loha_impl
from lycoris.modules.loha import LohaModule as LycorisLohaModule
from safetensors import safe_open

from networks import NETWORK_KWARGS, resolve_network_spec
from networks.lora_anima.factory import create_network, create_network_from_weights
from networks.lora_modules.loha import LoHaModule


class Block(torch.nn.Module):
    def __init__(self, in_dim: int = 512, out_dim: int = 512) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(in_dim, out_dim, bias=False)


class _TinyDiT(torch.nn.Module):
    def __init__(self, in_dim: int = 512, out_dim: int = 512) -> None:
        super().__init__()
        self.block = Block(in_dim, out_dim)

    def reset_mod_guidance(self) -> None:
        pass


def _make_module(
    *,
    in_dim: int = 512,
    out_dim: int = 512,
    rank: int = 4,
    alpha: float = 16,
    multiplier: float = 1.0,
    rank_dropout: float = 0.0,
) -> tuple[torch.nn.Linear, LoHaModule]:
    base = torch.nn.Linear(in_dim, out_dim, bias=False)
    base.weight.requires_grad_(False)
    module = LoHaModule(
        "lora_test",
        base,
        multiplier=multiplier,
        lora_dim=rank,
        alpha=alpha,
        rank_dropout=rank_dropout,
    )
    return base, module


def _make_delta_nonzero(module: LoHaModule) -> None:
    # hada_w2_a is the official zero-init chain end.
    with torch.no_grad():
        module.hada_w2_a.normal_(0, 0.1)


def _reference_delta(module: LoHaModule) -> torch.Tensor:
    return (
        (module.hada_w1_a @ module.hada_w1_b) * (module.hada_w2_a @ module.hada_w2_b)
    ) * module.scale


def test_backend_is_official_lycoris_module():
    assert issubclass(LoHaModule, LycorisLohaModule)


def test_kwarg_is_registered_and_resolver_routes_it():
    assert {"use_loha"} <= NETWORK_KWARGS
    assert resolve_network_spec({"use_loha": "true"}).name == "loha"
    assert resolve_network_spec({"use_loha": "false"}).name == "lora"
    # Documented precedence: lokr sits above loha in the resolver tail.
    assert resolve_network_spec({"use_lokr": "true", "use_loha": "true"}).name == "lokr"


def test_layout_and_state_keys_match_lycoris():
    _, module = _make_module(rank=4, alpha=16)

    assert module.scale == 4.0
    assert module.hada_w1_a.shape == (512, 4)
    assert module.hada_w1_b.shape == (4, 512)
    assert module.hada_w2_a.shape == (512, 4)
    assert module.hada_w2_b.shape == (4, 512)
    assert set(module.state_dict()) == {
        "alpha",
        "hada_w1_a",
        "hada_w1_b",
        "hada_w2_a",
        "hada_w2_b",
    }


def test_channel_scaling_is_rejected_at_direct_module_boundary():
    base = torch.nn.Linear(12, 20, bias=False)
    with pytest.raises(ValueError, match="channel scaling"):
        LoHaModule(
            "lora_test",
            base,
            lora_dim=4,
            alpha=4,
            channel_scale=torch.ones(12),
        )


def test_forward_is_identity_at_init():
    base, module = _make_module()
    module.apply_to()
    x = torch.randn(2, 3, 512)
    assert torch.allclose(base(x), module.org_forward(x), atol=1e-7, rtol=1e-7)


def test_forward_applies_scale_exactly_once():
    # Guards the 3.4.0 asymmetry: LoHa's official bypass already bakes in
    # ``self.scale`` — the wrapper must NOT re-add it (the LoKr fix would
    # double-scale here).
    base, module = _make_module(rank=4, alpha=16, multiplier=0.75)
    _make_delta_nonzero(module)
    module.apply_to()
    x = torch.randn(2, 3, 512)

    expected = module.org_forward(x) + torch.nn.functional.linear(
        x, _reference_delta(module) * 0.75
    )
    assert torch.allclose(base(x), expected, atol=1e-5, rtol=1e-5)


def test_wrapper_forward_and_gradients_equal_direct_lycoris():
    torch.manual_seed(123)
    wrapped_base = torch.nn.Linear(512, 512, bias=False)
    wrapped_base.weight.requires_grad_(False)
    official_base = copy.deepcopy(wrapped_base)

    torch.manual_seed(456)
    wrapped = LoHaModule(
        "wrapped",
        wrapped_base,
        multiplier=0.75,
        lora_dim=4,
        alpha=16,
    )
    official = LycorisLohaModule(
        "official",
        official_base,
        multiplier=0.75,
        lora_dim=4,
        alpha=16,
        # The official regular (rebuild) path is the numerical reference.
        bypass_mode=False,
    )
    _make_delta_nonzero(wrapped)
    official.load_state_dict(wrapped.state_dict())

    wrapped.apply_to()
    official.apply_to()
    x = torch.randn(2, 3, 512)
    wrapped_out = wrapped_base(x)
    official_out = official_base(x)

    assert torch.allclose(wrapped_out, official_out, atol=1e-6, rtol=1e-6)

    wrapped_out.square().mean().backward()
    official_out.square().mean().backward()
    wrapped_grads = dict(wrapped.named_parameters())
    official_grads = dict(official.named_parameters())
    assert wrapped_grads.keys() == official_grads.keys()
    for name, parameter in wrapped_grads.items():
        assert parameter.grad is not None
        assert torch.allclose(
            parameter.grad,
            official_grads[name].grad,
            atol=1e-6,
            rtol=1e-5,
        )


def test_zero_init_trains_the_official_w2_chain_end_first():
    base, module = _make_module(rank=4, alpha=4)
    module.apply_to()
    loss = base(torch.randn(2, 3, 512)).square().mean()
    loss.backward()

    # ΔW = (w1a@w1b) ⊙ (w2a@w2b): with hada_w2_a = 0 the second product is
    # zero, so only hada_w2_a receives gradient at step 0.
    assert module.hada_w2_a.grad is not None
    assert torch.count_nonzero(module.hada_w2_a.grad) > 0
    assert module.hada_w2_b.grad is not None
    assert torch.count_nonzero(module.hada_w2_b.grad) == 0
    assert torch.count_nonzero(module.hada_w1_a.grad) == 0


def test_get_diff_weight_scales_once():
    # LyCORIS 3.4.0's own get_diff_weight multiplies self.scale on top of a
    # get_weight that already applied it; the override must scale once.
    _, module = _make_module(rank=4, alpha=16, multiplier=0.75)
    _make_delta_nonzero(module)

    diff, extra = module.get_diff_weight(multiplier=1.0)
    assert extra is None
    assert torch.allclose(diff, _reference_delta(module), atol=1e-6, rtol=1e-5)


def test_fp32_compute_casts_all_official_operands(monkeypatch):
    base, module = _make_module(in_dim=64, out_dim=64, rank=4, alpha=16)
    base.half()
    module.half()
    module.fp32_compute = True
    module.train()
    _make_delta_nonzero(module)

    seen_dtypes: list[torch.dtype] = []
    original = loha_impl.lycoris_loha_diff_weight

    def spy(*weights, **kwargs):
        seen_dtypes.extend(w.dtype for w in weights if w is not None)
        gamma = kwargs.get("gamma")
        if gamma is not None:
            seen_dtypes.append(gamma.dtype)
        return original(*weights, **kwargs)

    monkeypatch.setattr(loha_impl, "lycoris_loha_diff_weight", spy)
    module.apply_to()
    x = torch.randn(2, 3, 64, dtype=torch.float16)
    # Reference in fp32 (matching the fp32 path) — module.get_weight() would
    # rebuild ΔW from the half params in fp16 and lose the precision the
    # fp32_compute branch exists to keep.
    expected_delta = (
        (module.hada_w1_a.float() @ module.hada_w1_b.float())
        * (module.hada_w2_a.float() @ module.hada_w2_b.float())
    ) * module.scale
    expected = module.org_forward(x) + torch.nn.functional.linear(
        x.float(), expected_delta
    ).to(torch.float16)
    output = base(x)

    assert output.dtype == torch.float16
    assert torch.allclose(output, expected, atol=1e-3, rtol=1e-3)
    assert seen_dtypes
    assert set(seen_dtypes) == {torch.float32}


def test_merge_uses_official_functional_delta():
    base, module = _make_module(rank=4, alpha=16, multiplier=0.75)
    _make_delta_nonzero(module)
    original = base.weight.detach().clone()
    expected = module.get_weight().clone()

    module.merge_to(module.state_dict(), dtype=torch.float32, device="cpu")

    assert torch.allclose(base.weight, original + expected, atol=1e-6, rtol=1e-5)


def test_fuse_unfuse_preserves_forward_and_restores_weight():
    base, module = _make_module(rank=4, alpha=16, multiplier=0.75)
    _make_delta_nonzero(module)
    module.apply_to()
    x = torch.randn(2, 3, 512)
    original = base.weight.detach().clone()
    expected = base(x)

    module.fuse_weight()
    assert torch.allclose(base(x), expected, atol=1e-5, rtol=1e-5)

    module.unfuse_weight()
    assert torch.allclose(base.weight, original, atol=1e-6, rtol=1e-6)
    assert torch.allclose(base(x), expected, atol=1e-5, rtol=1e-5)


def test_fuse_unfuse_is_reversible_with_rank_dropout_in_training_mode():
    base, module = _make_module(rank=4, alpha=16, multiplier=0.75, rank_dropout=0.5)
    _make_delta_nonzero(module)
    module.apply_to()
    module.train()
    original = base.weight.detach().clone()

    module.fuse_weight()
    module.unfuse_weight()

    assert module.training is True
    assert torch.allclose(base.weight, original, atol=1e-6, rtol=1e-6)


def test_rank_dropout_still_fires_in_training_forward():
    # The wrapper's get_weight override suppresses rank_dropout for merge/fuse
    # determinism; the training forward must keep official row-dropout (it
    # lives inside the official get_weight the bypass calls non-virtually).
    torch.manual_seed(0)
    base, module = _make_module(rank=4, alpha=16, rank_dropout=0.5)
    _make_delta_nonzero(module)
    module.apply_to()
    module.train()
    x = torch.randn(4, 8, 512)

    outputs = {base(x).sum().item() for _ in range(8)}
    assert len(outputs) > 1  # dropout mask varies per forward

    module.eval()
    eval_a = base(x)
    eval_b = base(x)
    assert torch.equal(eval_a, eval_b)


def test_native_checkpoint_keeps_keys_and_infers_rank():
    lora_name = "lora_unet_block_proj"
    native_sd = {
        f"{lora_name}.hada_w1_a": torch.randn(512, 4),
        f"{lora_name}.hada_w1_b": torch.randn(4, 512),
        f"{lora_name}.hada_w2_a": torch.randn(512, 4),
        f"{lora_name}.hada_w2_b": torch.randn(4, 512),
        f"{lora_name}.alpha": torch.tensor(16),
    }
    network, normalized_sd = create_network_from_weights(
        multiplier=1.0,
        file=None,
        weights_sd=native_sd,
        metadata={"ss_network_spec": "loha"},
        ae=None,
        text_encoders=[],
        unet=_TinyDiT(),
    )

    assert normalized_sd.keys() == native_sd.keys()
    assert network._network_spec.name == "loha"
    assert network.cfg.use_loha is True  # threads into module rebuild on resume
    module = network.unet_loras[0]
    assert isinstance(module, LoHaModule)
    assert module.lora_dim == 4
    assert module.scale == 4.0

    network.apply_to(text_encoders=[], unet=None, apply_text_encoder=False)
    info = network.load_state_dict(normalized_sd, strict=False)
    assert not any("hada_" in key for key in info.missing_keys)
    assert not any("hada_" in key for key in info.unexpected_keys)
    assert torch.equal(module.hada_w2_b.detach(), native_sd[f"{lora_name}.hada_w2_b"])


def test_loha_wins_over_for_inference_plain_lora_fallback():
    lora_name = "lora_unet_block_proj"
    native_sd = {
        f"{lora_name}.hada_w1_a": torch.randn(512, 4),
        f"{lora_name}.hada_w1_b": torch.randn(4, 512),
        f"{lora_name}.hada_w2_a": torch.randn(512, 4),
        f"{lora_name}.hada_w2_b": torch.randn(4, 512),
        f"{lora_name}.alpha": torch.tensor(16),
    }
    network, _ = create_network_from_weights(
        multiplier=1.0,
        file=None,
        weights_sd=native_sd,
        metadata={"ss_network_spec": "loha"},
        ae=None,
        text_encoders=[],
        unet=_TinyDiT(),
        for_inference=True,
    )
    # A plain LoRAModule cannot consume hada_* keys — the loha spec must win.
    assert network._network_spec.name == "loha"
    assert isinstance(network.unet_loras[0], LoHaModule)


def test_tucker_checkpoint_is_rejected():
    lora_name = "lora_unet_block_proj"
    native_sd = {
        f"{lora_name}.hada_w1_a": torch.randn(4, 512),
        f"{lora_name}.hada_w1_b": torch.randn(4, 512),
        f"{lora_name}.hada_t1": torch.randn(4, 4, 3, 3),
        f"{lora_name}.alpha": torch.tensor(16),
    }
    with pytest.raises(RuntimeError, match="Tucker"):
        create_network_from_weights(
            multiplier=1.0,
            file=None,
            weights_sd=native_sd,
            metadata={"ss_network_spec": "loha"},
            ae=None,
            text_encoders=[],
            unet=_TinyDiT(),
        )


def test_dora_checkpoint_is_rejected():
    # LyCORIS weight_decompose (DoRA) renormalizes the merged weight; the
    # wrapper has no dora_scale slot, and silently dropping the key would
    # produce materially wrong deltas — the loader must fail loudly.
    lora_name = "lora_unet_block_proj"
    native_sd = {
        f"{lora_name}.hada_w1_a": torch.randn(512, 4),
        f"{lora_name}.hada_w1_b": torch.randn(4, 512),
        f"{lora_name}.hada_w2_a": torch.randn(512, 4),
        f"{lora_name}.hada_w2_b": torch.randn(4, 512),
        f"{lora_name}.dora_scale": torch.randn(512, 1),
        f"{lora_name}.alpha": torch.tensor(16),
    }
    with pytest.raises(RuntimeError, match="DoRA"):
        create_network_from_weights(
            multiplier=1.0,
            file=None,
            weights_sd=native_sd,
            metadata={"ss_network_spec": "loha"},
            ae=None,
            text_encoders=[],
            unet=_TinyDiT(),
        )


def test_max_norm_regularization_is_a_clean_noop():
    # hada_* networks have no lora_down/lora_up keys; --scale_weight_norms
    # must degrade to a no-op instead of dividing by zero at step 1.
    network = create_network(
        multiplier=1.0,
        network_dim=4,
        network_alpha=4,
        vae=None,
        text_encoders=[],
        unet=_TinyDiT(),
        use_loha="true",
    )
    network.apply_to(text_encoders=[], unet=None, apply_text_encoder=False)
    assert network.apply_max_norm_regularization(1.0, "cpu") == (0, 0.0, 0.0)


def test_use_loha_triggers_split_fused_projections(monkeypatch):
    import networks.lora_modules.split_attn as split_attn

    calls: list[object] = []
    monkeypatch.setattr(
        split_attn, "split_fused_projections", lambda model: calls.append(model)
    )
    unet = _TinyDiT()
    create_network(
        multiplier=1.0,
        network_dim=4,
        network_alpha=4,
        vae=None,
        text_encoders=[],
        unet=unet,
        use_loha="true",
    )
    assert calls == [unet]


def test_fused_qkv_dit_gets_split_hada_modules_and_merge_writes_through():
    # Integration pin for the shared split_fused_projections path (the LoHa
    # analogue of test_lokr_split_attn): a fused-qkv DiT must yield per-
    # component LoHa modules with ComfyUI-compatible split names, and a
    # component merge must write through its narrow view into the fused
    # storage without contaminating sibling bands.
    class _SelfAttn(torch.nn.Module):
        def __init__(self, dim: int = 32, n_heads: int = 4, head_dim: int = 8):
            super().__init__()
            self.is_selfattn = True
            self.n_heads = n_heads
            self.head_dim = head_dim
            self.qkv_format = "bshd"
            inner = n_heads * head_dim
            self.inner_dim = inner
            self.qkv_proj = torch.nn.Linear(dim, 3 * inner, bias=False)
            self.output_proj = torch.nn.Linear(inner, dim, bias=False)
            self.q_norm = torch.nn.Identity()
            self.k_norm = torch.nn.Identity()
            self.v_norm = torch.nn.Identity()

        def compute_qkv(self, x, context, rope_cos_sin=None):
            qkv = self.qkv_proj(x).unflatten(-1, (3, self.n_heads, self.head_dim))
            q, k, v = qkv.unbind(dim=-3)
            return self.q_norm(q), self.k_norm(k), self.v_norm(v)

    class Block(torch.nn.Module):  # noqa: F811 — walker matches on class name
        def __init__(self):
            super().__init__()
            self.self_attn = _SelfAttn()

    class _AttnDiT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.block = Block()

    dit = _AttnDiT()
    fused_weight = dit.block.self_attn.qkv_proj.weight  # aliases split views
    inner = dit.block.self_attn.inner_dim

    network = create_network(
        multiplier=1.0,
        network_dim=4,
        network_alpha=4,
        vae=None,
        text_encoders=[],
        unet=dit,
        use_loha="true",
    )
    names = [lora.lora_name for lora in network.unet_loras]
    for component in ("q_proj", "k_proj", "v_proj"):
        assert any(n.endswith(f"_self_attn_{component}") for n in names)
    assert not any("qkv_proj" in n for n in names)
    assert all(isinstance(lora, LoHaModule) for lora in network.unet_loras)

    q_mod = next(
        lora for lora in network.unet_loras if lora.lora_name.endswith("_q_proj")
    )
    assert q_mod.org_module_ref[0].weight.data_ptr() == fused_weight.data_ptr()
    _make_delta_nonzero(q_mod)
    before = fused_weight.detach().clone()
    delta = q_mod.get_weight().clone()

    q_mod.merge_to(q_mod.state_dict(), dtype=torch.float32, device="cpu")

    after = fused_weight.detach()
    assert torch.allclose(after[:inner], before[:inner] + delta, atol=1e-6, rtol=1e-5)
    assert torch.equal(after[inner:], before[inner:])


def test_checkpoint_save_and_reload_preserves_official_delta(tmp_path):
    network = create_network(
        multiplier=1.0,
        network_dim=4,
        network_alpha=16,
        vae=None,
        text_encoders=[],
        unet=_TinyDiT(),
        use_loha="true",
    )
    network.apply_to(text_encoders=[], unet=None, apply_text_encoder=False)
    source = network.unet_loras[0]
    assert isinstance(source, LoHaModule)
    _make_delta_nonzero(source)
    expected = source.get_weight().clone()

    out = tmp_path / "loha.safetensors"
    network.save_weights(str(out), torch.float32, metadata={})

    with safe_open(str(out), framework="pt") as handle:
        keys = set(handle.keys())
        metadata = handle.metadata() or {}
        saved_alpha = handle.get_tensor(f"{source.lora_name}.alpha").item()
    assert metadata["ss_network_spec"] == "loha"
    assert metadata["ss_network_dim"] == "4"
    assert saved_alpha == 16
    assert any(key.endswith(".hada_w1_a") for key in keys)
    assert any(key.endswith(".hada_w2_b") for key in keys)
    assert not any(key.endswith(".inv_scale") for key in keys)

    restored, weights_sd = create_network_from_weights(
        multiplier=1.0,
        file=str(out),
        ae=None,
        text_encoders=[],
        unet=_TinyDiT(),
    )
    restored.apply_to(text_encoders=[], unet=None, apply_text_encoder=False)
    info = restored.load_state_dict(weights_sd, strict=False)
    assert not info.missing_keys
    assert not info.unexpected_keys
    actual = restored.unet_loras[0].get_weight()
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
