"""MOSS-Image LoRA + trainer pieces on a tiny random DiT (CPU)."""

import math
import re
from collections import Counter

import pytest
import torch

from library.moss.lora import (
    DEFAULT_TARGETS,
    LoKrAdapter,
    LoRANetwork,
    channel_scale,
    load_for_inference,
    load_network,
    ref_settings,
)
from library.moss.model import MOSSImageTransformer
from library.moss.train import checkpoint_blocks, sample_sigma, training_loss


def _tiny(layers=3):
    torch.manual_seed(0)
    return MOSSImageTransformer(
        dim=96,
        ffn_dim=128,
        num_heads=4,
        num_layers=layers,
        text_dim=32,
        latent_channels=8,
    ).eval()


def _inputs():
    g = torch.Generator().manual_seed(1)
    return torch.randn(1, 8, 4, 6, generator=g), torch.randn(1, 5, 32, generator=g)


def _out(model, x, e):
    with torch.no_grad():
        return model(x, torch.tensor([600.0]), e).sample


def test_targets_six_per_block():
    model = _tiny(layers=4)
    net = LoRANetwork(model, rank=4)
    per_block = Counter(
        int(re.match(r"layers\.(\d+)\.", n)[1]) for n in net.target_names
    )
    assert per_block == {i: 6 for i in range(4)}
    assert all(re.fullmatch(DEFAULT_TARGETS, n) for n in net.target_names)


def test_zero_init_and_multiplier_zero_are_base():
    model = _tiny()
    x, e = _inputs()
    base = _out(model, x, e)
    net = LoRANetwork(model, rank=4)
    net.apply_to()
    assert torch.equal(_out(model, x, e), base)
    for adapter in net.adapters.values():
        torch.nn.init.normal_(adapter.up.weight)
    assert not torch.allclose(_out(model, x, e), base)
    net.set_multiplier(0.0)
    assert torch.equal(_out(model, x, e), base)
    net.restore()
    assert torch.equal(_out(model, x, e), base)


def test_save_load_round_trip(tmp_path):
    model = _tiny()
    x, e = _inputs()
    net = LoRANetwork(model, rank=4, alpha=2.0)
    for adapter in net.adapters.values():
        torch.nn.init.normal_(adapter.up.weight, std=0.1)
    net.apply_to()
    want = _out(model, x, e)
    path = tmp_path / "l.safetensors"
    net.save(path, "fp32", base_variant="base")
    net.restore()

    fresh = _tiny()
    loaded = load_for_inference(fresh, path, multiplier=1.0)
    assert torch.allclose(_out(fresh, x, e), want, atol=1e-6)
    _, meta = load_network(_tiny(), path)
    assert (
        meta["rank"] == "4" and meta["alpha"] == "2.0" and meta["ss_num_blocks"] == "3"
    )
    loaded.set_multiplier(0.0)
    assert torch.equal(_out(fresh, x, e), _out(_tiny(), x, e))
    with pytest.raises(ValueError, match="block"):
        load_network(_tiny(layers=4), path)


def test_saved_keys_match_comfy_lora_map(tmp_path):
    # ComfyUI's model_lora_keys_unet maps lora_unet_<weight key, dots → _>;
    # the loader then reads .lora_down/.lora_up/.alpha under it.
    from safetensors import safe_open
    from safetensors.torch import save_file

    model = _tiny()
    x, e = _inputs()
    net = LoRANetwork(model, rank=4, alpha=2.0)
    for adapter in net.adapters.values():
        torch.nn.init.normal_(adapter.up.weight, std=0.1)
    path = tmp_path / "l.safetensors"
    net.save(path, "fp32")
    comfy = {
        "lora_unet_" + k.removesuffix(".weight").replace(".", "_")
        for k in model.state_dict()
        if k.endswith(".weight")
    }
    with safe_open(str(path), framework="pt") as handle:
        keys = set(handle.keys())
        meta = handle.metadata()
        stems = {k.split(".", 1)[0] for k in keys}
        assert all(handle.get_tensor(s + ".alpha") == 2.0 for s in stems)
    assert len(stems) == len(net.target_names) and stems <= comfy
    assert {k.split(".", 1)[1] for k in keys} == {
        "lora_down.weight",
        "lora_up.weight",
        "alpha",
    }

    # the earlier adapters.<name>.{down,up}.weight layout still loads
    legacy = tmp_path / "legacy.safetensors"
    save_file(
        {k: v.contiguous() for k, v in net.state_dict().items()}, str(legacy), meta
    )
    net.apply_to()
    want = _out(model, x, e)
    net.restore()
    fresh = _tiny()
    load_for_inference(fresh, legacy)
    assert torch.allclose(_out(fresh, x, e), want, atol=1e-6)


@pytest.mark.training_step
@pytest.mark.parametrize("algo", ["lora", "lokr", "loha"])
def test_one_step_decreases_loss(algo):
    model = _tiny()
    model.requires_grad_(False)
    net = LoRANetwork(model, rank=4, algo=algo)
    net.apply_to()
    checkpoint_blocks(model.layers)
    model.train()
    x, e = _inputs()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-2)

    def loss():
        return training_loss(model, x, e, 0.5, gen=torch.Generator().manual_seed(3))

    before = loss()
    before.backward()
    assert all(p.grad is not None for p in net.parameters())
    opt.step()
    with torch.no_grad():
        after = loss()
    assert after.item() < before.item()


def test_sigma_grid_and_bias():
    g = torch.Generator().manual_seed(0)
    draws = torch.tensor([sample_sigma(g, -1.09, 0.95) for _ in range(4000)])
    assert torch.all((draws >= 1e-3) & (draws <= 1.0))
    assert torch.allclose(draws * 1000, (draws * 1000).round(), atol=1e-6)
    # sigma = 1 - sigmoid(Z), mu = -1.09: noise-heavy, median ~0.75 before the
    # /0.95 rescale (~0.79 after).
    assert 0.72 < draws.median().item() < 0.85


def test_weight_svd_down_init():
    model = _tiny()
    x, e = _inputs()
    base = _out(model, x, e)
    torch.manual_seed(5)
    kaiming = LoRANetwork(model, rank=4)
    after_kaiming = torch.rand(())
    torch.manual_seed(5)
    net = LoRANetwork(model, rank=4, down_init="weight_svd")
    assert torch.equal(torch.rand(()), after_kaiming)  # same RNG stream either way
    assert net.metadata()["down_init"] == "weight_svd"
    for name, module in net._targets:
        down = net.adapters[name.replace(".", "_")].down.weight
        gram = down @ down.T * 3.0
        assert torch.allclose(gram, torch.eye(4), atol=1e-4)
        exact = torch.linalg.svd(module.weight.float(), full_matrices=False).Vh[:4]
        # Same subspace as the exact top-4 right singular vectors.
        proj = exact @ (down.T * math.sqrt(3.0))
        assert torch.allclose(proj @ proj.T, torch.eye(4), atol=1e-3)
        assert not torch.equal(
            down, kaiming.adapters[name.replace(".", "_")].down.weight
        )
    net.apply_to()
    assert torch.equal(_out(model, x, e), base)
    with pytest.raises(ValueError):
        LoRANetwork(model, rank=4, down_init="pissa")


def test_svd_slice_windows_are_orthogonal():
    model = _tiny()
    rows = {}
    for slice_, count in ((0, 0), (1, 0), (0, 3), (2, 3)):
        net = LoRANetwork(
            model,
            rank=4,
            down_init="weight_svd",
            svd_slice=slice_,
            svd_slice_count=count,
        )
        rows[slice_, count] = {
            name: net.adapters[name.replace(".", "_")].down.weight * math.sqrt(3.0)
            for name, _ in net._targets
        }
        meta = net.metadata()
        assert meta.get("ss_svd_slice") == (str(slice_) if slice_ else None)
        assert meta.get("ss_svd_slice_count") == (str(count) if count else None)
    for name, module in net._targets:
        vh = torch.linalg.svd(module.weight.float(), full_matrices=False).Vh
        for (slice_, count), by_name in rows.items():
            idx = (
                slice_ + count * torch.arange(4)
                if count
                else slice_ * 4 + torch.arange(4)
            )
            # Same subspace as the exact window (singular vectors up to sign).
            proj = vh[idx] @ by_name[name].T
            assert torch.allclose(proj.abs(), torch.eye(4), atol=1e-3)
        # Disjoint windows of one basis: no overlap.
        cross = rows[0, 0][name] @ rows[1, 0][name].T
        assert cross.abs().max() < 1e-3
        cross = rows[0, 3][name] @ rows[2, 3][name].T
        assert cross.abs().max() < 1e-3
    with pytest.raises(ValueError, match="window ends"):
        LoRANetwork(model, rank=4, down_init="weight_svd", svd_slice=24)
    with pytest.raises(ValueError, match="weight_svd"):
        LoRANetwork(model, rank=4, svd_slice=1)
    with pytest.raises(ValueError, match="weight_svd"):
        LoRANetwork(model, rank=4, down_init="weight_svd", algo="lokr", svd_slice=1)
    with pytest.raises(ValueError, match="must be <"):
        LoRANetwork(
            model, rank=4, down_init="weight_svd", svd_slice=3, svd_slice_count=3
        )


def _skewed_stats(model):
    """One loud input channel per target, like MOSS's ch 42 into ffn_up_gate."""
    g = torch.Generator().manual_seed(3)
    stats = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and re.fullmatch(DEFAULT_TARGETS, name):
            m = torch.rand(module.in_features, generator=g) + 0.1
            m[0] = 30.0
            stats[name] = m
    return stats


def test_channel_scaling_is_output_neutral_and_saves_plain(tmp_path):
    model = _tiny()
    x, e = _inputs()
    stats = _skewed_stats(model)
    torch.manual_seed(5)
    plain = LoRANetwork(model, rank=4, alpha=2.0, down_init="weight_svd")
    torch.manual_seed(5)
    net = LoRANetwork(
        model,
        rank=4,
        alpha=2.0,
        down_init="weight_svd",
        channel_stats=stats,
        channel_scaling_alpha=0.5,
    )
    for name, adapter in net.adapters.items():
        s = channel_scale(stats[adapter_name(net, name)], 0.5)
        assert torch.allclose(s.mean(), torch.tensor(1.0))
        assert torch.allclose(adapter.down.weight, plain.adapters[name].down.weight * s)
        assert torch.allclose(adapter.down_weight(), plain.adapters[name].down.weight)
        torch.nn.init.normal_(adapter.up.weight, std=0.1)
        plain.adapters[name].up.weight.data.copy_(adapter.up.weight)

    def loss_and_grads(network):
        network.apply_to()
        loss = model(x, torch.tensor([600.0]), e).sample.square().mean()
        loss.backward()
        network.restore()
        return loss.detach(), {
            k: a.down.weight.grad for k, a in network.adapters.items()
        }

    want = _out_with(model, plain, x, e)
    assert torch.allclose(_out_with(model, net, x, e), want, atol=1e-5)
    # Adam geometry only: down's gradient on input channel c is scaled by 1/s[c].
    loss_plain, g_plain = loss_and_grads(plain)
    loss_net, g_net = loss_and_grads(net)
    assert torch.allclose(loss_net, loss_plain, rtol=1e-5)
    for name, adapter in net.adapters.items():
        assert torch.allclose(g_net[name], g_plain[name] * adapter.inv_scale, atol=1e-7)

    path = tmp_path / "cs.safetensors"
    net.save(path, "fp32")
    fresh = _tiny()
    loaded = load_for_inference(fresh, path)
    assert all(a.inv_scale is None for a in loaded.adapters.values())  # plain LoRA
    assert torch.allclose(_out(fresh, x, e), want, atol=1e-5)
    assert load_network(_tiny(), path)[1]["channel_scaling_alpha"] == "0.5"

    with pytest.raises(ValueError, match="lora'"):
        LoRANetwork(
            model, rank=4, algo="lokr", channel_stats=stats, channel_scaling_alpha=0.5
        )
    with pytest.raises(ValueError, match="needs channel_stats"):
        LoRANetwork(model, rank=4, channel_scaling_alpha=0.5)


def _out_with(model, network, x, e):
    network.apply_to()
    try:
        return _out(model, x, e)
    finally:
        network.restore()


def adapter_name(net, key):
    return next(n for n, _ in net._targets if n.replace(".", "_") == key)


def _dense_delta(adapter):
    """ΔW written out densely, independent of the bypass / merge paths."""
    if isinstance(adapter, LoKrAdapter):
        w2 = (
            adapter.lokr_w2
            if hasattr(adapter, "lokr_w2")
            else adapter.lokr_w2_a @ adapter.lokr_w2_b
        )
        return torch.kron(adapter.lokr_w1, w2) * adapter.scale
    hada = (adapter.hada_w1_a @ adapter.hada_w1_b) * (
        adapter.hada_w2_a @ adapter.hada_w2_b
    )
    return hada * adapter.scale


@pytest.mark.parametrize("algo", ["lokr", "loha"])
def test_lycoris_matches_dense_and_round_trips(algo, tmp_path):
    from safetensors import safe_open

    model = _tiny()
    x, e = _inputs()
    base = _out(model, x, e)
    net = LoRANetwork(model, rank=4, alpha=2.0, algo=algo)
    net.apply_to()
    assert torch.equal(_out(model, x, e), base)
    for adapter in net.adapters.values():
        for param in adapter.parameters():
            torch.nn.init.normal_(param, std=0.1)
    got = _out(model, x, e)
    assert not torch.allclose(got, base)

    merged = _tiny()
    with torch.no_grad():
        for name, _module in net._targets:
            delta = _dense_delta(net.adapters[name.replace(".", "_")])
            merged.get_submodule(name).weight += delta
    assert torch.allclose(got, _out(merged, x, e), atol=1e-5)
    net.set_multiplier(0.0)
    assert torch.equal(_out(model, x, e), base)
    net.set_multiplier(1.0)

    path = tmp_path / f"{algo}.safetensors"
    net.save(path, "fp32")
    net.restore()
    prefix = {"lokr": "lokr_w", "loha": "hada_w"}[algo]
    with safe_open(str(path), framework="pt") as handle:
        keys = list(handle.keys())
        assert handle.get_tensor(keys[0].rsplit(".", 1)[0] + ".alpha").item() == 2.0
    assert all(
        k.endswith(".alpha") or k.split(".")[-1].startswith(prefix) for k in keys
    )
    fresh = _tiny()
    load_for_inference(fresh, path)
    assert torch.allclose(_out(fresh, x, e), got, atol=1e-6)
    assert load_network(_tiny(), path)[1]["algo"] == algo


def test_lokr_full_w2_scale_is_one():
    # rank >= half the larger factor's side: w2 is a full matrix and LyCORIS
    # drops alpha (scale 1), whatever alpha says.
    net = LoRANetwork(_tiny(), rank=16, alpha=2.0, algo="lokr")
    for adapter in net.adapters.values():
        assert hasattr(adapter, "lokr_w2") and adapter.scale == 1.0
        assert adapter.alpha.item() == 16.0
    with pytest.raises(ValueError, match="algo"):
        LoRANetwork(_tiny(), algo="glora")


def _dc_means(model):
    """A signed token mean per target with one dominant channel, like MOSS's x̄."""
    g = torch.Generator().manual_seed(4)
    means = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and re.fullmatch(DEFAULT_TARGETS, name):
            m = torch.randn(module.in_features, generator=g) * 0.1
            m[1] = -5.0
            means[name] = m
    return means


def test_dc_isolate_localises_the_dc_direction_and_saves_plain(tmp_path):
    model = _tiny()
    x, e = _inputs()
    means = _dc_means(model)
    stats = _skewed_stats(model)
    base_out = _out(model, x, e)

    def rows_vs_u(adapter, mean):
        u = mean / mean.norm()
        w = adapter.down_weight()
        return (w @ u) / w.norm(dim=1)  # cosine of every row with u

    for alpha in (0.0, 0.5):
        torch.manual_seed(5)
        net = LoRANetwork(
            model,
            rank=4,
            alpha=2.0,
            down_init="weight_svd",
            channel_mean=means,
            dc_isolate=True,
            channel_stats=stats if alpha > 0 else None,
            channel_scaling_alpha=alpha,
        )
        for name, adapter in net.adapters.items():
            cos = rows_vs_u(adapter, means[adapter_name(net, name)])
            assert torch.allclose(cos[0].abs(), torch.tensor(1.0), atol=1e-5)
            assert cos[1:].abs().max() < 1e-5
            norms = adapter.down_weight().norm(dim=1)
            assert torch.allclose(norms, norms[0].expand_as(norms), rtol=1e-4)
        # up is zero: the base model exactly
        assert torch.allclose(_out_with(model, net, x, e), base_out, atol=1e-6)
        # an optimizer step puts a little u back into rows 1..; reproject removes it
        with torch.no_grad():
            for adapter in net.adapters.values():
                adapter.down.weight.add_(torch.randn_like(adapter.down.weight) * 0.05)
        assert any(
            rows_vs_u(a, means[adapter_name(net, n)])[1:].abs().max() > 1e-3
            for n, a in net.adapters.items()
        )
        net.reproject_dc()
        for name, adapter in net.adapters.items():
            assert (
                rows_vs_u(adapter, means[adapter_name(net, name)])[1:].abs().max()
                < 1e-5
            )

    # plain LoRA on disk, dc_isolate stamped as provenance
    for adapter in net.adapters.values():
        torch.nn.init.normal_(adapter.up.weight, std=0.1)
    want = _out_with(model, net, x, e)
    path = tmp_path / "dc.safetensors"
    net.save(path, "fp32")
    fresh = _tiny()
    loaded = load_for_inference(fresh, path)
    assert all(
        a.dc_dir is None and a.inv_scale is None for a in loaded.adapters.values()
    )
    assert torch.allclose(_out(fresh, x, e), want, atol=1e-5)
    assert load_network(_tiny(), path)[1]["dc_isolate"] == "1"

    with pytest.raises(ValueError, match="lora'"):
        LoRANetwork(model, rank=4, algo="lokr", channel_mean=means, dc_isolate=True)
    with pytest.raises(ValueError, match="needs channel_mean"):
        LoRANetwork(model, rank=4, dc_isolate=True)


@pytest.mark.parametrize(
    "meta, want",
    [
        ({}, (None, False)),
        ({"ref_timestep": "-1.0"}, (None, False)),
        ({"ref_timestep": "0.0", "ref_isolate": "1"}, (0.0, True)),
        ({"ref_timestep": "250.0", "ref_isolate": "True"}, (250.0, True)),
        ({"ref_timestep": "0.0", "ref_isolate": "False"}, (0.0, False)),
    ],
)
def test_load_for_inference_sets_ref_settings(tmp_path, meta, want):
    """The reference settings a LoRA trained with land on the transformer;
    ``ref_isolate`` reads both the current ``"1"`` and the first LoRAs' ``"True"``."""
    path = tmp_path / "ref.safetensors"
    LoRANetwork(_tiny(), rank=4).save(path, "fp32", **meta)
    assert ref_settings(path) == want
    fresh = _tiny()
    fresh.ref_timestep, fresh.ref_isolate = 123.0, True
    load_for_inference(fresh, path)
    assert (fresh.ref_timestep, fresh.ref_isolate) == want
