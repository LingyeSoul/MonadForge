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
    load_for_inference,
    load_network,
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
