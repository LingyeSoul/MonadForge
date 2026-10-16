"""MOSS-Image LoRA + trainer pieces on a tiny random DiT (CPU)."""

import re
from collections import Counter

import pytest
import torch

from library.moss.lora import DEFAULT_TARGETS, LoRANetwork, load_for_inference, load_network
from library.moss.model import MOSSImageTransformer
from library.moss.train import checkpoint_blocks, sample_sigma, training_loss


def _tiny(layers=3):
    torch.manual_seed(0)
    return MOSSImageTransformer(
        dim=96, ffn_dim=128, num_heads=4, num_layers=layers, text_dim=32, latent_channels=8
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
    per_block = Counter(int(re.match(r"layers\.(\d+)\.", n)[1]) for n in net.target_names)
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
    assert meta["rank"] == "4" and meta["alpha"] == "2.0" and meta["ss_num_blocks"] == "3"
    loaded.set_multiplier(0.0)
    assert torch.equal(_out(fresh, x, e), _out(_tiny(), x, e))
    with pytest.raises(ValueError, match="block"):
        load_network(_tiny(layers=4), path)


def test_one_step_decreases_loss():
    model = _tiny()
    model.requires_grad_(False)
    net = LoRANetwork(model, rank=4)
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
