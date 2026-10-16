"""MOSS cache contract: latent pack/unpack inverse, sizing, flat-cache reading."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

from library.moss.cache import (
    LATENT_SUFFIX,
    TE_SUFFIX,
    find_pairs,
    fit,
    list_items,
    load_item,
    target_size,
)
from library.moss.model import pack_latents, unpack_latents


def _fake_vae(channels=32):
    g = torch.Generator().manual_seed(0)
    bn = SimpleNamespace(
        running_mean=torch.randn(channels * 4, generator=g),
        running_var=torch.rand(channels * 4, generator=g) + 0.5,
    )
    return SimpleNamespace(bn=bn, config=SimpleNamespace(batch_norm_eps=1e-4), dtype=torch.float32)


def test_pack_is_inverse_of_unpack():
    vae = _fake_vae()
    z = torch.randn(2, 32, 12, 18, dtype=torch.float64).float()
    packed = pack_latents(z, vae)
    assert packed.shape == (2, 128, 6, 9)
    torch.testing.assert_close(unpack_latents(packed, vae), z, rtol=1e-5, atol=1e-5)
    d = torch.randn(1, 128, 5, 7)
    torch.testing.assert_close(pack_latents(unpack_latents(d, vae), vae), d, rtol=1e-5, atol=1e-5)


def test_pack_matches_flux2_patchify():
    # Channel 4c + 2a + b of the packed latent holds pixel (2i + a, 2j + b) of channel c.
    vae = _fake_vae(channels=1)
    vae.bn.running_mean = torch.zeros(4)
    vae.bn.running_var = torch.ones(4) - 1e-4
    z = torch.arange(16.0).view(1, 1, 4, 4)
    packed = pack_latents(z, vae)
    for a in range(2):
        for b in range(2):
            torch.testing.assert_close(packed[0, 2 * a + b], z[0, 0, a::2, b::2])


@pytest.mark.parametrize(
    "size",
    [(896, 1152), (1024, 1024), (3000, 2000), (1200, 4000), (640, 480), (1023, 1023), (5000, 300), (17, 33)],
)
def test_target_size_invariants(size):
    w, h = size
    tw, th = target_size(w, h, 1024)
    assert tw % 16 == 0 and th % 16 == 0 and tw >= 16 and th >= 16
    assert tw <= max(16, w) and th <= max(16, h)  # never upscaled
    assert tw * th <= 1024 * 1024 * 1.10
    if min(w, h) >= 256:
        assert abs((tw / th) / (w / h) - 1) < 0.05


def test_target_size_keeps_grid_images():
    assert target_size(960, 1120, 1024) == (960, 1120)  # 1.075 MP, already on the grid
    assert target_size(1536, 1536, 1024) == (1024, 1024)


def test_fit_exact_size():
    im = Image.new("RGB", (1000, 1500))
    assert fit(im, 832, 1248).size == (832, 1248)
    assert fit(im, 1000, 1500) is im


def test_list_and_load_items(tmp_path):
    save_file({"latent": torch.randn(128, 72, 56)}, tmp_path / f"a_b_896x1152{LATENT_SUFFIX}")
    save_file({"embeds": torch.randn(37, 4096).bfloat16()}, tmp_path / f"a_b{TE_SUFFIX}")
    save_file({"latent": torch.randn(128, 64, 64)}, tmp_path / f"lonely_1024x1024{LATENT_SUFFIX}")
    save_file({"embeds": torch.randn(5, 4096).bfloat16()}, tmp_path / f"textonly{TE_SUFFIX}")
    items = list_items(tmp_path)
    assert [(i.stem, i.width, i.height, i.n_text) for i in items] == [("a_b", 896, 1152, 37)]
    latent, embeds = load_item(items[0])
    assert latent.dtype == torch.float32 and latent.shape == (128, 72, 56)
    assert embeds.dtype == torch.bfloat16 and embeds.shape == (37, 4096)


def test_find_pairs_prefixes_colliding_stems(tmp_path):
    for sub in ("x", "y"):
        (tmp_path / sub).mkdir()
        Image.new("RGB", (32, 32)).save(tmp_path / sub / "img.png")
        (tmp_path / sub / "img.txt").write_text(f" caption {sub}, tag \n", encoding="utf-8")
    Image.new("RGB", (32, 32)).save(tmp_path / "x" / "solo.png")
    (tmp_path / "x" / "solo.txt").write_text("solo", encoding="utf-8")
    Image.new("RGB", (32, 32)).save(tmp_path / "x" / "nocap.png")
    pairs = {p.stem: p.caption for p in find_pairs(tmp_path)}
    assert pairs == {"x__img": "caption x, tag", "y__img": "caption y, tag", "solo": "solo"}
