"""Tiny CPU coverage for the vendored SR helpers; no pretrained weights."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from sr.resshift.ldm.modules.vqvae.quantize import EMAVectorQuantizer
from sr.resshift.utils.util_common import str2bool
from sr.resshift.utils.util_image import convfft, psf2otf, tensor2img, zero_pad


def test_ema_quantizer_uses_declared_dimensions_and_updates_on_training_only():
    torch.manual_seed(19)
    quantizer = EMAVectorQuantizer(n_embed=4, embedding_dim=2, beta=0.25)
    assert quantizer.embedding.weight.shape == (4, 2)
    before = quantizer.embedding.weight.detach().clone()
    inputs = torch.randn(1, 2, 2, 2, requires_grad=True)
    output, loss, (perplexity, encodings, indices) = quantizer(inputs)
    loss.backward()
    assert output.shape == inputs.shape
    assert encodings.shape == (4, 4) and indices.shape == (4,)
    assert torch.isfinite(output).all() and torch.isfinite(perplexity)
    assert torch.isfinite(inputs.grad).all() and inputs.grad.abs().sum() > 0
    assert not torch.equal(before, quantizer.embedding.weight)
    quantizer.eval()
    before_eval = quantizer.embedding.weight.detach().clone()
    quantizer(inputs.detach())
    assert torch.equal(before_eval, quantizer.embedding.weight)


def test_ema_quantizer_remap_initializes_and_roundtrips_known_codes(tmp_path):
    remap = tmp_path / "used.npy"
    np.save(remap, [0, 2])
    quantizer = EMAVectorQuantizer(4, 2, 0.25, remap=str(remap))
    indices = torch.tensor([[0, 2]])
    mapped = quantizer.remap_to_used(indices)
    assert mapped.tolist() == [[0, 1]]
    assert torch.equal(quantizer.unmap_to_all(mapped), indices)


def test_boolean_parser_reports_argument_error_for_invalid_input():
    assert str2bool("YES") is True
    assert str2bool("false") is False
    with pytest.raises(argparse.ArgumentTypeError, match="Boolean value"):
        str2bool("perhaps")


@pytest.mark.parametrize("bgr", [False, True])
def test_tensor_batch_grid_preserves_both_images_and_channel_order(bgr):
    batch = torch.zeros(2, 3, 2, 2)
    batch[0, 0] = 1
    batch[1, 1] = 1
    grid = tensor2img(batch, rgb2bgr=bgr)
    assert grid.shape == (10, 6, 3)
    assert grid.dtype == np.uint8
    np.testing.assert_array_equal(grid[2, 2], [0, 0, 255] if bgr else [255, 0, 0])
    np.testing.assert_array_equal(grid[6, 2], [0, 255, 0])


@pytest.mark.parametrize("position", ["corner", "center"])
def test_zero_padding_preserves_kernel_values_and_dtype(position):
    kernel = np.array([[1, 2], [3, 4]], dtype=np.int16)
    padded = zero_pad(kernel, (4, 6), position)
    assert padded.shape == (4, 6) and padded.dtype == kernel.dtype
    row, column = (0, 0) if position == "corner" else (1, 2)
    np.testing.assert_array_equal(padded[row : row + 2, column : column + 2], kernel)
    assert padded.sum() == kernel.sum()


@pytest.mark.parametrize(
    ("shape", "position"),
    [((1, 2), "corner"), ((0, 2), "corner"), ((3, 4), "center"), ((4, 4), "invalid")],
)
def test_zero_padding_rejects_invalid_geometry(shape, position):
    with pytest.raises(ValueError, match="zero_pad"):
        zero_pad(np.ones((2, 2)), shape, position)


def test_fft_convolution_matches_periodic_spatial_convolution():
    image = np.arange(5 * 7 * 2, dtype=np.float64).reshape(5, 7, 2)
    kernel = np.array([[0, 1, 0], [2, 3, 4], [0, 5, 0]], dtype=np.float64)
    expected = sum(
        kernel[row, column] * np.roll(image, (row - 1, column - 1), axis=(0, 1))
        for row in range(3)
        for column in range(3)
    )
    np.testing.assert_allclose(convfft(image, kernel), expected, atol=1e-10)
    zero = np.zeros((3, 3))
    assert psf2otf(zero, image.shape[:2]).shape == image.shape[:2]
    np.testing.assert_array_equal(convfft(image, zero), np.zeros_like(image))


def test_swin_module_cpu_self_check_runs():
    result = subprocess.run(
        [sys.executable, "-m", "sr.resshift.models.swin_transformer"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        timeout=30,
        check=True,
    )
    assert "WindowAttention CPU self-check passed: (1, 4, 8)" in result.stdout
