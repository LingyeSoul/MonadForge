"""Real small-model round trips for official shards and Comfy single files."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch
from diffusers import QwenImage21Transformer2DModel
from safetensors.torch import save_file

from library.qwen21.checkpoint import (
    CheckpointFormatError,
    read_checkpoint,
    text_encoder_keys,
    transformer_keys,
    vae_keys,
)
from library.qwen21.loader import load_scheduler, load_transformer
from library.qwen21.lora import LoRANetwork, load_network
from library.qwen21.requests import (
    CONFIG_DIR,
    CacheRequest,
    GenerateRequest,
    TrainRequest,
    model_paths,
)
from library.qwen21.train import training_step


def tiny_transformer() -> QwenImage21Transformer2DModel:
    return (
        QwenImage21Transformer2DModel(
            in_channels=64,
            out_channels=64,
            num_layers=2,
            attention_head_dim=32,
            num_attention_heads=2,
            context_in_dim=32,
            axes_dims_rope=(8, 12, 12),
        )
        .eval()
        .requires_grad_(False)
    )


def comfy_transformer_state(
    model: QwenImage21Transformer2DModel,
) -> dict[str, torch.Tensor]:
    source = model.state_dict()
    result = {
        k: v.clone()
        for k, v in source.items()
        if not k.endswith((".gate_layer.weight", ".proj.weight"))
    }
    for key, value in source.items():
        if key.endswith(".gate_layer.weight"):
            result[key.replace(".gate_layer.", ".gate_up.")] = torch.cat(
                (value, source[key.replace(".gate_layer.", ".proj.")]),
                dim=0,
            )
    return result


def training_batch() -> dict[str, torch.Tensor | int]:
    return {
        "latents": torch.randn(1, 4, 64),
        "latent_h": 2,
        "latent_w": 2,
        "prompt_embeds": torch.randn(1, 3, 32),
        "prompt_embeds_mask": torch.ones(1, 3, dtype=torch.bool),
    }


def test_comfy_transformer_inference_round_trip(tmp_path: Path):
    torch.manual_seed(12)
    original = tiny_transformer()
    checkpoint = tmp_path / "comfy-inference.safetensors"
    save_file(comfy_transformer_state(original), checkpoint)
    loaded = tiny_transformer()
    loaded.load_state_dict(
        read_checkpoint(
            checkpoint, loaded.state_dict(), transformer_keys, torch.float32
        ),
        strict=True,
    )
    inputs = {
        "hidden_states": torch.randn(1, 4, 64),
        "encoder_hidden_states": torch.randn(1, 3, 32),
        "timestep": torch.tensor([0.4]),
        "img_shapes": [[(1, 2, 2)]],
        "img_mask": torch.tensor([[False, False, False, True]]),
        "encoder_hidden_states_mask": torch.ones(1, 3, dtype=torch.bool),
    }
    with torch.inference_mode():
        expected = original(**inputs).sample
        actual = loaded(**inputs).sample
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.isfinite(actual).all()


def test_comfy_round_trip_forward_backward_and_adapter_reload(tmp_path: Path):
    torch.manual_seed(12)
    original = tiny_transformer()
    checkpoint = tmp_path / "comfy.safetensors"
    save_file(comfy_transformer_state(original), checkpoint)
    loaded = tiny_transformer()
    loaded.load_state_dict(
        read_checkpoint(
            checkpoint, loaded.state_dict(), transformer_keys, torch.float32
        )
    )
    batch = training_batch()
    sigma = torch.tensor([0.4])
    torch.manual_seed(50)
    expected = training_step(original, batch, sigma)
    torch.manual_seed(50)
    actual = training_step(loaded, batch, sigma)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    adapter = LoRANetwork(loaded, rank=2, alpha=2.0)
    adapter.apply_to()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=0.01)
    before = {key: value.clone() for key, value in adapter.state_dict().items()}
    loss = training_step(loaded, batch, sigma)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in adapter.parameters()
    )
    optimizer.step()
    assert any(not torch.equal(before[k], v) for k, v in adapter.state_dict().items())
    path = tmp_path / "lora.safetensors"
    from library.qwen21.requests import DEFAULT_TARGETS

    save_file(
        adapter.state_dict(),
        path,
        metadata={
            "rank": "2",
            "alpha": "2.0",
            "targets": DEFAULT_TARGETS,
            "lora_dtype": "fp32",
        },
    )
    reloaded, _ = load_network(original, path)
    reloaded.apply_to()
    torch.manual_seed(51)
    expected = training_step(loaded, batch, sigma)
    torch.manual_seed(51)
    actual = training_step(original, batch, sigma)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    adapter.restore()
    reloaded.restore()


def test_official_sharded_transformer_round_trip(tmp_path: Path):
    original = tiny_transformer()
    model_dir = tmp_path / "official"
    original.save_pretrained(model_dir / "transformer", max_shard_size="10KB")
    assert list((model_dir / "transformer").glob("*.safetensors.index.json"))
    req = TrainRequest.from_argv(["--model_dir", str(model_dir)])
    restored = load_transformer(model_paths(req).dit, torch.float32)
    expected = original.state_dict()
    actual = restored.state_dict()
    assert actual.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


@pytest.mark.parametrize("request_type", [CacheRequest, TrainRequest, GenerateRequest])
def test_official_directory_and_component_override_paths(tmp_path: Path, request_type):
    model_dir = tmp_path / "official"
    req = request_type.from_argv(["--model_dir", str(model_dir)])
    assert request_type.from_argv(req.to_argv()) == req
    paths = model_paths(req)
    assert paths.dit == model_dir / "transformer"
    assert paths.text_encoder == model_dir / "text_encoder"
    assert paths.vae == model_dir / "vae"
    assert paths.processor == model_dir / "processor"
    assert paths.scheduler == model_dir / "scheduler"

    single_file = tmp_path / "comfy.safetensors"
    mixed = request_type.from_argv(
        ["--model_dir", str(model_dir), "--dit", str(single_file)]
    )
    mixed_paths = model_paths(mixed)
    assert mixed_paths.dit == single_file
    assert mixed_paths.scheduler == paths.scheduler
    assert mixed_paths.text_encoder == paths.text_encoder


def test_vae_temporal_axis_conversion_preserves_convolution(tmp_path: Path):
    torch.manual_seed(3)
    weight = torch.randn(2, 4, 1, 3, 3)
    path = tmp_path / "vae.safetensors"
    save_file({"encoder.conv1.weight": weight}, path)
    state = read_checkpoint(
        path,
        {"encoder.conv_in.weight": torch.empty(2, 4, 3, 3)},
        vae_keys,
        torch.float32,
    )
    pixels = torch.randn(1, 4, 1, 8, 8)
    expected = torch.nn.functional.conv3d(pixels, weight).squeeze(2)
    actual = torch.nn.functional.conv2d(
        pixels.squeeze(2), state["encoder.conv_in.weight"]
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "kind", ["missing", "shape", "unmarked_int8", "int8_convrot", "w4a8", "fp8"]
)
def test_invalid_checkpoint_fails_before_loading(tmp_path: Path, kind: str):
    expected = {"img_in.weight": torch.empty(4, 4)}
    state = {"img_in.weight": torch.zeros(4, 4)}
    if kind == "missing":
        state = {"unrelated": torch.zeros(1)}
    elif kind == "shape":
        state = {"img_in.weight": torch.zeros(3, 4)}
    elif kind == "unmarked_int8":
        state = {"img_in.weight": torch.zeros(4, 4, dtype=torch.int8)}
    elif kind == "fp8":
        state = {"img_in.weight": torch.zeros(4, 4).to(torch.float8_e4m3fn)}
    else:
        metadata = (
            b'{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":4}'
            if kind == "int8_convrot"
            else b'{"format":"asym_w4a8_int8"}'
        )
        state["img_in.comfy_quant"] = torch.tensor(list(metadata), dtype=torch.uint8)
    path = tmp_path / "invalid.safetensors"
    save_file(state, path)
    with pytest.raises(CheckpointFormatError, match="invalid.safetensors"):
        read_checkpoint(path, expected, transformer_keys, torch.float32)


@pytest.mark.parametrize("request_type", [CacheRequest, TrainRequest, GenerateRequest])
def test_single_file_cli_round_trip(request_type):
    req = request_type.from_argv(
        ["--dit", "weights/dit.safetensors", "--processor", "weights/processor"]
    )
    assert request_type.from_argv(req.to_argv()) == req
    assert model_paths(req).dit.is_absolute()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            req.SCRIPT.removesuffix(".py").replace("/", "."),
            "--help",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--dit" in result.stdout and "--model_dir" in result.stdout


def test_training_schedule_uses_qwen21_config():
    from library.qwen21.train import calculate_shift, sample_sigma

    scheduler = load_scheduler(CONFIG_DIR / "scheduler.json")
    assert calculate_shift(8192, scheduler.config) == pytest.approx(0.9)
    assert calculate_shift(256, scheduler.config) == pytest.approx(0.5)
    sigma = sample_sigma(scheduler, 0.7, 0.0, 1.0, torch.device("cpu"))
    assert torch.isfinite(sigma).all() and 0 < sigma.item() < 1


def test_qwen3vl_single_file_text_forward_round_trip(tmp_path: Path):
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    config = Qwen3VLConfig(
        text_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "vocab_size": 128,
            "rope_scaling": {
                "rope_type": "default",
                "mrope_section": [1, 1, 2],
                "mrope_interleaved": True,
            },
        },
        vision_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "out_hidden_size": 32,
            "depth": 1,
            "num_heads": 4,
            "deepstack_visual_indexes": [0],
            "patch_size": 2,
            "num_position_embeddings": 16,
        },
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(6)
    original = Qwen3VLForConditionalGeneration(config).eval()
    comfy = {
        key.replace("model.language_model.", "model.").replace(
            "model.visual.", "visual."
        ): tensor.clone()
        for key, tensor in original.state_dict().items()
    }
    path = tmp_path / "text_encoder.safetensors"
    save_file(comfy, path)
    loaded = Qwen3VLForConditionalGeneration(config).eval()
    loaded.load_state_dict(
        read_checkpoint(path, loaded.state_dict(), text_encoder_keys, torch.float32)
    )
    ids = torch.tensor([[2, 3, 4]])
    with torch.no_grad():
        expected = original(input_ids=ids, output_hidden_states=True).hidden_states[-1]
        actual = loaded(input_ids=ids, output_hidden_states=True).hidden_states[-1]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
