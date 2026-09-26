"""Load official or Comfy-Org Qwen-Image-2.1 weights, one phase at a time.

Ported from sorryhyun/anima_lora qwen21 at 50bf09d5. Architecture and scheduler
configs match Qwen/Qwen-Image-2.1. Official component directories retain the
native from_pretrained loader, including shard indexes. Comfy BF16 files use
meta construction and strict assignment without dequantization. Both paths
feed the same upstream block swapper; neither loads the other encoders during
cached training.
Anima FP16 policies are intentionally not applied to this different model.
"""

from __future__ import annotations

import gc
from pathlib import Path

import torch
from accelerate import init_empty_weights

from typing import TYPE_CHECKING

from library.qwen21.requests import CONFIG_DIR, ModelPaths
from library.qwen21.checkpoint import (
    read_checkpoint,
    transformer_keys,
    text_encoder_keys,
    vae_keys,
)

if TYPE_CHECKING:
    from diffusers import (
        AutoencoderKLQwenImage21,
        QwenImage21Pipeline,
        QwenImage21Transformer2DModel,
        FlowMatchEulerDiscreteScheduler,
    )
    from transformers import Qwen3VLForConditionalGeneration


def free_vram_gb(device: int = 0) -> float:
    free, _total = torch.cuda.mem_get_info(device)
    return free / 1024**3


def empty_cache() -> None:
    gc.collect()
    torch.cuda.empty_cache()


TEXT_ENCODER_BLOCKS = "model.language_model.layers"
TRANSFORMER_BLOCKS = "transformer_blocks"


def validate_component_path(path: Path) -> None:
    """Reject missing/unsupported sources before constructing any model."""
    if not path.exists():
        raise FileNotFoundError(
            f"Qwen component not found: {path}; select an official component directory "
            "with its config and weight shards, or a Comfy-Org BF16 safetensors file"
        )
    if not path.is_dir() and not (path.is_file() and path.suffix == ".safetensors"):
        raise ValueError(
            f"Unsupported Qwen component path: {path}; expected a Diffusers component "
            "directory or a .safetensors file"
        )


def load_text_encoder(
    path: Path, dtype: torch.dtype
) -> Qwen3VLForConditionalGeneration:
    """Load an official Qwen3-VL directory or strictly assign a Comfy file."""
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    validate_component_path(path)
    if path.is_dir():
        return (
            Qwen3VLForConditionalGeneration.from_pretrained(
                path, dtype=dtype, local_files_only=True
            )
            .eval()
            .requires_grad_(False)
        )

    config = Qwen3VLConfig.from_pretrained(
        CONFIG_DIR / "text_encoder.json", local_files_only=True
    )
    config._attn_implementation = "sdpa"
    with init_empty_weights():
        model = Qwen3VLForConditionalGeneration(config)
    state = read_checkpoint(path, model.state_dict(), text_encoder_keys, dtype)
    model.load_state_dict(state, strict=True, assign=True)
    return model.eval().requires_grad_(False)


def load_transformer(path: Path, dtype: torch.dtype) -> QwenImage21Transformer2DModel:
    """Load official shards natively, or split Comfy gate/up projections."""
    from diffusers import QwenImage21Transformer2DModel

    validate_component_path(path)
    if path.is_dir():
        return (
            QwenImage21Transformer2DModel.from_pretrained(
                path, dtype=dtype, low_cpu_mem_usage=True, local_files_only=True
            )
            .eval()
            .requires_grad_(False)
        )

    config = QwenImage21Transformer2DModel.load_config(CONFIG_DIR / "transformer.json")
    with init_empty_weights():
        model = QwenImage21Transformer2DModel.from_config(config)
    model.load_state_dict(
        read_checkpoint(path, model.state_dict(), transformer_keys, dtype),
        strict=True,
        assign=True,
    )
    return model.eval().requires_grad_(False)


def load_vae(path: Path, dtype: torch.dtype) -> AutoencoderKLQwenImage21:
    """Load the 64-channel, RGBA Qwen-Image-2.1 VAE, not Anima's VAE."""
    from diffusers import AutoencoderKLQwenImage21

    validate_component_path(path)
    if path.is_dir():
        return (
            AutoencoderKLQwenImage21.from_pretrained(
                path, dtype=dtype, low_cpu_mem_usage=True, local_files_only=True
            )
            .eval()
            .requires_grad_(False)
        )

    config = AutoencoderKLQwenImage21.load_config(CONFIG_DIR / "vae.json")
    with init_empty_weights():
        model = AutoencoderKLQwenImage21.from_config(config)
    model.load_state_dict(
        read_checkpoint(path, model.state_dict(), vae_keys, dtype),
        strict=True,
        assign=True,
    )
    return model.eval().requires_grad_(False)


def load_scheduler(path: Path) -> FlowMatchEulerDiscreteScheduler:
    from diffusers import FlowMatchEulerDiscreteScheduler

    if not path.is_dir() and not path.is_file():
        raise FileNotFoundError(
            f"Qwen scheduler configuration not found: {path}; the official model directory "
            "must contain scheduler/scheduler_config.json"
        )
    config = FlowMatchEulerDiscreteScheduler.load_config(path, local_files_only=True)
    return FlowMatchEulerDiscreteScheduler.from_config(config)


def block_devices(blocks) -> list[str]:
    return [next(block.parameters()).device.type for block in blocks]


def place(
    model,
    blocks_path: str,
    device: torch.device,
    *,
    blocks_to_swap: int | None = None,
    supports_backward: bool = False,
    activation_reserve_gb: float = 2.5,
    label: str = "model",
    minimal_schedule: bool = True,
):
    """Move ``model`` onto ``device``, block-swapping as little as fits.

    ``blocks_to_swap=None`` sizes the swap against what is actually free;
    0 keeps everything resident. Returns the ``Attached`` handle (or None) —
    the caller must ``detach()`` it to give the VRAM back.
    """
    from library.qwen21 import blockswap

    blocks = blockswap.find_blocks(model, blocks_path)
    if blocks_to_swap is None:
        blocks_to_swap = blockswap.auto_blocks_to_swap(
            model,
            blocks,
            free_vram_gb(),
            activation_reserve_gb=activation_reserve_gb,
        )
    per = blockswap.block_size_gb(blocks)
    resident = blockswap.resident_size_gb(model, blocks)
    print(
        f"{label}: {len(blocks)} blocks x {per:.2f} GB + {resident:.2f} GB resident; "
        f"swapping {blocks_to_swap}, "
        f"{resident + (len(blocks) - blocks_to_swap) * per:.2f} GB on card "
        f"({free_vram_gb():.2f} GB free)",
        flush=True,
    )

    attached, blocks = blockswap.attach(
        model,
        blocks_path,
        blocks_to_swap,
        device,
        supports_backward=supports_backward,
        minimal_schedule=minimal_schedule,
    )
    if attached is None:
        model.to(device)
        return None
    blockswap.to_device_except_blocks(model, blocks, device)
    attached.prepare()
    return attached


def build_pipeline(
    paths: ModelPaths,
    vae: AutoencoderKLQwenImage21 | None,
    text_encoder: Qwen3VLForConditionalGeneration | None,
    transformer: QwenImage21Transformer2DModel | None,
    scheduler: FlowMatchEulerDiscreteScheduler,
) -> QwenImage21Pipeline:
    """Assemble already-loaded components; never implicitly load other weights."""
    from diffusers import QwenImage21Pipeline
    from transformers import Qwen3VLProcessor

    if not paths.processor.is_dir():
        raise FileNotFoundError(
            f"Qwen processor assets not found: {paths.processor}; run "
            "python tasks.py qwen21-processor, or set --processor to a local "
            "Qwen-Image-2.1 processor folder (no model weights required)"
        )
    processor = Qwen3VLProcessor.from_pretrained(paths.processor, local_files_only=True)
    return QwenImage21Pipeline(
        scheduler=scheduler,
        processor=processor,
        vae=vae,
        text_encoder=text_encoder,
        transformer=transformer,
    )


def encode_prompts(
    pipe,
    prompts: list[str],
    *,
    device: str = "cuda",
    images: list | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]]:
    """Encode prompts and bring the results back to CPU."""
    out = []
    with torch.no_grad():
        for prompt in prompts:
            embeds, mask, pad_mask = pipe.encode_prompt(
                prompt=prompt, image=images, device=torch.device(device)
            )
            out.append(
                (
                    embeds.detach().to("cpu"),
                    None if mask is None else mask.detach().to("cpu"),
                    pad_mask.detach().to("cpu"),
                )
            )
    return out


def decode_latents(pipe, latents: torch.Tensor, height: int, width: int):
    """The tail of ``QwenImage21Pipeline.__call__``, run separately.

    ``output_type="latent"`` lets the caller unload the transformer before the
    VAE runs. Tiling is required at 1024: the decoder holds a feature cache
    across its 3D resnet stack and untiled it asks for >1 GiB contiguous on top
    of ~12 GB already allocated. Tiles keep the peak flat and the output is the
    same image.
    """
    pipe.vae.enable_tiling()
    latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
    latents = latents.to(pipe.vae.device, pipe.vae.dtype)
    shape = (1, pipe.vae.config.z_dim, 1, 1, 1)
    mean = torch.tensor(pipe.vae.config.latents_mean).view(shape).to(latents)
    std = torch.tensor(pipe.vae.config.latents_std).view(shape).to(latents)
    decoded = pipe.vae.decode(latents * std + mean, return_dict=False)[0][:, :, 0]
    return pipe.image_processor.postprocess(decoded, output_type="pil")


def drop_text_encoder(pipe) -> None:
    """Release the 17.5 GB text encoder and whatever it held on the card."""
    te = pipe.text_encoder
    pipe.text_encoder = None
    pipe.register_to_config(text_encoder=None)
    del te
    empty_cache()
