"""Anima DirectEdit node.

Image in, edited image out. Consumes ComfyUI's stock ``MODEL`` / ``CLIP`` /
``VAE`` sockets — pipe in the same loaders you'd use for a normal Anima
KSampler workflow (``UNETLoader``/``CLIPLoader``/``VAELoader`` or
``CheckpointLoaderSimple``). The tagger socket comes from the
``comfyui-anima-tagger`` package (``AnimaTaggerLoader``) — install it
alongside this package if you want on-image ψ_src derivation; otherwise
use ``prompt_src_override`` to supply ψ_src directly.

Pipeline mirrors ``scripts/edit.py``:

  IMAGE -> PIL                        (ComfyUI [B,H,W,C] in [0,1] -> PIL.RGB)
  tagger.predict_caption -> psi_src   (skipped if prompt_src_override is set;
                                       tagger comes from AnimaTaggerLoader in
                                       comfyui-anima-tagger)
  psi_tar = psi_src + ", " + edit_text
  bucket pick from source aspect ratio
  CLIP.tokenize / encode_from_tokens(return_dict=True) for psi_src/tar/neg
  diffusion_model.preprocess_text_embeds -> 512-padded crossattn embeds
  VAE.encode source -> z_clean
  invert(z_clean, embed_src) -> z_inv, delta_z
  edit_forward(z_inv[0], delta_z, embed_tar) -> z_edit
  VAE.decode z_edit -> ComfyUI IMAGE [1,H,W,3] in [0,1]
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

# Make ``anima_lora/`` importable. This file lives at
# ``anima_lora/custom_nodes/comfyui-anima-directedit/nodes.py`` so parents[2]
# is ``anima_lora/``.
ANIMA_LORA = Path(__file__).resolve().parents[2]
if str(ANIMA_LORA) not in sys.path:
    sys.path.insert(0, str(ANIMA_LORA))

from PIL import Image  # noqa: E402

import comfy.model_management  # noqa: E402  ComfyUI module; only resolvable inside ComfyUI

from library.captioning.anima_tagger import AnimaTagger  # noqa: E402
from library.datasets.buckets import CONSTANT_TOKEN_BUCKETS  # noqa: E402
from library.inference import directedit, sampling as inference_utils  # noqa: E402

logger = logging.getLogger(__name__)


def _pick_bucket(pil_img: Image.Image) -> Tuple[int, int]:
    """Closest CONSTANT_TOKEN_BUCKETS entry by aspect ratio. Returns (H, W)."""
    rw, rh = pil_img.size
    target = rw / rh
    best = min(CONSTANT_TOKEN_BUCKETS, key=lambda wh: abs(wh[0] / wh[1] - target))
    return best[1], best[0]  # bucket is (W, H); we return (H, W)


def _comfy_image_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """ComfyUI IMAGE [B, H, W, C] in [0,1] -> PIL.RGB (first batch element)."""
    arr = image_tensor[0].clamp(0, 1).cpu().numpy()
    return Image.fromarray((arr * 255).astype(np.uint8)).convert("RGB")


def _pil_to_comfy_image(pil_img: Image.Image) -> torch.Tensor:
    """PIL.RGB -> ComfyUI IMAGE [1, H, W, 3] in [0,1] (for VAE.encode input)."""
    arr = np.asarray(pil_img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _encode_prompt(clip, unet, text: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Tokenize + encode + LLMAdapter preprocess + pad-to-512.

    Returns the same ``[1, 512, 1024]`` crossattn tensor that
    ``library.inference.text.prepare_text_inputs`` produces, so the
    directedit primitives consume it unchanged.
    """
    tokens = clip.tokenize(text)
    out = clip.encode_from_tokens(tokens, return_pooled=False, return_dict=True)
    cond = out["cond"]  # [1, T_q, 1024]
    t5xxl_ids = out["t5xxl_ids"]  # [T_t5]
    t5xxl_weights = out["t5xxl_weights"]  # [T_t5]

    # Mirror model_base.Anima.extra_conds shape conventions.
    t5xxl_ids = t5xxl_ids.unsqueeze(0).to(device)  # [1, T_t5]
    t5xxl_weights = t5xxl_weights.unsqueeze(0).unsqueeze(-1).to(device, dtype=dtype)  # [1, T_t5, 1]
    cond = cond.to(device, dtype=dtype)

    # diffusion_model.preprocess_text_embeds runs LLMAdapter and pads to 512.
    embed = unet.preprocess_text_embeds(cond, t5xxl_ids, t5xxl_weights=t5xxl_weights)
    return embed


class AnimaDirectEdit:
    """Image + tag-to-add -> edited image, driven by stock MODEL/CLIP/VAE sockets."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Anima DiT (e.g. UNETLoader)."}),
                "clip": ("CLIP", {"tooltip": "Anima CLIP (Qwen3 06B + T5xxl tokenizer)."}),
                "vae": ("VAE", {"tooltip": "Qwen Image VAE."}),
                "image": ("IMAGE",),
                "edit_text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "double peace",
                        "tooltip": (
                            "Tag(s) to add to the source caption. "
                            "psi_tar = psi_src + ', ' + edit_text. Leave "
                            "empty to use psi_src as-is (sanity-check; "
                            "should reconstruct the source)."
                        ),
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {"multiline": True, "default": "worst quality"},
                ),
                "infer_steps": ("INT", {"default": 28, "min": 1, "max": 200}),
                "flow_shift": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.05},
                ),
                "guidance_scale": (
                    "FLOAT",
                    {
                        "default": 4.0,
                        "min": 1.0,
                        "max": 20.0,
                        "step": 0.1,
                        "tooltip": "CFG scale for the edit (target) pass.",
                    },
                ),
                "invert_guidance": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 1.0,
                        "max": 20.0,
                        "step": 0.1,
                        "tooltip": (
                            "CFG scale during inversion. Default 1.0 (no CFG); "
                            "raise only if the inverted noise must match a "
                            "high-CFG generation seed."
                        ),
                    },
                ),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**63 - 1}),
            },
            "optional": {
                "tagger": (
                    "ANIMA_TAGGER",
                    {
                        "tooltip": (
                            "AnimaTagger from AnimaTaggerLoader (in the "
                            "comfyui-anima-tagger package). Required unless "
                            "prompt_src_override is set."
                        ),
                    },
                ),
                "prompt_src_override": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": (
                            "Optional: replace the tagger's caption with your "
                            "own psi_src. Leave empty to use the tagger output. "
                            "Useful when the source is an Anima-generated image "
                            "and you already know the original prompt. If set, "
                            "the tagger socket is ignored."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "prompt_src", "prompt_tar")
    FUNCTION = "edit"
    CATEGORY = "anima"
    DESCRIPTION = (
        "DirectEdit (Yang & Ye arXiv:2605.02417) editor for Anima. Runs the "
        "AnimaTagger on the source image to derive psi_src, appends edit_text "
        "to form psi_tar, then performs flow inversion + delta_z-anchored "
        "resampling on the MODEL/CLIP/VAE sockets you wire in."
    )

    def edit(
        self,
        model,
        clip,
        vae,
        image: torch.Tensor,
        edit_text: str,
        negative_prompt: str,
        infer_steps: int,
        flow_shift: float,
        guidance_scale: float,
        invert_guidance: float,
        seed: int,
        tagger: AnimaTagger | None = None,
        prompt_src_override: str = "",
    ):
        device = comfy.model_management.get_torch_device()
        dtype = torch.bfloat16

        pil_src = _comfy_image_to_pil(image)

        if prompt_src_override.strip():
            psi_src = prompt_src_override.strip()
            logger.info("DirectEdit: using prompt_src_override: %r", psi_src)
        else:
            if tagger is None:
                raise ValueError(
                    "AnimaDirectEdit needs either a `tagger` socket "
                    "(from AnimaTaggerLoader in the comfyui-anima-tagger "
                    "package) or a non-empty `prompt_src_override` to "
                    "derive psi_src."
                )
            logger.info("DirectEdit: running AnimaTagger from %s", tagger.ckpt_dir)
            psi_src = tagger.predict_caption(pil_src)
            logger.info("DirectEdit: psi_src = %r", psi_src)

        edit = (edit_text or "").strip()
        psi_tar = f"{psi_src}, {edit}" if edit else psi_src
        logger.info("DirectEdit: psi_tar = %r", psi_tar)

        h_pix, w_pix = _pick_bucket(pil_src)
        pil_src_resized = pil_src.resize((w_pix, h_pix), Image.LANCZOS)
        logger.info(
            "DirectEdit: bucket %dx%d (HxW) for source aspect %.3f",
            h_pix, w_pix, pil_src.size[0] / pil_src.size[1],
        )

        # Bring DiT onto device so preprocess_text_embeds (LLMAdapter) and the
        # invert/edit forwards see resident weights. ComfyUI's patcher owns
        # eviction — we don't move it back.
        comfy.model_management.load_models_gpu([model])
        unet = model.model.diffusion_model

        torch.manual_seed(seed)

        logger.info("DirectEdit: encoding prompts via CLIP")
        with torch.no_grad():
            embed_src = _encode_prompt(clip, unet, psi_src, device, dtype)
            embed_tar = _encode_prompt(clip, unet, psi_tar, device, dtype)
            embed_neg = _encode_prompt(clip, unet, negative_prompt, device, dtype)

        logger.info("DirectEdit: encoding source image via VAE")
        img_in = _pil_to_comfy_image(pil_src_resized)
        z_clean = vae.encode(img_in)  # [1, C_lat, T=1, H/8, W/8]
        z_clean = z_clean.to(device, dtype=dtype)
        logger.info("DirectEdit: encoded source latent %s", tuple(z_clean.shape))

        timesteps, sigmas = inference_utils.get_timesteps_sigmas(
            infer_steps, flow_shift, device
        )
        sigmas = sigmas.to(device)

        logger.info(
            "DirectEdit: invert (T=%d, src_guidance=%.2f) -> edit (tar_guidance=%.2f)",
            infer_steps, invert_guidance, guidance_scale,
        )
        z_inv, delta_z = directedit.invert(
            anima=unet,
            z_clean=z_clean,
            embed_src=embed_src,
            embed_neg=embed_neg if invert_guidance != 1.0 else None,
            sigmas=sigmas,
            guidance_scale=invert_guidance,
        )
        z_edit = directedit.edit_forward(
            anima=unet,
            z_init=z_inv[0],
            delta_z=delta_z,
            embed_tar=embed_tar,
            embed_neg=embed_neg,
            sigmas=sigmas,
            guidance_scale=guidance_scale,
        )

        logger.info("DirectEdit: decoding edited latent via VAE")
        comfy_img = vae.decode(z_edit.to(dtype=dtype))  # [1, H, W, 3] in [0,1]
        return (comfy_img, psi_src, psi_tar)


NODE_CLASS_MAPPINGS = {
    "AnimaDirectEdit": AnimaDirectEdit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaDirectEdit": "Anima DirectEdit",
}
