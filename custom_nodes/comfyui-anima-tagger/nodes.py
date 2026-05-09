"""Anima Tagger ComfyUI nodes.

Two nodes:

* ``AnimaTaggerLoader`` — load an ``AnimaTagger`` checkpoint from disk and
  emit a reusable ``ANIMA_TAGGER`` socket. ComfyUI memoizes loader outputs
  by inputs, so the tagger persists across graph runs without per-call
  reload.
* ``AnimaTaggerCaption`` — take an ``ANIMA_TAGGER`` and an ``IMAGE`` and
  return the predicted caption as a ``STRING``. Useful outside DirectEdit
  (e.g., feeding any sampler's text input or pre-filling LoRA captions).

The ``ANIMA_TAGGER`` socket type is a plain string in ComfyUI's type
system, so the AnimaDirectEdit node in ``comfyui-anima-directedit`` consumes
the same socket without a code-level dependency on this package.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import torch

# Make ``anima_lora/`` importable. This file lives at
# ``anima_lora/custom_nodes/comfyui-anima-tagger/nodes.py`` so parents[2]
# is ``anima_lora/``.
ANIMA_LORA = Path(__file__).resolve().parents[2]
if str(ANIMA_LORA) not in sys.path:
    sys.path.insert(0, str(ANIMA_LORA))

from PIL import Image  # noqa: E402

import comfy.model_management  # noqa: E402  ComfyUI module; only resolvable inside ComfyUI

from library.captioning.anima_tagger import AnimaTagger  # noqa: E402

logger = logging.getLogger(__name__)


def _comfy_image_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """ComfyUI IMAGE [B, H, W, C] in [0,1] -> PIL.RGB (first batch element)."""
    arr = image_tensor[0].clamp(0, 1).cpu().numpy()
    return Image.fromarray((arr * 255).astype(np.uint8)).convert("RGB")


class AnimaTaggerLoader:
    """Load an AnimaTagger checkpoint as a reusable graph asset.

    ComfyUI memoizes loader outputs by inputs, so the tagger persists across
    invocations as long as ``tagger_dir`` doesn't change — no per-call
    reload from disk.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tagger_dir": (
                    "STRING",
                    {
                        "default": "models/captioners/anima-tagger-v1",
                        "tooltip": (
                            "AnimaTagger checkpoint directory. Relative paths "
                            "are resolved against the anima_lora/ project root; "
                            "absolute paths used as-is. Must contain "
                            "model.safetensors + config.json + vocab.json + "
                            "thresholds.safetensors + rules.yaml."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("ANIMA_TAGGER",)
    RETURN_NAMES = ("tagger",)
    FUNCTION = "load"
    CATEGORY = "anima"
    DESCRIPTION = (
        "Load an AnimaTagger checkpoint. Output socket is consumed by "
        "AnimaTaggerCaption (image -> caption) and AnimaDirectEdit. ComfyUI "
        "memoizes the output, so re-running the graph reuses the same "
        "instance without reloading."
    )

    def load(self, tagger_dir: str):
        tdir = Path(tagger_dir)
        if not tdir.is_absolute():
            tdir = ANIMA_LORA / tdir
        device = comfy.model_management.get_torch_device()
        logger.info("AnimaTaggerLoader: loading %s on %s", tdir, device)
        tagger = AnimaTagger(ckpt_dir=tdir, device=device)
        return (tagger,)


class AnimaTaggerCaption:
    """Run an AnimaTagger over an image and emit the caption as a STRING."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tagger": (
                    "ANIMA_TAGGER",
                    {"tooltip": "AnimaTagger from AnimaTaggerLoader."},
                ),
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    FUNCTION = "caption"
    CATEGORY = "anima"
    DESCRIPTION = (
        "Predict an Anima-formatted caption for an image using AnimaTagger. "
        "Output is a comma-separated tag string in Anima's canonical slot "
        "order (rating, count, characters, copyrights, @artists, generals) "
        "with underscores replaced by spaces — drop-in for any STRING input "
        "(CLIPTextEncode, prompt_src_override on AnimaDirectEdit, etc.)."
    )

    def caption(self, tagger: AnimaTagger, image: torch.Tensor):
        pil = _comfy_image_to_pil(image)
        text = tagger.predict_caption(pil)
        logger.info("AnimaTaggerCaption: %r", text)
        return (text,)


NODE_CLASS_MAPPINGS = {
    "AnimaTaggerLoader": AnimaTaggerLoader,
    "AnimaTaggerCaption": AnimaTaggerCaption,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaTaggerLoader": "Anima Tagger Loader",
    "AnimaTaggerCaption": "Anima Tagger Caption",
}
