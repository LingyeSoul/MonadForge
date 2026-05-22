"""Anima Block Compile ComfyUI custom node.

One node, ``AnimaBlockCompile``, that runs ``torch.compile`` on the Anima DiT
per transformer block (targeting ``diffusion_model.blocks.{i}``) rather than
compiling the whole ``diffusion_model`` at once. Faster to compile and far less
prone to graph breaks — the same reasoning behind anima_lora's training/
inference ``--compile_mode blocks`` default.

It is a thin, Anima-named wrapper over ComfyUI core's
``set_torch_compile_wrapper(model, keys=[...])`` and pulls in no library.*/
networks.* code, so no vendored tree is needed.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
