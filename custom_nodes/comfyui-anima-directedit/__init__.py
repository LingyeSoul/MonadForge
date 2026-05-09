"""Anima DirectEdit ComfyUI custom node.

One node:

* ``AnimaDirectEdit`` — takes an image, an "edit text" (tags to add), and
  optionally a tagger socket; runs the tagger to derive ``psi_src``
  (or uses ``prompt_src_override``), forms
  ``psi_tar = psi_src + ", " + edit_text``, then invokes the DirectEdit
  invert + edit_forward primitives on the wired-in MODEL to produce an
  edited image. Consumes ComfyUI's stock MODEL / CLIP / VAE sockets, so
  ``LoraLoader`` / ``comfyui-hydralora``'s adapter loader compose
  naturally upstream.

The ``ANIMA_TAGGER`` socket is produced by ``AnimaTaggerLoader`` in the
sibling ``comfyui-anima-tagger`` package — install both for image-driven
ψ_src derivation, or supply ψ_src as a string via ``prompt_src_override``
to skip the tagger entirely.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
