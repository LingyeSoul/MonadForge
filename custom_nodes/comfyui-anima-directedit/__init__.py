"""Anima DirectEdit ComfyUI custom node.

Two nodes:

* ``AnimaDirectEdit`` - takes an image plus two caption STRINGs
  (``source_tag`` describing the source, ``target_tag`` describing the
  edit; empty ``target_tag`` falls back to ``source_tag`` for a
  reconstruction sanity check) and invokes the DirectEdit invert +
  edit_forward primitives on the wired-in MODEL to produce an edited
  latent. Consumes ComfyUI's stock MODEL / CLIP / VAE sockets, so
  ``LoraLoader`` / ``comfyui-hydralora``'s adapter loader compose
  naturally upstream. Both caption inputs are plain STRINGs — any node
  that emits STRING can drive them.

* ``AnimaDirectEditAutoTag`` - convenience wrapper that auto-derives the
  source caption from an ``ANIMA_TAGGER`` (wired in from the Anima Tagger
  Loader node) so you only describe the edit as ``tags_to_add`` /
  ``tags_to_remove`` deltas. Returns the edited latent plus the derived
  source/target captions.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
