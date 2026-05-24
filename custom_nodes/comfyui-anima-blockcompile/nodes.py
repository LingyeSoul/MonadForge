"""Anima block-compile ComfyUI node.

A single node, ``AnimaBlockCompile``, that applies ``torch.compile`` to the
Anima DiT one transformer block at a time instead of compiling the whole
``diffusion_model`` in one frame.

Why per-block? Anima's ``diffusion_model.blocks`` is a plain ``nn.ModuleList``
of identical transformer blocks. Compiling each block separately:

* compiles far faster (one small graph reused across N blocks instead of one
  giant graph), and
* is much less likely to hit a graph break or recompile that silently falls
  back to eager, because each block is a small, regular subgraph.

This mirrors anima_lora's training/inference behavior, which compiles each
block's ``_forward`` (``DiT.compile_blocks``) for the same reasons.

Mechanically this is exactly what ComfyUI core already supports via
``set_torch_compile_wrapper(model, keys=[...])`` — it ``torch.compile``\\s each
listed submodule and swaps the compiled copies in at sample time through a
single ``APPLY_MODEL`` wrapper. This node is the one-purpose, Anima-named
equivalent: it targets ``diffusion_model.blocks.{i}`` directly with the safe
inductor defaults, no knobs.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Anima's MiniTrainDIT exposes its transformer stack as ``blocks``; the parent
# class is shared with cosmos/predict2. We probe these names in order so the
# node keeps working if a future Anima variant renames the stack, and so it
# degrades gracefully on a non-Anima model wired in by mistake.
_BLOCK_ATTR_CANDIDATES = ("blocks", "transformer_blocks", "layers")


def _skip_transformer_options_guards(guard_entries):
    """Drop ``transformer_options`` from dynamo's guard set.

    ComfyUI threads a fresh ``transformer_options`` dict (sampler step, sigmas,
    Spectrum/mod-guidance state, …) into every ``apply_model`` call. Guarding on
    it would recompile each block every step. The bundled ``TorchCompileModel``
    node uses the same filter. Mirrors comfy_extras/nodes_torch_compile.py.
    """
    return [("transformer_options" not in entry.name) for entry in guard_entries]


class AnimaBlockCompile:
    """Per-block ``torch.compile`` for the Anima DiT.

    Drop between the model loader and the sampler. Returns a cloned MODEL with
    a torch.compile wrapper installed; compilation itself happens lazily on the
    first sample, so the first generation after wiring this in is slow and
    every subsequent one is fast.

    No options: always the inductor backend with default settings, applied
    per transformer block — which is the safe, fast configuration.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "anima"
    DESCRIPTION = (
        "Apply torch.compile (inductor) to the Anima DiT one transformer block "
        "at a time. Faster compile and fewer graph breaks than whole-model compile."
    )
    EXPERIMENTAL = True

    def patch(self, model):
        from comfy_api.torch_helpers import set_torch_compile_wrapper

        # disable_dynamic=True turns off ComfyUI's dynamic VRAM (lazy) weight
        # loading. Without it the weights stay symbolic-shaped under dynamo and
        # the per-block matmuls fail the fake-tensor meta check
        # ("a and b must have same reduction dim"). Older ComfyUI builds don't
        # take the kwarg — fall back to a plain clone there.
        try:
            m = model.clone(disable_dynamic=True)
        except TypeError:
            logger.warning(
                "AnimaBlockCompile: this ComfyUI build can't disable dynamic VRAM "
                "loading via clone(); compile may fail under lazy weight loading."
            )
            m = model.clone()
        diffusion_model = m.get_model_object("diffusion_model")

        keys: list[str] = []
        for attr in _BLOCK_ATTR_CANDIDATES:
            blocks = getattr(diffusion_model, attr, None)
            if blocks is not None and len(blocks) > 0:
                keys = [f"diffusion_model.{attr}.{i}" for i in range(len(blocks))]
                logger.info(
                    "AnimaBlockCompile: per-block compile of diffusion_model.%s "
                    "(%d blocks)",
                    attr,
                    len(blocks),
                )
                break

        if not keys:
            logger.warning(
                "AnimaBlockCompile: no transformer block list found on "
                "diffusion_model (tried %s); falling back to whole-model compile.",
                ", ".join(_BLOCK_ATTR_CANDIDATES),
            )
            keys = ["diffusion_model"]

        set_torch_compile_wrapper(
            model=m,
            keys=keys,
            backend="inductor",
            options={"guard_filter_fn": _skip_transformer_options_guards},
        )
        return (m,)


NODE_CLASS_MAPPINGS = {
    "AnimaBlockCompile": AnimaBlockCompile,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaBlockCompile": "Anima Block Compile",
}
