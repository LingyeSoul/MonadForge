"""Anima ComfyUI custom nodes.

Three single-purpose loader nodes; chain them via the MODEL socket when
a workflow needs more than one:

  - ``AnimaAdapterLoader`` — LoRA / HydraLoRA / ReFT (auto-detected
    from safetensors keys). HydraLoRA supports both σ-conditional and
    FeRA-style FEI-conditional live routing on the Hydra stack.
  - ``AnimaFeraLoader`` — author-faithful FeRA (Yin et al.,
    arXiv:2511.17979): global router on the latent's spectral energy +
    per-Linear stacked independent experts. Incompatible save format
    with the FEI-on-Hydra variant above; mutually exclusive with
    HydraLoRA-moe at load time.
  - ``AnimaSoftTokensLoader`` — SoftREPA-parameterization soft tokens
    (Lee et al., arXiv:2503.08250): per-layer × per-timestep-bucket
    learned tokens spliced into the crossattn embedding inside the first
    n_layers DiT blocks via per-block forward pre-hooks.

``AnimaFeraLoader`` was added in v3.1.0; ``AnimaSoftTokensLoader`` in
v3.6.0. The ``AnimaPostfixLoader`` node was retired when the postfix
training method was archived (no live trainer).
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
