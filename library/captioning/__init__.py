"""Image-to-caption helpers used by editing/inversion paths.

Ships :class:`AnimaTagger` — trained on the Anima caption distribution,
the ψ_src provider for DirectEdit when a checkpoint is present at
``models/captioners/anima-tagger-v1/``.

Exposes ``predict(pil_img)`` and ``predict_caption(pil_img)`` for a
comma-separated tag string.
"""

# AnimaTagger's import touches torch/safetensors; expose it lazily (PEP 562) so torch-free siblings (notably library.captioning.taxonomy, used by the caption-index preprocess script) import without dragging torch through this __init__.

__all__ = ["AnimaTagger"]


def __getattr__(name: str):
    if name == "AnimaTagger":
        from library.captioning.anima_tagger import AnimaTagger

        return AnimaTagger
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
