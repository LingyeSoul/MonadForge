"""Image-to-caption helpers used by editing/inversion paths.

Ships :class:`AnimaTagger` — trained on the Anima caption distribution,
the ψ_src provider for DirectEdit when a checkpoint is present at
``models/captioners/anima-tagger-v1/``.

Exposes ``predict(pil_img)`` and ``predict_caption(pil_img)`` for a
comma-separated tag string.
"""

# ``AnimaTagger`` lives in ``anima_tagger``, whose import touches
# torch/safetensors. Expose it lazily (PEP 562) so that
# ``from library.captioning import AnimaTagger`` still works, while torch-free
# siblings — notably ``library.captioning.taxonomy`` (pure-stdlib tag-shape
# primitives, imported by the caption-index preprocessing script) — can be
# imported without dragging torch in through this package ``__init__``.
# Callers in environments without a built checkpoint still handle the
# ``FileNotFoundError`` raised by ``AnimaTagger.__init__`` at construction time.

__all__ = ["AnimaTagger"]


def __getattr__(name: str):
    if name == "AnimaTagger":
        from library.captioning.anima_tagger import AnimaTagger

        return AnimaTagger
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
