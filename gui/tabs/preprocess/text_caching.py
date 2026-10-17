"""Text-caching section: caption shuffle variants + tag dropout rate."""

from __future__ import annotations

from gui.i18n import t
from gui.tabs.preprocess._section import KnobSection, line, spin
from gui.tabs.preprocess.knobs import (
    DEFAULT_TE_SHUFFLE_VARIANTS,
    DEFAULT_TE_TAG_DROPOUT,
)


class TextCachingSection(KnobSection):
    def __init__(self, help_cb, *, settings: dict):
        self._settings = settings
        super().__init__(t("preprocess_text_caching"), help_cb)

    def _build(self) -> None:
        s = self._settings
        self.add_knob(
            "caption_shuffle_variants",
            spin(
                0,
                64,
                int(s.get("caption_shuffle_variants", DEFAULT_TE_SHUFFLE_VARIANTS)),
            ),
            t("preprocess_caption_shuffle_variants"),
            tooltip="",
        )
        # Free text (not a spin) so the exact user string reaches the env export.
        self.add_knob(
            "caption_tag_dropout_rate",
            line(
                f"{float(s.get('caption_tag_dropout_rate', DEFAULT_TE_TAG_DROPOUT)):g}"
            ),
            t("preprocess_caption_tag_dropout_rate"),
            tooltip="",
        )
