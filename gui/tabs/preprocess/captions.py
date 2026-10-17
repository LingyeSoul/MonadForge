"""Caption-mirror section drawn from the ``correct`` stage schema: order
correction, @no-artist, trigger word, tag groups to drop. The stage writes the
revised caption (and its variant sidecars) TE encodes; curation stages that
rewrite captions (autotag, position clauses) run from the ``anime_tools``
panel."""

from __future__ import annotations

from gui.i18n import t
from gui.tabs.preprocess.stage_form import StageFormSection


class CaptionEditingSection(StageFormSection):
    def __init__(self, schema: dict, help_cb, *, defaults: dict):
        super().__init__(
            schema,
            help_cb,
            title=t("preprocess_caption_editing"),
            defaults=defaults,
        )
