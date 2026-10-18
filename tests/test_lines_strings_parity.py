"""The sidecar lines' Basic fields keep Chinese labels and Anima's tiers.

Two invariants the MOSS / Qwen-Image-2.1 forms share with the Anima tab:

* a field a form shows under **Basic** (no ``advanced=True`` in its metadata) has a
  Chinese label in ``gui/<line>/strings.py`` — English falls back to the request's own
  ``help`` metadata, so a missing entry silently shows English text in the cn UI;
* the tier of the knobs the lines share matches Anima (``gui.core.config_io._BASIC``
  is the Basic set of the training tab).

Plain dataclass metadata and dicts only: no Qt, no torch.
"""

from __future__ import annotations

import dataclasses

import pytest

from gui.core.config_io import _BASIC
from gui.moss import strings as moss_strings
from gui.qwen21 import strings as qwen21_strings
from library.moss import requests as moss_requests
from library.qwen21 import requests as qwen21_requests

# ``line.Class`` → the request class, its line's FIELDS_CN, and the Test form's own
# FIELDS_CN_GENERATE (None for the Cache / Train forms, which only read FIELDS_CN).
_FORMS: dict[str, tuple[type, dict, dict | None]] = {
    "moss.CacheRequest": (moss_requests.CacheRequest, moss_strings.FIELDS_CN, None),
    "moss.TrainRequest": (moss_requests.TrainRequest, moss_strings.FIELDS_CN, None),
    "moss.GenerateRequest": (
        moss_requests.GenerateRequest,
        moss_strings.FIELDS_CN,
        moss_strings.FIELDS_CN_GENERATE,
    ),
    "qwen21.CacheRequest": (
        qwen21_requests.CacheRequest,
        qwen21_strings.FIELDS_CN,
        None,
    ),
    "qwen21.TrainRequest": (
        qwen21_requests.TrainRequest,
        qwen21_strings.FIELDS_CN,
        None,
    ),
    "qwen21.GenerateRequest": (
        qwen21_requests.GenerateRequest,
        qwen21_strings.FIELDS_CN,
        qwen21_strings.FIELDS_CN_GENERATE,
    ),
}

# ``line.Class.field`` → why that Basic field deliberately has no Chinese label.
# Empty: every non-advanced field is labelled today. A name may only be added here
# with a concrete reason (English ``help`` is the only text the cn form shows for it),
# never as a blanket skip — ``test_unlabelled_basic_whitelist_has_no_stale_entries``
# fails once such an entry stops applying.
_UNLABELLED_BASIC: dict[str, str] = {}

# The model-path overrides both lines keep in Advanced: Anima's equivalents
# (pretrained_model_name_or_path / qwen3 / vae) are Advanced, MOSS's model_dir too.
_QWEN21_ADVANCED_PATHS = ("model_dir", "dit", "text_encoder", "vae", "processor")
_QWEN21_CLASSES = (
    qwen21_requests.CacheRequest,
    qwen21_requests.TrainRequest,
    qwen21_requests.GenerateRequest,
)


def _fields(cls) -> dict[str, dataclasses.Field]:
    return {f.name: f for f in dataclasses.fields(cls)}


def _is_advanced(f: dataclasses.Field) -> bool:
    return bool(f.metadata.get("advanced"))


def _labelled(fields_cn: dict, fields_cn_test: dict | None) -> dict:
    return fields_cn if fields_cn_test is None else {**fields_cn, **fields_cn_test}


@pytest.mark.parametrize("form", sorted(_FORMS))
def test_every_basic_field_has_a_chinese_label(form):
    cls, fields_cn, fields_cn_test = _FORMS[form]
    labelled = _labelled(fields_cn, fields_cn_test)
    missing = [
        f.name
        for f in dataclasses.fields(cls)
        if not _is_advanced(f)
        and f.name not in labelled
        and f"{form}.{f.name}" not in _UNLABELLED_BASIC
    ]
    assert missing == [], (
        f"{form}: Basic fields without a Chinese label: {missing}. Add them to "
        f"gui/{form.split('.')[0]}/strings.py (FIELDS_CN, or FIELDS_CN_GENERATE for the "
        "Test form), or list them in _UNLABELLED_BASIC with a reason — the cn form "
        "would otherwise show the English help for them"
    )


def test_unlabelled_basic_whitelist_has_no_stale_entries():
    """A whitelisted field that is gone, Advanced again or now labelled must be removed."""
    stale = []
    for entry in _UNLABELLED_BASIC:
        form, _, name = entry.rpartition(".")
        cls, fields_cn, fields_cn_test = _FORMS[form]
        field = _fields(cls).get(name)
        if (
            field is None
            or _is_advanced(field)
            or name in _labelled(fields_cn, fields_cn_test)
        ):
            stale.append(entry)
    assert stale == [], (
        f"_UNLABELLED_BASIC entries that no longer apply: {stale}. Drop them so the "
        "whitelist keeps documenting real exemptions only"
    )


def test_qwen21_model_paths_stay_advanced():
    fields = {cls: _fields(cls) for cls in _QWEN21_CLASSES}
    basic = [
        f"{cls.__name__}.{name}"
        for cls in _QWEN21_CLASSES
        for name in _QWEN21_ADVANCED_PATHS
        if fields[cls].get(name) is None or not _is_advanced(fields[cls][name])
    ]
    assert basic == [], (
        f"Qwen-Image-2.1 model-path fields that left Advanced: {basic}. Anima keeps "
        "pretrained_model_name_or_path / qwen3 / vae in Advanced and MOSS keeps "
        "model_dir there; these paths are per-machine, not per-run, so the three tabs "
        "must agree"
    )


def test_blocks_to_swap_is_basic_on_both_train_requests():
    """Anima lists blocks_to_swap under Basic; the two lines' Train tabs follow.

    Only the Train forms: the generate side's swap knob is a VRAM-tuning override and
    stays Advanced on both lines.
    """
    assert "blocks_to_swap" in _BASIC, (
        "Anima's Basic set is what the sidecar lines mirror, and it must keep "
        "blocks_to_swap"
    )
    advanced = [
        f"{cls.__name__}.blocks_to_swap"
        for cls in (moss_requests.TrainRequest, qwen21_requests.TrainRequest)
        if _is_advanced(_fields(cls)["blocks_to_swap"])
    ]
    assert advanced == [], (
        f"blocks_to_swap must be Basic (no advanced=True) on both Train requests, "
        f"like Anima's _BASIC: {advanced}"
    )
