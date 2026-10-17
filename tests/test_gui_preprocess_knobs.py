"""Unit tests for the Qt-free knob table (``gui/tabs/preprocess/knobs.py``).

The table holds only the trainer-native rows — dataset roots / scope and the
TE-cache variant knobs (the curation chain gates left with the tab's curation
sections; the env pins them off); the stage forms are ``stage_form``'s and
tested in ``test_gui_stage_form.py``.
Feeds the pure functions hand-built value dicts and checks them against the
characterization fixture — so the table reproduces the tab's contract
without constructing a single widget.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gui.tabs.preprocess import knobs as K
from tests.test_gui_preprocess_characterization import FIXTURE, SCENARIOS

EXPECTED = json.loads(Path(FIXTURE).read_text(encoding="utf-8"))

# The widget values `_flip_every_knob` in the characterization test produces
# (trainer knobs only; the stage forms are flipped there too but live under
# ``meta["stages"]`` and the ``PREPROCESS_STAGES_JSON`` env, not here).
FLIPPED_VALUES = {
    "source_image_dir": "flipped_images",
    "path_scope": "artist_a",
    "preprocess_path_pattern": "artist_a/**",
    "caption_shuffle_variants": 11,
    "caption_tag_dropout_rate": "0.45",
}


def _defaults(scenario: str) -> dict:
    s = SCENARIOS[scenario]
    return K.resolved_defaults(s["preprocess_toml"], s["gui_settings"])


def _default_widget_values(scenario: str) -> dict:
    """What the widgets hold right after ``set_variant`` on an empty variant."""
    values = K.load_values({}, _defaults(scenario))
    # Free-text numerics are shown with :g and read back as text.
    values["caption_tag_dropout_rate"] = (
        f"{float(values['caption_tag_dropout_rate']):g}"
    )
    return values


def _flipped_widget_values(scenario: str) -> dict:
    values = _default_widget_values(scenario)
    values.update(FLIPPED_VALUES)
    return values


def _roundtrip(data):
    return json.loads(json.dumps(data, sort_keys=True))


def _env_without_stages(env: dict) -> dict:
    return {k: v for k, v in env.items() if k != "PREPROCESS_STAGES_JSON"}


@pytest.mark.parametrize("scenario", list(SCENARIOS))
@pytest.mark.parametrize("state", ["defaults", "flipped"])
def test_env_and_overrides_match_fixture(scenario, state):
    values = (
        _default_widget_values(scenario)
        if state == "defaults"
        else _flipped_widget_values(scenario)
    )
    expected = EXPECTED[scenario][state]
    assert K.to_env(values, _defaults(scenario)) == _env_without_stages(expected["env"])
    assert _roundtrip(K.to_overrides(values)) == expected["overrides"]


def _persistable(values: dict) -> dict:
    """The tab validates the free-text numerics before persisting."""
    values = dict(values)
    values["caption_tag_dropout_rate"] = float(values["caption_tag_dropout_rate"])
    return values


def _flat(meta: dict) -> dict:
    """The fixture's meta minus the stage tables (``stage_form``'s)."""
    return {k: v for k, v in meta.items() if k != K.STAGES_KEY}


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_merge_into_meta_matches_fixture(scenario):
    """Same save sequence as the characterization run, on one meta table:
    inputs-only save, full save, flip, inputs-only save, full save."""
    defaults = _defaults(scenario)
    expected = EXPECTED[scenario]
    meta = {"family": "lora"}
    for state, values in (
        ("defaults", _persistable(_default_widget_values(scenario))),
        ("flipped", _persistable(_flipped_widget_values(scenario))),
    ):
        K.merge_into_meta(meta, values, defaults)
        assert _roundtrip(meta) == _flat(expected[state]["meta_inputs_only"]), state
        K.merge_into_meta(meta, values, defaults)
        assert _roundtrip(meta) == _flat(expected[state]["meta_full"]), state


def test_load_values_is_a_fixed_point_of_merge_on_bare_checkout():
    """Save → load reproduces the flipped values (the tab's reload invariant)."""
    defaults = _defaults("bare")
    values = _persistable(_flipped_widget_values("bare"))
    meta = K.merge_into_meta({}, values, defaults)
    loaded = K.load_values(meta, defaults)
    for knob in K.KNOBS:
        assert K._coerce(knob, loaded[knob.key]) == K._coerce(knob, values[knob.key]), (
            knob.key
        )


def test_retired_knob_loads_as_nothing_and_is_dropped_on_save():
    """A variant saved by an older GUI still carries ``drop_lowres_images`` or
    a curation chain gate: none is a knob any more, so each loads as nothing,
    the next save pops it, and ConfigTab still strips it from training."""
    defaults = _defaults("populated")
    stale = {key: True for key in K.RETIRED_KEYS}
    loaded = K.load_values(stale, defaults)
    values = _persistable(_flipped_widget_values("populated"))
    meta = K.merge_into_meta(dict(stale), values, defaults)
    for key in K.RETIRED_KEYS:
        assert key not in loaded and key not in meta, key
        assert key in K.PREPROCESS_ONLY_KEYS, key


def test_env_pins_the_curation_gates_off():
    """A ``preprocess.toml`` turning a curation stage on never reaches a GUI
    run: env beats the config chain in ``tasks.py``."""
    pp = {"caption_position_clauses": True, "caption_autotag": True}
    defaults = K.resolved_defaults(pp, {})
    env = K.to_env(K.load_values({}, defaults), defaults)
    assert env["CAPTION_AUTOTAG"] == "0"
    assert env["CAPTION_POSITION_CLAUSES"] == "0"


def test_elision_keeps_a_plain_checkout_empty():
    """All-defaults on a bare checkout writes nothing (the tiers, always
    written, live in the resize stage's table now)."""
    defaults = _defaults("bare")
    meta = K.merge_into_meta({}, _default_widget_values("bare"), defaults)
    assert meta == {}


def test_table_invariants():
    keys = [k.key for k in K.KNOBS]
    assert len(keys) == len(set(keys))
    assert K.PREPROCESS_ONLY_KEYS == set(keys) | {K.STAGES_KEY} | set(K.RETIRED_KEYS)
    for knob in K.KNOBS:
        assert knob.enabled_by is None or knob.enabled_by in K.KNOBS_BY_KEY, knob.key
    env_names = [k.env for k in K.ENV_KNOBS]
    assert len(env_names) == len(set(env_names))
    # Nothing a stage form shows is a knob row any more (the trainer-owned
    # dests it hides — the variant sidecar knobs — are exactly the overlap).
    from gui.tabs.preprocess.stage_form import (
        STAGE_IDS,
        load_stage_schemas,
        visible_fields,
    )

    schemas = load_stage_schemas()
    shown = {f["dest"] for sid in STAGE_IDS for f in visible_fields(schemas[sid])}
    assert not shown & set(keys)
