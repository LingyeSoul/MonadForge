"""The torch-free dataset-key allow-lists in ``library.config.dataset_keys``
must stay in sync with the real voluptuous schemas built by
``ConfigSanitizer`` — otherwise the GUI lint would report false positives /
miss real ones. This test ties them together so they can't drift.
"""

from library.config import dataset_keys as dk
from library.config.loader import ConfigSanitizer


def _schema_keys(schema: dict) -> set:
    """Voluptuous schema dicts key on plain strings or Marker objects
    (``Required("image_dir")``). Markers carry the real key on ``.schema``."""
    out = set()
    for k in schema:
        out.add(getattr(k, "schema", k))
    return out


def test_dataset_key_allowlists_match_sanitizer():
    # train.py builds the sanitizer with support_dropout=True.
    s = ConfigSanitizer(support_dropout=True)

    assert dk.GENERAL_KEYS == _schema_keys(s.general_schema)
    # dataset_schema adds the nested `subsets` array key on top of general.
    assert dk.DATASET_TABLE_KEYS == _schema_keys(s.dataset_schema)
    assert dk.SUBSET_KEYS == _schema_keys(s.db_subset_schema)


def test_lint_flags_unknown_dataset_key():
    raw = {
        "general": {"caption_extension": ".txt"},
        "datasets": [
            {
                "batch_size": 1,
                "resolution": 1024,  # removed from the schema — should flag
                "subsets": [{"image_dir": "x", "bogus_subset_key": 1}],
            }
        ],
    }
    issues = dk.lint_dataset_sections(raw, source="base.toml")
    flagged = {(i.key, i.section) for i in issues}
    assert ("resolution", "datasets[0]") in flagged
    assert ("bogus_subset_key", "datasets[0].subsets[0]") in flagged
    # valid keys are not flagged
    assert not any(i.key in {"batch_size", "caption_extension", "image_dir"} for i in issues)
