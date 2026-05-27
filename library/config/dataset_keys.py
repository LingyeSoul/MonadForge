"""Torch-free allow-lists for the dataset blueprint (``[general]`` /
``[[datasets]]`` / ``[[datasets.subsets]]``).

These mirror the voluptuous schemas built by
:class:`library.config.loader.ConfigSanitizer` (with ``support_dropout=True``,
which is what ``train.py`` uses). They live here — not in ``loader`` — so the
GUI can lint a config for unknown dataset keys *without* importing the training
stack (``loader`` pulls in ``library.datasets`` → torch).

``tests/test_dataset_keys.py`` asserts these stay in sync with the real
sanitizer schemas, so this file can't silently drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from library.config.schema import find_line

# ── Leaf key groups (mirror ConfigSanitizer class attributes) ──────────────
_DATASET_ASCENDABLE_KEYS = frozenset(
    {
        "batch_size",
        "validation_seed",
        "validation_split",
        "validation_split_num",
        "network_multiplier",
        "resize_interpolation",
    }
)
_SUBSET_ASCENDABLE_KEYS = frozenset(
    {
        "color_aug",
        "face_crop_aug_range",
        "flip_aug",
        "num_repeats",
        "sample_ratio",
        "random_crop",
        "keep_tokens",
        "keep_tokens_separator",
        "secondary_separator",
        "caption_separator",
        "enable_wildcard",
        "token_warmup_min",
        "token_warmup_step",
        "caption_prefix",
        "caption_suffix",
        "custom_attributes",
        "resize_interpolation",
        "path_pattern",
    }
)
_DB_SUBSET_ASCENDABLE_KEYS = frozenset({"caption_extension", "class_tokens", "cache_info"})
_DB_SUBSET_DISTINCT_KEYS = frozenset(
    {"image_dir", "is_reg", "alpha_mask", "cache_dir", "mask_dir", "recursive"}
)
_DO_SUBSET_KEYS = frozenset(
    {
        "caption_dropout_every_n_epochs",
        "caption_dropout_rate",
        "caption_tag_dropout_rate",
    }
)

# ── Composite allow-lists per section level ────────────────────────────────
GENERAL_KEYS = (
    _DATASET_ASCENDABLE_KEYS
    | _SUBSET_ASCENDABLE_KEYS
    | _DB_SUBSET_ASCENDABLE_KEYS
    | _DO_SUBSET_KEYS
)
#: Keys allowed directly on a ``[[datasets]]`` table (general keys + ``subsets``).
DATASET_TABLE_KEYS = GENERAL_KEYS | {"subsets"}
#: Keys allowed on a ``[[datasets.subsets]]`` table.
SUBSET_KEYS = (
    _SUBSET_ASCENDABLE_KEYS
    | _DB_SUBSET_DISTINCT_KEYS
    | _DB_SUBSET_ASCENDABLE_KEYS
    | _DO_SUBSET_KEYS
)


@dataclass(frozen=True)
class DatasetKeyIssue:
    """An unknown key found in a dataset-blueprint section."""

    key: str
    section: str  # e.g. "general", "datasets[0]", "datasets[0].subsets[1]"
    source: Optional[str] = None  # file path, for display
    line: Optional[int] = None

    @property
    def location(self) -> str:
        loc = self.source or "<config>"
        return f"{loc}:{self.line}" if self.line is not None else loc

    def __str__(self) -> str:
        return (
            f"{self.location}: unknown dataset key {self.key!r} in [{self.section}] "
            f"(will fail training validation)"
        )


def lint_dataset_sections(
    raw: dict, *, source: Optional[str] = None, text: Optional[str] = None
) -> list[DatasetKeyIssue]:
    """Return unknown keys in the ``[general]`` / ``[[datasets]]`` /
    ``[[datasets.subsets]]`` sections of a parsed TOML mapping.

    ``text`` (the raw file contents) is used only to attach best-effort line
    numbers; pass it when available. Keys that *are* valid, and any non-dataset
    top-level sections, are ignored.
    """
    issues: list[DatasetKeyIssue] = []

    def _check(table: Any, allowed: frozenset, section: str) -> None:
        if not isinstance(table, dict):
            return
        for key in table:
            if key not in allowed:
                issues.append(
                    DatasetKeyIssue(
                        key=key,
                        section=section,
                        source=source,
                        line=find_line(text, key),
                    )
                )

    _check(raw.get("general"), GENERAL_KEYS, "general")

    datasets = raw.get("datasets")
    if isinstance(datasets, list):
        for di, ds in enumerate(datasets):
            _check(ds, DATASET_TABLE_KEYS, f"datasets[{di}]")
            if isinstance(ds, dict) and isinstance(ds.get("subsets"), list):
                for si, sub in enumerate(ds["subsets"]):
                    _check(sub, SUBSET_KEYS, f"datasets[{di}].subsets[{si}]")

    return issues
