"""Taking an image out of the pipeline, and putting it back.

An exclusion is a curation decision, not a stage: no model, no report, no diff to
agree to. It moves every file the image has out of the live workspace trees into
``workspace/_excluded/`` and writes down where each came from
(:data:`~anime_tools.workspace.EXCLUDED`); un-excluding moves them back.

| Half | What is in it |
|---|---|
| :mod:`._ledger` | the ledger — the state an exclusion *is*, and the rel that keys it |
| :mod:`._artifacts` | what an image is made of: every file that has to move |
| :mod:`._move` | the move engine — ``exclude_one`` / ``restore_one``, and the ``_many`` pair |
| :mod:`._cli` | ``python -m anime_tools.exclude`` |

The enforcement is one chokepoint: ``resize`` reads :func:`excluded_rels` into
its own ``--skip``, and every other stage walks ``workspace/resized/``, which an
excluded image has left. Torch-free.
"""

from __future__ import annotations

from library.curation_compat.exclude._artifacts import artifacts
from library.curation_compat.exclude._cli import build_parser, main
from library.curation_compat.exclude._ledger import (
    MANIFEST_VERSION,
    Entry,
    ExclusionError,
    Trees,
    excluded_rels,
    manifest_path,
    read_entries,
    rel_key,
    write_entries,
)
from library.curation_compat.exclude._move import (
    Result,
    exclude_many,
    exclude_one,
    restore_many,
    restore_one,
)

__all__ = [
    "MANIFEST_VERSION",
    "Entry",
    "ExclusionError",
    "Result",
    "Trees",
    "artifacts",
    "build_parser",
    "exclude_many",
    "exclude_one",
    "excluded_rels",
    "main",
    "manifest_path",
    "read_entries",
    "rel_key",
    "restore_many",
    "restore_one",
    "write_entries",
]
