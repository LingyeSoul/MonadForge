"""What an image is *made of* — every file an exclusion has to move. Torch-free.

One function per live tree, each matching on directory + stem rather than on the
rel itself, because resize re-encodes (``a.jpg`` → ``a.png``) and each tree
spells the tail its own way (``a.txt``, ``a_mask.png``, ``a.ocr.txt``). The
per-tree lookups are deliberately the same ones the readers use, so an exclusion
cannot miss a file a stage would still find.

:func:`artifacts` is the answer the move engine (:mod:`._move`) walks.
"""

from __future__ import annotations

import stat
from collections.abc import Iterator
from pathlib import Path

from library.curation_compat import workspace as WS
from library.curation_compat._walk import CAPTION_EXTENSIONS, IMAGE_EXTENSIONS
from library.curation_compat.captions.history import history_sidecar_path
from library.curation_compat.captions.ocr_sidecar import ocr_sidecar_path
from library.curation_compat.captions.variants import variants_sidecar_path
from library.curation_compat.exclude._ledger import Trees, rel_key
from library.curation_compat.masking._masks import mask_name

__all__ = ["artifacts"]


def _resized_files(root: Path, rel: Path) -> Iterator[Path]:
    """The resized image and every sidecar keyed to its stem.

    Matched on directory + stem, not on ``rel`` itself: resize re-encodes to PNG,
    so the master's ``a.jpg`` is ``a.png`` here. Spelled out rather than globbed
    ``a.*``, which would also sweep up the neighbouring image ``a.b.png``.
    """
    directory = root / rel.parent
    caption = directory / f"{rel.stem}.txt"
    for ext in (*IMAGE_EXTENSIONS, *CAPTION_EXTENSIONS):
        yield directory / f"{rel.stem}{ext}"
    yield variants_sidecar_path(caption)
    yield history_sidecar_path(caption)


def _mask_files(root: Path, rel: Path) -> Iterator[Path]:
    """The mirrored mask, and the legacy flat one — the same two-step lookup
    ``gui.dataset.mask_path`` and Export's ``_mask_source`` do."""
    yield root / rel.parent / mask_name(rel.stem)
    yield root / mask_name(rel.stem)


def _ocr_files(root: Path, rel: Path) -> Iterator[Path]:
    yield ocr_sidecar_path(root / rel.parent / f"{rel.stem}.txt")


_FILES = {"resized": _resized_files, "masks": _mask_files, "ocr": _ocr_files}


def _slot(trees: Trees, tree: str, path: Path) -> str:
    return f"{tree}/{path.relative_to(trees.live(tree)).as_posix()}"


def artifacts(trees: Trees, rel: str) -> list[tuple[str, Path]]:
    """``[(slot, live path)]`` for every file this image has *right now*.

    Deduplicated by what the path resolves *to*, not by how it is spelled: the
    mask lookup offers a mirrored and a flat path, which are one file for an
    image at the root of the tree, and on a case-insensitive filesystem
    ``a.png`` and ``a.PNG`` — both in ``IMAGE_EXTENSIONS`` — are one file under
    two names. The first spelling wins, and the candidates are ordered so that is
    the lowercase one.
    """
    key = Path(rel_key(rel))
    out: list[tuple[str, Path]] = []
    seen: set[tuple[int, int]] = set()
    for tree in WS.EXCLUDED_TREES:
        for path in _FILES[tree](trees.live(tree), key):
            try:
                st = path.stat()
            except OSError:  # not there, or not reachable — the same answer here
                continue
            if not stat.S_ISREG(st.st_mode):
                continue
            ident = (st.st_dev, st.st_ino)
            if ident in seen:
                continue
            seen.add(ident)
            out.append((_slot(trees, tree, path), path))
    return out
