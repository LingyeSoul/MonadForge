"""What every mask generator does around the model: plan, run, write, read.

The flags themselves are the fields of ``masking.requests.MaskWalkRequest``; this
module reads them back by attribute in :func:`mask_run`.
"""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from library.curation_compat._env import resolve_path
from library.curation_compat._progress import ProgressBar
from library.curation_compat._walk import walk_images

MASK_SUFFIX = "_mask.png"
"""The contract between the generators, ``merge_masks`` and the GUI's mask lookup:
this suffix plus "mirror the source subdir"."""


def mask_name(stem: str) -> str:
    return f"{stem}{MASK_SUFFIX}"


# ---- the write side ----------------------------------------------------


def mask_path_for(image_path: Path, image_dir: Path, mask_dir: Path) -> Path:
    """Where ``image_path``'s mask belongs: the source subdir, mirrored.

    An image outside ``image_dir`` (which ``walk_images`` will not hand back,
    but a direct caller might) lands flat at the root.
    """
    try:
        rel = image_path.parent.relative_to(image_dir)
    except ValueError:
        rel = Path("")
    target_dir = mask_dir if str(rel) in ("", ".") else mask_dir / rel
    return target_dir / mask_name(image_path.stem)


def plan_mask_jobs(
    image_dir: Path,
    mask_dir: Path,
    *,
    recursive: bool = False,
    pattern: str | None = None,
    force: bool = False,
) -> list[tuple[Path, Path]]:
    """``[(image_path, mask_path)]`` for the masks still to write.

    ``walk_images`` raises on same-stem collisions *within* one folder, which
    would have the two images overwrite each other's mask; the same stem across
    folders is fine, since the mirrored layout disambiguates it. Output
    directories are created here, so the caller's write loop is a plain save.
    """
    jobs: list[tuple[Path, Path]] = []
    for image_path in walk_images(image_dir, recursive=recursive, pattern=pattern):
        mask_path = mask_path_for(image_path, image_dir, mask_dir)
        if mask_path.exists() and not force:
            continue
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        jobs.append((image_path, mask_path))
    return jobs


def save_mask(path: Path, alpha_mask: np.ndarray) -> None:
    """Thread-safe enough for the I/O pool."""
    Image.fromarray(alpha_mask, mode="L").save(path)


def write_mask(path: Path, keep: np.ndarray, *, pool=None):
    """Save ``keep`` (1 = train here) as the 8-bit alpha mask ``keep * 255``.

    With ``pool``, the save is submitted to it and the future returned; without
    one it is written inline and ``None`` comes back.
    """
    alpha_mask = (np.asarray(keep) * 255).astype(np.uint8)
    if pool is None:
        save_mask(path, alpha_mask)
        return None
    return pool.submit(save_mask, path, alpha_mask)


def write_ignore_mask(path: Path, detected: np.ndarray, *, pool=None):
    """Save the *inverse* of a detection mask, the polarity the trainer reads.

    detected=1 → alpha=0 (ignored in the loss), no detection → alpha=255 (trained on).
    """
    return write_mask(path, 1 - np.asarray(detected), pool=pool)


# ---- the run -----------------------------------------------------------


def coverage_pct(mask: np.ndarray) -> float:
    """What share of the frame ``mask`` covers, as a percentage.

    Denominated in the mask's *own* size rather than a ``(w, h)`` passed alongside it.
    Polarity-blind: the generators call it on a keep mask and on an ignore mask, and what
    the number is called on the progress line is theirs to say.
    """
    return 100 * np.count_nonzero(mask) / mask.size


@dataclass(slots=True)
class MaskRun:
    """One generator's pass: where it reads, where it writes, what is left to do.

    ``items`` is the plan (:func:`plan_mask_jobs`) — a list rather than an iterator
    because a batching generator indexes ahead of the loop to prefetch. ``pool`` is the
    shared I/O pool ``write_mask`` submits saves to; the progress bar is private, reached
    through the two methods below.
    """

    image_dir: Path
    mask_dir: Path
    items: list[tuple[Path, Path]]
    pool: ThreadPoolExecutor
    workers: int
    """The pool's size, already clamped to at least 1 — the one answer to "how
    far ahead may I prefetch?", so a generator never re-reads ``--workers`` and
    never computes a prefetch depth of zero."""
    _bar: ProgressBar

    @property
    def total(self) -> int:
        return len(self.items)

    def advance(self) -> None:
        """One image dealt with — called before the branches, so an image that
        gets no mask still moves the count."""
        self._bar.advance()

    def note(self, image_path: Path, what: str) -> None:
        """The image's own ``  [n/total] name: what`` line. ``what`` is the
        stage's own wording (``train 41.2%``, ``skipped``, ``focus not found``).
        The shared bar, not ``tqdm``: carriage returns are noise in the GUI's
        log and match nothing its progress bar reads."""
        self._bar.note(f"{image_path.name}: {what}")


@contextmanager
def mask_run(args: Any, *, desc: str = "Generating masks") -> Iterator[MaskRun]:
    """The scaffolding a generator wraps its inner loop in.

    ``args`` is anything carrying the walk attributes of
    ``masking.requests.MaskWalkRequest`` (``image_dir`` / ``mask_dir`` / ``recursive`` /
    ``path_pattern`` / ``force`` / ``workers``) — the request itself, or a parsed
    namespace.

    Both roots are home-anchored (the ``--mask-dir`` defaults are home-relative, so a run
    from another directory still means the tree the GUI and the merge do), the output root
    exists before the first write, and the plan comes from the request's walk fields.

    Draining the saves is the caller's job, inside the ``with``: a future whose exception
    nobody reads is a mask that silently did not get written.

    Nothing to do is not an error — ``items`` is empty, the caller's loop does not run,
    and the closing line says so.
    """
    image_dir = resolve_path(args.image_dir)
    mask_dir = resolve_path(args.mask_dir)
    mask_dir.mkdir(parents=True, exist_ok=True)

    items = plan_mask_jobs(
        image_dir,
        mask_dir,
        recursive=args.recursive,
        pattern=args.path_pattern,
        force=args.force,
    )

    # At least one: ``ThreadPoolExecutor(0)`` refuses outright, and a prefetch
    # depth of zero leaves a batching generator indexing an empty list.
    workers = max(1, int(args.workers))
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        yield MaskRun(
            image_dir, mask_dir, items, pool, workers, ProgressBar(len(items))
        )
    finally:
        pool.shutdown()
    # Past the `finally`, so a run that raised does not sign off as if it had not.
    print(f"Masks saved to {mask_dir}/" if items else "No images to process.")


# ---- the read side -----------------------------------------------------


def iter_masks(mask_dir: Path):
    """``(rel_dir, path)`` for every mask under ``mask_dir``, recursively.

    ``rel_dir`` is the mirrored source subdir as a string, ``""`` at the root — the key
    half of ``merge_masks``' ``(rel_dir, name)`` collision rule, so two inputs merge only
    when the mask sits at the same relative path in both.
    """
    for p in sorted(mask_dir.rglob(f"*{MASK_SUFFIX}")):
        rel = p.parent.relative_to(mask_dir)
        yield ("" if str(rel) in ("", ".") else str(rel)), p
