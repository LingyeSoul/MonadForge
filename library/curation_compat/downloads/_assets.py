"""What a downloadable row *is*, and the fetch engine behind it. Torch-free.

An :class:`Asset` is one model: where its files come from, where they land, and
an offline probe for whether they are already there. :meth:`Asset.fetch` is the
only thing here that touches the network, and every failure it raises is a
``FileNotFoundError`` naming the asset and the recovery rather than a hub
traceback — the GUI prints it straight into a job log.

The rows themselves are :mod:`anime_tools.downloads._catalog`'s; a :class:`Pack`
is how a Models pane groups them under one button.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from library.curation_compat.downloads._locations import http_timeout

__all__ = ["PACKS", "PACK_BY_ID", "Asset", "Pack"]


def _say(msg: str) -> None:
    print(msg, flush=True)


def _size(n: int) -> str:
    return f"{n / 1e6:,.0f} MB" if n >= 1e6 else f"{n / 1e3:,.0f} KB"


@dataclass(frozen=True)
class Pack:
    """A group of rows that install together — what a Download button on a
    Models pane is a button *for*. Every :class:`Asset` names one."""

    id: str
    title: str
    description: str = ""


PACKS: tuple[Pack, ...] = (
    Pack(
        "tagger",
        "Tagger",
        "The Anima Tagger: its checkpoint and the gated dbv4 backbone.",
    ),
    Pack(
        "tags",
        "Danbooru tag DB",
        "The ~114k-row tag table caption correction types against, and its "
        "English descriptions.",
    ),
    Pack(
        "masking",
        "Masking",
        "SAM3 subject masks, and the subject soft prompt the position stages "
        "detect with.",
    ),
    Pack(
        "ocr",
        "OCR",
        "The AnimeText text-block detector and the manga VL reader "
        "(PaddleOCR-VL-1.6 base + the SFX fine-tune).",
    ),
    Pack("grouping", "Grouping", "PE-Spatial-B16-512, the near-twin grouping tower."),
)
"""Display order. A row's :attr:`Asset.pack` is one of these ids; the CLI and
both GUIs accept a pack id wherever they accept a row id."""

PACK_BY_ID: dict[str, Pack] = {p.id: p for p in PACKS}

REVISION_STAMP = "REVISION"
"""File a pinned ``dest`` row writes after a fetch: the Hub commit its files
came from, one line. :meth:`Asset.missing` compares it to :attr:`Asset.revision`."""


@dataclass(frozen=True)
class Asset:
    """One downloadable model, resolved against the current curation home."""

    id: str
    title: str
    repo: str
    files: tuple[str, ...]
    """Required files; all of them present means installed."""
    used_by: str
    """Which stages stop working without it."""
    stages: tuple[str, ...] = ()
    """The same, as GUI stage ids, so the stage bar can warn before a run."""
    pack: str = ""
    """Which :data:`PACKS` entry this row installs under. Every catalog row
    names one (pinned by test); the default only spares a caller building a
    row of its own."""
    dest: Path | None = None
    """Directory the files are flattened into; ``None`` = the HF hub cache."""
    url: str = ""
    """Base URL each file hangs off for rows that are not on the Hub. Set it and
    ``repo`` is only a label; requires ``dest``."""
    subfolder: str = ""
    """Path prefix inside the repo (``""`` = root; the AnimeText detector
    ships under one)."""
    repo_type: str = "model"
    """Hub repo kind — ``dataset`` for the Danbooru wiki mirror."""
    derived: tuple[str, ...] = ()
    """Files this row *makes* under ``dest`` via :attr:`build`. The probe asks
    for these, not the downloads, so a hub-cache sweep can't turn a built row
    back to "missing"."""
    build: Callable[[Path, Callable[[str], None]], None] | None = None
    """Post-fetch step that writes :attr:`derived` into ``dest``."""
    optional: tuple[str, ...] = field(default_factory=tuple)
    """Best-effort files: a 404 means this checkpoint doesn't ship one."""
    gated: str = ""
    """Accept-the-terms URL when the repo is gated; empty when it is public."""
    revision: str = ""
    """Hub commit the files are fetched at. Empty = ``main``, and the probe is
    file-existence only. Set, and a ``dest`` row is installed only when its
    :data:`REVISION_STAMP` names this commit — so moving the pin re-fetches an
    install whose files exist under the old name (a Hub ``main`` that moved
    under an existing install was otherwise invisible to :meth:`missing`)."""
    notes: str = ""

    @property
    def location(self) -> str:
        return str(self.dest) if self.dest is not None else "Hugging Face cache"

    def missing(self) -> list[str]:
        """Required files that are not here yet. Never touches the network."""
        if self.derived:
            assert self.dest is not None, "a built row needs a dest to write into"
            return [f for f in self.derived if not (self.dest / f).exists()]
        if self.dest is None:
            from library.curation_compat._hf import hf_file_cached

            return [f for f in self.files if not hf_file_cached(self.repo, f)]
        if self.revision and self._stamped_revision() != self.revision:
            return list(self.files)
        return [f for f in self.files if not (self.dest / Path(f).name).exists()]

    def _stamped_revision(self) -> str:
        """The commit the files under ``dest`` were fetched at, or ``""``."""
        assert self.dest is not None
        try:
            return (self.dest / REVISION_STAMP).read_text().strip()
        except OSError:
            return ""

    @property
    def installed(self) -> bool:
        return not self.missing()

    def to_dict(self) -> dict[str, Any]:
        missing = self.missing()
        return {
            "id": self.id,
            "title": self.title,
            "repo": self.repo,
            "files": list(self.files),
            "used_by": self.used_by,
            "stages": list(self.stages),
            "pack": self.pack,
            "location": self.location,
            "installed": not missing,
            "missing": missing,
            "gated": self.gated,
            "revision": self.revision,
            "notes": self.notes,
        }

    @property
    def _hint(self) -> str:
        return (
            f"hf auth login, then accept the terms at {self.gated}"
            if self.gated
            else f"python -m anime_tools.downloads {self.id}"
        )

    def _fetch_http(self, log: Callable[[str], None]) -> None:
        """Plain-HTTPS download for a ``url`` row. Bounded: a stalled socket
        must raise, not hang the job slot the GUI runs this in."""
        import urllib.error
        import urllib.request

        assert self.dest is not None, "a url row needs a dest to download into"
        for name in (*self.files, *self.optional):
            src = f"{self.url}/{name}"
            final = self.dest / Path(name).name
            log(f"  {src}")
            part = final.with_name(final.name + ".part")
            try:
                with urllib.request.urlopen(src, timeout=http_timeout()) as r:
                    part.write_bytes(r.read())
            except (OSError, urllib.error.URLError) as exc:
                part.unlink(missing_ok=True)
                if name in self.optional:
                    log(f"    optional — {type(exc).__name__}, skipped")
                    continue
                raise FileNotFoundError(
                    f"{self.title} ({name}): download from {src} failed "
                    f"({type(exc).__name__}: {exc}). Check connectivity, then "
                    f"re-run `{self._hint}`."
                ) from exc
            part.replace(final)
            log(f"    ok  {final}  ({_size(final.stat().st_size)})")

    def fetch(self, log: Callable[[str], None] = _say) -> None:
        """Download every required file; optional ones are best-effort.

        Network and gated-repo failures raise ``FileNotFoundError`` naming the
        asset and the recovery, not a hub traceback.
        """
        from huggingface_hub.utils import EntryNotFoundError

        from library.curation_compat._hf import hf_download

        if self.dest is not None:
            self.dest.mkdir(parents=True, exist_ok=True)
        if self.url:
            self._fetch_http(log)
            self._build(log)
            return

        hint = self._hint
        # A built row's downloads are inputs and stay in the hub cache; only
        # what ``build`` writes belongs in ``dest``.
        into = None if self.build else self.dest
        for name in (*self.files, *self.optional):
            remote = f"{self.subfolder}/{name}" if self.subfolder else name
            log(f"  {self.repo}/{remote}")
            try:
                got = Path(
                    hf_download(
                        what=f"{self.title} ({name})",
                        hint=hint,
                        repo_id=self.repo,
                        repo_type=self.repo_type,
                        filename=remote,
                        **({"local_dir": str(into)} if into else {}),
                        **({"revision": self.revision} if self.revision else {}),
                    )
                )
            except EntryNotFoundError:
                if name in self.optional:
                    log(f"    optional — not published by {self.repo}, skipped")
                    continue
                raise
            if into is not None:
                # local_dir keeps the repo's subfolder layout; the loaders want
                # a flat checkpoint dir.
                final = into / Path(name).name
                if got.resolve() != final.resolve():
                    shutil.move(str(got), str(final))
                got = final
            log(f"    ok  {got}  ({_size(got.stat().st_size)})")
        self._build(log)
        if into is not None and self.revision:
            (into / REVISION_STAMP).write_text(self.revision + "\n")
        if into is not None and self.subfolder:
            # Drop the now-empty subfolder local_dir the files were moved out of.
            leftover = into / self.subfolder
            if leftover.is_dir() and not any(leftover.iterdir()):
                leftover.rmdir()

    def _build(self, log: Callable[[str], None]) -> None:
        """Run the post-fetch step, if this row has one."""
        if self.build is None:
            return
        assert self.dest is not None, "a built row needs a dest to write into"
        self.build(self.dest, log)
