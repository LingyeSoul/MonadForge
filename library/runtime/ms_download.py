"""Bounded, fail-fast ModelScope (modelscope.cn) downloads.

Zero-dependency counterpart of ``hf_download``: talks to ModelScope's public
resolve URLs (``https://modelscope.cn/models/<repo>/resolve/<revision>/<path>``
— verified live 2026-09, including ``Range``/206 resume behind the 302 to the
CDN) and the repo-files listing API with nothing but urllib, so no
``modelscope`` SDK install is required. This exists because huggingface.co is
slow or unreachable from some networks where ModelScope is fast; the source is
selected with ``--source modelscope`` (task CLIs) or
``ANIMA_DOWNLOAD_SOURCE=modelscope`` (env, honored by auto-fetch too).

Verified mirrors of this project's assets: ``circlestone-labs/Anima`` (the
official repo, same ``split_files/`` layout as HF), ``facebook/sam3`` (public
there, unlike the gated HF repo) and ``AI-ModelScope/PE-Core-L14-336``.
An asset without an entry in ``MIRRORS`` is never silently re-fetched from
HuggingFace while the ModelScope source is selected — task CLIs print a skip
notice, auto-fetch (``maybe_ms_download``) raises.

Guards mirror ``hf_download``: pinned socket timeouts so a stalled connection
raises in seconds (the daemon job queue must never wedge on a download), and
transport failures become actionable ``FileNotFoundError`` exceptions.
Downloads are
atomic (``.part`` temp, rename on completion) and resumable (Range); complete
files are skipped by exact size so a re-run verifies instead of re-fetching
gigabytes (GH #21 idempotency contract).

Known limits: the repo-files listing endpoint returns a single page
(``PageSize`` is ignored), so repos with thousands of files are not supported
— every asset this project targets is well under 100 files. Authentication for
gated repos: set ``MODELSCOPE_TOKEN``; note urllib forwards headers (including
Authorization) across the 302 to the CDN host, so only set the token when the
target repo actually requires it.
"""

from __future__ import annotations

import fnmatch
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# Per-request socket timeout (connect + read) for ModelScope traffic, in
# seconds. Tunable via ANIMA_MS_TIMEOUT; bounds a fully stalled connection (a
# slow trickle is caught by the daemon stall watchdog, as with hf downloads).
_CHUNK = 1 << 20

# HuggingFace repo id -> ModelScope repo id, for the assets this project
# downloads. Values verified against the ModelScope API (repo exists + file
# layout matches). A repo missing here has no known mirror.
MIRRORS: dict[str, str] = {
    "circlestone-labs/Anima": "circlestone-labs/Anima",
    "facebook/sam3": "facebook/sam3",
    "facebook/PE-Core-L14-336": "AI-ModelScope/PE-Core-L14-336",
}


def _base() -> str:
    return os.environ.get("ANIMA_MS_ENDPOINT", "https://modelscope.cn").rstrip("/")


def source_enabled() -> bool:
    """True when ``ANIMA_DOWNLOAD_SOURCE=modelscope`` selects ModelScope."""
    return os.environ.get("ANIMA_DOWNLOAD_SOURCE", "").strip().lower() == "modelscope"


def mirror_repo(hf_repo_id: str) -> str | None:
    """ModelScope repo mirroring ``hf_repo_id``, or None when unknown."""
    return MIRRORS.get(hf_repo_id)


def resolve_url(repo_id: str, path: str, revision: str = "master") -> str:
    """Public download URL of one file (302s to the CDN; urllib follows it)."""
    return f"{_base()}/models/{repo_id}/resolve/{revision}/{urllib.parse.quote(path)}"


def _timeout() -> float:
    """Per-request socket timeout, re-read per call so ``ANIMA_MS_TIMEOUT``
    set after import (wrappers, tests) still applies."""
    return float(os.environ.get("ANIMA_MS_TIMEOUT", "30"))


def _headers() -> dict[str, str]:
    token = os.environ.get("MODELSCOPE_TOKEN", "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def _open(url: str, headers: dict[str, str] | None = None):
    """``urlopen`` with the pinned timeout and the auth headers applied."""
    req = urllib.request.Request(url, headers=dict(headers or {}))
    return urllib.request.urlopen(req, timeout=_timeout())


def _file_complete(dst: Path, size: int) -> bool:
    """True when ``dst`` already holds exactly ``size`` bytes."""
    return dst.is_file() and dst.stat().st_size == size


def _range_start(content_range: str | None) -> int | None:
    """Start offset from a ``bytes <start>-<end>/<total>`` Content-Range."""
    if not content_range:
        return None
    try:
        return int(content_range.split(" ", 1)[1].split("-", 1)[0])
    except (IndexError, ValueError):
        return None


def repo_files(repo_id: str, revision: str = "master") -> list[tuple[str, int]]:
    """List ``(path, size)`` for every file (blob) in the repo."""
    url = f"{_base()}/api/v1/models/{repo_id}/repo/files?Revision={revision}&Recursive=true"
    try:
        with _open(url) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise FileNotFoundError(
                f"ModelScope repo {repo_id}@{revision} not found"
            ) from exc
        raise
    files = (payload.get("Data") or {}).get("Files") or []
    return [(f["Path"], int(f["Size"])) for f in files if f.get("Type") == "blob"]


def download_file(
    url: str,
    dst: Path,
    *,
    expected_size: int | None = None,
    what: str = "asset",
) -> Path:
    """Stream ``url`` to ``dst`` with Range resume and an atomic rename.

    A leftover ``<dst>.part`` is resumed from its current size (a 200 response
    to the Range request means the server ignored it — restart from zero; a
    206 is only trusted when its Content-Range starts exactly at the requested
    offset, otherwise the append would corrupt the midsection the final size
    check can't see — restart from zero too). Raises ``FileNotFoundError`` on
    transport failure (actionable, like ``hf_download``); size mismatch raises
    ``IOError``.
    """
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    written = part.stat().st_size if part.exists() else 0
    if expected_size is not None and written == expected_size:
        # A previous run downloaded everything but died before the rename.
        part.replace(dst)
        return dst
    if expected_size is not None and written > expected_size:
        written = 0

    from library.runtime.hf_download import is_network_error

    try:
        headers = _headers()
        requested = written
        if written:
            headers["Range"] = f"bytes={written}-"
        with _open(url, headers) as resp:
            if resp.status == 206:
                start = _range_start(resp.headers.get("Content-Range"))
                if start != requested:
                    written = 0  # resumed at the wrong offset — restart clean
            elif resp.status == 200 and requested:
                written = 0  # Range ignored — restart the part from scratch
            append = written > 0
            with open(part, "ab" if append else "wb") as f:
                while True:
                    chunk = resp.read(_CHUNK)
                    if not chunk:
                        break
                    f.write(chunk)
                    written += len(chunk)
    except Exception as exc:  # noqa: BLE001
        if is_network_error(exc):
            raise FileNotFoundError(
                f"{what}: download from ModelScope stalled or failed "
                f"({type(exc).__name__}: {exc}). Check connectivity (or unset "
                f"ANIMA_DOWNLOAD_SOURCE to use HuggingFace), then re-run; "
                f"the .part file resumes where it left off."
            ) from exc
        raise
    if expected_size is not None and written != expected_size:
        raise IOError(
            f"{what}: incomplete download from ModelScope "
            f"({written}/{expected_size} bytes)"
        )
    part.replace(dst)
    return dst


def snapshot_download(
    repo_id: str,
    local_dir: str | Path,
    *,
    revision: str = "master",
    include: list[str] | None = None,
    filenames: list[str] | None = None,
    force: bool = False,
) -> list[Path]:
    """Download repo files into ``local_dir``, preserving repo paths.

    ``hf download --local-dir`` equivalent (plain files, no symlink cache).
    ``include`` filters by fnmatch against repo paths; ``filenames`` is an
    explicit allowlist of exact repo paths. Files already present with the
    exact expected size are skipped unless ``force`` — re-runs verify instead
    of re-fetching. Returns the paths downloaded by this call.
    """
    all_files = repo_files(repo_id, revision)
    selected = [
        (path, size)
        for path, size in all_files
        if (filenames is None or path in filenames)
        and (include is None or any(fnmatch.fnmatch(path, pat) for pat in include))
    ]
    if filenames is not None:
        found = {path for path, _ in selected}
        missing = [p for p in filenames if p not in found]
        if missing:
            raise FileNotFoundError(
                f"ModelScope repo {repo_id} has no file(s): {', '.join(missing)}"
            )
    local_dir = Path(local_dir)
    downloaded: list[Path] = []
    for path, size in selected:
        dst = local_dir / path
        if not force and _file_complete(dst, size):
            continue
        download_file(
            resolve_url(repo_id, path, revision),
            dst,
            expected_size=size if size > 0 else None,
            what=f"{repo_id}/{path}",
        )
        downloaded.append(dst)
    return downloaded


def fetch_file(
    *,
    ms_repo: str,
    filename: str,
    local_dir: str | Path,
    revision: str = "master",
    what: str = "asset",
    force: bool = False,
) -> Path:
    """Fetch one file — matched by basename anywhere in the repo — to
    ``local_dir/filename``. Skips an already-complete copy (exact size) unless
    ``force``. When several repo paths share the basename, the shallowest wins
    (deterministic, instead of whichever the listing happened to end on)."""
    candidates = [
        (path, size)
        for path, size in repo_files(ms_repo, revision)
        if Path(path).name == filename
    ]
    if not candidates:
        raise FileNotFoundError(
            f"{what}: ModelScope repo {ms_repo} has no file named {filename}"
        )
    path, size = min(candidates, key=lambda c: (c[0].count("/"), c[0]))
    dst = Path(local_dir) / filename
    if not force and _file_complete(dst, size):
        return dst
    return download_file(
        resolve_url(ms_repo, path, revision),
        dst,
        expected_size=size if size > 0 else None,
        what=what,
    )


def maybe_ms_download(
    *,
    repo_id: str,
    filename: str,
    local_dir: str | Path,
    what: str = "asset",
) -> Path | None:
    """Fetch ``filename`` via the ModelScope mirror when the MS source is on.

    Returns the local path, or None when ``ANIMA_DOWNLOAD_SOURCE`` is not
    ``modelscope`` — callers fall back to ``hf_download`` in that case. When
    the source IS selected but this repo has no mirror, raises
    ``FileNotFoundError`` instead: the user picked ModelScope because
    HuggingFace is unreachable, so silently hitting HF again would contradict
    the selection (the task CLIs print the same "no mirror — skipped" notice).
    """
    if not source_enabled():
        return None
    ms_repo = mirror_repo(repo_id)
    if ms_repo is None:
        raise FileNotFoundError(
            f"{what}: ANIMA_DOWNLOAD_SOURCE=modelscope is set, but {repo_id} "
            f"has no ModelScope mirror (see ms_download.MIRRORS). Fetch it "
            f"from an MS repo via `python tasks.py download-ms <org/name>`, "
            f"or unset ANIMA_DOWNLOAD_SOURCE to use HuggingFace."
        )
    return fetch_file(
        ms_repo=ms_repo, filename=filename, local_dir=local_dir, what=what
    )
