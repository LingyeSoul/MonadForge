"""Bounded, fail-fast HuggingFace downloads.

A stalled connection inside ``hf_hub_download`` blocks the whole subprocess, so
this pins an explicit socket timeout (a slow trickle is still the stall
watchdog's job) and translates network failures into an error naming the
missing asset and the recovery command.
"""

from __future__ import annotations

import os

# Per-request socket timeout (connect + read) for hub traffic, in seconds.
# Tunable via ANIMA_HF_TIMEOUT; bounds a fully stalled connection.
_DEFAULT_TIMEOUT = os.environ.get("ANIMA_HF_TIMEOUT", "30")


def ensure_hf_timeouts() -> None:
    """Pin huggingface_hub's socket timeouts unless the user set them.

    ``HF_HUB_DOWNLOAD_TIMEOUT`` bounds the streaming file-download read;
    ``HF_HUB_ETAG_TIMEOUT`` the metadata HEAD/list call.
    """
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", _DEFAULT_TIMEOUT)
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", _DEFAULT_TIMEOUT)


def _is_network_error(exc: BaseException) -> bool:
    """True only for *transport* failures (the ones that hang): connection and
    read timeouts, refused/reset connections.

    Excludes HTTP-status errors (``EntryNotFoundError``,
    ``RepositoryNotFoundError``): callers catch those specifically, so they must
    propagate unchanged.
    """
    import socket

    net: list[type] = [socket.timeout, TimeoutError, ConnectionError]
    try:
        import requests  # huggingface_hub's transport

        net.append(requests.exceptions.ConnectionError)
        net.append(requests.exceptions.Timeout)
    except ImportError:
        pass
    return isinstance(exc, tuple(net))


def hf_download(*, what: str, hint: str = "python -m anime_tools.downloads", **kwargs):
    """``hf_hub_download`` with pinned timeouts and a fail-fast network error.

    ``what`` names the asset for the error message; ``hint`` is the suggested
    recovery command. Remaining kwargs pass straight through to
    ``hf_hub_download`` (``repo_id`` / ``filename`` / ``local_dir`` / ``token``
    / ``revision`` …). Non-network failures propagate unchanged.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import GatedRepoError

    ensure_hf_timeouts()
    try:
        return hf_hub_download(**kwargs)
    except GatedRepoError as exc:
        # Not a hang, but the raw hub traceback says nothing about how to get
        # access.
        repo = kwargs.get("repo_id", "?")
        raise FileNotFoundError(
            f"{what}: {repo} is a gated HuggingFace repo and this token cannot "
            f"access it ({type(exc).__name__}). Run `{hint}` and re-run."
        ) from exc
    except Exception as exc:
        if _is_network_error(exc):
            raise FileNotFoundError(
                f"{what}: download from HuggingFace stalled or failed "
                f"({type(exc).__name__}: {exc}). Check connectivity (or set "
                f"HF_HUB_OFFLINE=1 if it is already cached locally), then "
                f"re-run `{hint}`."
            ) from exc
        raise


def hf_file_cached(repo_id: str, filename: str, revision: str | None = None) -> bool:
    """True when ``filename`` from ``repo_id`` is already in the local hub cache.

    Never touches the network, so it is safe from UI code. Assets fetched
    through plain ``hf_hub_download`` never land under ``models/``, so a path
    check would report them missing forever.
    """
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    try:
        hit = try_to_load_from_cache(
            repo_id=repo_id, filename=filename, revision=revision
        )
    except Exception:  # noqa: BLE001 — a broken/unreadable cache is just "missing"
        return False
    # A hit is the path; ``None`` is absent and a sentinel object is a
    # remembered 404 — only the path counts as installed.
    return isinstance(hit, str)
