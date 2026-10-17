"""Bounded Hub downloads shared with the pinned model-free curation helpers."""

from library.curation_compat._hf import (
    _is_network_error,
    ensure_hf_timeouts,
    hf_download as _curation_hf_download,
    hf_file_cached,
)

# ModelScope callers use this public name; keep the historical alias too.
is_network_error = _is_network_error


def hf_download(*, what: str, hint: str = "make download-models", **kwargs):
    return _curation_hf_download(what=what, hint=hint, **kwargs)
