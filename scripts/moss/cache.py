#!/usr/bin/env python3
"""Precache MOSS-Image text embeddings + VAE latents — flags in
``library.moss.requests.CacheRequest``, work in ``library.moss.cache``."""

from library.moss.requests import CacheRequest


def main() -> None:
    req = CacheRequest.from_argv()
    from library.moss.validation import validate_request

    validate_request(req)
    from library.moss.cache import run_cache

    run_cache(req)


if __name__ == "__main__":
    main()
