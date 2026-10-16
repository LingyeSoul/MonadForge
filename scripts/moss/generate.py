#!/usr/bin/env python3
"""Render prompts with MOSS-Image — flags in ``library.moss.requests.GenerateRequest``,
work in ``library.moss.generate``."""

from library.moss.requests import GenerateRequest


def main() -> None:
    req = GenerateRequest.from_argv()
    from library.moss.generate import run_generate

    run_generate(req)


if __name__ == "__main__":
    main()
