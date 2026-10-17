#!/usr/bin/env python3
"""Find the fastest block swap / activation budget pair for MOSS LoRA training
on this machine — flags in ``library.moss.requests.TuneSwapRequest``, work in
``library.moss.tune``."""

from library.moss.requests import TuneSwapRequest


def main() -> None:
    req = TuneSwapRequest.from_argv()
    from library.moss.tune import run_tune

    run_tune(req)


if __name__ == "__main__":
    main()
