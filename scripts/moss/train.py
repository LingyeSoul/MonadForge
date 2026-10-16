#!/usr/bin/env python3
"""Train a MOSS-Image LoRA on a precached folder — flags in
``library.moss.requests.TrainRequest``, work in ``library.moss.train``."""

from library.moss.requests import TrainRequest


def main() -> None:
    req = TrainRequest.from_argv()
    from library.moss.train import run_train

    run_train(req)


if __name__ == "__main__":
    main()
