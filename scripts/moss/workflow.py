"""Cache then train in separate child processes on the same daemon serial job."""

from __future__ import annotations

import argparse
import dataclasses
import subprocess
import sys

from pydantic import TypeAdapter

from library.moss.requests import CacheRequest, TrainRequest
from library.moss.validation import validate_request


def parse_workflow(argv: list[str]) -> tuple[CacheRequest, TrainRequest]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-args", required=True)
    parser.add_argument("--train-args", required=True)
    values = parser.parse_args(argv)
    strings = TypeAdapter(list[str])
    cache = CacheRequest.from_argv(
        strings.validate_json(values.cache_args, strict=True)
    )
    train = TrainRequest.from_argv(
        strings.validate_json(values.train_args, strict=True)
    )
    train = dataclasses.replace(train, cache=cache.out)
    validate_request(cache)
    validate_request(train)
    return cache, train


def main() -> None:
    cache, train = parse_workflow(sys.argv[1:])
    for phase, req in (("cache", cache), ("train", train)):
        print(f"MOSS workflow: starting {phase}", flush=True)
        subprocess.run(
            [sys.executable, "-m", f"scripts.moss.{phase}", *req.to_argv()], check=True
        )
        print(f"MOSS workflow: {phase} completed", flush=True)


if __name__ == "__main__":
    main()
