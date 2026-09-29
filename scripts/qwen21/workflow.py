"""One daemon job, two isolated processes: cache must succeed before training.

The child-process boundary releases the encoders' host/GPU allocations before
starting the trainer. Closing the WebUI does not cancel this workflow. The
existing daemon process-tree cancellation remains authoritative.
"""

from __future__ import annotations

import argparse
import dataclasses
import subprocess
import sys

from pydantic import TypeAdapter

from library.qwen21.requests import CacheRequest, TrainRequest
from library.qwen21.validation import validate_request


def parse_workflow(argv: list[str]) -> tuple[CacheRequest, TrainRequest]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-args", required=True)
    parser.add_argument("--train-args", required=True)
    args = parser.parse_args(argv)
    strings = TypeAdapter(list[str])
    cache = CacheRequest.from_argv(strings.validate_json(args.cache_args, strict=True))
    train = TrainRequest.from_argv(strings.validate_json(args.train_args, strict=True))
    linked = dataclasses.replace(train, cache=cache.out)
    validate_request(cache)
    validate_request(linked)
    return cache, linked


def run_workflow(cache: CacheRequest, train: TrainRequest) -> None:
    for phase, req in (("cache", cache), ("train", train)):
        print(f"Qwen workflow: starting {phase}", flush=True)
        subprocess.run(
            [sys.executable, "-m", f"scripts.qwen21.{phase}", *req.to_argv()],
            check=True,
        )
        print(f"Qwen workflow: {phase} completed", flush=True)


def main() -> None:
    cache, train = parse_workflow(sys.argv[1:])
    run_workflow(cache, train)


if __name__ == "__main__":
    main()
