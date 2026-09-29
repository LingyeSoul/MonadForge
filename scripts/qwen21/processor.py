"""Download only tokenizer/processor assets; no official model shards are fetched."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError
from httpx import TransportError

from library.env import resolve_under_home

logger = logging.getLogger(__name__)
PROCESSOR_REPO = "Qwen/Qwen-Image-2.1"
PROCESSOR_REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"


def download_processor(destination: Path) -> Path:
    for attempt in range(1, 4):
        try:
            snapshot_download(
                repo_id=PROCESSOR_REPO,
                revision=PROCESSOR_REVISION,
                allow_patterns=["processor/*"],
                local_dir=destination,
            )
            return destination / "processor"
        except (HfHubHTTPError, TransportError) as exc:
            if attempt == 3:
                raise
            logger.warning(
                "Processor download failed; retrying",
                extra={
                    "repo": PROCESSOR_REPO,
                    "revision": PROCESSOR_REVISION,
                    "attempt": attempt,
                    "error": str(exc),
                },
            )
            time.sleep(attempt * 2)
    raise RuntimeError("Processor download exhausted its attempts")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default="models/qwen_image_2.1", help="parent directory for processor/"
    )
    args = parser.parse_args()
    print(download_processor(resolve_under_home(args.out)))


if __name__ == "__main__":
    main()
