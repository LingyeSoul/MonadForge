"""Resident autotag worker for the GUI Dataset tab.

Loads the Anima Tagger once, then serves single-image requests over a
line-based stdio protocol so consecutive "Autotag" clicks don't pay the
model-load cost each time. The GUI (``gui/tabs/image_tab.py``) owns the
process: it spawns one on first use, keeps it resident ("model loaded,
waiting"), and tears it down before any other GPU work (grouping / training /
preprocessing) so the card is free.

Protocol (all lines newline-terminated, UTF-8):

* On startup, once the model is loaded + warmed, the worker prints::

      ANIMA_AUTOTAG_READY

* The driver writes one request per line to stdin. A request is either a bare
  image path, or ``<min_confidence>\t<image path>`` where ``min_confidence`` is
  an extra probability floor (0–1) applied on top of the model's per-tag F1
  thresholds. For each, the worker replies with exactly one line::

      ANIMA_AUTOTAG_RESULT\t<comma-separated caption>
      ANIMA_AUTOTAG_ERROR\t<message>

* Closing stdin (EOF) makes the worker exit.

Only these sentinel lines go to **stdout**; all logging / download progress
goes to **stderr**, so the driver can parse stdout unambiguously.
"""

from __future__ import annotations

import argparse
import logging
import sys

from library.captioning.anima_tagger import (
    DEFAULT_TAGGER_DIR,
    AnimaTagger,
    ensure_tagger_checkpoint,
)
from library.env import resolve_under_home
from library.log import setup_logging

READY = "ANIMA_AUTOTAG_READY"
RESULT_PREFIX = "ANIMA_AUTOTAG_RESULT\t"
ERROR_PREFIX = "ANIMA_AUTOTAG_ERROR\t"

setup_logging()
logger = logging.getLogger(__name__)


def _emit(line: str) -> None:
    """Write one sentinel line to stdout and flush (the driver reads live)."""
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def main() -> None:
    p = argparse.ArgumentParser(description="Resident Anima autotag worker")
    p.add_argument("--tagger_dir", default=DEFAULT_TAGGER_DIR)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    import torch
    from PIL import Image

    ckpt_dir = ensure_tagger_checkpoint(resolve_under_home(args.tagger_dir))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tagger = AnimaTagger(ckpt_dir, device=device)
    # Warm the lazily-loaded PE encoders with a tiny dummy image so the first
    # real request is fast and READY genuinely means "ready to serve".
    try:
        tagger.predict_caption(Image.new("RGB", (64, 64)))
    except Exception:  # noqa: BLE001 — warm-up is best-effort
        logger.exception("autotag warm-up failed (continuing)")
    _emit(READY)
    logger.info("autotag worker ready (device=%s, ckpt=%s)", device, ckpt_dir)

    # readline() (not `for line in sys.stdin`): iterating read-aheads and blocks
    # until the buffer fills, stalling the interactive protocol.
    while True:
        raw = sys.stdin.readline()
        if not raw:  # EOF — driver closed stdin
            break
        request = raw.rstrip("\n")
        if not request.strip():
            continue
        # Optional ``<min_confidence>\t<path>`` prefix; bare paths keep the
        # model's own threshold decisions (min_confidence=0.0).
        min_confidence = 0.0
        if "\t" in request:
            conf_str, _, rest = request.partition("\t")
            try:
                min_confidence = max(0.0, min(1.0, float(conf_str)))
                request = rest
            except ValueError:
                pass  # not a confidence prefix — treat the whole line as a path
        path = request.strip()
        if not path:
            continue
        try:
            image_path = resolve_under_home(path)
            if not image_path.exists():
                _emit(ERROR_PREFIX + f"image not found: {image_path}")
                continue
            caption = tagger.predict_caption(
                Image.open(image_path), min_confidence=min_confidence
            )
            _emit(RESULT_PREFIX + caption)
        except Exception as e:  # noqa: BLE001 — report, keep serving
            logger.exception("autotag failed for %s", path)
            _emit(ERROR_PREFIX + str(e))


if __name__ == "__main__":
    main()
