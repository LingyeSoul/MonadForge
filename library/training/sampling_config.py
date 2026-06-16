"""Boundary normalization for the sample-image knobs.

The GUI drives ``sample_prompts`` / ``sample_every_n_epochs`` /
``sample_every_n_steps`` more loosely than the training core expects: prompts may
be inline rather than a file path, and ``0`` is used as a "disabled" sentinel for
the cadence knobs. :func:`normalize_sample_args` reconciles those into the shapes
the rest of the training path assumes, mutating the ``argparse.Namespace`` in
place. Kept torch-free so it stays a fast headless unit-test target.
"""

from __future__ import annotations

import argparse
import logging
import os

logger = logging.getLogger(__name__)


def normalize_sample_args(args: argparse.Namespace) -> None:
    """Normalize the sample-image knobs so the GUI can drive them naturally.

    Two concerns:

    1. ``sample_prompts`` may be given inline (a TOML list / multi-line string)
       instead of a file path — the GUI writes the user's prompts straight into
       the config rather than asking them to manage a separate file. The rest of
       the sampling path (``train_util.load_prompts`` etc.) expects a ``.txt``
       path, so a list/multi-line value is written to
       ``output_dir/sample_prompts.txt`` (one prompt per line) and
       ``args.sample_prompts`` is repointed at it. A plain existing-file path
       (the CLI case) is left untouched.

    2. Cadence knobs use ``0`` as the GUI's "disabled" sentinel. ``sample_images``
       does ``epoch % args.sample_every_n_epochs`` once the value ``is not None``,
       so a literal ``0`` would raise ``ZeroDivisionError``. Coerce any
       non-positive cadence to ``None`` (= disabled).
    """
    for knob in ("sample_every_n_epochs", "sample_every_n_steps"):
        v = getattr(args, knob, None)
        if v is not None and v <= 0:
            setattr(args, knob, None)

    val = getattr(args, "sample_prompts", None)
    if val is None:
        return

    if isinstance(val, (list, tuple)):
        lines = [str(p).strip() for p in val]
    elif isinstance(val, str):
        # A real file path stays as-is; only treat it as inline text when it's
        # not an existing file and actually contains prompt content.
        if os.path.isfile(val):
            return
        lines = [ln.strip() for ln in val.splitlines()]
    else:
        return

    lines = [ln for ln in lines if ln and not ln.startswith("#")]
    if not lines:
        # Nothing usable — disable sampling rather than point at a phantom file.
        args.sample_prompts = None
        return

    os.makedirs(args.output_dir, exist_ok=True)
    prompt_path = os.path.join(args.output_dir, "sample_prompts.txt")
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"Wrote {len(lines)} inline sample prompt(s) to {prompt_path}")
    args.sample_prompts = prompt_path
