"""Regression test for the resume-training progress-bar absolute-step fix.

Symptom (user report): resuming a run displayed impossible step reads like
``172/138`` (step > total) and ``epoch 8/12``. Root cause was that the tqdm
bar was sized to the *remaining* step count (``range(max - global_step)``)
while the JSONL progress sink logged the *absolute* ``global_step``. The
WebUI feeds ``metrics.step`` / ``metrics.total_steps`` from both channels, so
they raced and produced a mixed absolute/relative snapshot.

Fix: build the bar over the full ``range(max_train_steps)`` with
``initial=global_step`` so tqdm's internal ``n`` matches ``global_step``
after each ``update(1)``. This test pins that contract by reproducing the
exact update sequence the training loop uses
(``progress_bar.update(1); global_step += 1``) and asserting the rendered
stdout line parses back to the absolute step / total the JSONL would also
report.
"""

from __future__ import annotations

import io
import re

from tqdm import tqdm

from webui.services.training_log_parser import TrainingLogParser


def _capture_tqdm_line(bar: tqdm, capture: io.StringIO) -> str:
    """Force a tqdm redraw into *capture* and return the rendered line."""
    bar.refresh()
    line = capture.getvalue().splitlines()[-1]
    # tqdm ANSI left the carriage-return redraws; the last segment is current.
    # Strip any leading ``\r`` fragments from prior redraws.
    line = line.split("\r")[-1]
    return line


def test_resume_progress_bar_shows_absolute_step():
    """On resume the bar must show ``global_step / max_train_steps``.

    Reproduces a run that resumed at step 138 with ``max_train_steps = 300``:
    the old code showed ``1/162`` (remaining), the fix shows ``139/300``
    (absolute) — matching the JSONL ``global_step`` the sink logs alongside.
    """
    max_train_steps = 300
    global_step = 138  # resume offset

    capture = io.StringIO()
    bar = tqdm(
        range(max_train_steps),
        initial=global_step,
        smoothing=0,
        file=capture,
        desc="steps",
        miniters=1,
        mininterval=0,
    )

    # Mirror the loop body: update(1) then global_step += 1, for a few steps.
    for _ in range(4):
        bar.update(1)
        global_step += 1

    line = _capture_tqdm_line(bar, capture)
    bar.close()

    # The rendered bar must carry the absolute step (142) and the absolute
    # total (300). Under the old ``range(max - global_step)`` sizing the total
    # would be 162 (300-138) and the counter would be ~4 — neither matches the
    # JSONL absolute step the sink logs in the same wall-clock window.
    assert "142/300" in line, (
        f"expected absolute step 142/300 in tqdm line, got: {line!r}"
    )


def test_resume_progress_bar_parses_to_absolute_via_stdout_parser():
    """The WebUI stdout parser must read the absolute step/total off the bar.

    The dashboard's ``metrics.step`` / ``metrics.total_steps`` are written by
    BOTH the JSONL watcher (absolute ``global_step`` + ``run_start.total``)
    AND the stdout tqdm parser. For a resume these must agree — the stdout
    line parsed here has to yield ``step == global_step`` and
    ``total_steps == max_train_steps``, not the remaining count.
    """
    max_train_steps = 300
    global_step = 138

    capture = io.StringIO()
    bar = tqdm(
        range(max_train_steps),
        initial=global_step,
        smoothing=0,
        file=capture,
        desc="steps",
        miniters=1,
        mininterval=0,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
        "[{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    )
    bar.set_postfix(avr_loss=0.1234, lr=1.0e-4, refresh=False)
    for _ in range(4):
        bar.update(1)
        global_step += 1
    bar.refresh()
    line = capture.getvalue().splitlines()[-1].split("\r")[-1]
    bar.close()

    parser = TrainingLogParser()
    assert parser.feed(line), f"tqdm line did not match the parser: {line!r}"
    m = parser.metrics
    # The two channels now agree: stdout-parsed step/total == JSONL absolutes.
    assert m.step == 142, f"expected absolute step 142, got {m.step}"
    assert m.total_steps == 300, (
        f"expected absolute total 300 (not remaining {300 - 138}), "
        f"got {m.total_steps}"
    )


def test_fresh_run_progress_bar_unchanged():
    """The fix must not change the fresh-run display: ``n/total`` from 0."""
    max_train_steps = 100
    global_step = 0

    capture = io.StringIO()
    bar = tqdm(
        range(max_train_steps),
        initial=global_step,
        smoothing=0,
        file=capture,
        desc="steps",
        miniters=1,
        mininterval=0,
    )
    bar.update(1)
    global_step += 1
    line = _capture_tqdm_line(bar, capture)
    bar.close()

    assert "1/100" in line, f"fresh run should show 1/100, got: {line!r}"
