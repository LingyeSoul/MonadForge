"""The constants both halves of the seam spell: this package's CLIs and GUI, and
the trainer (``anima_lora``) that shells out to them.

**Stdlib only.** Nothing here imports another ``anime_tools`` module, so the GUI
server, the resident autotag worker's driver and the trainer can all read it
without pulling a stage (and its torch) into their process. Pinned by
``tests/test_boundary.py::test_contract_is_torch_free``.

Everything below is part of the CLI/stdio surface a consumer was written
against. That surface is append-only within one :data:`CONTRACT_VERSION`; a
removed or renamed name bumps it, and the trainer asserts the version it
expects.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

CONTRACT_VERSION = 2
"""Bumped on any incompatible change to a name in this module or a stage's flags."""

# --- resident autotag worker (``tagger/cli/autotag_server.py``) ---------------
# Only these sentinel lines reach **stdout**; logging goes to stderr. The driver
# reads live, so each is one line, flushed.
AUTOTAG_READY = "ANIMA_AUTOTAG_READY"
"""The worker has loaded its model and is serving requests."""
AUTOTAG_RESULT_PREFIX = "ANIMA_AUTOTAG_RESULT\t"
"""``prefix + caption`` — one per request, also the single-image CLI's output."""
AUTOTAG_ERROR_PREFIX = "ANIMA_AUTOTAG_ERROR\t"
"""``prefix + message`` — the request failed; the worker keeps serving."""

# --- batch autotag (``stages/autotag.py``) ------------------------------------
AUTOTAG_MODES = ("missing", "merge", "overwrite")
"""``--mode``: only ``missing`` is non-destructive (see ``docs/anima_tagger.md``)."""

# --- Export's published caption formats (``stages/export_workspace.py``) ------
EXPORT_CAPTION_FORMATS = ("anima", "layout_json")
"""``--caption_format``: ``anima`` publishes ``{stem}.txt`` only; ``layout_json``
also publishes ``{stem}`` + :data:`LAYOUT_JSON_SUFFIX` beside it. The ``.txt``
is the Anima caption under both."""
LAYOUT_JSON_SUFFIX = ".layout.txt"
"""``{stem}.layout.txt``, beside the published ``{stem}.txt``: the MOSS-Image
layout-JSON prompt of that caption — ``captions.layout_json.layout_json(text,
kb)`` of the ``.txt``'s stripped text, UTF-8, no trailing newline. The trainer's
MOSS cache reads it instead of converting the caption itself. It ends in
``.txt``, so a caption walk must skip it (:data:`CAPTION_SIDECAR_SUFFIXES`)."""

# --- caption sidecars ---------------------------------------------------------
CAPTION_SIDECAR_SUFFIXES = (
    ".variants.txt",
    ".history.txt",
    ".ocr.txt",
    LAYOUT_JSON_SUFFIX,
)
"""The ``{stem}<suffix>`` files that sit beside a ``{stem}.txt`` caption and end
in ``.txt`` without being one: variants, history, OCR lines, the layout-JSON
prompt. A walk that collects captions by suffix skips these
(:func:`is_caption_sidecar`); ``tests/test_contract_sidecars.py`` pins the tuple
to each owner's own constant."""


def is_caption_sidecar(name: str) -> bool:
    """Is the file ``name`` one of :data:`CAPTION_SIDECAR_SUFFIXES` rather than
    a caption?"""
    return name.lower().endswith(CAPTION_SIDECAR_SUFFIXES)


# --- tagger checkpoint layout (``tagger/dbv4_meta.py``) -----------------------
# Our half of the tagger — vocab / rules / groups / thresholds / sidecar — is
# fetched into ``models/captioners/`` when a required file is missing.
TAGGER_REQUIRED_FILES = ("config.json", "model.safetensors", "vocab.json", "rules.yaml")
TAGGER_OPTIONAL_FILES = ("thresholds.safetensors", "groups.yaml")
# A dbv4-backed checkpoint carries no ``model.safetensors`` (the weights come
# from the gated upstream repo); the sidecar pair is optional.
DBV4_REQUIRED_FILES = ("config.json", "vocab.json", "rules.yaml")
DBV4_OPTIONAL_FILES = TAGGER_OPTIONAL_FILES + ("sidecar.safetensors", "sidecar.json")
DBV4_BACKBONE_FILES = ("model.safetensors", "selected_tags.csv", "meta.json")
"""What the backbone loader pulls from the upstream repo."""

# --- stage reports and their replay (``stages/replay.py``) --------------------
REPLAY_REPORT_NAME = "apply_report.json"
"""What ``--from_report --apply`` writes, beside the dry run's ``report.json``
and never over it."""

GATE_ATTR = "gui_gate"
"""The attribute an argparse group carries when it is a *drawer*: the dest of the
boolean that switches the whole group on. The source is the request field's
``gate`` metadata (``anime_tools._request``); the generated parser stamps the
group with it for anyone introspecting argparse directly."""


@dataclass(frozen=True)
class ReplaySpec:
    """How to read one stage's report and where its proposals get written.

    ``target_root`` is the tree ``caption_path`` is relative to: ``"src"`` for
    a stage that writes the caption master, ``"dst"`` for the revised caption —
    which is every replaying stage today; none of them writes the master.
    """

    stage: str
    # Report container keys — ``rows``/``stats`` (autotag) or ``images``/``summary``
    # (position clauses, multiview audit).
    rows_key: str = "rows"
    stats_key: str = "stats"
    # A row is writable when its status matches, and/or ``row_filter`` says so.
    # The multiview audit gates on verdict/confidence instead of a status field,
    # so it supplies a closure over the CLI's own gate.
    ok_status: str | None = None
    row_filter: Callable[[Mapping[str, object]], bool] | None = None
    before_field: str = "existing"
    """The row key holding the *write target's* own text — the drift baseline,
    which is not the caption that spoke for the image: every caption stage reads
    revised-first with the master as a fallback, so a run that acts on a master
    writes a revised caption that was never there. Empty says exactly that, and
    is what makes the inverse of such a write a delete
    (``replay.undo_one``)."""
    after_field: str = "proposed"
    target_root: str = "src"
    # Passed straight to ``_caption_io.write_caption``; ``history_by`` names who
    # the superseded version is filed under, so a replay pushes the same history
    # entry the stage's own apply would have.
    newline: bool = False
    drop_variants: bool = False
    history_by: str | None = None


# Every proposal lands on the **revised** caption (``--dst``), and every stage
# reads the master as a fallback when there is no revised one, so all three name
# ``target_before`` — what the write target itself held — rather than the text
# that spoke for the image (``existing`` / ``original`` / ``caption``, still in
# the rows for the reader).
REPLAY_SHAPES: dict[str, ReplaySpec] = {
    "autotag": ReplaySpec(
        stage="autotag_captions",
        rows_key="rows",
        stats_key="stats",
        ok_status="ok",
        before_field="target_before",
        after_field="proposed",
        target_root="dst",
        drop_variants=True,
        history_by="autotag",
    ),
    # ``drop_variants`` mirrors the stage's own write: a stale
    # ``{stem}.variants.txt`` outranks ``{stem}.txt`` at encode time.
    "position": ReplaySpec(
        stage="position_captions",
        rows_key="images",
        stats_key="summary",
        ok_status="proposed",
        before_field="target_before",
        after_field="proposed",
        target_root="dst",
        drop_variants=True,
        history_by="position",
    ),
    # Text-only, so it has no ``--from_report`` of its own; the shape exists for
    # the GUI's Undo, which replays these rows backwards. ``drop_variants``
    # mirrors the stage's own write, which rebuilds the sidecar from v0.
    "correct": ReplaySpec(
        stage="correct_captions",
        rows_key="rows",
        stats_key="stats",
        ok_status="ok",
        before_field="target_before",
        after_field="proposed",
        target_root="dst",
        drop_variants=True,
        history_by="correct",
    ),
    # Text-only like ``correct``, and so no ``--from_report`` either. Its write
    # drops the variants sidecar rather than rebuilding it: the sidecar's v0 is
    # the caption before the cut.
    "drop_groups": ReplaySpec(
        stage="drop_group_captions",
        rows_key="rows",
        stats_key="stats",
        ok_status="ok",
        before_field="target_before",
        after_field="proposed",
        target_root="dst",
        drop_variants=True,
        history_by="drop_groups",
    ),
    # The writable set is the verdict/confidence gate, not a row ``status``, so
    # ``row_filter`` is left open here and closed over the gate at replay time.
    # Same target and same bytes as ``apply_findings``: the revised caption, no
    # trailing newline, the stale variants sidecar dropped.
    "audit": ReplaySpec(
        stage="audit_multiview",
        rows_key="images",
        stats_key="summary",
        before_field="target_before",
        after_field="proposed",
        target_root="dst",
        drop_variants=True,
        history_by="audit",
    ),
}
"""GUI stage id → the shape of the report its CLI writes. OCR is absent: it writes
only its own sidecar tree and touches no caption."""
