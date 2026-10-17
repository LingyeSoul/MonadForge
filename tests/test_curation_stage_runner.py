"""Report replay safety without a stage runner, model import or real dataset."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.curation import runner
from scripts.curation.context import checked_path, validate_roots


@pytest.fixture
def export_case(tmp_path):
    package = pytest.importorskip("anime_tools.stages.export_workspace")
    replay = pytest.importorskip("anime_tools.stages.replay")
    roots = {name: str(tmp_path / name) for name in ("src", "master", "dst", "masks", "out")}
    for root in roots.values():
        Path(root).mkdir()
    (tmp_path / "src" / "one.txt").write_text("original")
    (tmp_path / "master" / "one.txt").write_text("revised")
    (tmp_path / "src" / "one.png").write_bytes(b"new pixels")
    (tmp_path / "out" / "resized").mkdir()
    (tmp_path / "out" / "resized" / "one.png").write_bytes(b"old pixels")
    context = {"home": str(tmp_path), "workspace": str(tmp_path / "workspace"), "roots": roots,
               "job_dir": str(tmp_path / "jobs" / "preview"), "id": "preview", "stage": "export",
               "report_path": str(tmp_path / "jobs" / "preview" / "report.json")}
    report = {"out": roots["out"], "src": roots["src"], "dst": roots["dst"], "apply": False, "rows": [
        {"rel": "one.png", "kind": "master", "src": str(tmp_path / "master" / "one.txt"), "dst": str(tmp_path / "src" / "one.txt"), "status": "would-overwrite", "before": "original"},
        {"rel": "one.png", "kind": "image", "src": str(tmp_path / "src" / "one.png"), "dst": str(tmp_path / "out" / "resized" / "one.png"), "status": "would-overwrite"},
    ]}
    loaded = SimpleNamespace(E=package, R=replay)
    runner.seal_preview(loaded, context, report)
    return loaded, context, report


def operation_context(source, name):
    context = copy.deepcopy(source)
    directory = Path(source["job_dir"]).parent / name
    context.update(id=name, job_dir=str(directory), report_path=str(directory / "report.json"))
    return context


def test_export_apply_and_undo_restore_text_and_pixel_overwrite(export_case):
    loaded, source, report = export_case
    applied = operation_context(source, "applied")
    runner._apply_export(loaded, applied, source, report)
    assert (Path(source["roots"]["src"]) / "one.txt").read_text() == "revised"
    assert (Path(source["roots"]["out"]) / "resized" / "one.png").read_bytes() == b"new pixels"
    _, applied_report = runner.sealed_report(loaded, applied)
    runner._undo_export(loaded, operation_context(source, "undo"), applied, applied_report)
    assert (Path(source["roots"]["src"]) / "one.txt").read_text() == "original"
    assert (Path(source["roots"]["out"]) / "resized" / "one.png").read_bytes() == b"old pixels"


def test_export_drift_refuses_all_writes(export_case):
    loaded, source, report = export_case
    target = Path(source["roots"]["src"]) / "one.txt"
    target.write_text("edited after preview")
    with pytest.raises(ValueError, match="drifted"):
        runner.validate_operation(loaded, source, report, "apply")
    assert (Path(source["roots"]["out"]) / "resized" / "one.png").read_bytes() == b"old pixels"


def test_report_drift_and_symlink_escape_refused(export_case, tmp_path):
    loaded, source, _ = export_case
    Path(source["report_path"]).write_text("{}")
    with pytest.raises(ValueError, match="report changed"):
        runner.sealed_report(loaded, source)
    outside = tmp_path / "private.txt"
    outside.write_text("private")
    link = Path(source["roots"]["dst"]) / "escape.txt"
    link.symlink_to(outside)
    with pytest.raises(ValueError, match="outside"):
        checked_path(link, source)
    with pytest.raises(ValueError, match="overlap"):
        validate_roots({**source["roots"], "master": source["roots"]["src"]})


def test_interrupted_export_keeps_a_recoverable_journal(export_case, monkeypatch):
    loaded, source, report = export_case
    applied = operation_context(source, "interrupted")
    original = Path.replace
    publications = []

    def replace(path, target):
        if str(target).endswith("/out/resized/one.png"):
            raise OSError("fixture simulated publication failure")
        publications.append(str(target))
        return original(path, target)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", replace)
        with pytest.raises(OSError, match="simulated"):
            runner._apply_export(loaded, applied, source, report)
    assert (Path(source["roots"]["src"]) / "one.txt").read_text() == "revised"
    _, recovery = runner.sealed_report(loaded, applied)
    runner._undo_export(loaded, operation_context(source, "recover"), applied, recovery)
    assert (Path(source["roots"]["src"]) / "one.txt").read_text() == "original"
    assert (Path(source["roots"]["out"]) / "resized" / "one.png").read_bytes() == b"old pixels"


def test_undo_rejects_changed_published_file(export_case):
    loaded, source, report = export_case
    applied = operation_context(source, "applied")
    runner._apply_export(loaded, applied, source, report)
    _, applied_report = runner.sealed_report(loaded, applied)
    (Path(source["roots"]["src"]) / "one.txt").write_text("later user edit")
    with pytest.raises(ValueError, match="drifted"):
        runner.validate_operation(loaded, applied, applied_report, "undo")


@pytest.mark.parametrize("cancelled", [False, True])
def test_stage_sequence_stops_on_first_error_or_cancellation(tmp_path, monkeypatch, cancelled):
    roots = {name: str(tmp_path / name) for name in ("src", "master", "dst", "masks", "out")}
    directory = tmp_path / "job"
    context = {
        "home": str(tmp_path), "roots": roots, "workspace": str(tmp_path / "workspace"),
        "job_dir": str(directory), "operation": "run", "report_path": None,
        "settings": {}, "environment": {},
        "steps": [{"stage": name, "module": f"fixture.{name}", "argv": [], "report_path": None}
                  for name in ("resize", "masks_sam", "masks_merge")],
    }
    stages = {step["stage"]: SimpleNamespace(module=step["module"]) for step in context["steps"]}

    @contextmanager
    def tools(*args):
        yield SimpleNamespace(S=SimpleNamespace(BY_ID=stages))

    monkeypatch.setattr(runner, "scoped_tools", tools)
    calls = []

    def execute(argv, **kwargs):
        calls.append(argv)
        # No new process group: the daemon can cancel the entire child tree.
        assert "start_new_session" not in kwargs and "creationflags" not in kwargs
        assert kwargs["check"] is True
        if len(calls) == 2:
            if cancelled:
                raise runner.CurationCancelled("fixture daemon cancellation")
            raise subprocess.CalledProcessError(7, argv)

    monkeypatch.setattr(runner.subprocess, "run", execute)
    directory.mkdir()
    path = directory / "snapshot.json"
    path.write_text(json.dumps(context))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(runner.CurationCancelled if cancelled else subprocess.CalledProcessError):
        runner.run(path, digest)
    assert len(calls) == 2
    assert json.loads((directory / "state.json").read_text())["state"] == ("cancelled" if cancelled else "failed")
