"""Execute a frozen curation plan as one serial daemon command job.

No official JobManager, training, or process-global GUI state is used here.
Every child stage runs with the same saved roots, scope and environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import shutil
import subprocess
import sys
import uuid
from dataclasses import fields
from pathlib import Path

from scripts.curation.context import checked_path, checked_report, scoped_tools, validate_roots, validate_tree


class CurationCancelled(RuntimeError):
    """The daemon stopped this wrapper and its process tree."""


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected an object: {path}")
    return data


def signature(path: Path) -> dict:
    if not path.exists():
        return {"exists": False}
    if not path.is_file():
        raise ValueError(f"Expected an artifact file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return {"exists": True, "sha256": digest.hexdigest(), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _seal(context: dict, report: dict) -> None:
    path = checked_path(context["report_path"], context, bases=[context["job_dir"]])
    _write(path, report)
    _write(Path(context["job_dir"]) / "report-seal.json", {
        "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    })


def sealed_report(loaded, context: dict) -> tuple[Path, dict]:
    if not context.get("report_path"):
        raise FileNotFoundError("This stage has no report")
    path = checked_path(context["report_path"], context, bases=[context["job_dir"]])
    seal = _read(Path(context["job_dir"]) / "report-seal.json")
    if seal.get("path") != str(path) or hashlib.sha256(path.read_bytes()).hexdigest() != seal.get("sha256"):
        raise ValueError("The saved report changed; create a new preview")
    return path, loaded.R.load_report(path)


def _export_rows(loaded, report: dict):
    # Official rows_from_report omits the OCR threshold fields. Preserve every
    # dataclass field so replay is exactly what the preview rendered.
    names = {field.name for field in fields(loaded.E.ExportRow)}
    return [loaded.E.ExportRow(**{key: value for key, value in row.items() if key in names}) for row in report.get("rows", [])]


def _export_paths(context: dict, report: dict) -> list[Path]:
    out = checked_path(report.get("out") or context["roots"]["out"], context, bases=[context["roots"]["out"]])
    src = Path(context["roots"]["src"])
    if out.is_relative_to(src) or src.is_relative_to(out):
        raise ValueError("Export output cannot overlap the source tree")
    found = set()
    destinations = set()
    for row in report.get("rows", []):
        if not isinstance(row, dict) or not row.get("dst"):
            raise ValueError("Malformed export artifact row")
        destination = checked_path(row["dst"], context, bases=[src] if row.get("kind") == "master" else [out])
        if row.get("kind") == "master":
            rel = Path(str(row.get("rel", "")))
            if rel.is_absolute() or ".." in rel.parts or destination != src / rel.with_suffix(".txt"):
                raise ValueError("Invalid master publication path")
        if destination in destinations:
            raise ValueError(f"Duplicate export destination: {destination}")
        destinations.add(destination)
        found.add(destination)
        for name in ("src", "ocr", "ref", "fit", "tag_csv"):
            if row.get(name):
                found.add(checked_path(row[name], context, bases=[*context["roots"].values(), context["workspace"], Path(context["home"]) / "models"]))
    return sorted(found)


def seal_preview(loaded, context: dict, report: dict) -> None:
    if context["stage"] == "export":
        paths = _export_paths(context, report)
        report["_monadforge"] = {"version": 1, "inputs": {str(path): signature(path) for path in paths}}
        for row in report.get("rows", []):
            if row.get("kind") in loaded.E.TEXT_KINDS and not row.get("text") and row.get("src"):
                path = Path(row["src"])
                if path.is_file():
                    row["preview_text"] = path.read_text(encoding="utf-8")
    _seal(context, report)


def _caption_preflight(loaded, context: dict, report: dict, operation: str) -> None:
    _, _, roots = checked_report(loaded, context)
    shape = loaded.P.SHAPES[context["stage"]]
    if operation == "apply":
        loaded.R.validate_report(report, spec=shape, src=roots.src, dst=roots.dst)
        rows, _ = loaded.R.replay_rows(report, spec=shape, src=roots.src, dst=roots.dst,
                                       path_pattern=context.get("path_pattern"), apply=False)
        refused = [row.caption_path for row in rows if row.status in {"skip:drifted", "skip:missing-caption", "skip:no-caption-path"}]
    else:
        refused = []
        for row, before, after in loaded.P._walk(report, shape):
            if str(row.get("status", "")).startswith("skip:"):
                continue
            target = checked_path(roots.dst / row["caption_path"], context, bases=[roots.dst])
            status = loaded.R.undo_one(target, after, before, apply=False)
            if status in {"drifted", "missing-caption"}:
                refused.append(str(target))
    if refused:
        raise ValueError(f"Caption drift detected; no changes applied: {', '.join(refused[:5])}")


def validate_operation(loaded, context: dict, report: dict, operation: str) -> None:
    if context["stage"] != "export":
        _caption_preflight(loaded, context, report, operation)
        return
    _export_paths(context, report)
    guard = report.get("_monadforge") or {}
    if operation == "apply":
        expected = guard.get("inputs")
        if not isinstance(expected, dict):
            raise ValueError("Export has no immutable preview baseline; create a new preview")
        for path, recorded in expected.items():
            checked_path(path, context, bases=[*context["roots"].values(), context["workspace"], Path(context["home"]) / "models"])
            if signature(Path(path)) != recorded:
                raise ValueError(f"Export input or destination drifted: {path}; create a new preview")
    else:
        journal = guard.get("journal")
        if not isinstance(journal, list):
            raise ValueError("Export has no recovery journal")
        for entry in journal:
            target = checked_path(entry["dst"], context)
            current = signature(target)
            if current not in (entry["before"], entry["after"]):
                raise ValueError(f"Export destination drifted: {target}; refusing undo")
            if entry["before"]["exists"]:
                backup = checked_path(entry["backup"], context, bases=[context["job_dir"]])
                if signature(backup) != entry["before"]:
                    raise ValueError(f"Export recovery backup changed: {backup}")


def _apply_export(loaded, context: dict, source: dict, report: dict) -> None:
    """Stage output and backup first, then journal before any destination write.

    If killed between replacements, Undo compares each target with both sides
    of this journal. Thus an interrupted/failed apply can be recovered too.
    """
    rows = _export_rows(loaded, report)
    directory = Path(context["job_dir"])
    journal = []
    for index, row in enumerate(rows):
        if not row.status.startswith("would-"):
            continue
        target = checked_path(row.dst, source)
        before = signature(target)
        backup = directory / "backups" / f"{index}{target.suffix}"
        if before["exists"]:
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, backup)
        temporary = directory / "staged" / f"{index}{target.suffix}"
        after = {"exists": False}
        if not row.remove:
            original = row.dst
            row.dst = str(temporary)
            loaded.E.export_one(row, apply=True, decided=True)
            row.dst = original
            after = signature(temporary)
        row.status = "removed" if row.remove else ("overwrote" if before["exists"] else "created")
        journal.append({"dst": str(target), "before": before, "after": after,
                        "backup": str(backup) if before["exists"] else "", "staged": str(temporary),
                        "kind": row.kind, "rel": row.rel})
    # Recheck after potentially slow rendering/copying, before publication.
    validate_operation(loaded, source, report, "apply")
    result = {**report, "apply": True, "rows": [row.to_dict() for row in rows],
              "_monadforge": {"version": 1, "journal": journal, "source_report": source["report_path"]}}
    _seal(context, result)
    for entry in journal:
        target = checked_path(entry["dst"], context)
        if signature(target) != entry["before"]:
            raise ValueError(f"Export destination changed during publication: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if entry["after"]["exists"]:
            # Copy to the destination filesystem before an atomic replace;
            # os.replace across mounted dataset/export roots would be EXDEV.
            adjacent = target.with_name(f".{target.name}.{context['id']}.tmp")
            try:
                shutil.copy2(entry["staged"], adjacent)
                adjacent.replace(target)
            finally:
                adjacent.unlink(missing_ok=True)
        else:
            target.unlink(missing_ok=True)
    print(f"Export applied: {len(journal)} artifacts; recovery report {context['report_path']}", flush=True)


def _undo_export(loaded, context: dict, source: dict, report: dict) -> None:
    validate_operation(loaded, source, report, "undo")
    restored = []
    for entry in report["_monadforge"]["journal"]:
        target = checked_path(entry["dst"], source)
        current = signature(target)
        if current == entry["before"]:
            continue
        if current != entry["after"]:
            raise ValueError(f"Export destination changed during undo: {target}")
        if entry["before"]["exists"]:
            adjacent = target.with_name(f".{target.name}.{context['id']}.tmp")
            try:
                shutil.copy2(entry["backup"], adjacent)
                adjacent.replace(target)
            finally:
                adjacent.unlink(missing_ok=True)
        else:
            target.unlink(missing_ok=True)
        restored.append(str(target))
    _seal(context, {"stage": "export", "operation": "undo", "restored": restored, "source_report": source["report_path"]})


def _replay(loaded, context: dict) -> None:
    source = context["source"]
    path, report = sealed_report(loaded, source)
    if hashlib.sha256(path.read_bytes()).hexdigest() != context["source_report_sha256"]:
        raise ValueError("The reviewed report changed after submission")
    operation = context["operation"]
    validate_operation(loaded, source, report, operation)
    if context["stage"] == "export":
        (_apply_export if operation == "apply" else _undo_export)(loaded, context, source, report)
        return
    _, _, roots = checked_report(loaded, source)
    if operation == "undo":
        result = loaded.P.undo(path, roots, source["stage"])
    else:
        shape = loaded.P.SHAPES[source["stage"]]
        rows, stats = loaded.R.replay_rows(report, spec=shape, src=roots.src, dst=roots.dst,
                                          path_pattern=source.get("path_pattern"), apply=True)
        result = loaded.R.build_replay_report(
            spec=shape, source_report=report, source_report_path=path, src=roots.src, dst=roots.dst,
            path_pattern=source.get("path_pattern"), apply=True, rows=rows, stats=stats,
        )
    _seal(context, result)


def run(snapshot_path: Path, digest: str) -> None:
    raw = snapshot_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("Curation snapshot changed after submission")
    context = json.loads(raw)
    directory = Path(context["job_dir"])
    if snapshot_path.resolve() != (directory / "snapshot.json").resolve():
        raise ValueError("Snapshot is outside its job directory")
    state = directory / "state.json"
    _write(state, {"state": "running"})
    try:
        with scoped_tools(context["home"], context["settings"], context["environment"]) as loaded:
            validate_roots(context["roots"])
            for root in [*context["roots"].values(), context["workspace"]]:
                validate_tree(root)
            if context["operation"] != "run":
                _replay(loaded, context)
            else:
                for step in context["steps"]:
                    stage = loaded.S.BY_ID.get(step["stage"])
                    if stage is None or step["module"] != stage.module:
                        raise ValueError("Snapshot contains an unregistered stage")
                    if step.get("report_path"):
                        checked_path(step["report_path"], context, bases=[directory]).parent.mkdir(parents=True, exist_ok=True)
                    print(f"[curation] {step['stage']}", flush=True)
                    subprocess.run([sys.executable, "-m", stage.module, *step["argv"]],
                                   cwd=context["home"], env=dict(os.environ), check=True)
                if context.get("report_path"):
                    path = checked_path(context["report_path"], context, bases=[directory])
                    seal_preview(loaded, context, loaded.R.load_report(path))
        _write(state, {"state": "success"})
    except BaseException as exc:
        _write(state, {"state": "cancelled" if isinstance(exc, (KeyboardInterrupt, CurationCancelled)) else "failed", "error": str(exc)})
        raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args(argv)

    def cancel(signum, frame):
        raise CurationCancelled(f"Curation job cancelled by signal {signum}")

    previous = signal.signal(signal.SIGTERM, cancel)
    try:
        run(args.snapshot, args.sha256)
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
