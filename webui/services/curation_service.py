"""Native WebUI adapter over anime-tools' official, import-light contracts.

Each submission freezes its bindings and ordered stages in a separate file.
Only the existing daemon runs work; importing this module starts no job manager.
"""

from __future__ import annotations

import hashlib
import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from scripts.curation.context import (
    ENV_KEYS, checked_path, checked_report, run_context, scoped_tools,
    validate_roots, validate_tree,
)
from scripts.curation.bridge import in_curation_runtime

ROOT = Path(__file__).resolve().parents[2]
HOME = ROOT
JOBS = ROOT / "output" / "curation" / "jobs"
ROOT_NAMES = ("src", "master", "dst", "masks", "out")


def write_json(path: Path, data: dict, *, exclusive: bool = False) -> str:
    payload = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(payload)
    else:
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(path)
    return hashlib.sha256(payload.encode()).hexdigest()


def _read(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def _settings(loaded) -> dict:
    data = loaded.settings.load_settings()
    default = dict(loaded.D.DEFAULT_ROOTS)
    default["src"] = str(ROOT / "image_dataset")
    default["out"] = str(ROOT / "post_image_dataset")
    data["dataset"] = {**default, **data.get("dataset", {})}
    # The canonical spelling makes an explicit root symlink a registration of
    # its current target, not an invitation to retarget old jobs later.
    data["dataset"] = {
        name: str(loaded.D.lexical(value).resolve())
        for name, value in data["dataset"].items() if name in ROOT_NAMES
    }
    validate_roots(data["dataset"])
    return data


def _context(loaded, settings: dict, *, job_dir: Path | None = None) -> dict:
    workspace = Path(os.environ.get("ANIME_TOOLS_WORKSPACE") or HOME / "workspace").resolve()
    source = Path(settings["dataset"]["src"])
    for path in (workspace, JOBS.resolve()):
        if path.is_relative_to(source) or source.is_relative_to(path):
            raise ValueError("Source root must not overlap the curation workspace or job records")
    return {
        "home": str(HOME.resolve()), "settings": settings,
        "roots": dict(settings["dataset"]), "workspace": str(workspace),
        "job_dir": str((job_dir or JOBS).resolve()),
        "environment": {key: os.environ[key] for key in ENV_KEYS if key in os.environ},
    }


@in_curation_runtime
def get_settings() -> dict:
    with scoped_tools(HOME) as loaded:
        return _settings(loaded)


@in_curation_runtime
def get_roots() -> dict:
    with scoped_tools(HOME) as loaded:
        return {"roots": run_context(loaded, _settings(loaded)).roots.as_dict()}


@in_curation_runtime
def save_roots(values: dict) -> dict:
    if set(values) - set(ROOT_NAMES):
        raise ValueError("Unknown dataset root")
    with scoped_tools(HOME) as loaded:
        settings = _settings(loaded)
        merged = {**settings["dataset"], **values}
        # Only this explicit registration can widen the trusted roots.
        for value in values.values():
            if not isinstance(value, str) or not value.strip() or ".." in Path(value).parts:
                raise ValueError("Roots must be nonempty paths without parent traversal")
        roots = loaded.D.resolve_roots(merged, trusted=True)
        paths = {name: str(path.resolve()) for name, path in roots.items()}
        validate_roots(paths)
        for name, path in paths.items():
            if not Path(path).is_relative_to(HOME.resolve()) and not Path(path).is_dir():
                raise ValueError(f"Register an existing external directory for {name}: {path}")
        settings["dataset"] = paths
        _context(loaded, settings)
        loaded.settings.save_settings(settings)
        return {"roots": run_context(loaded, settings).roots.as_dict()}


@in_curation_runtime
def save_settings(values: dict) -> dict:
    allowed = {"stage_defaults", "preprocess", "values"}
    if set(values) - allowed:
        raise ValueError("Only stage_defaults, preprocess and values may be changed here; use /roots to register roots")
    if any(not isinstance(value, dict) for value in values.values()):
        raise ValueError("Settings sections must be objects")
    with scoped_tools(HOME) as loaded:
        data = _settings(loaded)
        data.update(values)
        ctx = _context(loaded, data)
        defaults = data.get("stage_defaults", {})
        for key in ("report_root", "mask_root"):
            if defaults.get(key):
                path = checked_path(defaults[key], ctx)
                if path.is_relative_to(Path(ctx["roots"]["src"])):
                    raise ValueError(f"{key} cannot write under the source root")
        loaded.settings.save_settings(data)
        return data


@in_curation_runtime
def get_schema() -> dict:
    with scoped_tools(HOME) as loaded:
        settings = _settings(loaded)
        ctx = run_context(loaded, settings)
        return {
            "stages": [loaded.S.resolved_schema(sc, **ctx.bindings()) for sc in loaded.S.load_schemas().values()],
            "roots": ctx.roots.as_dict(), "settings": settings,
            "ladder": loaded.D.ladder_schema(),
        }


def _item_paths(loaded, context: dict, rel: str):
    key = loaded.D._rel_key(rel)
    for root in context["roots"].values():
        checked_path(Path(root) / key, context, bases=[root])
        checked_path((Path(root) / key).with_suffix(".txt"), context, bases=[root])
    return key


@in_curation_runtime
def dataset(search: str = "", limit: int = 20000) -> dict:
    with scoped_tools(HOME) as loaded:
        settings = _settings(loaded)
        ctx = run_context(loaded, settings)
        validate_tree(ctx.roots.src)
        return loaded.D.list_items(ctx.roots, pattern=ctx.defaults.get("path_pattern"), query=search, limit=limit)


@in_curation_runtime
def item(rel: str) -> dict:
    with scoped_tools(HOME) as loaded:
        settings = _settings(loaded)
        _item_paths(loaded, _context(loaded, settings), rel)
        roots = run_context(loaded, settings).roots
        for root in (roots.master, roots.dst, roots.masks):
            validate_tree(root)
        return loaded.D.item_detail(roots, rel)


@in_curation_runtime
def caption(rel: str, kind: str, text: str) -> dict:
    with scoped_tools(HOME) as loaded:
        settings = _settings(loaded)
        context = _context(loaded, settings)
        key = _item_paths(loaded, context, rel)
        roots = run_context(loaded, settings).roots
        if kind not in loaded.D.CAPTION_KINDS:
            raise ValueError(f"Caption rung is not editable: {kind}")
        path = checked_path(loaded.D.caption_write_path(roots, key, kind), context)
        if path.is_relative_to(roots.src):
            raise ValueError("Caption edits must use the workspace overlay")
        validate_tree(roots.master if kind == "master" else roots.dst)
        return loaded.D.write_caption(roots, rel, kind, text)


@in_curation_runtime
def parse(text: str) -> dict:
    with scoped_tools(HOME) as loaded:
        return loaded.D.parsed_caption(text)


@in_curation_runtime
def compose(flat_tags: list[str], clauses: list[dict]) -> dict:
    with scoped_tools(HOME) as loaded:
        grammar = loaded.grammar
        values = []
        for row in clauses:
            prefix = row.get("prefix", "On the ")
            if prefix not in (*grammar.CLAUSE_PREFIXES, *grammar.TEXT_PREFIXES):
                raise ValueError("Unknown caption clause prefix")
            values.append(grammar.PositionClause(position=row.get("position", ""), tags=tuple(row.get("tags", [])), prefix=prefix))
        text = grammar.compose_caption(flat_tags, values)
        return {"text": text, "parsed": loaded.D.parsed_caption(text)}


@in_curation_runtime
def exclude(rel: str, excluded: bool, note: str = "") -> dict:
    with scoped_tools(HOME) as loaded:
        settings = _settings(loaded)
        context = _context(loaded, settings)
        _item_paths(loaded, context, rel)
        for root in [*context["roots"].values(), context["workspace"]]:
            validate_tree(root)
        return loaded.D.set_excluded(run_context(loaded, settings).roots, rel, excluded=excluded, note=note)


@in_curation_runtime
def file_path(path: str) -> Path:
    with scoped_tools(HOME) as loaded:
        context = _context(loaded, _settings(loaded))
        resolved = checked_path(path, context)
        if not resolved.is_file():
            raise FileNotFoundError(path)
        # The dataset surface serves artifacts, never credentials or configs
        # that happen to live under a broad user-registered directory.
        if resolved.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".avif", ".txt", ".json"}:
            raise ValueError("Unsupported dataset artifact type")
        return resolved


@in_curation_runtime
def thumbnail(path: str, size: int = 256) -> bytes:
    resolved = file_path(path)
    with scoped_tools(HOME) as loaded:
        return loaded.D._thumb_bytes(str(resolved), resolved.stat().st_mtime, max(16, min(size, 1024)))


@in_curation_runtime
def groups() -> dict:
    with scoped_tools(HOME) as loaded:
        settings = _settings(loaded)
        root = run_context(loaded, settings).report_root
        # Group reports live under each immutable job, not a mutable shared path.
        for job in list_jobs()["jobs"]:
            if job["stage"] == "groups" and job["state"] == "success" and {key: value["path"] for key, value in job["roots"].items()} == settings["dataset"]:
                root = _load_job(job["id"])["report_root"]
                break
        return loaded.D.load_groups(root)


def _load_job(job_id: str) -> dict:
    if not job_id or any(c not in "0123456789abcdef" for c in job_id) or len(job_id) != 32:
        raise FileNotFoundError("Unknown curation job")
    path = JOBS / job_id / "snapshot.json"
    data = _read(path)
    record = _read(path.parent / "record.json")
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["snapshot_sha256"]:
        raise ValueError("Job snapshot changed; refusing to use it")
    return data


def _job_info(snapshot: dict) -> dict:
    directory = Path(snapshot["job_dir"])
    record = _read(directory / "record.json")
    state = _read(directory / "state.json") if (directory / "state.json").exists() else {}
    info = {key: snapshot.get(key) for key in ("id", "stage", "apply", "roots", "values", "created_at", "report_path", "operation", "scope", "steps")}
    info.update(task_id=record.get("task_id"), state=state.get("state", record.get("state", "pending")), error=state.get("error", record.get("error")))
    info["roots"] = {key: {"path": value, "exists": Path(value).is_dir()} for key, value in snapshot["roots"].items()}
    # The daemon owns cancellation and failures before the wrapper starts.
    from webui.services.task_service import task_service
    task = task_service.get_task(record.get("task_id", ""))
    if task is not None and task.state.value != "pending":
        info["state"] = task.state.value
    return info


def list_jobs() -> dict:
    jobs = []
    for path in sorted(JOBS.glob("*/snapshot.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            jobs.append(_job_info(_load_job(path.parent.name)))
        except (OSError, ValueError, KeyError):
            continue
    return {"jobs": jobs}


def _new_context(loaded, settings: dict) -> dict:
    job_id = uuid.uuid4().hex
    context = _context(loaded, settings, job_dir=JOBS / job_id)
    context.update(id=job_id, created_at=datetime.now(timezone.utc).isoformat(), version=1)
    return context


def _validate_values(loaded, sc: dict, values: dict, context: dict) -> None:
    fields = {f["dest"]: f for f in sc["fields"]}
    unknown = set(values) - fields.keys()
    if unknown:
        raise ValueError(f"Unknown stage fields: {', '.join(sorted(unknown))}")
    if values.get("from_report") or values.get("apply"):
        raise ValueError("Use the reviewed job's apply operation instead of raw replay flags")
    for name, value in values.items():
        field = fields[name]
        if name in {"path_pattern", "category_paths"} or not field.get("path") or not value or (field.get("root") or field.get("setting")) and not field.get("overridable"):
            continue
        for path in value if isinstance(value, list) else [value]:
            resolved = checked_path(str(path), context, bases=[*context["roots"].values(), context["workspace"], context["job_dir"], Path(context["home"]) / "models"])
            if sc["id"] == "ocr" and name == "ocr_dir" and resolved.is_relative_to(Path(context["roots"]["src"])):
                raise ValueError("OCR output cannot write under the source root")


@in_curation_runtime
def prepare(stage_id: str, values: dict, apply: bool = False, rel: str = "") -> dict:
    with scoped_tools(HOME) as loaded:
        settings = _settings(loaded)
        schemas = loaded.S.load_schemas()
        sc = schemas.get(stage_id)
        if sc is None:
            raise ValueError(f"Unknown curation stage: {stage_id}")
        if not sc["available"]:
            raise ValueError(f"Stage unavailable: {sc.get('error', stage_id)}")
        if stage_id == "export" and apply:
            raise ValueError("Export requires a preview job; review its report and apply that job")
        context = _new_context(loaded, settings)
        _validate_values(loaded, sc, values, context)
        if stage_id == "export" and values.get("out"):
            checked_path(values["out"], context, bases=[context["roots"]["out"]])
        values = loaded.S.form_values(sc["fields"], values)
        ctx = run_context(loaded, settings)
        mask_root = checked_path(ctx.mask_root, context)
        if mask_root.is_relative_to(Path(context["roots"]["src"])):
            raise ValueError("Mask output cannot write under the source root")
        if rel:
            if not sc.get("scoped"):
                raise ValueError(f"{stage_id} cannot run on a single image")
            _item_paths(loaded, context, rel)
            ctx = ctx.scoped_to(rel)
        bindings = ctx.bindings()
        # Export reads the previous caption index, not its own new report tree.
        if stage_id == "export" and not values.get("index"):
            resolved = loaded.S.resolved_schema(sc, **bindings)
            values["index"] = next(f["default"] for f in resolved["fields"] if f["dest"] == "index")
        report_root = str(Path(context["job_dir"]) / "reports")
        bindings["report_root"] = report_root
        stage_ids = [loaded.S.preprocess_for(stage_id), stage_id, loaded.S.FOLLOW_UPS.get(stage_id)]
        steps = []
        for key in filter(None, stage_ids):
            schema = schemas[key]
            if not schema["available"]:
                raise ValueError(f"Required stage unavailable: {key}")
            vals = values if key == stage_id else (settings.get("preprocess", {}) if key == "resize" else settings.get("values", {}).get(key, {}))
            _validate_values(loaded, schema, vals, context)
            argv = loaded.S.build_argv(schema, vals, apply=apply if key == stage_id else False, **bindings)
            steps.append({"stage": key, "argv": argv, "module": loaded.S.BY_ID[key].module,
                          "report_path": loaded.S.report_path(loaded.S.BY_ID[key], schema["fields"], vals, report_root)})
        report_path = next(step["report_path"] for step in steps if step["stage"] == stage_id)
        context.update(stage=stage_id, values=values, apply=bool(apply), scope=rel, steps=steps,
                       operation="run", report_root=report_root, report_path=report_path,
                       mask_root=ctx.mask_root, path_pattern=ctx.defaults.get("path_pattern", "*"))
        return context


async def _submit(context: dict) -> dict:
    from webui.services.task_service import task_service
    directory = Path(context["job_dir"])
    digest = write_json(directory / "snapshot.json", context, exclusive=True)
    record = {"snapshot_sha256": digest, "state": "pending"}
    write_json(directory / "record.json", record, exclusive=True)
    try:
        task = await task_service.start_task(
            "curation-stage", ["--snapshot", str(directory / "snapshot.json"), "--sha256", digest],
            context={"curation": context}, runtime_profile="curation",
        )
    except Exception as exc:
        write_json(directory / "record.json", {**record, "state": "failed", "error": str(exc)})
        raise
    write_json(directory / "record.json", {**record, "task_id": task.id})
    return _job_info(context)


async def submit(stage: str, values: dict, apply: bool = False, rel: str = "") -> dict:
    context = await asyncio.to_thread(prepare, stage, values, apply, rel)
    result = await _submit(context)
    await asyncio.to_thread(remember_values, stage, context["values"])
    return result


@in_curation_runtime
def remember_values(stage: str, values: dict) -> dict:
    with scoped_tools(HOME) as loaded:
        data = _settings(loaded)
        data.setdefault("values", {})[stage] = values
        loaded.settings.save_settings(data)
        return {}


@in_curation_runtime
def report(job_id: str) -> dict:
    context = _load_job(job_id)
    from scripts.curation.runner import sealed_report
    with scoped_tools(context["home"], context["settings"], context["environment"]) as loaded:
        path, data = sealed_report(loaded, context)
        return {"path": str(path), "report": data}


def _proposals(job_id: str) -> tuple[dict, dict]:
    context = _load_job(job_id)
    data = report(job_id)["report"]
    with scoped_tools(context["home"], context["settings"], context["environment"]) as loaded:
        if context["stage"] == "export":
            # Every artifact has its own key: one image may publish a mask,
            # caption, layout, variants and a master revision simultaneously.
            rows = {}
            for index, row in enumerate(data.get("rows", [])):
                key = f"{index}:{row.get('kind')}:{row.get('rel')}"
                rows[key] = {"rel": key, "image": row.get("src", ""), "kind": row.get("kind", "export"),
                             "path": row.get("dst", ""), "before": row.get("before", ""),
                             "after": row.get("text") or row.get("preview_text") or f"{row.get('status')}: {row.get('src')} → {row.get('dst')}",
                             "status": row.get("status", "proposed")}
            return context, rows
        path, _, roots = checked_report(loaded, context)
        return context, {key: value.to_dict() for key, value in loaded.P.read(path, roots, context["stage"]).items()}


@in_curation_runtime
def proposals(job_id: str) -> dict:
    context, found = _proposals(job_id)
    return {"stage": context["stage"], "apply": context["apply"], "kind": "export" if context["stage"] == "export" else "revised", "total": len(found), "rels": sorted(found)}


@in_curation_runtime
def proposal(job_id: str, rel: str) -> dict:
    _, found = _proposals(job_id)
    if rel not in found:
        raise FileNotFoundError(f"No proposal for {rel}")
    return found[rel]


@in_curation_runtime
def prepare_replay(job_id: str, operation: str) -> dict:
    source = _load_job(job_id)
    if operation not in {"apply", "undo"}:
        raise ValueError("Unknown report operation")
    if operation == "undo" and (not source["apply"] or source["operation"] == "undo"):
        raise ValueError("Only a job that applied changes can be undone")
    if operation == "apply" and source["apply"]:
        raise ValueError("This job already applied changes; create a new preview")
    from scripts.curation.runner import sealed_report, validate_operation
    with scoped_tools(source["home"], source["settings"], source["environment"]) as loaded:
        path, data = sealed_report(loaded, source)
        validate_operation(loaded, source, data, operation)
        context = _new_context(loaded, source["settings"])
        # Everything historical stays historical, including workspace/env.
        context.update({key: source[key] for key in ("home", "roots", "workspace", "environment", "stage", "values", "scope", "path_pattern")})
        context.update(operation=operation, apply=True, steps=[], source=source,
                       source_report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                       report_path=str(Path(context["job_dir"]) / "report.json"),
                       report_root=str(Path(context["job_dir"]) / "reports"))
    return context


async def replay(job_id: str, operation: str) -> dict:
    source = _load_job(job_id)
    if _job_info(source)["state"] in {"pending", "queued", "running", "stopping"}:
        raise ValueError("Wait for this job to finish before applying or undoing it")
    return await _submit(await asyncio.to_thread(prepare_replay, job_id, operation))
