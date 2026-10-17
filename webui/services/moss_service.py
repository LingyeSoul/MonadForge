"""MOSS schema, path validation, task identity and result manifests; no torch."""

from __future__ import annotations

import dataclasses
import json
import uuid
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ConfigDict, TypeAdapter

from library.env import resolve_under_home
from library.moss.requests import (
    CacheRequest,
    GenerateRequest,
    TrainRequest,
    TuneSwapRequest,
    resolve_model_dir,
    variant_dir,
)
from library.moss.validation import validate_request
from scripts.moss.workflow import parse_workflow
from webui.services.qwen21_service import FormField, form_fields
from webui.services.qwen21_workspace import check_component_headers

Scalar = str | int | float | bool | None
RequestT = TypeVar(
    "RequestT", CacheRequest, GenerateRequest, TrainRequest, TuneSwapRequest
)
REQUEST_TYPES = {
    "moss-cache": CacheRequest,
    "moss-train": TrainRequest,
    "moss-generate": GenerateRequest,
    "moss-tune": TuneSwapRequest,
}
MOSS_COMMANDS = frozenset({*REQUEST_TYPES, "moss-cache-train"})


class ValuesRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore", allow_inf_nan=False)
    values: dict[str, Scalar]
    runtime_profile: str = "default"


class ChainRequest(BaseModel):
    cache: ValuesRequest
    train: ValuesRequest
    runtime_profile: str = "default"


class FormSchema(BaseModel):
    cache: list[FormField]
    train: list[FormField]
    generate: list[FormField]
    tune: list[FormField]


class ResolvedPaths(BaseModel):
    model_dir: str
    dit: str
    text_encoder: str
    vae: str
    tokenizer: str


def read_schema() -> FormSchema:
    return FormSchema(
        **{
            phase: form_fields(cls)
            for phase, cls in (
                ("cache", CacheRequest),
                ("train", TrainRequest),
                ("generate", GenerateRequest),
                ("tune", TuneSwapRequest),
            )
        }
    )


def request_from_values(
    request_type: type[RequestT], values: dict[str, Scalar]
) -> RequestT:
    req = TypeAdapter(request_type).validate_json(json.dumps(values), strict=True)
    validate_request(req)
    return req


def resolve_values(values: dict[str, Scalar]) -> ResolvedPaths:
    req = request_from_values(TrainRequest, values)
    root = resolve_model_dir(req.model_dir)
    base = variant_dir(req.model_dir, "base")
    return ResolvedPaths(
        model_dir=str(root),
        dit=str(variant_dir(req.model_dir, req.model) / "transformer"),
        text_encoder=str(base / "text_encoder"),
        vae=str(base / "vae"),
        tokenizer=str(base / "tokenizer"),
    )


def output_path(value: str) -> Path:
    path = resolve_under_home(value).resolve()
    roots = [
        resolve_under_home("output/moss").resolve(),
        resolve_under_home("output/ckpt_moss").resolve(),
    ]
    if not any(path == root or path.is_relative_to(root) for root in roots):
        raise ValueError(
            "Browser MOSS outputs must stay inside output/moss or output/ckpt_moss"
        )
    return path


def require_directory(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} directory not found: {path}")


def require_component(path: Path, label: str) -> None:
    require_directory(path, label)
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"{label} config.json not found: {path}")
    check_component_headers(path)


def validate_paths(req) -> None:
    base = variant_dir(req.model_dir, "base")
    if isinstance(req, CacheRequest):
        require_directory(resolve_under_home(req.src), "Image source")
        output_path(req.out)
        if not req.skip_text:
            require_component(base / "text_encoder", "MOSS text encoder")
            require_directory(base / "tokenizer", "MOSS tokenizer")
        if not req.skip_latents:
            require_component(base / "vae", "MOSS VAE")
    elif isinstance(req, TrainRequest):
        require_component(
            variant_dir(req.model_dir, req.model) / "transformer", "MOSS transformer"
        )
        output_path(req.output)
    elif isinstance(req, GenerateRequest):
        output_path(req.out_dir)
        require_component(base / "vae", "MOSS VAE")
        if not req.decode_only:
            require_component(
                variant_dir(req.model_dir, req.model) / "transformer",
                "MOSS transformer",
            )
            require_component(base / "text_encoder", "MOSS text encoder")
            require_directory(base / "tokenizer", "MOSS tokenizer")
        for value in (req.lora, req.prompts_file, *req.example.split(",")):
            if value and not resolve_under_home(value.strip()).is_file():
                raise FileNotFoundError(f"MOSS input file not found: {value}")
    elif isinstance(req, TuneSwapRequest):
        require_component(
            variant_dir(req.model_dir, req.model) / "transformer", "MOSS transformer"
        )
        output_path(req.output)
        if not req.sizes:
            require_directory(resolve_under_home(req.cache), "MOSS cache")


def pin_request_paths(req: RequestT) -> RequestT:
    names = {
        "src",
        "out",
        "cache",
        "output",
        "out_dir",
        "model_dir",
        "lora",
        "prompts_file",
    }
    values = {
        field.name: str(resolve_under_home(getattr(req, field.name)))
        for field in dataclasses.fields(req)
        if field.name in names and getattr(req, field.name)
    }
    values["model_dir"] = str(resolve_model_dir(req.model_dir))
    if isinstance(req, GenerateRequest) and req.example:
        values["example"] = ",".join(
            str(resolve_under_home(value.strip()))
            for value in req.example.split(",")
            if value.strip()
        )
    return dataclasses.replace(req, **values)


def prepare_job_args(command: str, args: list[str]) -> list[str]:
    try:
        if command == "moss-cache-train":
            cache, train = parse_workflow(args)
            validate_paths(cache)
            validate_paths(train)
            return [
                "--cache-args",
                json.dumps(pin_request_paths(cache).to_argv()),
                "--train-args",
                json.dumps(pin_request_paths(train).to_argv()),
            ]
        req = REQUEST_TYPES[command].from_argv(args)
    except (SystemExit, KeyError) as exc:
        raise ValueError(f"Invalid MOSS command or arguments: {command}") from exc
    validate_request(req)
    validate_paths(req)
    if isinstance(req, TrainRequest):
        from .moss_workspace import inspect_cache

        status = inspect_cache(req.cache)
        if not status.ready:
            raise ValueError(f"MOSS cache is not ready: {status.model_dump_json()}")
    if isinstance(req, GenerateRequest) and not req.decode_only:
        req = dataclasses.replace(
            req, out_dir=str(output_path(req.out_dir) / uuid.uuid4().hex)
        )
    return pin_request_paths(req).to_argv()


def task_context(command: str, args: list[str]) -> dict:
    if command == "moss-cache-train":
        cache, req = parse_workflow(args)
        return {
            "model_type": "moss",
            "phase": "cache-train",
            "cache_dir": cache.out,
            "output": req.output,
            "model": req.model,
        }
    req = REQUEST_TYPES[command].from_argv(args)
    context = {
        "model_type": "moss",
        "phase": command.removeprefix("moss-"),
        "model": getattr(req, "model", "base"),
    }
    for attr in ("out_dir", "output", "cache", "out"):
        if hasattr(req, attr):
            context[{"out_dir": "output_dir", "out": "cache_dir"}.get(attr, attr)] = (
                getattr(req, attr)
            )
    return context


class ResultImage(BaseModel):
    file: str
    prompt: str = ""
    seed: int = 0
    width: int = 0
    height: int = 0
    cfg: float = 0
    model: str = "base"


class ResultManifest(BaseModel):
    directory: str
    images: list[ResultImage]


def result_image_path(directory: str, file: str) -> Path:
    root = output_path(directory)
    path = (root / file).resolve()
    if path.parent != root or path.suffix.lower() != ".png":
        raise ValueError(
            "MOSS result images must be PNGs directly inside the task output directory"
        )
    if not path.is_file():
        raise FileNotFoundError(f"MOSS result image does not exist: {file}")
    return path


def read_results(directory: str) -> ResultManifest:
    root = output_path(directory)
    manifest_path = root / "manifest.json"
    if manifest_path.resolve().parent != root:
        raise ValueError("MOSS manifest must stay inside the result directory")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("images"), list):
        raise ValueError("Invalid MOSS result manifest")
    argv = data.get("request", [])
    if not isinstance(argv, list) or any(not isinstance(arg, str) for arg in argv):
        raise ValueError("Invalid MOSS result request")
    try:
        model = GenerateRequest.from_argv(argv).model
    except SystemExit as exc:
        raise ValueError("Invalid MOSS result request") from exc
    images = []
    for record in data.get("images", []):
        if not isinstance(record, dict):
            raise ValueError("Invalid MOSS result image")
        image = record.get("image")
        if not image:
            continue
        path = Path(image)
        if not path.is_absolute():
            path = root / path
        file = path.name
        if path.resolve() != result_image_path(str(root), file):
            raise ValueError("MOSS manifest image leaves the task result directory")
        images.append(
            ResultImage.model_validate(
                {"model": model, **record, "file": file}, strict=True
            )
        )
    return ResultManifest(directory=str(root), images=images)
