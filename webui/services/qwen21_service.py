"""Torch-free Qwen forms and submission validation, independent of Anima presets.

Browser writes are confined to output/qwen21; CLI jobs retain their explicit
path choices. Model checks inspect paths only, never load a weight tensor.
"""

from __future__ import annotations

import dataclasses
import json
import uuid
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict, TypeAdapter

from library.env import resolve_under_home
from library.qwen21.requests import (
    CacheRequest,
    GenerateRequest,
    TrainRequest,
    model_paths,
)
from library.qwen21.validation import validate_request
from scripts.qwen21.workflow import parse_workflow

Scalar = str | int | float | bool | None
Request = CacheRequest | TrainRequest | GenerateRequest
RequestT = TypeVar("RequestT", CacheRequest, TrainRequest, GenerateRequest)
QWEN_COMMANDS = frozenset(
    {"qwen21-cache", "qwen21-train", "qwen21-generate", "qwen21-cache-train"}
)
REQUEST_TYPES: dict[
    str, type[CacheRequest] | type[TrainRequest] | type[GenerateRequest]
] = {
    "qwen21-cache": CacheRequest,
    "qwen21-train": TrainRequest,
    "qwen21-generate": GenerateRequest,
}


class ValuesRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")
    values: dict[str, Scalar]


class ChainRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")
    cache: ValuesRequest
    train: ValuesRequest


class FormField(BaseModel):
    name: str
    kind: Literal["str", "int", "float", "bool"]
    value: Scalar
    nullable: bool
    choices: list[str]
    help: str
    advanced: bool
    multiline: bool


class FormSchema(BaseModel):
    cache: list[FormField]
    train: list[FormField]
    generate: list[FormField]


class ResolvedPaths(BaseModel):
    dit: str
    text_encoder: str
    vae: str
    processor: str
    scheduler: str


def form_fields(request_type: type[RequestT]) -> list[FormField]:
    req = request_type()
    result: list[FormField] = []
    for field in dataclasses.fields(req):
        annotation = str(field.type)
        kind: Literal["str", "int", "float", "bool"] = "str"
        if "bool" in annotation:
            kind = "bool"
        elif "int" in annotation:
            kind = "int"
        elif "float" in annotation:
            kind = "float"
        result.append(
            FormField(
                name=field.name,
                kind=kind,
                value=getattr(req, field.name),
                nullable="None" in annotation,
                choices=list(field.metadata.get("choices") or ()),
                help=str(field.metadata["help"]),
                advanced=bool(field.metadata.get("advanced")),
                multiline=bool(field.metadata.get("multiline")),
            )
        )
    return result


def read_schema() -> FormSchema:
    return FormSchema(
        cache=form_fields(CacheRequest),
        train=form_fields(TrainRequest),
        generate=form_fields(GenerateRequest),
    )


def request_from_values(
    request_type: type[RequestT], values: dict[str, Scalar]
) -> RequestT:
    req = TypeAdapter(request_type).validate_json(json.dumps(values), strict=True)
    validate_request(req)
    return req


def resolve_values(values: dict[str, Scalar]) -> ResolvedPaths:
    req = request_from_values(TrainRequest, values)
    paths = model_paths(req)
    return ResolvedPaths(
        **{
            field.name: str(getattr(paths, field.name))
            for field in dataclasses.fields(paths)
        }
    )


def output_path(value: str) -> Path:
    path = resolve_under_home(value).resolve()
    root = resolve_under_home("output/qwen21").resolve()
    if not path.is_relative_to(root):
        raise ValueError(
            f"WebUI Qwen output must stay under {root}; got {path}. Use the CLI for other output locations"
        )
    return path


def require_directory(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} directory does not exist: {path}")


def require_component(path: Path, label: str) -> None:
    if path.is_dir():
        if not (path / "config.json").is_file():
            raise FileNotFoundError(
                f"{label} configuration missing: {path / 'config.json'}"
            )
    elif not (path.is_file() and path.suffix == ".safetensors"):
        raise FileNotFoundError(
            f"{label} must be a local Diffusers component directory or Comfy BF16 safetensors file: {path}"
        )


def validate_cache_paths(req: CacheRequest) -> None:
    from webui.services.qwen21_workspace import check_component_headers

    paths = model_paths(req)
    require_directory(resolve_under_home(req.src), "Dataset")
    if not req.skip_text:
        require_component(paths.text_encoder, "Qwen3-VL")
        check_component_headers(paths.text_encoder)
    if not req.skip_latents:
        require_component(paths.vae, "RGBA VAE")
        check_component_headers(paths.vae)
    if not (req.skip_text and req.skip_latents):
        require_directory(paths.processor, "Processor")
    output_path(req.out)


def validate_training_model(req: TrainRequest) -> None:
    from webui.services.qwen21_workspace import check_component_headers

    paths = model_paths(req)
    require_component(paths.dit, "DiT")
    check_component_headers(paths.dit)
    scheduler = (
        paths.scheduler / "scheduler_config.json"
        if paths.scheduler.is_dir()
        else paths.scheduler
    )
    if not scheduler.is_file():
        raise FileNotFoundError(f"Scheduler configuration does not exist: {scheduler}")
    if output_path(req.output).suffix != ".safetensors":
        raise ValueError("Qwen LoRA output must end in .safetensors")


def validate_generate_paths(req: GenerateRequest) -> None:
    from webui.services.qwen21_workspace import check_component_headers

    paths = model_paths(req)
    for path, label in (
        (paths.dit, "DiT"),
        (paths.text_encoder, "Qwen3-VL"),
        (paths.vae, "RGBA VAE"),
    ):
        require_component(path, label)
        check_component_headers(path)
    require_directory(paths.processor, "Processor")
    scheduler = (
        paths.scheduler / "scheduler_config.json"
        if paths.scheduler.is_dir()
        else paths.scheduler
    )
    if not scheduler.is_file():
        raise FileNotFoundError(f"Scheduler configuration does not exist: {scheduler}")
    for value, label in ((req.lora, "LoRA"), (req.prompts_file, "Prompts")):
        if value and not resolve_under_home(value).is_file():
            raise FileNotFoundError(
                f"{label} file does not exist: {resolve_under_home(value)}"
            )
    output_path(req.out_dir)


def prepare_job_args(command: str, args: list[str]) -> list[str]:
    try:
        if command == "qwen21-cache-train":
            cache, train = parse_workflow(args)
            validate_cache_paths(cache)
            validate_training_model(train)
            return [
                "--cache-args",
                json.dumps(pin_request_paths(cache).to_argv()),
                "--train-args",
                json.dumps(pin_request_paths(train).to_argv()),
            ]
        request_type = REQUEST_TYPES[command]
        req = request_type.from_argv(args)
    except (SystemExit, KeyError) as exc:
        raise ValueError(f"Invalid Qwen command/arguments: {command} {args!r}") from exc
    validate_request(req)
    if isinstance(req, CacheRequest):
        validate_cache_paths(req)
    elif isinstance(req, TrainRequest):
        validate_training_model(req)
        require_directory(resolve_under_home(req.cache), "Training cache")
        from webui.services.qwen21_workspace import inspect_cache

        status = inspect_cache(req.cache)
        if not status.ready:
            raise ValueError(
                f"Qwen training cache is not ready: {status.model_dump_json()}"
            )
    else:
        validate_generate_paths(req)
        req = dataclasses.replace(
            req, out_dir=str(output_path(req.out_dir) / uuid.uuid4().hex)
        )
    return pin_request_paths(req).to_argv()


class ResultImage(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore", allow_inf_nan=False)
    file: str
    index: int
    multiplier: float
    seed: int
    prompt: str
    denoise_seconds: float


class ResultManifest(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")
    images: list[ResultImage]
    size: str
    steps: int
    lora: str


def result_image_path(directory: str, file: str) -> Path:
    root = output_path(directory)
    path = (root / file).resolve()
    if path.parent != root or path.suffix.lower() != ".png":
        raise ValueError(
            f"Result image must be a PNG directly inside {root}; got {file!r}"
        )
    if not path.is_file():
        raise FileNotFoundError(f"Qwen result image does not exist: {path}")
    return path


def read_results(directory: str) -> ResultManifest:
    path = output_path(str(output_path(directory) / "manifest.json"))
    manifest = ResultManifest.model_validate_json(
        path.read_text(encoding="utf-8"), strict=True
    )
    for image in manifest.images:
        result_image_path(directory, image.file)
    return manifest


def pin_request_paths(req: RequestT) -> RequestT:
    """Snapshot resolved paths in job argv; leave all hardware auto knobs unset."""
    paths = model_paths(req)
    updates = {
        field.name: str(getattr(paths, field.name).resolve())
        for field in dataclasses.fields(paths)
    }
    for name in ("src", "out", "cache", "output", "out_dir", "lora", "prompts_file"):
        value = getattr(req, name, "")
        if value:
            updates[name] = str(resolve_under_home(value).resolve())
    return dataclasses.replace(req, **updates)
