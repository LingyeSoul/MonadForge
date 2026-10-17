"""The typed-request base the non-Anima lines' ``requests.py`` build on.

Torch-free on purpose: the GUI builds its forms from the request dataclasses and
must not pull torch in at launch. Each field carries its CLI help in
``metadata``, so a line's sidecar CLI and the GUI read one definition:

    req = TrainRequest(cache="…", epochs=4)
    argv = req.to_argv()                 # → daemon job argv
    req = TrainRequest.from_argv(argv)   # ← sidecar CLI

``to_argv`` writes only fields that differ from their defaults, so a job record
reads as the choices that were made.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
from dataclasses import field
from pathlib import Path
from typing import Any, ClassVar

from library.env import load_dotenv, resolve_under_home


def env_model_dir(env_name: str, fallback: Path) -> Path:
    """``env_name`` (env or ``.env``) if set, else ``fallback``."""
    load_dotenv()
    override = os.environ.get(env_name)
    if override:
        return resolve_under_home(override)
    return fallback


def request_field(
    default: Any,
    help: str,
    *,
    choices: tuple | None = None,
    advanced=False,
    multiline=False,
):
    """A request field: ``help`` / ``choices`` / ``advanced`` / ``multiline``
    ride in metadata (the last two are GUI layout only)."""
    meta = {
        "help": help,
        "choices": choices,
        "advanced": advanced,
        "multiline": multiline,
    }
    return field(default=default, metadata=meta)


class Request:
    """argv round-trip over the dataclass fields.

    Types map to argparse by the field's default: ``bool`` is a flag (``--x`` when
    the default is False, ``--no-x`` when it is True), ``None`` defaults are
    typed from the annotation, the rest from the default's own type.
    """

    SCRIPT: ClassVar[str]

    @classmethod
    def parser(cls) -> argparse.ArgumentParser:
        ap = argparse.ArgumentParser(description=cls.__doc__)
        for f in dataclasses.fields(cls):
            flag = f"--{f.name}"
            help = f.metadata.get("help")
            if isinstance(f.default, bool):
                ap.add_argument(
                    flag,
                    action=argparse.BooleanOptionalAction,
                    default=f.default,
                    help=help,
                )
                continue
            ap.add_argument(
                flag,
                type=_field_type(f),
                default=f.default,
                choices=f.metadata.get("choices"),
                required=f.default is dataclasses.MISSING,
                help=help,
            )
        return ap

    @classmethod
    def from_argv(cls, argv: list[str] | None = None):
        return cls(**vars(cls.parser().parse_args(argv)))

    def to_argv(self) -> list[str]:
        argv: list[str] = []
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if f.default is not dataclasses.MISSING and value == f.default:
                continue
            if isinstance(value, bool):
                argv.append(f"--{f.name}" if value else f"--no-{f.name}")
            elif value is not None:
                argv += [f"--{f.name}", str(value)]
        return argv


def _field_type(f: dataclasses.Field) -> type:
    annotation = str(f.type)
    if "int" in annotation:
        return int
    if "float" in annotation:
        return float
    return str
