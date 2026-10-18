"""The typed-request base the non-Anima lines' ``requests.py`` build on.

Torch-free on purpose: the GUI builds its forms from the request dataclasses and
must not pull torch in at launch. Each field carries its CLI help in
``metadata``, so a line's sidecar CLI and the GUI read one definition:

    req = TrainRequest(cache="…", epochs=4)
    argv = req.to_argv()                 # → daemon job argv
    req = TrainRequest.from_argv(argv)   # ← sidecar CLI

``to_argv`` writes only fields that differ from their defaults, so a job record
reads as the choices that were made.

``--config <file.toml>`` (e.g. ``configs/moss/lora.toml``) is a flat table of
field names; it replaces the field defaults and the CLI still wins over it:
``field default → --config → CLI``.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import re
import tomllib
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


# The form's group headings, borrowed from Anima (``gui/core/config_io.py::
# _GROUPS`` plus its trailing ``Other``) so the same kind of knob reads the same
# way on every line. The tuple order is the order the GUI draws the boxes in.
_GROUP_FIELDS: dict[str, tuple[str, ...]] = {
    "Paths": (
        "src",
        "out",
        "out_dir",
        "cache",
        "output",
        "model_dir",
        "lora",
        "prompts_file",
        "example",
        "scheduler",
        "dit",
        "text_encoder",
        "vae",
        "processor",
    ),
    "Training": (
        "epochs",
        "max_steps",
        "lr",
        "warmup_ratio",
        "save_every_epochs",
        "caption_dropout",
        "max_grad_norm",
        "logit_mean",
        "logit_std",
        "sigma_mu",
        "sigma_s",
        "sigma_cutoff",
        "sigma_uniform",
        "seed",
    ),
    "Architecture": (
        "algo",
        "rank",
        "alpha",
        "factor",
        "down_init",
        "targets",
        "model",
    ),
    "Performance": (
        "blocks_to_swap",
        "te_blocks_to_swap",
        "activation_memory_budget",
        "activation_reserve_gb",
        "grad_checkpointing",
        "compile",
        "compile_seq",
        "compile_mode",
        "lora_dtype",
        "vae_dtype",
        "batched",
        "spectrum",
        "spectrum_window_size",
        "spectrum_flex_window",
        "spectrum_warmup",
        "spectrum_stop_caching_step",
        "spectrum_w",
        "spectrum_m",
        "spectrum_lam",
    ),
    "Other": (
        "resolution",
        "caption_format",
        "prompt",
        "negative_prompt",
        "prompt_format",
        "lora_multiplier",
        "multipliers",
        "width",
        "height",
        "steps",
        "cfg",
        "true_cfg_scale",
        "sampler",
        "uncond_skip",
        "shift",
        "decode_only",
        "overwrite",
        "skip_text",
        "skip_latents",
        "save_crops",
        "max_text_tokens",
        "log_every",
        "sizes",
        "swap_grid",
        "patience",
        "margin_gb",
        "tolerance",
        "reference",
        "budget_step",
        "warmup",
    ),
}

# Group headings, in form order.
GROUPS: tuple[str, ...] = tuple(_GROUP_FIELDS)

# Field name → group. Keyed by name because the lines' request modules
# (``library/moss/requests.py``, ``library/qwen21/requests.py``) declare their
# fields through :func:`request_field` without naming a group; a name is mapped
# once so a knob that several requests share (``model_dir``, ``seed``, …) reads
# the same everywhere. ``group=`` on the field itself still wins.
_GROUPS_BY_FIELD: dict[str, str] = {
    name: group for group, names in _GROUP_FIELDS.items() for name in names
}


def field_group(f: dataclasses.Field) -> str | None:
    """The form group of ``f``: its own ``group`` metadata if set, else the
    shared name table, else ``None`` — an ungrouped field keeps the flat form."""
    return f.metadata.get("group") or _GROUPS_BY_FIELD.get(f.name)


def request_field(
    default: Any,
    help: str,
    *,
    choices: tuple | None = None,
    advanced=False,
    multiline=False,
    group: str | None = None,
):
    """A request field: ``help`` / ``choices`` ride in metadata (``advanced`` /
    ``multiline`` / ``group`` are GUI layout only). ``group`` overrides the
    shared :func:`field_group` table for this field."""
    meta = {
        "help": help,
        "choices": choices,
        "advanced": advanced,
        "multiline": multiline,
        "group": group,
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
    def parser(cls, defaults: dict | None = None) -> argparse.ArgumentParser:
        """``defaults`` (a loaded ``--config``) replace the field defaults."""
        defaults = defaults or {}
        ap = argparse.ArgumentParser(description=cls.__doc__)
        ap.add_argument(
            "--config",
            default=None,
            help="TOML of field defaults (flat table of flag names); CLI flags win",
        )
        for f in dataclasses.fields(cls):
            flag = f"--{f.name}"
            help = re.sub(r"(?<!%)%(?!%)", "%%", str(f.metadata.get("help") or ""))
            default = defaults.get(f.name, f.default)
            if isinstance(f.default, bool):
                ap.add_argument(
                    flag,
                    action=argparse.BooleanOptionalAction,
                    default=default,
                    help=help,
                )
                continue
            ap.add_argument(
                flag,
                type=_field_type(f),
                default=default,
                choices=f.metadata.get("choices"),
                required=default is dataclasses.MISSING,
                help=help,
            )
        return ap

    @classmethod
    def from_argv(cls, argv: list[str] | None = None):
        pre = argparse.ArgumentParser(add_help=False)
        pre.add_argument("--config", default=None)
        config = pre.parse_known_args(argv)[0].config
        defaults = cls.load_config(config) if config else None
        args = vars(cls.parser(defaults).parse_args(argv))
        args.pop("config")
        return cls(**args)

    @classmethod
    def load_config(cls, path: str | Path) -> dict[str, Any]:
        """A ``--config`` TOML, checked against the fields (names, types, choices)."""
        path = resolve_under_home(path)
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
        fields = {f.name: f for f in dataclasses.fields(cls)}
        unknown = sorted(set(data) - set(fields))
        if unknown:
            raise ValueError(f"{path}: not {cls.__name__} fields: {unknown}")
        out = {}
        for name, value in data.items():
            f = fields[name]
            if isinstance(f.default, bool):
                want = bool
            else:
                want = _field_type(f)
            if want is float and type(value) is int:
                value = float(value)
            if type(value) is not want:
                raise ValueError(f"{path}: {name} = {value!r} is not {want.__name__}")
            choices = f.metadata.get("choices")
            if choices and value not in choices:
                raise ValueError(f"{path}: {name} = {value!r} not in {choices}")
            out[name] = value
        return out

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
