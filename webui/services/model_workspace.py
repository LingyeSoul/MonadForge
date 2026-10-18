"""Shared TOML profile store for the typed model lines (Qwen-Image-2.1, MOSS).

Both lines keep one profile per name under ``configs/custom/<line>/``, validated
against their own request dataclasses: name validation, symlink refusal, the
model/section split, choice validation and the atomic write are identical, so
they live here. A line passes its ``section_types`` / ``model_fields`` /
``Profile`` class in; everything else is shared.

``root_resolver`` is called on every access (not captured once) so a caller —
or a test — can redirect the profile tree without rebuilding the store.
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import toml
from pydantic import BaseModel, TypeAdapter

from webui.services.config_storage import atomic_write_text

# Same rule for every line: a profile name is a file stem, never a path.
PROFILE_NAME_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")


@dataclass(frozen=True)
class ProfileStore:
    """One line's profile tree."""

    root: str  # relative config root, e.g. "configs/custom/qwen21"
    label: str  # user-facing line name, e.g. "Qwen" / "MOSS"
    section_types: dict[str, type]  # section name → request dataclass
    model_fields: frozenset[str]  # keys that live in the `models` section
    profile_cls: type[BaseModel]
    empty_factory: Callable[[], BaseModel]
    root_resolver: Callable[[str], Path]

    def path(self, name: str) -> Path:
        """The profile file for ``name``; rejects paths and symlinks."""
        if not PROFILE_NAME_RE.fullmatch(name):
            raise ValueError(
                "Profile name must contain 1–64 letters, digits, underscores or hyphens"
            )
        base = self.root_resolver(self.root).resolve()
        path = base / f"{name}.toml"
        if path.is_symlink():
            raise ValueError(f"{self.label} profiles cannot be symbolic links")
        return path

    def list(self) -> list[str]:
        directory = self.path("default").parent
        return sorted(
            {
                "default",
                *(
                    path.stem
                    for path in directory.glob("*.toml")
                    if not path.is_symlink()
                ),
            }
        )

    def read(self, name: str) -> BaseModel:
        path = self.path(name)
        if not path.exists():
            if name != "default":
                raise FileNotFoundError(f"{self.label} profile does not exist: {name}")
            return self.normalize(self.empty_factory())
        return self.normalize(
            self.profile_cls.model_validate(
                toml.loads(path.read_text(encoding="utf-8")), strict=True
            )
        )

    def save(self, name: str, profile: BaseModel) -> BaseModel:
        normalized = self.normalize(profile)
        path = self.path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, toml.dumps(normalized.model_dump(exclude_none=True)))
        return normalized

    def normalize(self, profile: BaseModel) -> BaseModel:
        """Validate every section against its request and re-collect the model keys.

        A missing optional key restores the request default; a value outside a
        field's ``choices`` is refused here rather than at submission time.
        """
        models: dict[str, Any] = {
            key: value
            for key, value in profile.models.items()
            if key in self.model_fields
        }
        sections: dict[str, dict[str, Any]] = {}
        for name, cls in self.section_types.items():
            values = {
                key: value
                for key, value in getattr(profile, name).items()
                if key not in self.model_fields
            }
            req = TypeAdapter(cls).validate_json(
                json.dumps({**values, **models}), strict=True
            )
            for field in dataclasses.fields(req):
                choices = field.metadata.get("choices")
                if choices is not None and getattr(req, field.name) not in choices:
                    raise ValueError(f"{name}.{field.name} must be one of {choices}")
            full = dataclasses.asdict(req)
            sections[name] = {
                key: value
                for key, value in full.items()
                if key not in self.model_fields
            }
            models = {
                key: value for key, value in full.items() if key in self.model_fields
            }
        return self.profile_cls(models=models, **sections)
