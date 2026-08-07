"""Snapshot TOML source — the merged training config with section provenance.

``train.py --print-config`` writes ``<run>.snapshot.toml`` next to the
tensorboard dir. Keys are grouped by ``# --- from <source> ---`` header
comments (base preset / method variant / custom overlay); the tool renders
them as editorial section blocks instead of one flat table.
"""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, field
from typing import Any, Optional

_HEADER_RE = re.compile(r"^\s*#\s*---\s*from\s+(.+?)\s*---\s*$")

# Keys that are noise for the UI (long absolute paths / boilerplate)
_SKIP_KEYS = {
    "pretrained_model_name_or_path",
    "qwen3",
    "vae",
    "vae_cache_dir",
    "cache_dir",
    "lora_cache_dir",
    "sdxl_cache_text_encoder_outputs",
}

# C10：参数页顶部关键微标行（白名单，存在则显示）
KEYLINE_KEYS = (
    "learning_rate",
    "network_dim",
    "alpha_rank_scale",
    "max_train_epochs",
    "lr_scheduler",
    "optimizer_type",
    "timestep_sampling",
    "network_module",
    "save_every_n_epochs",
    "gradient_accumulation_steps",
    "masked_loss",
    "sample_sampler",
)


@dataclass
class Snapshot:
    sections: list[dict] = field(default_factory=list)  # [{source, keys:[{k,v}]}]
    file: Optional[str] = None
    parse_error: Optional[str] = None


def keyline(snapshot: "Snapshot") -> list[dict]:
    """从合并参数中提取常用关键键（C10）。"""
    merged: dict = {}
    for sec in snapshot.sections:
        for item in sec["keys"]:
            merged.setdefault(item["k"], item["v"])
    out = []
    for k in KEYLINE_KEYS:
        if k in merged:
            v = merged[k]
            out.append({"k": k, "v": v if not isinstance(v, (list, dict)) else str(v)})
    return out


def _jsonable(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return str(v)


def parse(path: str) -> Optional[Snapshot]:
    """Parse a snapshot.toml, preserving per-source section grouping."""
    if not os.path.isfile(path):
        return None
    out = Snapshot(file=path)
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError:
        return None

    # split raw text into (source_header, body) chunks by comment lines
    text = raw.decode("utf-8", errors="replace")
    chunks: list[tuple[Optional[str], list[str]]] = []
    cur_header: Optional[str] = None
    cur_body: list[str] = []
    header_rows: list[str] = []
    for line in text.splitlines():
        m = _HEADER_RE.match(line)
        if m:
            if cur_body or header_rows:
                chunks.append((cur_header, header_rows + cur_body))
                cur_body = []
                header_rows = []
            cur_header = m.group(1).strip()
        elif line.lstrip().startswith("# --- from"):
            continue
        elif cur_header is None:
            header_rows.append(line)
        else:
            cur_body.append(line)
    if cur_body or header_rows:
        chunks.append((cur_header, header_rows + cur_body))

    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        out.parse_error = str(exc)
        data = {}
        for _h, body in chunks:
            try:
                data.update(tomllib.loads("\n".join(body)))
            except tomllib.TOMLDecodeError:
                pass

    seen = set()
    for header, body in chunks:
        if header is None:
            continue
        try:
            part = tomllib.loads("\n".join(body))
        except tomllib.TOMLDecodeError:
            part = {}
        keys = []
        for k, v in part.items():
            if k in seen or k in _SKIP_KEYS:
                continue
            seen.add(k)
            keys.append({"k": k, "v": _jsonable(v)})
        if keys:
            out.sections.append({"source": header, "keys": keys})

    # keys only present in the merged data (defensive)
    merged_keys = [k for k in data if k not in seen and k not in _SKIP_KEYS]
    if merged_keys:
        out.sections.append(
            {"source": "(merged)", "keys": [{"k": k, "v": _jsonable(data[k])} for k in merged_keys]}
        )
    return out
