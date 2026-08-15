"""Header-only Anima checkpoint inspection and identity helpers."""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from library.io.safetensors import MemoryEfficientSafeOpen, get_split_weight_filenames

_KEY_PREFIXES = ("net.", "model.diffusion_model.", "")
_BLOCK_RE = re.compile(r"^blocks\.(\d+)\.(.+)$")
_BLOCK_SHAPES = {
    "adaln_modulation_cross_attn.1.weight": (256, 2048),
    "adaln_modulation_cross_attn.2.weight": (6144, 256),
    "adaln_modulation_mlp.1.weight": (256, 2048),
    "adaln_modulation_mlp.2.weight": (6144, 256),
    "adaln_modulation_self_attn.1.weight": (256, 2048),
    "adaln_modulation_self_attn.2.weight": (6144, 256),
    "cross_attn.k_norm.weight": (128,),
    "cross_attn.k_proj.weight": (2048, 1024),
    "cross_attn.output_proj.weight": (2048, 2048),
    "cross_attn.q_norm.weight": (128,),
    "cross_attn.q_proj.weight": (2048, 2048),
    "cross_attn.v_proj.weight": (2048, 1024),
    "mlp.layer1.weight": (8192, 2048),
    "mlp.layer2.weight": (2048, 8192),
    "self_attn.k_norm.weight": (128,),
    "self_attn.k_proj.weight": (2048, 2048),
    "self_attn.output_proj.weight": (2048, 2048),
    "self_attn.q_norm.weight": (128,),
    "self_attn.q_proj.weight": (2048, 2048),
    "self_attn.v_proj.weight": (2048, 2048),
}
_BLOCK_SUFFIXES = frozenset(_BLOCK_SHAPES)


@dataclass(frozen=True)
class AnimaCheckpointLayout:
    arch: str
    variant: str
    num_blocks: int
    model_channels: int
    num_heads: int
    key_prefix: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def resolve_checkpoint_files(
    path_or_files: str | os.PathLike | Sequence[str | os.PathLike],
) -> tuple[Path, ...]:
    """Expand a single checkpoint or an existing ``00001-of-N`` shard set."""

    raw = (
        [path_or_files]
        if isinstance(path_or_files, (str, os.PathLike))
        else list(path_or_files)
    )
    if not raw:
        raise ValueError("Anima checkpoint path list is empty")

    files: list[Path] = []
    seen: set[Path] = set()
    for item in raw:
        path = Path(item).expanduser().resolve()
        split = get_split_weight_filenames(str(path))
        candidates = [Path(p) for p in split] if split is not None else [path]
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate in seen:
                continue
            if not candidate.is_file():
                raise FileNotFoundError(f"Anima checkpoint file not found: {candidate}")
            if candidate.suffix.lower() != ".safetensors":
                raise ValueError(
                    f"Anima checkpoint must use safetensors for header inspection: {candidate}"
                )
            files.append(candidate)
            seen.add(candidate)
    return tuple(files)


def _prefix_for_key(key: str) -> str | None:
    for prefix in _KEY_PREFIXES:
        if not key.startswith(prefix):
            continue
        if _BLOCK_RE.match(key[len(prefix) :]):
            return prefix
    return None


def inspect_anima_checkpoint(
    path_or_files: str | os.PathLike | Sequence[str | os.PathLike],
) -> AnimaCheckpointLayout:
    """Inspect only safetensors headers and return the supported Anima layout."""

    files = resolve_checkpoint_files(path_or_files)
    key_shapes: dict[str, tuple[int, ...]] = {}
    block_prefixes: set[str] = set()

    for path in files:
        with MemoryEfficientSafeOpen(str(path)) as handle:
            for key in handle.keys():  # noqa: SIM118 - safe-open wrapper method
                if key in key_shapes:
                    raise ValueError(
                        f"Duplicate tensor key across Anima checkpoint shards: {key}"
                    )
                header = handle.header.get(key) or {}
                shape = header.get("shape")
                if not isinstance(shape, list) or not all(
                    isinstance(dim, int) and dim >= 0 for dim in shape
                ):
                    raise ValueError(f"Invalid safetensors shape for {key!r} in {path}")
                key_shapes[key] = tuple(shape)
                prefix = _prefix_for_key(key)
                if prefix is not None:
                    block_prefixes.add(prefix)

    if not block_prefixes:
        raise ValueError("Checkpoint has no Anima blocks.N.* tensors")
    if len(block_prefixes) != 1:
        display = [prefix or "<none>" for prefix in sorted(block_prefixes)]
        raise ValueError(f"Checkpoint mixes Anima key prefixes: {display}")
    key_prefix = next(iter(block_prefixes))

    blocks: dict[int, dict[str, tuple[int, ...]]] = {}
    for key, shape in key_shapes.items():
        if not key.startswith(key_prefix):
            continue
        match = _BLOCK_RE.match(key[len(key_prefix) :])
        if match is None:
            continue
        index = int(match.group(1))
        suffix = match.group(2)
        block = blocks.setdefault(index, {})
        if suffix in block:
            raise ValueError(f"Duplicate Anima block tensor after normalization: {key}")
        block[suffix] = shape

    indices = sorted(blocks)
    if indices not in (list(range(28)), list(range(40))):
        raise ValueError(
            "Unsupported or incomplete Anima block layout: expected exactly "
            f"0..27 or 0..39, found {indices[:4]}...{indices[-4:]} ({len(indices)} blocks)"
        )

    reference = blocks[0]
    if set(reference) != _BLOCK_SUFFIXES:
        missing = sorted(_BLOCK_SUFFIXES - set(reference))
        extra = sorted(set(reference) - _BLOCK_SUFFIXES)
        raise ValueError(
            "Anima block 0 tensor layout mismatch: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    invalid_reference_shapes = sorted(
        suffix
        for suffix, expected_shape in _BLOCK_SHAPES.items()
        if reference[suffix] != expected_shape
    )
    if invalid_reference_shapes:
        suffix = invalid_reference_shapes[0]
        raise ValueError(
            f"Anima block 0 shape mismatch for {suffix}: "
            f"{reference[suffix]} != {_BLOCK_SHAPES[suffix]}"
        )
    for index in indices[1:]:
        current = blocks[index]
        if set(current) != _BLOCK_SUFFIXES:
            missing = sorted(_BLOCK_SUFFIXES - set(current))
            extra = sorted(set(current) - _BLOCK_SUFFIXES)
            raise ValueError(
                f"Anima block {index} tensor layout mismatch: "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
        mismatched = [
            suffix for suffix in _BLOCK_SUFFIXES if current[suffix] != reference[suffix]
        ]
        if mismatched:
            suffix = min(mismatched)
            raise ValueError(
                f"Anima block {index} shape mismatch for {suffix}: "
                f"{current[suffix]} != {reference[suffix]}"
            )

    q_shape = reference["self_attn.q_proj.weight"]
    norm_shape = reference["self_attn.q_norm.weight"]
    if len(q_shape) != 2 or q_shape[0] != q_shape[1]:
        raise ValueError(f"Invalid Anima self-attention projection shape: {q_shape}")
    model_channels = q_shape[1]
    if len(norm_shape) != 1 or not norm_shape[0] or model_channels % norm_shape[0]:
        raise ValueError(
            f"Invalid Anima attention head shape: q={q_shape}, norm={norm_shape}"
        )
    num_heads = model_channels // norm_shape[0]
    if (model_channels, num_heads) != (2048, 16):
        raise ValueError(
            "Unsupported Anima width/head layout: "
            f"model_channels={model_channels}, num_heads={num_heads}"
        )

    num_blocks = len(indices)
    return AnimaCheckpointLayout(
        arch=f"anima-{model_channels}-{num_blocks}",
        variant="anima-2.9b-preview-v1" if num_blocks == 40 else "anima-base-v1.0",
        num_blocks=num_blocks,
        model_channels=model_channels,
        num_heads=num_heads,
        key_prefix=key_prefix,
    )


def _sha256_file(path: str) -> bytes:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.digest()


def anima_checkpoint_sha256(
    path_or_files: str | os.PathLike | Sequence[str | os.PathLike],
) -> str:
    """Return a raw SHA256 for one file or a deterministic aggregate for shards."""

    files = resolve_checkpoint_files(path_or_files)
    digests = []
    for path in files:
        digests.append(_sha256_file(str(path)))
    if len(digests) == 1:
        return digests[0].hex()
    combined = hashlib.sha256()
    for digest in digests:
        combined.update(digest)
    return combined.hexdigest()


def apply_layout_to_args(args, layout: AnimaCheckpointLayout, base_sha256: str) -> str:
    """Attach the derived, read-only model identity to an argparse namespace."""

    args.anima_arch = layout.arch
    args.anima_variant = layout.variant
    args.anima_num_blocks = layout.num_blocks
    args.anima_model_channels = layout.model_channels
    args.anima_num_heads = layout.num_heads
    args.anima_base_sha256 = base_sha256
    args._anima_checkpoint_layout = layout
    payload = f"{layout.arch}:{base_sha256}"
    args.anima_model_signature = hashlib.sha256(payload.encode("ascii")).hexdigest()[
        :16
    ]
    return args.anima_model_signature
