"""Mask generation: SAM3 + MIT/ComicTextDetector → merged.

``make mask`` is a one-shot orchestrator: it runs SAM and MIT into a
``tempfile.TemporaryDirectory()`` (cross-platform — honors ``TMPDIR`` /
``TEMP``) and writes only the merged result to
``post_image_dataset/masks/<rel>/{stem}_mask.png``. Per-tool intermediates
are never persisted under the project root.

Either backend can be turned off via the ``RUN_SAM_MASK`` /
``RUN_MIT_MASK`` env vars (set by the GUI's Preprocessing tab) — values
``"0"`` / ``"false"`` / ``"no"`` (case-insensitive) skip that backend.
When only one runs, the merge step still fires; ``merge_masks.py`` is a
no-op for single-source inputs.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

from ._common import PY, ROOT, _path, run

MASK_OUTPUT_DIR = ROOT / "post_image_dataset" / "masks"
RESIZED_IMAGE_DIR = ROOT / "post_image_dataset" / "resized"
SAM_CONFIG = ROOT / "configs" / "sam_mask.yaml"
_UNSET = object()


def _resized_image_dir() -> Path:
    """Scoped resized dir to mask, honoring GUI ``path_scope``.

    Reads ``resized_image_dir`` from the merged config chain (the GUI passes a
    config snapshot via ``CONFIG_FILE`` whose ``resized_image_dir`` is already
    scoped to ``post_image_dataset/resized/<path_scope>``). Scoping the input
    is what stops a scoped run from re-masking every other folder. Without a
    snapshot (direct ``make mask``) this falls back to the unscoped default, so
    CLI behavior is unchanged.
    """
    return ROOT / _path("resized_image_dir", "post_image_dataset/resized")


def _scoped_mask_output_dir(resized_dir: Path) -> Path:
    """Re-apply the ``path_scope`` offset onto the mask output root.

    SAM/MIT emit masks with rel paths taken **relative to the scoped resized
    dir** (``resized/<scope>``), so a scoped run drops the ``<scope>`` prefix.
    But training resolves masks relative to the **unscoped** cache root
    (``lora/<scope>/<rel>`` → ``masks/<scope>/<rel>``, see
    ``CachedDataset._resolve_mask_path``), so masking must land them under
    ``masks/<scope>`` — not flat in ``masks/`` — or the trainer won't find
    them. Mirror whatever scope ``resized_dir`` carries over the unscoped
    ``post_image_dataset/resized`` default. Unscoped (direct ``make mask``)
    returns the bare output dir, so CLI behavior is unchanged.
    """
    try:
        scope = resized_dir.resolve().relative_to(RESIZED_IMAGE_DIR.resolve())
    except ValueError:
        return MASK_OUTPUT_DIR
    if str(scope) == ".":
        return MASK_OUTPUT_DIR
    return MASK_OUTPUT_DIR / scope


def _runtime_sam_config() -> dict | None:
    """GUI queue jobs can pass an immutable SAM config snapshot via env.

    Direct CLI usage leaves this unset and continues to read
    ``configs/sam_mask.yaml``.
    """
    raw = os.environ.get("SAM_MASK_CONFIG_JSON")
    if not raw:
        return None
    try:
        cfg = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid SAM_MASK_CONFIG_JSON: {exc}") from exc
    if not isinstance(cfg, dict):
        raise SystemExit("Invalid SAM_MASK_CONFIG_JSON: expected an object")
    return cfg


def _load_sam_config(runtime: dict | None | object = _UNSET) -> dict:
    if runtime is _UNSET:
        runtime = _runtime_sam_config()
    if runtime is not None:
        return runtime
    try:
        import yaml

        with open(SAM_CONFIG, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except (OSError, ImportError):
        return {}


def _config_path_pattern(cfg: dict) -> str | None:
    """Read ``path_pattern`` so both backends filter alike.

    The key lives with the SAM config but is a dataset-level filter, so ``make
    mask`` forwards it to the MIT backend too (both run on the same resized
    dir). Missing key / ``"*"`` means mask everything.
    """
    pattern = cfg.get("path_pattern")
    return pattern if pattern and pattern != "*" else None


def _sam_config_path(cfg: dict, tmp_root: str, *, from_env: bool) -> str:
    if not from_env:
        return "configs/sam_mask.yaml"
    path = Path(tmp_root) / "sam_mask.yaml"
    path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    return str(path)


def _run_sam(
    image_dir: Path, out_dir: Path, extra: list[str], config_path: str
) -> None:
    run(
        [
            PY,
            "scripts/preprocess/generate_masks.py",
            "--config",
            config_path,
            "--image-dir",
            str(image_dir),
            "--mask-dir",
            str(out_dir),
            "--checkpoint",
            "models/sam3/sam3.pt",
            "--batch-size",
            "4",
            "--recursive",
            *extra,
        ]
    )


def _run_mit(image_dir: Path, out_dir: Path, extra: list[str]) -> None:
    # MIT_TEXT_THRESHOLD / MIT_DILATE let the GUI tune the MIT masker; defaults
    # match the script's argparse so direct CLI use is unchanged.
    cmd = [
        PY,
        "scripts/preprocess/generate_masks_mit.py",
        "--image-dir",
        str(image_dir),
        "--mask-dir",
        str(out_dir),
        "--model-path",
        "models/mit/model.pth",
        "--recursive",
    ]
    text_threshold = os.environ.get("MIT_TEXT_THRESHOLD")
    if text_threshold:
        cmd += ["--text-threshold", text_threshold]
    dilate = os.environ.get("MIT_DILATE")
    if dilate:
        cmd += ["--dilate", dilate]
    cmd += list(extra)
    run(cmd)


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def cmd_mask(extra):
    """Run SAM + MIT into a tempdir, merge, write to post_image_dataset/masks/.

    ``RUN_SAM_MASK`` / ``RUN_MIT_MASK`` env vars gate each backend
    independently (default on). If both are disabled the command is a no-op.
    """
    run_sam = _env_flag("RUN_SAM_MASK")
    run_mit = _env_flag("RUN_MIT_MASK")
    if not (run_sam or run_mit):
        print("Both SAM and MIT masking are disabled — nothing to do.")
        return
    runtime_sam_cfg = _runtime_sam_config()
    sam_cfg = _load_sam_config(runtime_sam_cfg)
    pattern = _config_path_pattern(sam_cfg)
    pattern_args = ["--path-pattern", pattern] if pattern else []
    resized_dir = _resized_image_dir()
    mask_output_dir = _scoped_mask_output_dir(resized_dir)
    with tempfile.TemporaryDirectory(prefix="anima-masks-") as tmp_root:
        sam_config_path = _sam_config_path(
            sam_cfg,
            tmp_root,
            from_env=runtime_sam_cfg is not None,
        )
        merge_sources: list[str] = []
        if run_sam:
            tmp_sam = Path(tmp_root) / "sam"
            _run_sam(resized_dir, tmp_sam, [*pattern_args], sam_config_path)
            merge_sources.append(str(tmp_sam))
        if run_mit:
            tmp_mit = Path(tmp_root) / "mit"
            _run_mit(resized_dir, tmp_mit, [*pattern_args])
            merge_sources.append(str(tmp_mit))
        mask_output_dir.mkdir(parents=True, exist_ok=True)
        run(
            [
                PY,
                "scripts/preprocess/merge_masks.py",
                *merge_sources,
                "--output-dir",
                str(mask_output_dir),
                *extra,
            ]
        )


def cmd_mask_clean(_extra):
    if MASK_OUTPUT_DIR.exists():
        shutil.rmtree(MASK_OUTPUT_DIR)
        print(f"  Removed {MASK_OUTPUT_DIR.relative_to(ROOT)}/")
