"""Model download entry-points (Anima base, SAM3, MIT, PE-Core, Tagger vocab).

All targets shell out to ``hf download`` (rather than the SDK) so the user's
``hf auth login`` cache is honored — unless ``--source modelscope`` (or
``ANIMA_DOWNLOAD_SOURCE=modelscope``) selects the ModelScope (魔搭) mirror,
in which case the mapped targets fetch in-process via
``library.runtime.ms_download`` (zero-dependency, resumable, same skip rules).
Mirrors are verified in ``ms_download.MIRRORS``; components without one print
a skip notice instead of silently falling back to a network the user just
said they can't reach.

Idempotency contract (see GH #21): every target skips when its final
destination files already exist, so a re-run *verifies* rather than re-fetching
gigabytes. This matters because several targets ``shutil.move`` files out of
``hf``'s ``--local-dir`` layout after download — once moved, ``hf download``
no longer sees them at the path it checks and would otherwise re-pull the whole
repo. Pass ``--force`` (e.g. ``make download-anima ARGS=--force``) to re-fetch
regardless. ``download-models`` continues past a failed component (a gated SAM3
without granted access shouldn't abort the Anima download) and reports the
failures at the end.
"""

from __future__ import annotations

import shutil
import urllib.error
import urllib.request
from pathlib import Path

from library.runtime import ms_download

from ._common import PY, ROOT, explicit_option_value, run


DANBOORU_TAGS_PATH = ROOT / "models" / "danbooru_tags_classified.csv"
DANBOORU_TAGS_EN_PATH = ROOT / "models" / "danbooru_tags_classified.en.csv"
DANBOORU_TAGS_URLS = (
    "https://raw.githubusercontent.com/Localsmile/danbooru_KR_wiki_tag_search/main/danbooru_tags_classified.csv",
)


def _present(paths: list[Path]) -> bool:
    """True when every expected destination path already exists."""
    return all(p.exists() for p in paths)


def _force(extra) -> bool:
    """True when ``--force`` re-fetches regardless of existing files."""
    return "--force" in (extra or [])


def _skip(name: str, paths: list[Path], extra) -> bool:
    """Return True (caller should skip) when files exist and ``--force`` absent."""
    if _force(extra):
        return False
    if _present(paths):
        print(f"  ✓ {name} already present (pass --force to re-download)")
        return True
    return False


def _wants_modelscope(extra) -> bool:
    """True when ``--source modelscope`` (flag) or ``ANIMA_DOWNLOAD_SOURCE``
    (env) selects the ModelScope mirror for this invocation."""
    if ms_download.source_enabled():
        return True
    source = explicit_option_value(extra or [], "--source")
    return (source or "").strip().lower() == "modelscope"


def _ms_unmapped(name: str) -> None:
    """Notice for a component the ModelScope source can't serve (no mirror)."""
    print(f"  ✗ {name}: no ModelScope mirror (HuggingFace only) — skipped")


def _ms_fetch(name: str, hf_repo: str, filename: str, dst_dir: Path, extra) -> bool:
    """Fetch one file from the component's ModelScope mirror; False when it
    has none (caller should skip). Honors ``--force``."""
    ms_repo = ms_download.mirror_repo(hf_repo)
    if ms_repo is None:
        _ms_unmapped(name)
        return False
    ms_download.fetch_file(
        ms_repo=ms_repo,
        filename=filename,
        local_dir=dst_dir,
        what=name,
        force=_force(extra),
    )
    return True


def cmd_download_sam3(_extra):
    dst = ROOT / "models" / "sam3"
    # SAM3 is a gated repo on HF; the full snapshot lands a config.json + weights.
    # The ModelScope mirror is public — no token needed there.
    if _skip("SAM3", [dst / "config.json"], _extra):
        return
    dst.mkdir(parents=True, exist_ok=True)
    if _wants_modelscope(_extra):
        ms_download.snapshot_download("facebook/sam3", dst, force=_force(_extra))
        return
    run(["hf", "download", "facebook/sam3", "--local-dir", "models/sam3"])


def cmd_download_pe(_extra):
    # Only the .pt is needed; vision tower is vendored at library/models/pe.py.
    dst = ROOT / "models" / "pe"
    # Skip only the PE-Core fetch — still fall through to PE-Spatial below, which
    # may be missing even when PE-Core is on disk.
    if not _skip("PE-Core", [dst / "PE-Core-L14-336.pt"], _extra):
        dst.mkdir(parents=True, exist_ok=True)
        if _wants_modelscope(_extra):
            _ms_fetch(
                "PE-Core", "facebook/PE-Core-L14-336", "PE-Core-L14-336.pt", dst, _extra
            )
        else:
            run(
                [
                    "hf",
                    "download",
                    "facebook/PE-Core-L14-336",
                    "PE-Core-L14-336.pt",
                    "--local-dir",
                    "models/pe",
                ]
            )
    # PE-Spatial is the default REPA alignment encoder — fetch it alongside PE-Core.
    cmd_download_pe_spatial(_extra)


def cmd_download_pe_spatial(_extra):
    # Auxiliary encoder for the Anima Tagger's dual-encoder config; only the .pt.
    dst = ROOT / "models" / "pe"
    if _skip("PE-Spatial", [dst / "PE-Spatial-B16-512.pt"], _extra):
        return
    if _wants_modelscope(_extra):
        # No known mirror — _ms_fetch prints the skip notice.
        _ms_fetch(
            "PE-Spatial",
            "facebook/PE-Spatial-B16-512",
            "PE-Spatial-B16-512.pt",
            dst,
            _extra,
        )
        return
    dst.mkdir(parents=True, exist_ok=True)
    run(
        [
            "hf",
            "download",
            "facebook/PE-Spatial-B16-512",
            "PE-Spatial-B16-512.pt",
            "--local-dir",
            "models/pe",
        ]
    )


def cmd_download_tagger(_extra):
    # Just the Tagger ``vocab.json`` (~0.7 MB) that caption-index/preprocess need.
    # The full model is not fetched here, so this won't clobber a local model.safetensors.
    dst = ROOT / "models" / "captioners" / "anima-tagger-v2"
    if _skip("Anima Tagger vocab", [dst / "vocab.json"], _extra):
        return
    if _wants_modelscope(_extra):
        # No known mirror — _ms_fetch prints the skip notice.
        _ms_fetch(
            "Anima Tagger vocab",
            "sorryhyun/anima-tagger",
            "vocab.json",
            dst,
            _extra,
        )
        return
    dst.mkdir(parents=True, exist_ok=True)
    run(
        [
            "hf",
            "download",
            "sorryhyun/anima-tagger",
            "vocab.json",
            "--local-dir",
            "models/captioners/anima-tagger-v2",
        ]
    )


def _download_danbooru_base(_extra):
    """Fetch the Korean-description base CSV from Localsmile (idempotent)."""
    if _skip("Danbooru classified tags", [DANBOORU_TAGS_PATH], _extra):
        return
    DANBOORU_TAGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = DANBOORU_TAGS_PATH.with_suffix(".csv.tmp")
    last_error = ""
    for url in DANBOORU_TAGS_URLS:
        print(f"  download {url}")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "anima-lora"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                tmp.write_bytes(resp.read())
            if tmp.stat().st_size <= 0:
                raise OSError("downloaded file is empty")
            tmp.replace(DANBOORU_TAGS_PATH)
            print(f"  ✓ wrote {DANBOORU_TAGS_PATH}")
            return
        except (OSError, urllib.error.URLError) as exc:
            last_error = str(exc)
            if tmp.exists():
                tmp.unlink()
            print(f"  ✗ failed: {last_error}")
    raise SystemExit(
        "failed to download danbooru_tags_classified.csv from "
        "Localsmile/danbooru_KR_wiki_tag_search"
    )


def cmd_download_danbooru_tags(_extra):
    """Fetch the Danbooru tag table for caption correction — both languages.

    Downloads the Korean-description base CSV (``danbooru_tags_classified.csv``)
    from Localsmile, then builds the English sibling
    (``danbooru_tags_classified.en.csv``) by joining tag names against the
    ``isek-ai/danbooru-wiki-2024`` wiki mirror so the GUI tag-explanation tooltip
    works for non-Korean UIs. Both steps are idempotent (``--force`` re-fetches).
    """
    _download_danbooru_base(_extra)
    if _skip("Danbooru English tags", [DANBOORU_TAGS_EN_PATH], _extra):
        return
    # Pass through only the builder's own flags (e.g. --revision); --force is a
    # task-runner concept the build script doesn't accept.
    build_args = [a for a in (_extra or []) if a != "--force"]
    run(
        [PY, "-m", "scripts.anima_tagger.build_english_tag_csv", *build_args],
        cwd=ROOT,
    )


def cmd_download_mit(_extra):
    dst = ROOT / "models" / "mit"
    if _skip("MIT", [dst / "model.pth"], _extra):
        return
    if _wants_modelscope(_extra):
        # No known mirror — _ms_fetch prints the skip notice.
        _ms_fetch(
            "MIT", "a-b-c-x-y-z/Manga-Text-Segmentation-2025", "model.pth", dst, _extra
        )
        return
    dst.mkdir(parents=True, exist_ok=True)
    run(
        [
            "hf",
            "download",
            "a-b-c-x-y-z/Manga-Text-Segmentation-2025",
            "model.pth",
            "--local-dir",
            "models/mit",
        ]
    )


def cmd_download_anima(_extra):
    models = ROOT / "models"
    # Final (post-move) destinations — this is what we verify against, NOT the
    # transient split_files/ layout hf downloads into (see module docstring).
    finals = [
        models / "diffusion_models" / "anima-base-v1.0.safetensors",
        models / "text_encoders" / "qwen_3_06b_base.safetensors",
        models / "vae" / "qwen_image_vae.safetensors",
    ]
    if _skip("Anima base (DiT + TE + VAE, ~5GB)", finals, _extra):
        return
    for d in ["diffusion_models", "text_encoders", "vae"]:
        (models / d).mkdir(parents=True, exist_ok=True)
    if _wants_modelscope(_extra):
        # The official ModelScope repo mirrors the HF split_files/ layout exactly,
        # so the shared move below works for both sources.
        ms_download.snapshot_download(
            "circlestone-labs/Anima",
            models,
            filenames=[
                "split_files/diffusion_models/anima-base-v1.0.safetensors",
                "split_files/text_encoders/qwen_3_06b_base.safetensors",
                "split_files/vae/qwen_image_vae.safetensors",
            ],
            force=_force(_extra),
        )
    else:
        run(
            [
                "hf",
                "download",
                "circlestone-labs/Anima",
                "split_files/diffusion_models/anima-base-v1.0.safetensors",
                "split_files/text_encoders/qwen_3_06b_base.safetensors",
                "split_files/vae/qwen_image_vae.safetensors",
                "--local-dir",
                "models",
                "--include",
                "split_files/*",
            ]
        )
    split = models / "split_files"
    for subdir in ["diffusion_models", "text_encoders", "vae"]:
        src = split / subdir
        dst = models / subdir
        if src.exists():
            for f in src.iterdir():
                shutil.move(str(f), str(dst / f.name))
    if split.exists():
        shutil.rmtree(split)


def cmd_download_models(_extra):
    # Continue-on-failure: a gated/un-authed component (SAM3) must not abort the
    # rest. ``run`` sys.exits on a non-zero subprocess, so catch SystemExit per
    # component; each is skip-if-present so the retry doesn't re-download successes.
    components = [
        ("Anima base", cmd_download_anima),
        ("SAM3 (gated)", cmd_download_sam3),
        ("MIT", cmd_download_mit),
        ("PE-Core", cmd_download_pe),
        ("PE-Spatial", cmd_download_pe_spatial),
        ("Anima Tagger vocab", cmd_download_tagger),
        ("Danbooru classified tags", cmd_download_danbooru_tags),
    ]
    failed: list[str] = []
    for name, fn in components:
        try:
            fn(_extra)
        except SystemExit as e:
            if e.code:
                failed.append(name)
                print(f"  ✗ {name} failed (exit {e.code}); continuing")
    if failed:
        print()
        print("The following downloads did not complete:")
        for name in failed:
            print(f"  - {name}")
        print()
        print("Common causes:")
        print("  - not authenticated: run `hf auth login` and re-run")
        print(
            "  - SAM3 is gated: request access at https://huggingface.co/facebook/sam3"
        )
        print("Successful components are cached; re-running only retries the failures.")
        raise SystemExit(1)


def cmd_download_ms(extra):
    """``download-ms <org/name> [--local-dir DIR] [--include GLOB]... [--force]``

    Fetch any ModelScope repo (e.g. a base model only mirrored there) into
    ``models/<name>`` (or ``--local-dir``). Plain files, no symlink cache;
    complete files are skipped by size unless ``--force``.
    """
    args = list(extra or [])
    positional: list[str] = []
    local_dir: str | None = None
    include: list[str] = []
    force = False
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--local-dir" and i + 1 < len(args):
            local_dir = args[i + 1]
            i += 2
        elif a.startswith("--local-dir="):
            local_dir = a.split("=", 1)[1]
            i += 1
        elif a == "--include" and i + 1 < len(args):
            include.append(args[i + 1])
            i += 2
        elif a.startswith("--include="):
            include.append(a.split("=", 1)[1])
            i += 1
        elif a == "--force":
            force = True
            i += 1
        elif a.startswith("--"):
            raise SystemExit(
                f"unknown option {a} — supported: --local-dir DIR, "
                "--include GLOB, --force"
            )
        else:
            positional.append(a)
            i += 1
    if not positional:
        raise SystemExit(
            "usage: tasks.py download-ms <modelscope_repo_id> "
            "[--local-dir DIR] [--include GLOB]... [--force]"
        )
    repo_id = positional[0]
    if "/" not in repo_id:
        raise SystemExit(
            f"{repo_id!r} is not a ModelScope repo id — expected 'org/name' "
            "(see https://modelscope.cn/models/...)"
        )
    if local_dir is None:
        local_dir = str(ROOT / "models" / repo_id.split("/")[-1])
    print(f"  source: ModelScope {repo_id} -> {local_dir}")
    try:
        downloaded = ms_download.snapshot_download(
            repo_id, local_dir, include=include or None, force=force
        )
    except FileNotFoundError as e:
        raise SystemExit(str(e))
    if downloaded:
        print(f"  ✓ {len(downloaded)} file(s) downloaded from ModelScope")
    else:
        print("  ✓ all files already present (pass --force to re-download)")
