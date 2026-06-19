"""Model download entry-points (Anima base, SAM3, MIT, PE-Core, Tagger vocab).

All targets shell out to ``hf download`` (rather than the SDK) so the user's
``hf auth login`` cache is honored.

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

from ._common import ROOT, run


DANBOORU_TAGS_PATH = ROOT / "models" / "danbooru_tags_classified.csv"
DANBOORU_TAGS_URLS = (
    "https://raw.githubusercontent.com/Localsmile/danbooru_KR_wiki_tag_search/main/danbooru_tags_classified.csv",
)


def _present(paths: list[Path]) -> bool:
    """True when every expected destination path already exists."""
    return all(p.exists() for p in paths)


def _skip(name: str, paths: list[Path], extra) -> bool:
    """Return True (caller should skip) when files exist and ``--force`` absent."""
    if "--force" in (extra or []):
        return False
    if _present(paths):
        print(f"  ✓ {name} already present (pass --force to re-download)")
        return True
    return False


def cmd_download_sam3(_extra):
    dst = ROOT / "models" / "sam3"
    # SAM3 is a gated repo; the full snapshot lands a config.json + weights.
    if _skip("SAM3", [dst / "config.json"], _extra):
        return
    dst.mkdir(parents=True, exist_ok=True)
    run(["hf", "download", "facebook/sam3", "--local-dir", "models/sam3"])


def cmd_download_pe(_extra):
    # Only the .pt is needed; vision tower is vendored at library/models/pe.py.
    dst = ROOT / "models" / "pe"
    # Skip only the PE-Core fetch — still fall through to PE-Spatial below, which
    # may be missing even when PE-Core is on disk.
    if not _skip("PE-Core", [dst / "PE-Core-L14-336.pt"], _extra):
        dst.mkdir(parents=True, exist_ok=True)
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


def cmd_download_danbooru_tags(_extra):
    """Download the classified Danbooru tag table used by caption correction."""

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


def cmd_download_mit(_extra):
    dst = ROOT / "models" / "mit"
    if _skip("MIT", [dst / "model.pth"], _extra):
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
