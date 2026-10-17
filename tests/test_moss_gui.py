"""``make gui-moss``: torch-free at launch, and the GUI's scan agrees with the cache."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import save_file

from library.moss.cache import LATENT_SUFFIX, TE_SUFFIX, caption_sha
from library.moss.scan import cache_counts, scan

ROOT = Path(__file__).resolve().parents[1]


def test_gui_is_torch_free():
    code = (
        "import sys, gui.moss.app, library.moss.scan; sys.exit('torch' in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT)
    assert result.returncode == 0, "gui.moss.app imports torch"


def _te(path: Path, caption: str) -> None:
    save_file(
        {"embeds": torch.zeros(3, 8, dtype=torch.bfloat16)},
        path,
        metadata={"caption_sha256": caption_sha(caption), "max_text_tokens": "4096"},
    )


def test_scan_counts_prefixed_stems_and_stale_captions(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    for sub in ("x", "y"):
        (src / sub).mkdir(parents=True)
        Image.new("RGB", (32, 32)).save(src / sub / "img.png")
        (src / sub / "img.txt").write_text(f"caption {sub}\n", encoding="utf-8")
    Image.new("RGB", (32, 32)).save(src / "x" / "nocap.png")
    out.mkdir()
    _te(out / f"x__img{TE_SUFFIX}", "caption x")  # current
    _te(out / f"y__img{TE_SUFFIX}", "old caption")  # caption edited since
    save_file({"latent": torch.zeros(1)}, out / f"x__img_32x32{LATENT_SUFFIX}")

    s = scan(src, out)
    assert (s.images, s.pairs, s.duplicates) == (3, 2, 2)
    assert (s.text_cached, s.stale_text, s.latents_cached) == (1, 1, 1)
    c = cache_counts(out)
    assert (c.text, c.latents, c.samples) == (2, 1, 1)
