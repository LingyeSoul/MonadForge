from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from library import curation_compat
from library.curation_compat.captions.position_clauses import compose_caption, parse_caption
from library.curation_compat.exclude import Trees, exclude_one, read_entries, restore_one


ROOT = Path(curation_compat.__file__).parent


def test_compatibility_copy_preserves_pinned_upstream_source():
    manifest = json.loads((ROOT / "SOURCE.json").read_text())
    assert manifest["commit"] == curation_compat.UPSTREAM_COMMIT
    assert manifest["version"] == curation_compat.UPSTREAM_VERSION
    for relative, hashes in manifest["files"].items():
        raw = (ROOT / relative).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == hashes["vendored_sha256"], relative
        original = re.sub(
            r"(?m)^(\s*(?:from|import)\s+)library\.curation_compat(?=[.\s])",
            r"\1anime_tools", raw.decode(),
        ).encode()
        assert hashlib.sha256(original).hexdigest() == hashes["upstream_sha256"], relative


def test_caption_round_trip_keeps_quoted_punctuation_and_subject_clauses():
    source = '1girl, blue hair. On the left, yellow eyes. Text: "Hello, world.".'
    caption = parse_caption(source)
    assert compose_caption(caption.flat_tags, caption.clauses) == source


def test_exclusion_restore_keeps_source_and_preserves_live_collisions(tmp_path):
    trees = Trees(*(tmp_path / part for part in ("resized", "masks", "ocr", "excluded")))
    trees.resized.mkdir()
    original = trees.resized / "a.png"
    original.write_bytes(b"old-image")
    moved = exclude_one(trees, "a.jpg")
    assert moved.action == "excluded"
    assert not original.exists()
    assert "a.jpg" in read_entries(trees.excluded)
    original.write_bytes(b"new-image")
    restored = restore_one(trees, "a.jpg")
    assert restored.skipped
    assert original.read_bytes() == b"new-image"
    assert "a.jpg" not in read_entries(trees.excluded)
    assert (trees.excluded / "resized" / "a.png").read_bytes() == b"old-image"
    # Upstream deliberately un-excludes the image even when an old slot stays
    # behind. A non-colliding image exercises the normal full restore path.
    second = trees.resized / "b.png"
    second.write_bytes(b"second-image")
    exclude_one(trees, "b.jpg")
    assert not second.exists()
    assert restore_one(trees, "b.jpg").action == "restored"
    assert second.read_bytes() == b"second-image"


@pytest.mark.parametrize("relative", ["../a.jpg", "/a.jpg", "C:\\a.jpg", "//host/share/a.jpg"])
def test_exclusion_refuses_paths_outside_dataset(tmp_path, relative):
    trees = Trees(*(tmp_path / part for part in ("resized", "masks", "ocr", "excluded")))
    with pytest.raises(ValueError, match="bad relative path"):
        exclude_one(trees, relative)


def test_training_primitives_import_without_anime_tools_or_torch():
    project = ROOT.parents[1]
    code = f"""
import sys
sys.path.insert(0, {str(project)!r})
from library.curation_compat import buckets, contract, downloads, exclude, workspace
from library.curation_compat.captions import correction, layout_json, shuffle, variants
from library.runtime.hf_download import is_network_error, hf_file_cached
assert 'torch' not in sys.modules
assert not any(x == 'anime_tools' or x.startswith('anime_tools.') for x in sys.modules)
assert downloads.DEFAULT_SUBJECT_PROMPT_EMBED
assert is_network_error(TimeoutError())
"""
    subprocess.run([sys.executable, "-I", "-c", code], check=True, timeout=30)
