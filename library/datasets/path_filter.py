# Pure-stdlib path filtering, split out of `subsets.py` so torch-free callers
# (notably the GUI process — see gui/CLAUDE.md) can use it without paying the
# torch/cv2 import that subsets.py needs for its dataset classes.

import fnmatch
import os
from typing import List, Optional


def filter_paths_by_glob(
    img_paths: List[str],
    image_dir: Optional[str],
    pattern: Optional[str],
) -> List[bool]:
    """Return a per-path boolean mask: True keeps the file, False drops it.

    The pattern is matched against each file's path relative to ``image_dir``
    (with forward slashes, no leading "./") via ``fnmatch``. ``|`` separates
    alternatives — ``char_a/*|char_b/*`` keeps anything under either folder.
    Default ``*``, empty, or None all keep everything. Returns a mask rather
    than a filtered list so callers can keep parallel arrays (sizes,
    captions) aligned.
    """
    if not pattern:
        return [True] * len(img_paths)
    alternatives = [alt.strip() for alt in pattern.split("|")]
    alternatives = [alt for alt in alternatives if alt]
    if not alternatives or any(alt == "*" for alt in alternatives):
        return [True] * len(img_paths)
    base = os.path.abspath(image_dir) if image_dir else None
    keep: List[bool] = []
    for p in img_paths:
        if base is not None:
            try:
                rel = os.path.relpath(p, base)
            except ValueError:
                rel = os.path.basename(p)
        else:
            rel = os.path.basename(p)
        rel = rel.replace(os.sep, "/")
        keep.append(any(fnmatch.fnmatchcase(rel, alt) for alt in alternatives))
    return keep
