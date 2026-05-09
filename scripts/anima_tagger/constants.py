"""Tag-taxonomy / caption-format constants shared across tagger CLI modes.

These are the single source of truth for the trainer's view of the corpus —
``vocab.py`` writes them into ``vocab.json`` and inference reads them back
through the snapshot, so changes here invalidate existing checkpoints.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional, Tuple

# Booru-style tag-type integers from the corpus's tag-taxonomy cache.
TAG_TYPE_NAMES: Dict[int, str] = {
    0: "general",
    1: "artist",
    3: "copyright",
    4: "character",
    5: "metadata",
    6: "deprecated",
}

# Anima caption-format slot order. The inference emitter joins by this
# order; categories not in the list (``deprecated``, ``metadata``) are
# either filtered out or treated as ``general`` depending on context.
SLOT_ORDER: Tuple[str, ...] = (
    "rating",
    "count",
    "character",
    "copyright",
    "artist",
    "general",
)

# 3-class rating set (post-``questionable→sensitive`` collapse).
RATINGS: Tuple[str, ...] = ("general", "sensitive", "explicit")

# Count-tag detection. Matches ``1girl``, ``2girls``, ``1boy``, ``3others``,
# ``multiple_girls``, ``multiple_boys``. Underscores or spaces both fine.
_COUNT_RE = re.compile(
    r"^(?:\d+(?:girl|boy|other)s?|multiple[_ ](?:girls|boys|others))$"
)

# Image extensions we look for next to each .txt caption file. Order is
# preference; first hit wins.
IMAGE_EXTS: Tuple[str, ...] = (".webp", ".jpg", ".jpeg", ".png")


def find_image_for_caption(caption_path: Path) -> Optional[Path]:
    """Return the sibling image file matching ``{stem}.<ext>``, or None."""
    for ext in IMAGE_EXTS:
        candidate = caption_path.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


def is_count_tag(tag: str) -> bool:
    return bool(_COUNT_RE.match(tag))
