"""Compat shim — the real code lives in :mod:`scripts.anima_tagger`.

This file is kept so existing invocations like
``python scripts/train_anima_tagger.py --mode build_vocab`` keep working.
New entry points should prefer ``python -m scripts.anima_tagger.cli``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make ``anima_lora/`` importable when invoked as a script (Python only
# adds ``scripts/`` to sys.path in that case, not the parent).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.anima_tagger.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
