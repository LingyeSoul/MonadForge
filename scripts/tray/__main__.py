"""Tray entry point: ``pythonw -m scripts.tray``.

Mirrors ``scripts.tray.app:main`` so both ``-m scripts.tray`` and the installed
``monadforge-tray`` gui-script resolve to the same ``main``.
"""

from __future__ import annotations

import sys

# sys.path bootstrap so ``-m scripts.tray`` resolves sibling imports even
# without the package installed (matches the repo's other tooling entrypoints).
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.tray.app import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
