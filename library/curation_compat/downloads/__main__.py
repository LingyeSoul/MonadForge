"""``python -m anime_tools.downloads``. The CLI itself is
:mod:`anime_tools.downloads._cli`, so importing the package does not run it."""

from library.curation_compat.downloads._cli import main

raise SystemExit(main())
