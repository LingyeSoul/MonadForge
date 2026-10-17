"""``python -m anime_tools.exclude``. The CLI itself is
:mod:`anime_tools.exclude._cli`, so importing the package does not run it."""

from library.curation_compat.exclude._cli import main

raise SystemExit(main())
