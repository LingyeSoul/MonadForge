"""near_twins — in-artist near-twins variant-pair miner (EasyControl curation tool).

Mines near-duplicate *variant* pairs within a single artist where the two members
differ by a **specified attribute** (e.g. one has a speech bubble, the other
doesn't). See ``near_twins.__main__`` for the full pipeline + CLI. Run from the
repo root with ``python -m easycontrol_adapters.tools.near_twins``.

Layout:
  - ``engine``  — discovery → embed/cache → Stage-B grid match → discriminators → run_artist
  - ``outputs`` — ``_tags``/``_no_tags`` pair-tree export + EasyControl dataset blueprint
  - ``__main__``— config ([staging] toml) + argparse CLI + orchestration

See ``README.md`` for the end-to-end (mine → preprocess → train) workflow.
"""
