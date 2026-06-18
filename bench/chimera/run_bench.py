#!/usr/bin/env python
"""ChimeraHydra Phase-0 probes — dispatcher.

Usage::

    python bench/chimera/run_bench.py expert_capacity      [--steps ...]
    python bench/chimera/run_bench.py full_spectrum        [--lora ... --dit ...]
    python bench/chimera/run_bench.py content_specificity  [--ckpt ... --n-prompts ...]

Each probe is a self-contained CPU module under ``bench/chimera/`` that drops
the standard ``bench/_common`` envelope into ``bench/chimera/results/`` under a
``<probe>[-<label>]`` run dir. See ``bench/chimera/__init__.py`` for the map.
"""

from __future__ import annotations

import argparse

from bench.chimera import content_specificity, expert_capacity, full_spectrum

PROBES = {m.NAME: m for m in (expert_capacity, full_spectrum, content_specificity)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="probe", required=True)
    for name, mod in PROBES.items():
        sp = sub.add_parser(name, help=(mod.__doc__ or "").strip().splitlines()[0])
        mod.add_args(sp)
    args = p.parse_args()
    PROBES[args.probe].run(args)


if __name__ == "__main__":
    main()
