# Model-free compatibility primitives

This directory retains pure caption grammar, resize geometry, sidecars, workspace
contracts, exclusion/restore and download catalog helpers from anime-tools v0.7.15,
commit `7ea30612ebec736bbc3a243d153dca4d0d2e3853` (MIT; see LICENSE). Only Python
import paths are relocated. SOURCE.json records the original and relocated SHA256
of every retained source/data file. No models or weights are included.

The pinned upstream package requires Torch >=2.12. The V100 Torch 2.10 environment
must not install that package by bypassing its metadata. Training code instead
imports these primitives via `library.curation_compat`. Full model curation runs
in the separately registered `curation` runtime. Its GPU stages require compatible
hardware; isolating an interpreter does not make Torch 2.14 support a V100.

Module names preserve the upstream layout: `buckets`, `_walk`, `_hf`, `contract`,
`workspace`, `captions.*`, `downloads`, and `exclude`. `tagger.dbv4_meta` is catalog
metadata, not the tagger. `masking._masks` contains NumPy/Pillow mask I/O, not SAM.
No CLI runner for model stages is supplied by this compatibility package.

To update: take these exact paths from the intended upstream commit; rewrite only
imports beginning with `anime_tools` to `library.curation_compat`; update source
hashes, version and commit together; run source-integrity and behavior tests. Do
not widen this copy to include Torch models or private GUI implementation.
