"""The workspace layout: where the tools write, and where Export publishes to::

    <home>/
      (source_dir)/                   INPUT -- read-only for the tools (the
                                      ``src`` root's default name)
      workspace/                      everything the tools produce
        master/<rel>.txt                revised master
        resized/<rel>.{png,txt,variants.txt}
        masks_sam/<rel>/{stem}_mask.png  the SAM3 generator's own tree
        masks/<rel>/{stem}_mask.png      the merge
        captions/<stage>/report.json    the diffs
        groups/groups.json
        export/report.json              the export ledger
        _excluded/excluded.json         what was taken out of the pipeline
        _excluded/{resized,masks,ocr}/    and the files it was taken out of
      (export_target_dir)/            OUTPUT -- written only by Export (the
                                      ``out`` root's default name)
        resized/  masks/
        _excluded/                      the excluded tree, republished as it is

The invariant: **no stage writes outside** ``workspace/``; Export is the only
thing that touches the ``src`` or ``out`` roots.

Nothing here reads the filesystem or the settings file: these are the *default*
paths, which ⚙ Settings may override per root. The two names outside
``workspace/`` are placeholders inherited from the trainer's layout -- this is
the only module that spells them, and everything else names the root
(``src`` / ``out``, ``--src`` / ``--out``) rather than the directory.
Stdlib only, torch-free.
"""

from __future__ import annotations

WORKSPACE = "workspace"
"""The workspace directory, home-relative. ``_env.workspace_dir()`` resolves it
and honours ``ANIME_TOOLS_WORKSPACE``."""

SOURCE_ROOT = "image_dataset"
"""The input tree, home-relative. Read-only for the tools from the workspace
phase onward. The name is only the default ⚙ Settings starts from; import this
rather than spelling it."""

EXPORT_ROOT = "post_image_dataset"
"""The output tree, and the one the trainer reads. Written only by Export, and
likewise a default name — import it rather than spelling it."""

DEFAULT_ROOTS: dict[str, str] = {
    # input -> workspace -> output, the order ⚙ Settings shows them in.
    "src": SOURCE_ROOT,
    "master": f"{WORKSPACE}/master",
    "dst": f"{WORKSPACE}/resized",
    "masks": f"{WORKSPACE}/masks",
    "out": EXPORT_ROOT,
}
"""The five dataset roots, home-relative.

``src`` / ``dst`` / ``masks`` are the three trees the sidebar joins by the same
relative path, and the names ``gui.stages.ROOT_FIELDS`` binds stage flags to;
``master`` is the revised-master overlay and ``out`` the export destination.
"""

OUTPUT_ROOTS = frozenset({"master", "dst", "masks"})
"""The roots a stage run may create on the way past — not ``src`` (input) and
not ``out`` (Export-only, and it mkdirs its own destination).
``gui._context.make_output_dirs`` intersects a stage's bound roots with this.
"""

EXPORT_ROOTS = frozenset({"out"})
"""Roots only Export writes, and which nothing else may create on its behalf —
⚙ Settings' "make my roots" gesture stops short of these."""

REPORTS_SUBDIR = "captions"
"""The workspace subdirectory the stage reports land in, as the tail of every
CLI's ``--report_dir`` default (``captions/autotag``, ``captions/position``, …).
``gui._context.report_root`` derives the root from the parent of ``dst``."""

RESIZED = DEFAULT_ROOTS["dst"]
MASKS = DEFAULT_ROOTS["masks"]
REPORTS = f"{WORKSPACE}/{REPORTS_SUBDIR}"
GROUPS = f"{WORKSPACE}/groups"
OCR_SUBDIR = "ocr"
OCR = f"{WORKSPACE}/{OCR_SUBDIR}"
"""The workspace paths the CLI defaults are written in terms of.

Every ``--dst`` / ``--report_dir`` / ``--out`` default is one of these plus the
stage's own tail, which keeps ``gui.stages.report_subpath`` — it drops the first
component of a report default to get that tail — seeing exactly one component in
front of it.

:data:`OCR` is its own tree, not a sidecar beside each caption: what it holds is
text *in the picture*, so it can be published, deleted or regenerated without
touching a caption.
"""

MASKS_SAM = f"{WORKSPACE}/masks_sam"
"""The SAM3 generator's own output tree, and ``merge_masks``' default input.

Not a root, and separate from :data:`MASKS`: any second tree merged in beside it
names a mask ``{stem}_mask.png`` under the same relative path, so sharing the
root would have the second run overwrite the first. :data:`MASKS` holds the
merge (a pixel-wise minimum over the inputs, i.e. the union of what they mask)
and is the root the sidebar joins and Export publishes.
"""

EXCLUDED_SUBDIR = "_excluded"
EXCLUDED = f"{WORKSPACE}/{EXCLUDED_SUBDIR}"
"""Where an image the curator took *out* of the dataset goes.

Not a root and not a stage output: a tree the GUI's Exclude gesture moves files
into, holding one mirror per tree it took them from
(:data:`EXCLUDED_TREES`) plus the ledger that says which source image each came
from. The ledger is the state — a rel listed in it is skipped by ``resize``,
which is the only stage that could put the image back in
:data:`RESIZED` and so the one chokepoint every later stage is downstream of.

The leading underscore keeps it out of the way of a dataset subfolder and sorts
it to the top of ``workspace/``; Export republishes the whole tree under
``<out>/_excluded/``, so an excluded image is still *there* — beside the
trainer's tree rather than inside it.
"""

EXCLUDED_TREES: tuple[str, ...] = ("resized", "masks", "ocr")
"""The subdirectories of :data:`EXCLUDED`, one per tree an exclusion empties.

Named rather than merged flat because the three collide: a mask is
``{stem}_mask.png``, which is a legal image name in the resized tree. Each holds
the same relative path its own tree used, which is what makes putting one back a
move and not a lookup.
"""

EXCLUDED_MANIFEST = "excluded.json"
"""The ledger, at the root of :data:`EXCLUDED`. Inside the tree rather than
beside it, so moving the directory moves the record of what is in it."""

LEGACY_ROOTS: dict[str, str] = {
    # root name -> where its default pointed before the workspace.
    "dst": f"{EXPORT_ROOT}/resized",
    "masks": f"{EXPORT_ROOT}/masks",
}
"""Pre-workspace defaults for the two roots that moved. A root explicitly saved
in ⚙ Settings is unaffected, so the migrate CLI warns about one instead of
moving it."""

LEGACY_DIRS: tuple[str, ...] = (REPORTS_SUBDIR, "groups")
"""Workspace subdirectories that are not roots, so nothing in Settings names
them. Both follow ``dst``'s parent, so moving the tree is all it takes."""
