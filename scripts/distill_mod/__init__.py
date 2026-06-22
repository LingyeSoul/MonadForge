"""Modulation-guidance distillation scripts.

Two CLIs (matching the ``make distill-prep`` / ``make distill-mod`` targets):

* ``python -m scripts.distill_mod.prep``    — Phase 1 (T5("") sidecar) + Phase 2
  (teacher-synthesized clean latents) staging.
* ``python -m scripts.distill_mod.distill`` — train ``pooled_text_proj`` against
  the frozen teacher.

Shared modules:

* :mod:`library.anima.uncond`              — T5("") sidecar encode/load helpers.
* :mod:`library.preprocess.uncond`         — T5("") sidecar staging (produce-to-disk).
* :mod:`scripts.distill_mod.synth`         — Phase 2 teacher-driven synthesis.
* :mod:`scripts.distill_mod.teacher_cache` — train + val teacher prediction caches.
* :mod:`scripts.distill_mod.validation`    — fixed-sigma teacher↔student MSE pass.
"""
