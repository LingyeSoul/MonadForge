"""ChimeraHydra Phase-0 probes.

Three CPU-only structural probes of the dual-pool chimera adapter, dispatched
by ``run_bench.py``:

  * ``expert_capacity``      — do the frozen-Cayley capability levers
    (``expert_basis_mult`` / ``expert_diag``) deepen each ortho expert without
    freeing the bases (the ``use_ortho_init`` collapse mode)?
  * ``full_spectrum``        — is there ΔW energy outside the top-slice SVD seed
    that MedQwen-style full-spectrum segment seeding would capture?
  * ``content_specificity``  — does the shared-A content half actually learn a
    content-specific representation (responsive router × distinct B-heads)?

Run: ``python bench/chimera/run_bench.py <probe> [args]``. All three drop the
standard ``bench/_common`` envelope into ``bench/chimera/results/`` with a
``<probe>-<label>`` run dir.
"""
