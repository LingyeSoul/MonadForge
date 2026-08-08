# Technical Specification for Issue #43

## Issue Summary

- Title: NaN from finite Q/K/V on V100 FP16 self-attention (head_dim=128, seq_len=2925)
- Description: The published V100 FlashAttention `26.06` wheel returns NaN/Inf
  for finite FP16 dense attention inputs when the sequence length is not a
  multiple of 16. Exact captures, raw API replay, and a complete modulo-16
  sweep reproduce the failure; Torch SDPA remains finite.
- Labels: None
- Priority: High. The backend silently returns invalid training values on a
  supported public API path and arbitrary native image shapes commonly produce
  non-aligned token counts.

## Problem Statement

The release artifact and the current source tree must be treated separately.
Issue #43 tested the wheel published from tag `26.06` at commit `d89800e` on
2026-06-05. That artifact predates the 2026-06-25 commit `8a86213` (`Softmax:
Fix compile/runtime tail process in dense kernel`). The report therefore proves
a release-wheel defect, but current `main` has not yet been tested against the
same captured tensors.

Static analysis of the release tag explains the exact modulo-16 boundary. In
`include/softmax.h`, `WMMA_GEMM_SOFTMAX` defaults its compile-time `TAIL`
parameter to `false`. The dense forward call does not override that default, so
it neither processes the 1-3 scalar softmax remainder nor zeroes `sP` from
`VALID_KV` through `BLOCK_N`. Later, `WMMA_GEMM_GRADIENTS` consumes K in
16-wide WMMA fragments and executes the last fragment whenever
`k_offset < VALID_K`. For `VALID_K % 16 != 0`, lanes past `VALID_K` therefore
read stale/garbage `sP` values. Multiples of 16 stop before such a partial
fragment, matching the observed sweep exactly.

Current `main` removes the compile-time `TAIL` switch, performs tail handling at
runtime, and unconditionally zeroes the unused `sP` range. This is the expected
kernel-side fix, but it remains a hypothesis until replayed on SM70. Explicitly
setting `softmax_scale=1/sqrt(128)` cannot help because the wrapper already
computes that value. Head-dimension padding is unrelated because 128 is already
divisible by 8. The captured Q/K/V ranges rule out an input-magnitude contract
violation.

## Technical Approach

1. Reproduce three builds on the same V100 environment:
   - the published `26.06` wheel (negative control),
   - tag `26.06` plus only commit `8a86213`,
   - current `main` at or after `c91cad4`.
2. Replay `first_failure.pt` through the raw public API and compare to Torch
   SDPA. Record finiteness, NaN/Inf counts, maximum/mean absolute error over
   finite output, timing, and peak memory.
3. Run the exact 4112..4128 prefix sweep. The patched builds must be finite for
   all residues, not only the two aligned controls.
4. Add upstream tests that cover every `M % 16` and `N % 16` residue. Include
   self-attention and `M != N` cross-attention so query and key tails are tested
   independently.
5. If `8a86213` passes, release a versioned wheel built from a traceable commit
   and close #43 with the build SHA, wheel digest, and replay results. If it
   fails, instrument the final `sP` fragment and continue at the softmax-to-MMA
   boundary instead of changing FP16 accumulation policy globally.
6. Keep MonadForge's production V100 recommendation on Torch SDPA until a new
   upstream wheel passes the capture, modulo sweep, backward test, and a real
   Anima training smoke test.

The kernel fix should preserve the existing online-softmax algorithm. Do not
pad Q/K/V at the Python layer without an explicit key mask: zero-valued padded
keys would still alter the softmax denominator. Do not use `nan_to_num`, because
that hides invalid forward state and corrupts gradients.

## Implementation Plan

1. Add a standalone upstream replay test that loads Q/K/V from a capture and
   calls `flash_attn_func` without MonadForge wrappers.
2. Extend the upstream dense test matrix with compact shapes spanning residues
   0..15 for M and N, plus the issue's 4112..4128 regression sweep.
3. Build and run the release tag, cherry-picked fix, and current main on the
   same Tesla V100 with the same compiler flags.
4. Confirm whether `8a86213` alone fixes forward and backward. If not, dump the
   last KV tile after softmax and before `WMMA_GEMM_GRADIENTS` to identify any
   remaining non-zero or non-finite lanes beyond `VALID_KV`.
5. Add a release version bump and publish a replacement wheel only after the
   SM70 matrix passes. Record source commit and wheel SHA-256 in release notes.
6. Update MonadForge documentation from "backend is unstable" to a versioned
   compatibility statement. Retain the Torch SDPA fallback for old wheels.
7. Optionally add a MonadForge fail-fast guard for the known-bad `26.06` wheel:
   reject Flash mode for non-16-aligned dense calls with an actionable error,
   or route the whole V100 run to Torch SDPA before compilation.

## Test Plan

1. Unit Tests:
   - Assert the upstream public wrapper computes the same scale for `None` and
     explicit `1/sqrt(128)`.
   - Assert a known-bad release identifier is not considered production-safe by
     any MonadForge compatibility gate.
   - Test MonadForge routing/fail-fast behavior with a mocked Flash backend; no
     CUDA hardware is required for these host-side checks.
2. Component Tests (Tesla V100 / SM70 required):
   - Dense FP16 forward for D in 16, 32, 64, 128, and 256 across all N residues
     modulo 16; assert finite output and tolerance against FP32 SDPA reference.
   - Independent M-tail and N-tail cases (`M != N`) for causal and non-causal
     attention.
   - Backward for all tail classes; assert finite dQ/dK/dV and tolerance against
     the reference gradients.
   - Exact captured tensors at lengths 4112..4128; all patched outputs must be
     finite while the release build remains a reproducible negative control.
3. Integration Tests:
   - Replay `first_failure.pt` through raw eager, compatibility eager, and
     compatibility compiled paths.
   - Run at least five Anima forward/backward steps on native-shape buckets that
     include sequence lengths 986, 2925, and 4130.
   - Compare aligned-shape throughput and peak memory before/after the kernel
     fix; reject a material regression or document the tradeoff.

## Files to Modify

- Upstream `include/softmax.h`: keep runtime tail processing and unconditional
  zero-fill for all lanes from `VALID_KV` to `BLOCK_N`.
- Upstream `include/mat_mul.h`: add assertions or debug instrumentation around
  the final 16-wide fragment if the existing fix does not pass replay.
- Upstream `test.py`: add non-aligned forward/backward and cross-attention cases.
- Upstream `setup.py`: bump the backend/package version for a new traceable
  release instead of replacing the `26.06` artifact in place.
- `bench/v100_flash/README.md`: document the affected wheel and the first
  verified fixed version/commit.
- `networks/attention_dispatch.py`: only if a local version-aware guard or
  fallback is retained after upstream validation.
- `tests/test_v100_flash_stability.py`: cover any new version gate or fallback.

## Files to Create

- Upstream `tests/test_dense_tail.py`: focused modulo-16 forward/backward matrix.
- Upstream `tests/replay_issue_43.py`: optional capture replay kept separate from
  the lightweight synthetic test suite.
- `bench/v100_flash/replay_capture.py`: local reproducible raw/compat/compiled
  replay utility if the diagnostic implementation is not already committed.

## Existing Utilities to Leverage

- `bench/v100_flash/run_probe.py`: existing Anima V100 smoke harness.
- `networks/attention_dispatch.py::dispatch_attention`: public-call boundary for
  a local fail-fast or fallback policy.
- `tests/test_v100_flash_stability.py`: host-side V100 backend detection and
  stability-mode tests.
- Diagnostic release `diagnostic-v100-20260730`: exact Q/K/V captures and the
  4112..4128 sweep used as acceptance data.
- Torch `scaled_dot_product_attention`: finite reference and production fallback.

## Success Criteria

- [ ] The published `26.06` wheel reproduces the reported failure as a negative control.
- [ ] Tag `26.06` plus `8a86213` and current main are tested on a real SM70 GPU.
- [ ] Every sequence length from 4112 through 4128 returns finite forward output.
- [ ] All M/N modulo-16 residues pass forward and backward reference checks.
- [ ] The exact 4130-token capture is finite and within the agreed SDPA tolerance.
- [ ] A five-step Anima training smoke test remains finite at 986, 2925, and 4130 tokens.
- [ ] The fixed wheel has a new version, source commit, and published SHA-256.
- [ ] MonadForge continues to default to or clearly recommend Torch SDPA for affected wheels.

## Out of Scope

- Changing Anima's independent FP16 residual stabilization.
- Making `torch.compile` trace unsupported private FlashAttention wrapper APIs.
- Hiding non-finite output with clamping or `nan_to_num`.
- Claiming current upstream `main` is fixed before a V100 replay is completed.
- Treating the maintainer's discussion style as evidence for or against the bug.
