# Upstream sync — 2026-09-19

Selective absorption from `sorryhyun/anima_lora` (upstream/main @ `f96ef473`,
2026-09-18). Last sync point was merge base `cece8e10` (2026-06-28); 499
commits were triaged into keep / reject / adapt below. Every absorbed commit
landed as a `cherry-pick -x` on branch `sync-upstream-2026-09`, so each
carries `(cherry picked from commit …)` provenance.

## Absorbed (P0 / P1 / P3 per the sync review)

| Feature | Upstream commits |
|---|---|
| Regex escape fix (#101, SyntaxWarning on py3.13) | `3d0a12b4` |
| Anima-2.9B support (depth/width from checkpoint header; `ss_num_blocks` stamp) | `48317add` |
| Memorization countermeasures (`mem_reweight.py`, uncond-gap reweighting) | `edb3b9de` |
| `min_snr` v-pred weighting + `grad_svd`/`basis_file` lora_down init | `01581e83` |
| Position captions, the full in-tree era: v1 → v2 clause rewrite → clause-vocabulary policy → bag-relax defaults (`0.35`/`0.85`) → SAM3 soft-prompt subject detector | `a21c2e12 64cb50a1 d2de2879 f5e76a2e e2e78506 4b0faeb0 28621a35 d4ff9f75 2094d514 4a4304c7 b03b7502 d7ba9a8a 597d7894 8142ed3f 13659c31 0acdb05b 6ce71030 2f9df1cc 80666f7a` |
| Tagger over-sensitivity guards (dependency-adjacent to the position chain) | `a9e19990` |
| Aesthetic/turbo base-DiT prefix loading | `0cf70811` |
| `--compile_seq_bands` per-band dynamic-seq dispatch | `7e2728c7` |
| Comfy-layout adaln LoRA keys on the static-merge path | `0e8c38ba` |

Dependency addenda pulled in for coherence (not in the original review):
`scripts/preprocess/autotag_captions.py` + `library/preprocess/autotag.py`
(the caption-autotag stage the position-chain wiring and webui strings lean
on), and the multiview audit toolset
(`library/preprocess/multiview_{audit,sheet}.py`, `scripts/preprocess/{audit_multiview,probe_nms_pairs,probe_sam_masks}.py`)
referenced by position-captions v2.

## Adapted (intent ported, patch text deliberately not)

- **`5a41be95` (GH #92, Windows CUDA)** — upstream moved backend deps to
  dependency-groups because their lock could resolve ROCm torch on Windows.
  This tree is single-backend: `uv.lock` already pins `torch 2.12.0+cu132`
  for `sys_platform != 'darwin'`, so that failure mode cannot occur here and
  the pyproject/lock/install.ps1 restructure was skipped. Taken instead:
  `library/runtime/backend.py` with `diagnose_cuda_unavailable()` /
  `warn_if_cuda_unavailable()` wired into `train.py` and the three cache
  entry points (the "NVIDIA GPU but non-CUDA torch venv" loud diagnosis),
  plus the diagnostics slice of `tests/test_rocm_backend.py`.
- **`d0d0ad15` (v2 masking)** — §1 taken in full: `masked_loss` defaults to
  `false` (base + turbo pin), and `train.py::_prepare_dataset` strips subset
  mask dirs when the key is off, so a leftover `make mask` tree can never
  silently re-mask. §2 adapted to this pre-split tree: the in-tree MIT text
  masker (stage, `generate_masks_mit.py`, model download/catalog entry,
  `masks/mit` legacy resolution, near-twins `--signal mit_text`) is removed.
  The webui settings fields (`run_mit_mask`, `mit_*`) still round-trip so the
  Vue client keeps working; they are inert.
- **Rating vocabulary** — upstream renamed the band to Anima spellings
  (`safe`/`nsfw`); this tree keeps danbooru literals (`general`/`questionable`)
  canonical and maps the Anima spellings as legacy aliases
  (`library/captioning/taxonomy.py`).
- **2.9B (`48317add`)** — this tree already read DiT depth/width via
  `inspect_anima_checkpoint`, so `probe_dit_arch` stays as the imported helper
  but the build keeps the local mechanism; only the `ss_num_blocks` metadata
  stamp and the depth-aware `mod_end_layer` default were behavior changes.

## Rejected (P2 and out-of-scope)

- The **anime_tools curation split** (Phases 0–3: captions/tagger/masking/
  grouping/resize → external package) and everything downstream of it —
  `library/captioning/` stays in-tree; `library/downloads.py` Asset catalog
  (kept our ModelScope-aware `scripts/tasks/downloads.py`); dbv4 tagger
  backend; `anima_daemon/` package move (our `scripts/daemon/` continues).
- `project/` research lines (cjk_renderable_anima, SteerPE, MIST, SR),
  `gui/` (superseded by webui), ComfyUI custom_nodes, sigma_lowres /
  sigma_demote, ROCm support, register tokens (DSR), new merge-method
  caption-full work.

## Follow-ups recorded

- Prune the inert MIT fields from `webui/frontend/src/views/PreprocessView.vue`
  (+ webui_settings defaults) — needs the frontend build/test/preview loop.
- `docs/README.md` still lists upstream-only experimental docs referenced
  from merged comments; `test_doc_refs` was failing before this sync and its
  failure list should be re-baselined separately.

## Verification

Full suite on the merge result: **2170 passed, 9 failed, 9 skipped**. All 9
failures reproduce on the pre-sync `main` (8) or stem from ignored local
state (`configs/webui_settings.json` trigger word — not a regression). Two
`test_downloads` danbooru failures that existed on `main` are fixed by this
sync. Baseline was captured via a `git worktree` on `main`.
