# bench/v100_flash — V100 FlashAttention stability probe

`flash-attention-v100` is a third-party/drop-in FA2-style backend for Volta
(`sm_70`). The published `26.06` wheel is not production-stable for Anima. The
latest main removes the original non-finite dense-tail failure, but repeated
full-capture replay still fails the upstream numerical limit on this V100.

## Issue 43 validation status (2026-07-30)

The strict validation used a Tesla V100-SXM2-16GB, Python 3.12.13, PyTorch
`2.10.0+cu129`, CUDA toolkit 12.9.1, and the exact issue capture. No Q/K/V
padding, `nan_to_num`, custom softmax scale, or global FP16 accumulation change
was used.

| Candidate | Source / wheel SHA-256 | Capture and prefixes | Dense tail matrix | Real Anima LoRA |
| --- | --- | --- | --- | --- |
| Published `26.06` | `d89800e`; `74b4cdbd...a6a2` | 4130-token raw/eager/compiled non-finite; only 4112 and 4128 finite | Performance control only | Expected self-attention failure at 986 tokens |
| `26.06` plus tail fix | `88100ac` (conflict-resolved port of `8a86213`); `388a914c...8005` | raw/eager/compiled finite; 17/17 prefixes finite | 480/480 forward/backward cases passed | 5/5 optimizer steps passed; 986/2925/4130 covered |
| Fixed main | `c91cad40`; `08d5e4da...f307` | raw/eager/compiled finite; 17/17 prefixes finite | 480/480 forward/backward cases passed | 5/5 optimizer steps passed; 986/2925/4130 covered |

The matrix covers head dimensions 16/32/64/128/256, every query and key
residue modulo 16 through self/query-tail/key-tail shapes, causal and
non-causal attention, and forward plus dQ/dK/dV. Both fixed candidates passed
the upstream FP16-relative numerical limits. Current main's tightest full
capture result was `0.001953125 <= 0.001963125`.

That was a single-run result. The Python 3.13 landing below added repeated
full-capture replay and found intermittent errors above the same upstream limit;
the single passing sample must not be read as production approval.

The published wheel reproduced the defect class and non-finite mask parity
across all three Flash paths, while Torch SDPA stayed finite for all 17
prefixes. Its new raw replay counts were stable at 947,328 NaN, 450 +Inf, and
446 -Inf, not the capture's saved 1,412,224 / 562 / 590 counts. The strict
negative-control count check therefore remains failed and is not normalized
away.

Aligned-length timing used three rounds of ten runs after three warmups:

| Length | Candidate | Forward median | Backward median | Peak allocated |
| ---: | --- | ---: | ---: | ---: |
| 4112 | Published | 6.949 ms | 26.705 ms | 273.721 MiB |
| 4112 | Tail fix | 7.026 ms (+1.11%) | 26.667 ms (-0.14%) | 289.971 MiB (+5.94%) |
| 4112 | Main | 6.597 ms (-5.08%) | 24.352 ms (-8.81%) | 289.971 MiB (+5.94%) |
| 4128 | Published | 7.034 ms | 26.844 ms | 274.786 MiB |
| 4128 | Tail fix | 7.093 ms (+0.83%) | 26.895 ms (+0.19%) | 291.036 MiB (+5.91%) |
| 4128 | Main | 6.644 ms (-5.55%) | 25.217 ms (-6.06%) | 291.036 MiB (+5.91%) |

There is no sustained timing regression above 5%, but both fixed local wheels
show a material 5.9% peak-allocated and 6.5% peak-reserved increase in this
benchmark. Together with the exact negative-control count mismatch and the
absence of a new versioned upstream wheel, this prevents a full release
acceptance claim.

**Recommendation:** continue to use `attn_mode="torch"` for V100 production
training until an official, versioned fixed wheel passes the same suite. The
local fixed builds are diagnostic evidence, not release artifacts.

The validation tools are:

```bash
python -m bench.v100_flash.replay_capture --help
python -m bench.v100_flash.run_tail_matrix --help
python -m bench.v100_flash.run_anima_smoke --help
```

The complete local report, JSON, wheels, source trees, build records, and logs
are under `output/v100-flash-validation/`.

## Python 3.13 local landing

MonadForge pins the V100 source landing to current upstream main
`c91cad40c0539805754819e6ea96c75184d816a6`; the upstream kernel source is not
patched. Build it for the existing Python 3.13 / Torch 2.10 V100 environment:

```bash
make v100-flash-install
make v100-flash-validate
```

The installer writes a cp313 wheel, full build logs, compiler/toolkit details,
and hashes under `output/v100-flash-install/`. Installation alone records
`installed_unvalidated`; the validator must pass the full capture, 480-case
dense-tail matrix, fullgraph compatibility tests, and eager/compiled Anima
steps before it records `validated` and creates `V100_flash` plus `V100_sdpa`
under `configs/custom/presets/`.

The V100 wheel and the normal Ampere `flash-attn` distribution both provide the
same top-level `flash_attn` package. The installer refuses an existing provider
instead of overwriting it. A later `uv sync` follows `pyproject.toml` and can
replace the local provider, so rerun the install and validation tasks after any
environment resync.

### Python 3.13 result on this V100 (2026-07-30)

The local build used upstream main unchanged:

- source: `c91cad40c0539805754819e6ea96c75184d816a6`
- wheel: `flash_attn_v100-26.6-cp313-cp313-linux_x86_64.whl`
- wheel SHA-256: `e3dc0155e902fe9f769038214e044f64eddb991708a8da6a3867beb5a99137ab`
- installed extension SHA-256: `07056c01805d9554c1fbdab51670da1ac27963e84e3d465c597a1bf8d3866d5f`
- Python `3.13.14`, PyTorch `2.10.0+cu129`, CUDA toolkit `12.9.1`, driver
  `580.173.02`, Tesla V100-SXM2-16GB (`sm_70`)

Strict result: `validation_failed`. The final evidence is in
`output/v100-flash-install/validation/20260730T144316Z/` and the current state is
recorded in `output/v100-flash-install/current.json`.

| Gate | Result |
| --- | --- |
| Host/GPU integration tests | 36 passed |
| 4112..4128 prefix replay | 17/17 finite on raw/eager/compiled Flash; SDPA finite |
| 480-case dense-tail matrix | 480/480 forward/backward cases passed |
| Real Anima eager Flash | 5/5 optimizer steps passed; 986/2925/4130 covered |
| Real Anima compiled Flash | 5/5 passed; dynamic sequence, 8 swapped blocks, no graph break |
| Real Anima compiled SDPA | 5/5 passed with the same production compile order |
| 4130 full-capture repeats | Failed upstream FP16-relative tolerance |

Every full-capture output was finite. The failure is intermittent accuracy, not
the old NaN/Inf defect: over ten repeats per path, `compat_flash_eager` failed at
repeats 6 and 8 with a worst FP32-relative max error of `0.003173828125`, and
`compat_flash_compiled` failed at repeat 4 with a worst error of `0.0029296875`.
The allowed limit was `0.001963125`. An immediately preceding three-repeat run
happened to pass, demonstrating why the strict validator now fixes the repeat
count at ten.

The MonadForge compile integration was repaired without changing the Flash
kernel: compiled finite checks use an in-graph asynchronous assertion, while
eager checks retain detailed tensor ranges and block indices. Stable compiled
labels prevent per-block guard recompilation. Both compiled Flash and compiled
SDPA then completed all five real Anima steps.

Against the existing cp312 main baseline, aligned 4112/4128 forward deltas were
`+0.279%` / `+0.501%`, backward deltas were `-0.019%` / `+0.045%`, and peak
allocated memory changed `0%`; no sustained regression exceeded 5%.

Because the repeated capture gate failed, `V100_flash.toml` and
`V100_sdpa.toml` were revoked and the manifest remains `validation_failed`.
Continue to use the existing Torch SDPA V100 configuration for production.

### MonadForge training evidence (2026-07-31)

The following three raw daemon records are a same-machine Anima comparison. All
three runs used the same 144-step configuration and completed with return code
zero; only the attention backend changed. The configuration used
`target_res=[768]`, with native buckets of 2128 and 2196 tokens. It did not run
the 1024x1024 / 4096-token tier; that sample was explicitly skipped because it
was outside the compiled dynamic-sequence range.

| Backend | Job | End-to-end time | Steady step median |
| --- | --- | ---: | ---: |
| Flash | `20260731-000709-c946d5` | 181 s | 0.5875 s |
| Torch SDPA | `20260731-001142-85ff73` | 165 s | 0.4850 s |
| Memory-efficient SDPA | `20260731-001509-e84e33` | 147 s | 0.4850 s |

Each directory contains the submitted `job.json`, `progress.jsonl`, and
`stdout.log` files:

- [`flash` logs](../../output/daemon/jobs/20260731-000709-c946d5/)
- [`torch` logs](../../output/daemon/jobs/20260731-001142-85ff73/)
- [`mem_efficient` logs](../../output/daemon/jobs/20260731-001509-e84e33/)

The progress stream records every two optimizer steps, so its event intervals
must not be read as single-step timings. These runs show that Flash is about
21% slower than Torch on this V100 configuration; they do not establish a
regression from an older 1024-tier run.

### Optional memory-efficient SDPA

`attn_mode="mem_efficient"` forces PyTorch's native Efficient Attention backend
for Anima attention calls only. Unlike the process-global CUDA backend flags, it
does not change backend selection for the text encoder, VAE, or third-party
components. The public `sdpa_kernel` context is traceable by current
`torch.compile` (one graph, no graph break in the invariant test).

Validate the installed V100 PyTorch build before adopting it:

```bash
python -m bench.v100_flash.run_probe \
  --attn_mode mem_efficient --device cuda --steps 5 --label mem-efficient
```

If every step is finite and peak VRAM/step time improve over `--attn_mode torch`,
set `attn_mode = "mem_efficient"` in the custom V100 preset. A build without a
compatible Efficient Attention CUDA kernel fails explicitly instead of falling
back to the O(N^2) math backend.

## Production V100 recipe

Keep the hardware profile in the user-owned (and git-ignored)
`configs/custom/presets/V100.toml`, then run:

```bash
PRESET=V100 make lora
```

PowerShell / cross-platform task runner:

```powershell
$env:PRESET = "V100"
python tasks.py lora
```

Use this profile:

```toml
mixed_precision = "fp16"
save_precision = "fp16"
attn_mode = "torch"
torch_compile = true
gradient_checkpointing = true
unsloth_offload_checkpointing = false
blocks_to_swap = 8
```

Leave PyTorch's SDPA backend selection intact. Do not set CUDA SDPA backend
flags globally: other components need the math fallback, and current MonadForge
does not disable memory-efficient SDPA.

Operational notes from the verified V100 environment:

- Use a V100-compatible PyTorch build/venv (for example `torch==2.10.0+cu129` in
  `.venv`). Newer CUDA/PyTorch wheels may omit SM 7.0 kernels and fail with
  `no kernel image is available for execution on the device`.
- `torch_compile=true` can remain enabled with `attn_mode="torch"`; disable it
  only when debugging the unsupported `flash-attention-v100` path.
- Keep the V100 preset hardware-focused. Avoid method-level conflicts such as an
  empty `network_weights = ""` path or init-strategy overrides that fight the
  selected method.
- If a method TOML fully replaces `[[datasets]]`, include `image_dir` and
  `cache_dir`; otherwise the base dataset blueprint paths are dropped.

## Diagnostic quick checks on a V100

These commands are for investigation, not production recommendation:

```bash
# Baseline: should be finite.
python -m bench.v100_flash.run_probe --attn_mode torch --device cuda

# Candidate: force native memory-efficient SDPA (must not silently use math).
python -m bench.v100_flash.run_probe --attn_mode mem_efficient --device cuda

# Diagnostic: self-attn flash, cross-attn torch SDPA.
python -m bench.v100_flash.run_probe --attn_mode flash --stability hybrid --debug_finite --device cuda

# Diagnostic: full flash with finite checks around q/k/v and attention outputs.
python -m bench.v100_flash.run_probe --attn_mode flash --stability safe --debug_finite --device cuda
```

Results are written under `bench/v100_flash/results/<timestamp>/result.json`.

If the V100 venv is minimal, install full project deps first (the tiny Anima
fixture still needs normal runtime deps such as `einops`):

```bash
# Example only — use the same venv that contains the V100-compatible torch build.
pip install -r requirements.txt
# or install the project with its normal dependency workflow, without replacing torch.
```

## Stability modes

| Mode | Meaning | V100 Anima fp16 status |
|---|---|---|
| `off` | Normal `attn_mode=flash` behavior. | Not recommended; full flash was not pursued after hybrid failed in self-attn. |
| `hybrid` | Keep self-attention on FlashAttention, route cross-attention through torch SDPA. | Still failed: first non-finite tensor came from self-attn flash output. |
| `safe` | Keep flash but enable finite checks around q/k/v, attention output, projection output, block residuals, loss, and gradients. | Diagnostic only; it cannot make an unstable kernel numerically safe. |

Training accepts the same mode through config/CLI:

```toml
attn_mode = "flash"
v100_flash_stability = "hybrid"  # off | hybrid | safe, diagnostics only on V100
```

or temporarily via environment variable:

```bash
ANIMA_V100_FLASH_STABILITY=hybrid ANIMA_DEBUG_FINITE=1 python tasks.py lora-gui tlora
```

Do not use `nan_to_num` to hide NaNs in training. If the probe or training fails,
keep the first `FloatingPointError` location; it tells whether the first non-finite
value appeared before attention, after attention, after a residual add, in the
loss, or in gradients.

## Import compatibility note

`flash_attn_v100` exposes the public functions used by the main backend:

- `flash_attn_func`
- `flash_attn_varlen_func`

Some releases do **not** expose official FlashAttention internal helpers such as
`_flash_attn_forward`, `_wrapped_flash_attn_forward`, or
`_wrapped_flash_attn_backward`. MonadForge should treat those internal wrappers as
optional: their absence should not block public `flash_attn_func` import, but
features that need the wrapped internals must fall back or report that the V100
fork lacks them.
