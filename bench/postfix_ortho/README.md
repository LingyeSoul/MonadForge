# `bench/postfix_ortho/` — structural orthogonality validation

Validation harness for the ortho-postfix variant proposed in
[`docs/proposal/orthogonal_postfix.md`](../../docs/proposal/orthogonal_postfix.md).
Tier 1.5 (numerics revision to an existing method, per `CONTRIBUTING.md`):
this bench is the evidence the proposal's design intent (K orthonormal
slots by construction, no symmetry-breaking hyperparameter) survived
training + bf16 save/load roundtrip on real checkpoints.

The harness is intentionally **weights-only** — it doesn't run inference.
The qualitative A/B comparison the proposal asks for (DiT vs DiT +
ortho-postfix vs DiT + legacy collapsed postfix) needs human judgment and
is left to manual eyeballing via `make exp-test-postfix`.

## What it checks

Wraps `archive/bench/postfix/analyze_ortho_postfix.py` and writes the
standard `result.json` envelope. The analyzer covers all three validation
gates for the v2 (C1 + svd_te) layout:

1. **Structural orthogonality after roundtrip.**
   `‖postfix @ postfix.T - lambda_global² · I‖_F < 1e-4` — confirms Cayley
   + frozen SVD basis survived bf16 save/load. `ortho_basis` is persisted
   in fp32 (a one-time ~128 KB cost; the bf16-truncated basis blew the gate
   at ~9e-4 in initial testing). Note the gate target is `λ_g² · I` in v2,
   not `diag(λ²)` — every slot has identical magnitude by construction.

2. **lambda_global is alive.** `|lambda_global| > 1e-3`, i.e. the optimizer
   didn't kill the entire postfix outright. v1's max/min ratio + pinned-zero
   gates are gone — magnitudes are uniform per construction in C1.

3. **T5-token NN per slot.** Counts distinct top-1 nearest tokens across
   the K slots. The K=1-collapse signature on the legacy postfix was
   "every slot picks the same nearest token" (see
   `archive/bench/postfix/initial_postfix_problems.md`). Under structural
   orthogonality + SVD basis (shuffled) this should disagree across slots —
   if it doesn't, that's the splice-position-symmetry failure mode the
   proposal flags as the diagnostic-value flip side (proposal §B / §3).

The proposal also calls for re-running the legacy
`analyze_sigma_tokens.py` / `analyze_cond_postfix.py` analyzers, but those
target `cond` / `cond-timestep` checkpoints — they don't apply to the
plain-`postfix` ortho variant in v1. The proposal's own validation table
notes the cond-mode `pairwise cos across captions` check is "N/A in
default `postfix` mode". v2 (cond-timestep ortho) will add those back.

## Usage

```bash
# After training a checkpoint via:
#   make lora-gui GUI_PRESETS=postfix_ortho
# (or by uncommenting the ortho block in configs/methods/postfix.toml then
# running `make exp-postfix`):

uv run python bench/postfix_ortho/run_bench.py \
    --postfix_weight output/ckpt/anima_postfix_ortho_v2.safetensors \
    --label first-run
```

Writes:

```
bench/postfix_ortho/results/<YYYYMMDD-HHMM>[-first-run]/
    result.json     — standard bench envelope (metrics + git/env info)
    analyze.json    — full analyzer payload (per-slot data, NN top-k tokens)
```

## Pass / fail summary

The proposal's three v1 outcomes correspond to three cells in `result.json`:

| Field | Pass condition | Failure interpretation |
|-------|----------------|------------------------|
| `metrics.ortho_pass` | true | bf16 roundtrip broke orthogonality — investigate basis persistence dtype |
| `metrics.lambda_alive_pass` | true | optimizer killed the entire postfix (`\|lambda_global\| < 1e-3`) — capacity got driven to zero |
| `metrics.t5_nn_distinct_top1_per_slot` | ≥ K/4 (≥ 8 at K=32) | splice-position symmetry dominates — points to proposal §B (positional fixes are the next move) |

The first two are tooling/training-loop checks. The third is the
proposal's actual scientific question: did orthogonal slots translate
into K-rank cross-attention behavior, or did the splice-position
symmetry collapse it back to rank-1?
