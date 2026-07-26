# bench/loha

Tier 2 numerics gate for the LoHa (LyCORIS Hadamard-product) wrapper —
`networks/lora_modules/loha.py` over `lycoris-lora==3.4.0`.

Pure parameter math — no DiT load (`bench/_anima.py` is opt-in per
`CONTRIBUTING.md`, this is an analytical simulator). CPU-friendly; runs in
seconds.

## Run

```bash
python bench/loha/run_bench.py
python bench/loha/run_bench.py --label <date> --lora_dim 4 8 16 32
```

## What it quantifies

| Surface | Headline numbers |
|---|---|
| Wrapper (official bypass ops) vs official rebuild forward | max abs output error at fixed seed (`~1e-6`, fp32 accumulation only) |
| The 3.4.0 LoKr/LoHa scale asymmetry | output error IF the LoKr wrapper's `multiplier*scale` bypass fix were copied to LoHa (double-scale blowup — must stay large) |
| `get_diff_weight` double-scale bug | official/fixed norm ratio (equals `alpha/rank` ⇒ upstream applies scale twice; the wrapper override applies it once) |
| Merge equivalence | `merge_to(state_dict)` vs `base_W + get_weight()` max abs error (must be exactly 0) |
| Capacity vs plain LoRA | param count (2× LoRA-r), rank ceiling `r²`, and the numerical rank actually achieved by random-init deltas at real Anima DiT shapes |

Baseline run: `results/20260726-1724-baseline/` — wrapper-vs-official max abs
err `7.6e-06`, `gdw_ratio = scale = 4.0` on both probes, merge-vs-fuse err
`0.0`, rank ceiling hit 12/12 capacity rows.

An empirical Anima training side-by-side (LoHa vs LoRA at matched budget) is
still open — this bench is the numerics gate only, per the LoKr precedent.

Drops a `result.json` envelope (schema from `bench/_common.py`) into
`bench/loha/results/<ts>[-<label>]/`.
