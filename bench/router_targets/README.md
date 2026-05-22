# bench/router_targets

**Question.** Is routing a given Linear (e.g. `cross_attn.output_proj`) worth it,
or would it behave the same as a plain LoRA there? `router_targets` (a regex in
`configs/methods/chimera.toml` / the Hydra configs) selects which Linears get the
multi-expert routed pool; everything else falls back to single-expert LoRA. This
bench answers "is module M worth being in that set" **analytically** — no image
generation, no eyeballing, no FM val loss (which doesn't track quality on Anima).

## The idea

Within one expert pool the experts share a single down-projection `A`, so the
pool's output is exactly a single rank-r LoRA whose B is the gate-weighted sum of
the per-expert B-heads:

```
y(x) = ( Σ_k g_k(x)·B_k )·A·x  =  B_eff(g(x))·A·x
```

Freeze the gate to its dataset mean `ḡ` and you get **literally a plain LoRA**:
`B_eff(ḡ)·A·x`. That is the null model "M isn't worth routing." So define the
**routing leverage ratio**

```
ρ_M = E_x‖ΔB(x)·A·x‖² / E_x‖B_eff(ḡ)·A·x‖² ,   ΔB(x) = Σ_k (g_k(x)−ḡ_k)·B_k
```

- `ρ ≈ 0` → routing at M is numerically indistinguishable from plain LoRA → drop
  M from `router_targets` (keep the rank, lose the router).
- `ρ` large → the per-sample gate genuinely re-steers M's output → routing earns it.

`ρ` is cheap and exact: `‖Σ c_k B_k a‖² = Σ c_k c_l⟨T_kl, S⟩` with the static
pairwise Gram `T_kl = B_kᵀB_l` and the per-forward rank-space input Gram
`S = aᵀa` (both r×r). The bench stores only `(gate, S)` per forward.

When `ρ` is low, two sub-diagnostics say *why* (necessary conditions):
- `gate_drift` ≈ 0 → gate barely moves across samples ⇒ exactly plain LoRA;
- `expert_subspace_overlap` ≈ 1 → the B-heads span the same subspace ⇒ clones.

**The verdict is relative.** Rank `ρ` across every `router_target` and compare
`cross_attn.out` against `mlp.layer1/2`. The script prints that head-to-head plus
a decision column.

## Run

```bash
python bench/router_targets/analyze_router_target_leverage.py \
    --lora_weight output/ckpt/<routed-checkpoint>.safetensors \
    --dataset_dir post_image_dataset/lora \
    --num_samples 32
```

Drops `bench/router_targets/results/<ts>-<label>/result.json`
(`metrics.per_pool[*].rho`, `expert_subspace_overlap`, `gate_drift`,
`gate_norm_entropy`, plus `per_group_pool` aggregates and the routing axes).

To compare router configs (the `route_per_layer=true|false` and source axes the
question is really about), train one checkpoint per cell and run the bench on
each — the `routing_axes` block in `result.json` stamps which cell each run was.

## Coverage

- `HydraLoRAModule` — shared-A Hydra / FeRA-shared, `route_per_layer` either way
  (per-Linear router or network-level GlobalRouter), `router_source` ∈
  {input, sigma, fei, crossattn_emb}. Chimera-runtime form (`num_experts_content>0`)
  is split into content/freq pools.
- `ChimeraHydraInferenceModule` — dual free-form content + freq pools.

Not covered: the Cayley training-form `ChimeraHydraLoRAModule` (run on a *saved*
checkpoint, which always loads as the inference form) and `independent_A` FeRA's
`StackedExpertsLoRAModule` (per-expert A breaks the single-`S` trick).
