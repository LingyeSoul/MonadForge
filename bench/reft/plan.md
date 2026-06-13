# ReFT — does block-level residual-stream intervention earn its place vs plain LoRA?

**Status:** decide-before-keep. ReFT is being **downgraded from a shipped variant to a bench probe** — this plan is the evidence gate that decides whether it gets retired, re-scoped, or promoted back. Not currently a recommended method.

**Owner question:** *At a **matched trainable-parameter budget**, can ReFT (a) converge faster than plain LoRA, (b) reach competitive final quality, or (c) reach states plain LoRA structurally cannot — on Anima personalization? If none of the three, retire it with evidence.*

---

## Where this came from

ReFT (LoReFT, Wu et al. NeurIPS 2024) was imported and wired into the LoRA family but **never benched on Anima** — no on/off comparison, no `bench/`, structural tests only. It's off-by-default (`lora.toml:145` is commented out despite the `lora.toml:3` header claiming otherwise), unproven, and not foldable (`merge` refuses it). Before it accretes more cross-cutting surface (GUI fields, 4-language i18n, merge special-casing), it needs to justify itself or go.

The theory (from the design discussion that spawned this bench): ReFT is the **only adapter that edits the post-residual fused representation** `h = x + attn_out + ffn_out`, with an **affine** form `h + R^T(W_s·h + b)` confined to a learned-orthonormal `reft_dim`-subspace, applied **uniformly to every patch token** (no spatial structure). That structural profile predicts:

- **Strong** at *global* tone/style/palette steering — the same lever class as mod-guidance ("global-tone lever, not a content lever" — memory `mod_guidance_sigma_film`), DCW, channel-scaling, PiD color drift.
- **Weak** at *spatial / compositional / fine-identity* edits — its per-token map can't express "red here, blue there."
- **Possibly param-efficient** on the global-style axis because it intervenes once per block on the fused stream instead of re-skilling weights.

This bench falsifies or confirms that profile against plain LoRA, the current workhorse for personalization and the turbo/EasyControl stacks.

---

## The two adapters, at matched params

Anima DiT: `model_channels = 2048`, `num_blocks = 28`. ReFT wraps **block outputs**, so its `embed_dim = 2048`.

**ReFT param count** (`networks/lora_modules/reft.py`): per block = `rotate_layer (2048·d)` + `learned_source (2048·d + d)` ≈ `4097·d`. For `reft_layers="last_8"`:

| `reft_dim` d | params/block | last_8 total |
|---|---|---|
| 16 | 65.6k | **0.52M** |
| 32 | 131k | **1.05M** |
| 64 | 262k | **2.10M** |

**Plain LoRA** param count depends on `network_dim` × targeted Linears. **Do not hand-compute — instrument it:** after `network.apply_to`, read `sum(p.numel() for p in network.parameters() if p.requires_grad)` and tune `network_dim` until it lands within **±5%** of the ReFT arm's budget. The harness asserts the match and records both counts in `result.json`.

**Isolation gotcha (load-bearing):** the shipped `configs/gui-methods/reft.toml` keeps **LoRA at rank 16 alongside ReFT** — useless for attribution. The ReFT-only arm MUST disable LoRA entirely and assert **zero `lora_*` tensors / zero LoRA trainable params** at build time. If the network refuses to build with no LoRA modules, document the minimal-LoRA floor and subtract it.

**Fairness controls:**
- Per-method **LR sweep** (small grid, e.g. {5e-5, 1e-4, 2e-4}) — report best-of-grid per method, never a single shared LR (ReFT's post-residual site and orthogonality reg may want a different LR than LoRA's weight delta).
- Identical dataset, step budget, seed set, eval prompts, optimizer family, σ-sampling.
- ReFT's `regularization()` (`‖RRᵀ−I‖²`) term **included** with the same weight the shipped path uses.
- Both zero-init to identity at step 0 (ReFT `delta=0`, LoRA `up=0`) — already fair.
- Report **wall-clock steps/sec** for each arm, not just step count — ReFT adds a per-block linear on every token, so "faster per step" and "faster per second" can disagree. A convergence win must be honest in both.

---

## What the bench measures

**Primary signal: CMMD** (paired PE-Core MMD², the repo's live val signal — memory `cmmd_val_signal`). **Not FM val loss** — it doesn't track quality on Anima (memory `fm_val_loss_uninformative`). Plus a fixed **prompt×seed eyeball grid** at each checkpoint so every number is sanity-checked by eye.

### Q1 — Convergence speed
Train both arms (matched params), log **val-CMMD vs step** every N steps. Report:
- **steps-to-threshold** = steps for each arm to reach the *other* arm's final CMMD (and to reach a fixed absolute CMMD).
- **AUC** of the CMMD-vs-step curve (lower = faster overall descent).
- the same in **wall-clock seconds**, not just steps.

### Q2 — Competitive final quality
At plateau (matched params, matched wall-clock budget): **final CMMD** + grid. ReFT within CMMD noise of LoRA → "competitive"; clearly worse → "not."

### Q3 — Compensate something LoRA cannot  *(the scientific core)*
Three probes, escalating in mechanistic depth:

**3a — Task-type split.** Run Q1/Q2 on **two concepts**:
- a **global-style** concept — a palette/lighting/texture "look" (the open `configs/easycontrol/colorize.toml` is the natural candidate-class: a global recolor/tone target);
- a **spatial-identity** concept — a specific character/object whose value is structure/identity, not tone.

Predicted matrix (the thing we're testing):

| | global-style | spatial-identity |
|---|---|---|
| **ReFT** | closes most of gap | underfits |
| **LoRA** | competitive | dominates |

If ReFT **ever beats matched-param LoRA on any axis**, that's a real niche.

**3b — Complementarity at matched *total* params.** `LoRA(rank r)` alone vs `LoRA(rank r′) + ReFT` where `params(r′) + params(ReFT) = params(r)` (steal rank from LoRA to pay for ReFT). If `LoRA+ReFT > LoRA-only` at equal total budget on the global-style concept, ReFT buys something extra LoRA rank cannot. Cleanest apples-to-apples "compensate" test — same artifact, same budget.

**3c — Representation-reachability probe (optional, mechanistic).** After fitting the same concept with each arm, measure the induced per-token block-output delta `Δh = h_with − h_without` over a prompt set. Ask: is `Δh_ReFT`'s subspace **contained in** LoRA's reachable edit set, or **complementary**? Project `Δh_ReFT` onto `span(Δh_LoRA)` and report residual energy + principal-angle cosines. High residual ⇒ ReFT reaches directions a matched-param LoRA cannot, even if end-quality ties. (Mirrors the repo's mechanistic benches: `mod_guidance/text_jacobian.py`, `timestep_mask/learned_rank.py`.)

---

## Running it  *(to be implemented in `run_bench.py`)*

```bash
# correctness smoke — tiny dataset, ~few-hundred steps, one concept
uv run python -m bench.reft.run_bench \
    --concept colorize --arms lora reft --param_budget 2.1M \
    --max_steps 300 --eval_every 50 --label smoke

# real read — both concepts, matched params, LR grid, block-compile
uv run python -m bench.reft.run_bench \
    --concept colorize identity --arms lora reft lora+reft \
    --param_budget 2.1M --lr_grid 5e-5 1e-4 2e-4 \
    --max_steps 1500 --eval_every 100 --compile --label reft-vs-lora
```

Drives training via the existing trainer (matched configs built on the fly, `--method` overrides), logs val-CMMD per checkpoint, decodes the fixed eval grid. Standard `bench/_common.py` envelope.

### Output (`results/<ts>-<label>/`)
- `result.json` — envelope; `metrics.param_counts` (both arms, ±% match), `metrics.cmmd_curves` (per arm, per concept), `metrics.steps_to_threshold`, `metrics.final_cmmd`, `metrics.complementarity` (3c), one-line `metrics.verdict`.
- `cmmd_vs_step.png` — Q1 curves, one line per arm × concept.
- `grid_<concept>.png` — eyeball montage at the final checkpoint.
- `reachability.png` — 3c principal-angle / residual-energy plot (if run).

---

## How to read it — decision gates

- **`REFT EARNS A NICHE`** — wins or ties matched-param LoRA on the **global-style** concept, *or* `LoRA+ReFT > LoRA-only` at matched total (3b), *or* 3c shows genuinely complementary reach. → **Keep ReFT, but re-scope it:** docs say "global-style / tone concepts," drop it from any default stack, keep it opt-in. The header lie at `lora.toml:3` gets fixed either way.
- **`REFT CONVERGES FASTER, CAPS LOWER`** (or the reverse) — keep narrowly as a fast-warmup / few-step option; document the exact tradeoff with numbers.
- **`REFT IS REDUNDANT`** — LoRA ≥ ReFT on every concept at matched params **and** `LoRA+ReFT ≤ LoRA-only` (3b) **and** 3c shows ReFT's reach ⊆ LoRA's. → **Retire it with evidence:** remove module, configs, GUI fields, i18n, merge special-case, docs; keep this bench record as the obituary. The likely outcome given the prior that uniform low-rank residual edits keep collapsing to global-tone levers on this model.

---

## Scope & named follow-ups (out of phase 1)

- **Distillation polish (phase 3).** The one place ReFT might add to the turbo/DP-DMD student: a `last_8` ReFT **active only at the low-σ step** (Anima's x0 is visually resolved by σ≈0.45 — memory `sigma_signal_resolves_by_045`), polishing systematic few-step tone drift *on top of* the LoRA student. Test `student = LoRA + low-σ-only ReFT` vs `LoRA + matched-rank LoRA` by CMMD. **Caveat:** ReFT isn't foldable, so a win here costs turbo's "output is a plain LoRA" deployability — weigh against the free training-free correctors (DCW/CNS/channel-scaling) that already own this lever.
- **EasyControl:** *not* a candidate to replace cond-LoRA — spatial conditioning is structurally outside ReFT's uniform-per-token map. Excluded by design, not by experiment.
- **CFG / aspect:** held fixed in phase 1; revisit only if a niche is found (per the DCW lesson, optima move with CFG/aspect).

---

## References

- `networks/lora_modules/reft.py` — the module (`forward` is the whole story).
- `docs/methods/reft.md`, `docs/structure/reft.md` — current (about-to-be-re-scoped) docs.
- memories: `cmmd_val_signal`, `fm_val_loss_uninformative`, `mod_guidance_sigma_film` (the global-tone-lever prior), `sigma_signal_resolves_by_045`, `turbo_per_step_expert`.
- `bench/sigma_reshape/README.md`, `bench/mod_guidance/text_jacobian.py` — format + mechanistic-probe precedents.
