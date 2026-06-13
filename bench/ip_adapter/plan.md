# IP-Adapter — does zero-shot image-prompt conditioning earn its place vs just training a LoRA?

**Status:** removed-pending-evidence. IP-Adapter was **downgraded from an experimental shipped method to this bench probe on 2026-06-10** (live wiring removed, full restore map in `INTEGRATION.md`). This plan is the evidence gate that decides whether it gets retired for good or restored. Not currently a recommended method.

**Owner question:** *On Anima's actual use case — personalization from a handful of reference images — does IP-Adapter's zero-shot "drop in a reference, get the look" beat the obvious baseline of just **training a quick personalization LoRA** on the same reference(s)? If a 5-minute LoRA matches its reference fidelity on unseen prompts, IP-Adapter has no niche on this model and stays retired.*

---

## Why it was pulled (read this before re-litigating)

Three reasons, in order of weight. **Note which one is load-bearing** — the first is, the others reinforce it.

1. **Data-fit mismatch — the real reason.** IP-Adapter learns a *generalizable* image conditioner: given an arbitrary reference image at inference, produce a matching generation with no per-subject training. Learning that generalization needs a **large, diverse, paired** corpus (the distinct-pair / identity-pair contract in `impl/docs_experimental_ip-adapter.md` and `pair_audit.py` exists precisely because the signal is pair-hungry). Anima is a **personalization** pipeline where the workhorse — a LoRA — fits a concept from a *handful* of images in minutes. IP-Adapter's data appetite structurally misfits the product. The capability it buys (zero-shot reference) is exactly the capability the product doesn't need if a quick LoRA is good enough.

2. **Never benched on Anima.** No on/off comparison, no quality number against any baseline. It shipped on the strength of the upstream IP-Adapter paper, not on evidence here. (Phase-0 `pair_audit.py` only measured whether the *dataset can supply pairs* — never whether the trained adapter is *good*.)

3. **Cross-cutting surface it hadn't earned.** A `configs/methods/` + `gui-methods/` + `configs/datasets/` config trio, a GUI tab + 4-language i18n, a PE-LoRA prototype, distinct-pair orchestration threaded through `train.py` + `datasets/base.py`, a `_setup_ip_adapter` inference path, and three `exp-*` task targets. Same accretion-without-proof pattern that retired ReFT (`bench/reft/`).

### What is NOT a reason — don't justify the removal this way

**"EasyControl is the better referencing task" is wrong and must not be the argument.** They are orthogonal conditioning axes, not substitutes:

- **EasyControl** = *spatial / structural* control (extended self-attention, ControlNet-like) — "follow this layout."
- **IP-Adapter** = *semantic / style / identity* image-prompt (decoupled cross-attention, uniform over the canvas) — "match this subject/vibe," no spatial constraint.

EasyControl does not cover IP-Adapter's use. If IP-Adapter stays retired, the thing that replaces it is **personalization-LoRA**, not EasyControl. This is the opposite of the ReFT case: ReFT was genuinely *redundant* with live training-free levers (DCW / channel-scaling / mod-guidance); IP-Adapter is **not redundant — it's the only thing that did zero-shot image-prompt** — it's just expensive and unproven. So the gate is not "is something else better at the same job," it's **"does the product want this job at all, given the cheaper alternative."**

---

## The comparison — IP-Adapter vs a quick LoRA, on the same reference

The honest baseline is not "matched params" (as in the ReFT bench) — IP-Adapter and a personalization LoRA have different deployment contracts (zero-shot vs per-concept train). The honest baseline is **matched *user effort to a result*** and **matched reference fidelity on held-out prompts**.

**Arm A — IP-Adapter (zero-shot).** Load the trained IP-Adapter weight, feed reference image `R`, generate the eval prompt set. Zero per-reference training; inference-time conditioning only.

**Arm B — quick LoRA (the product's actual move).** Train a small personalization LoRA on `R` (and, where the reference is a multi-image concept, the few images of it) for a short fixed budget (e.g. ≤ 5 min / a few hundred steps), then generate the same eval prompt set.

**Reference set `R`:** drawn from the **held-out** identities in `pair_audit.py`'s cross-artist character pool (the tightest identity signal) — concepts IP-Adapter did **not** see paired in training, so Arm A is genuinely zero-shot and Arm B is genuinely few-shot on the same images. Run **both** a single-image-reference condition (the IP-Adapter-favourable case) and a 3–5-image-concept condition (the LoRA-favourable case).

**Fairness controls**
- Identical eval prompts × seeds, identical sampler / CFG / steps / aspect at generation.
- Arm B's training budget is **fixed and reported in wall-clock** — the whole point is "cheap." If Arm B needs an LR/step sweep to be fair, report best-of-a-small-grid and the total wall-clock it cost.
- Same VAE / text-encoder / decode path; same PE cache for the metric (see below).
- Report Arm A's one-time training cost separately — it's amortized over all references, so it doesn't count against per-reference effort, but it must be visible (it's the data-appetite the removal is premised on).

---

## What the bench measures

**Primary signal: CMMD** (paired PE-Core MMD², the repo's live val signal — memory `cmmd_val_signal`; **not FM val loss**, memory `fm_val_loss_uninformative`). Score each arm's generations against the reference identity. Plus a fixed **prompt × seed eyeball grid** per arm so every number is sanity-checked by eye — reference fidelity vs prompt adherence is a two-axis judgment a single scalar will flatten.

### Q1 — Reference fidelity on held-out prompts
For each reference `R`, CMMD(generations, reference-identity) for Arm A vs Arm B, single-image and multi-image conditions. The question: does zero-shot IP-Adapter hold identity as well as a quick LoRA across prompts it must generalize to?

### Q2 — Prompt adherence / editability
IP-Adapter's decoupled cross-attention is known to trade prompt adherence for reference strength (the `ip_scale` knob). At the `ip_scale` that matches Arm B's fidelity (Q1), does Arm A still follow the *text* prompt as well as the LoRA? Eyeball grid + a CLIP/PE text-alignment proxy on the eval prompts. A reference win that costs prompt control is not a win for this product.

### Q3 — Where (if anywhere) IP-Adapter wins
The one regime IP-Adapter should dominate by construction: **many distinct references, each used once** (a gallery / batch-restyle workload where per-reference LoRA training is amortization-hostile). If the product ever wants that, IP-Adapter's amortized one-time train beats N short LoRA trains. Estimate the **break-even N** (references-per-session above which Arm A's amortized cost wins) and state plainly whether Anima's workflows ever reach it. This is the only gate branch that keeps IP-Adapter alive.

---

## Running it  *(to be implemented in `run_bench.py`)*

```bash
# correctness smoke — one held-out identity, single-image ref, both arms
uv run python -m bench.ip_adapter.run_bench \
    --refs cross_artist --n 1 --arms ip lora \
    --lora_budget_steps 300 --label smoke

# real read — held-out identities, single + multi-image, ip_scale sweep on arm A
uv run python -m bench.ip_adapter.run_bench \
    --refs cross_artist --n 12 --conditions single multi \
    --arms ip lora --ip_scale_grid 0.6 0.8 1.0 \
    --lora_budget_steps 400 --label ip-vs-lora
```

Arm B drives training through the existing trainer (a plain-LoRA config built on the fly per reference). Arm A loads the preserved IP-Adapter weight (restore `impl/ip_adapter.py` onto the live tree first — see `INTEGRATION.md` — or run the bench against a checkpoint trained before removal). Standard `bench/_common.py` envelope.

### Output (`results/<ts>-<label>/`)
- `result.json` — envelope; `metrics.cmmd` (per arm, per condition, per reference), `metrics.prompt_align`, `metrics.lora_wallclock`, `metrics.break_even_n` (Q3), one-line `metrics.verdict`.
- `fidelity_<condition>.png` — Q1 grid, IP vs LoRA per reference.
- `editability_<ref>.png` — Q2 prompt-adherence grid across `ip_scale`.

---

## How to read it — decision gates

- **`IP-ADAPTER IS REDUNDANT`** — a quick LoRA matches or beats IP-Adapter on held-out reference fidelity (Q1) without the prompt-adherence tax (Q2), and Anima's workflows never reach the break-even N (Q3). → **Stay retired.** This bench is the obituary; `impl/` is the only record kept. The likely outcome given the prior that personalization-LoRA is the product's center of gravity.
- **`IP-ADAPTER WINS A NICHE`** — it clearly holds identity *better* than a short LoRA on single-image references **and** keeps prompt adherence at a usable `ip_scale`, **or** Q3's break-even N lands inside a real Anima workflow (batch restyle / large gallery). → **Restore it (re-scoped):** re-apply `INTEGRATION.md`, but document it as "zero-shot / many-reference only, not the personalization path," and **fix the data story first** — the removal premise was data appetite, so a restore must come with the corpus to feed it.
- **`MIXED`** — wins fidelity but loses editability, or wins only multi-reference. → keep retired in-tree; record the exact tradeoff here so it isn't re-discovered. Revisit only if a concrete product surface needs the niche.

---

## Scope & named follow-ups (out of phase 1)

- **PE cache stays regardless.** `scripts/preprocess/cache_pe_encoder.py` (+ `make preprocess-pe`) and the `{stem}_anima_pe.safetensors` / `anima_pe_centroid_*.safetensors` sidecars are **not** IP-Adapter-private — CMMD validation (`library/training/cmmd.py`) and DCW v4 read them. They were left in the live tree. This bench reuses them for the metric.
- **PE-LoRA stays too.** `networks/methods/ip_adapter_pe_lora.py` is vendored into the live Anima-Tagger ComfyUI node (`scripts/sync_vendor.py`, `comfyui-anima-tagger/nodes.py` `pe_lora` UI) — left in place. The IP-Adapter import of it was the thing removed.
- **EasyControl:** *not* a candidate to replace IP-Adapter — spatial conditioning is a different axis (see above). Excluded by design, not by experiment.
- **`ip_scale` / CFG / aspect:** swept in phase 1 only enough to find the fidelity-matched operating point; not a full optimization.

---

## References

- `impl/ip_adapter.py` — the module (PE-Core vision encoder → Perceiver resampler → per-block `to_k_ip`/`to_v_ip` + init-0 `ip_gate`). `forward` + `set_ip_tokens` are the story.
- `impl/docs_experimental_ip-adapter.md` — the (removed) method deep-dive: distinct-pair training, validation baselines, inference flow.
- `pair_audit.py` / `pair_audit.md` — Phase-0 dataset pair-coverage audit (the only IP-Adapter measurement ever taken).
- memories: `cmmd_val_signal`, `fm_val_loss_uninformative`, `post_image_dataset_role`, `caption_index_shared_artifact`, `tagger_dual_hardrouted` (PE-LoRA already removed from the tagger).
- `bench/reft/plan.md` — the demotion precedent this follows.
