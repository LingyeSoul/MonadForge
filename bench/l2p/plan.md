# L2P (Latent-to-Pixel) integration — staged plan & go/no-go gates

L2P (Chen et al., *Unlocking Latent Potential for Pixel Generation*,
arXiv:2605.12013): convert a pretrained **latent** DiT into a **pixel-space**
DiT by discarding the VAE, feeding raw RGB through large-patch tokenization,
**freezing the DiT core**, and re-training only the shallow input projection +
the first-n ∪ last-n blocks + a lightweight "Detailer Head" (DiP-style U-Net) that replaces
the final projection. Trained on **LDM-self-generated** synthetic images (zero
real data). The payoff is **native 4K** without the VAE decode memory wall:
because patch size scales with resolution (16² at 1K → 64² at 4K), the DiT sees
a **constant ~4096 tokens** at any resolution — flat transformer cost.

Why Anima is a natural fit: the constant-token trick is *already* Anima's core
invariant (`CONSTANT_TOKEN_BUCKETS`, 2 compile graphs keyed on token count). A
1024² image is `/8 VAE → /2 patch = 64² = 4096` tokens today; L2P's 1K pixel
path is `/16 patch = 64²` — same grid, same RoPE. The recipe is also
architecture-agnostic and uses the *same* FM objective Anima already trains on
(`target = ε − x0`), so nothing depends on Anima being the paper's source LDM
(it isn't — Anima is 2048-dim/28-block vs the paper's Z-Image 3840/30).

This plan is **gated** — each phase is cheap relative to the next and can kill
the idea before we pay for integration. Don't skip ahead. Full proposal:
`docs/proposal/l2p_pixel_anima.md`.

---

## Phase 0 — shell-reachability precondition  [the cheapest falsifier]

**Question.** The paper's load-bearing premise is that the *frozen* DiT core,
trained on VAE-latent token statistics, still "functions within its native
optimization manifold" when fed RGB-patch tokens through a re-trained input
shell. Proven empirically on a 3840/30 source LDM. **Does it survive at Anima's
smaller 2048/28 scale on a single-GPU budget?** That, not "does the concept
work," is the Anima-specific risk.

**Test** (`probe_shell_feasibility.py`): freeze the entire DiT; replace
`x_embedder` with a fresh RGB patch-embed (input shell) and `final_layer` +
`unpatchify` with a fresh **pure token→pixel decoder** (output shell). Overfit a
tiny image set (≤64) with the exact Anima FM objective. **Shells only** by
default (`--train_blocks 0`) — the absolute minimal L2P.

Two deliberate stiffenings vs the real method, so a PASS here is conservative:
- output shell is a **pure token decoder** (no DiP input-skip U-Net) → the only
  path to sharp pixels runs *through* the frozen core;
- **zero** unfrozen blocks (paper's Fig-9b shallow tuning is Phase 1).

**Gate.**
- **PASS** — FM loss drops >30% and lands <0.85, *and* the Euler-sampled
  montages show recognizable global structure → the frozen priors are reachable
  from pixel space → Phase 1.
- **WEAK** — loss drops 10–30%, montages are blobby-but-organized → the core is
  reachable but shells-only underfits; Phase 1's shallow (first-n ∪ last-n) block
  tuning is likely load-bearing, not optional. Proceed cautiously.
- **FAIL** — loss plateaus near the pure-noise floor / montages stay noise →
  latent→pixel gap too wide for cheap transfer at this scale. **Stop**; the
  paper's result doesn't transplant to Anima's budget.

```bash
# --dit defaults to Anima 1.0 (models/diffusion_models/anima-base-v1.0.safetensors)
python bench/l2p/probe_shell_feasibility.py \
    --num_images 64 --steps 2000 --resolution 1024
# first sanity check: a SMALL run must not NaN and loss must move
python bench/l2p/probe_shell_feasibility.py --num_images 8 --steps 200
```

> Run the 8-image/200-step smoke first — it confirms the shell swap + FM
> plumbing run end-to-end before committing to the 2k-step overfit.

### Result (2026-05-26, 64 imgs × 2000 steps @ 1024², bs=1, Anima 1.0) — WEAK

`results/20260526-1219-shell-feasibility/`. FM loss **1.425 → 1.052 (−26%)** but
**hard-plateaued at ~1.05 from step ~850 onward**; the step-2000 Euler montage is
monochrome noise-blobs — coarse luminance structure, **no content, no color**
(zero-init `proj_out` starts gray and never recovers chroma). ~1.05 is essentially
the `‖ε−x0‖²` "predict-pure-noise" floor, so shells-only learned only the
low-frequency mean velocity field and cannot reconstruct image content through the
frozen core.

**Verdict: WEAK** (loss moved → core is *reachable*; output unusable → shells-only
is *insufficient*). This is the minimal, deliberately-stiffened config (pure token
decoder, **0** unfrozen blocks, no AdaLN), so the noise output is the expected
shells-only ceiling — it does **not** falsify L2P, it confirms that for Anima's
2048/28 at single-GPU budget the **last-n block tuning + real DiP detailer (input
skips) are load-bearing, not optional**. → Phase 1, but enter it expecting those
two to be the difference between this plateau and a working transfer.

**Fit note:** 1024²/bs=1 lands ~32 MB under 16 GB. bf16 is already on; block-swap
trims weight (not the activation) memory and has the extra-forward desync bug
(`project_blockswap_extra_forwards_gradcache`); `compile_blocks()` is on but only
marginally helps. **bs=1 is the load-bearing knob.** Unfreezing blocks in Phase 1
adds grad+optim memory on top of this marginal fit → will need gradient
checkpointing (the activation-axis lever this probe sidestepped).

### Cheapest next probe — does unfreezing the shallow blocks break the plateau?

Before building the real DiP detailer, adapt the shallow blocks at **both I/O
ends** (paper §3.3: *"the first **and** last n blocks"*, Fig-9b shallow tuning —
the input-side blocks matter most here, since the RGB modality swap is at the
*input*). Prefer the **LoRA** path — it leaves the activation graph identical to
shells-only, so it still fits bs=1 with no gradient checkpointing:

```bash
# n per-end: --lora_blocks 2 ≈ paper's 5-layer shallow default on 28 blocks; 3 brackets it
python bench/l2p/probe_shell_feasibility.py --num_images 64 --steps 2000 --lora_blocks 2
python bench/l2p/probe_shell_feasibility.py --num_images 64 --steps 2000 --lora_blocks 3
```

If this drops loss well below 1.0 and yields *structured/colored* montages, it
pinpoints shallow-block tuning as the lever (and Phase 1 is worth it). If the
plateau holds even with the both-ends blocks adapted, the pure-token decoder is
the bottleneck → the DiP input-skip U-Net is mandatory, test that next.
`--train_blocks N` full-tunes the same first-N ∪ last-N set instead (adds
grad+optimizer memory → auto-enables gradient checkpointing); LoRA is the
lower-risk first cut. **Caveat:** the prior result tuned the *last* blocks only
in spirit — re-running last-only would test the wrong end, so the symmetric set
is the faithful next probe.

### Result (2026-05-26, `lora_blocks=2(r32)`, 64 imgs @ 1024²) — PLATEAU HELD

13.41M trainable, both-ends shallow LoRA. FM loss tracked **shells-only almost
exactly**: ma50 ~1.05–1.06 by step ~850, flat through 1050 — the same `‖ε−x0‖²`
noise floor (shells-only: 1.052). Adapting the first-n ∪ last-n blocks bought
**~nothing**. This is the plan's "plateau holds" branch: it falsifies the
shallow-block-capacity hypothesis and leaves the **pure-token decoder as the
bottleneck**. `lora_blocks=3` skipped — when n=2 lands *on* the floor the issue
isn't block count, so bracketing it can't move the verdict.

### Next — DiP input-skip decoder (wired 2026-05-26, `--dip_skip`)

`L2PDiPDecoder` is now in the probe: a U-Net encoder runs on the noisy input
(down `3→64→128→256→512` to the 64² token grid), the frozen core's output tokens
are fused at the bottleneck (`(512+hidden)→512`, paper's `512+3840`), and a
skip-connected decoder (`512→256→128→64→64→3`) upsamples back. ~19M params,
zero-init `proj_out` (first forward = zero velocity, same safe start as the pure
decoder). The skip gives pixels a path that **bypasses the frozen core** — the
suspected lever for the plateau. The full-res head activations **fit without
gradient checkpointing**: `--lora_blocks 2 --dip_skip` lands ~15.3 GB at
1024²/bs=1 (tight but clears 16 GB), so it stays in the eager eval() path —
checkpointing stays reserved for `--train_blocks`.

```bash
python bench/l2p/probe_shell_feasibility.py --num_images 64 --steps 2000 --dip_skip
# + --lora_blocks 2 to stack the (now-known-insufficient) shallow blocks on top
```

> **Note (2026-05-26):** `--flow_shift` (default **3**) is now wired and on by
> default — the reference shifts the FM schedule already at 1K (paper §3.4: pixel
> space under-corrupts without it), which our base stack never did. **All results
> above predate it** (unshifted, effective shift=1). Re-run the DiP-skip probe with
> the default shift before re-reading the plateau verdict; pass `--flow_shift 1` to
> reproduce the old unshifted runs. See `further_wiring.md` §1.

**Read the montage, not just the loss.** The input-skip lets the head trivially
copy the near-clean input at low σ, so FM loss will *overstate* the win (the §3.4
under-corruption cheat — benign at 1K). The gate is structured/colored montages:
- loss breaks well below ~1.0 **and** montages show content/color → the decoder
  was the bottleneck → Phase 1 is worth building → write the real detailer +
  self-gen corpus.
- plateau *still* holds with the input-skip → the latent→pixel gap is too wide
  for cheap transfer at Anima's 2048/28 single-GPU budget → **FAIL**, shelve with
  the negative result recorded.

---

## Phase 1 — minimal viable transfer  [conditional on Phase 0 PASS/WEAK]

Promote the probe toward the real recipe and ask: can it reach *striking
distance* of source-Anima quality at 1K?

1. **Real DiP Detailer Head** with noisy-input skip connections + frozen-block
   intermediate-feature fusion at the bottleneck. Paper Table 3 ladder, adapted
   to Anima's 2048 width: down-channels `3→64→128→256→512`, bottleneck
   `(512 + 2048) → 512` (paper's is `512+3840`), up-channels `512→256→128→64→64`,
   output `64→3`; symmetric down/up sampling `16→8→4→2→1` / `1→2→4→8→16`.
2. **Shallow tuning** — unfreeze the input proj + the **first n ∪ last n blocks**
   + detailer (paper §3.3 "first and last n blocks"; Fig-9b: 5-layer shallow beats
   10-layer and full-layer — full-layer *degrades*). Sweep `--lora_blocks {0,2,3}`
   (n per-end; LoRA keeps bs=1) to confirm the ordering holds on Anima.
3. **Self-generated corpus** — ~1–2k images sampled from source Anima (data
   saturates by ~20k in the paper; on our budget probe the 1–2k knee). Reuse the
   teacher-synthetic pool machinery (`make distill-prep`, the mod-guidance
   synth path).
4. **Gate:** CMMD (the live val signal — `project_cmmd_val_signal`) + DPG-lite /
   eyeball within ~10% of source-Anima at 1K. If yes → the transfer is viable;
   write `scripts/distill_l2p.py` + `configs/methods/l2p.toml`. If it stalls far
   below source → the small-scale / single-GPU budget can't buy the transfer;
   shelve with the negative result recorded.

---

## Phase 2 — native 4K  [the actual payoff; conditional on Phase 1 viable]

The capability unlock, not a 1K speedup (at 1K you still run the same DiT; the
only 1K win is dropping VAE decode).

- **Patch-size scaling** 16²→64² so 4K stays at 4096 tokens (flat transformer
  cost). Wire a per-resolution patch into the input shell + decoder upsample
  factor; the DiT block stack is untouched (token count constant).
- **Noise-shift increase** — 4K pixels are densely locally correlated; the
  standard schedule under-corrupts and the model degenerates to trivial local
  reconstruction (paper §3.4). Skew `discrete_flow_shift` toward higher noise.
  Paper Fig-12 sweeps the shift parameter 1→5: FID minimizes at **≈4** (slight
  over-corruption degrade at 5) — start the Anima sweep bracketed around there.
- **Gate:** native 4K visibly crisper than 1K-bicubic-upsample (paper Fig 8),
  and the flat-cost claim holds (single-step latency ≈ 1K, vs the latent path's
  quadratic blowup). Measure peak VRAM + step latency vs a latent-4K baseline.

---

## Phase 3 — full method (Tier 2)

`scripts/distill_l2p.py` + `configs/methods/l2p.toml` (bespoke sectioned schema
like turbo/spd — **don't** `print-config`), `make exp-l2p` / `exp-test-l2p`,
`docs/experimental/l2p.md`, standard `bench/l2p/results/` envelope. Note the
output is a **separate pixel-space model artifact**, not a foldable adapter — it
abandons the VAE and reshapes the I/O, so it can't merge into the latent DiT.

---

## Open question / bonus — LoRA reuse on the pixel model

Because the DiT *core* is frozen and bit-unchanged by L2P, existing identity /
style LoRAs that sit on intermediate blocks might apply to an L2P-pixel Anima
unchanged. Early-block adapters that depend on latent-space input statistics
probably won't transfer (the input shell changed modality), but mid-stack ones
might. Cheap to probe once a Phase-1 model exists: attach a known-good LoRA,
sample, eyeball whether identity survives. High-value if it holds — it would mean
the whole adapter library carries over to pixel-space 4K for free.
