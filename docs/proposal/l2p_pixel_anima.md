# L2P — Latent-to-Pixel transfer for Anima (native 4K, VAE-free)

**Status:** proposal / gated. Phase 0 probe shipped (`bench/l2p/`); nothing wired
into the live pipeline. **Source:** Chen et al., *L2P: Unlocking Latent Potential
for Pixel Generation*, arXiv:2605.12013.

## TL;DR

Convert the trained latent Anima DiT into a **pixel-space** DiT by discarding the
VAE, tokenizing raw RGB with large patches, **freezing the DiT core**, and
re-training only (a) a new RGB input projection, (b) the last *n* blocks, and (c)
a lightweight DiP-style "Detailer Head" that replaces the final projection. Train
on **Anima-self-generated** synthetic images — zero real data. The payoff is
**native 4K generation without the VAE-decode memory wall**, at *flat transformer
cost*: patch size scales with resolution (16² at 1K → 64² at 4K) so the DiT always
sees ~4096 tokens. This is a new **capability** (native ultra-high-res), not a 1K
speedup — at 1K you still run the same DiT and only save the VAE decode.

## Why this fits Anima specifically

1. **The constant-token trick is already Anima's core invariant.** L2P's whole
   efficiency mechanism — *keep token count fixed, scale patch size with
   resolution* — is exactly what `CONSTANT_TOKEN_BUCKETS` + `compile_blocks()`
   already do (2 graphs keyed on token count, not resolution). A 1024² image is
   `/8 VAE → /2 patch = 64² = 4096` tokens today; L2P's 1K pixel path is `/16
   patch = 64²` — identical grid and RoPE. The bucketing/flash machinery L2P needs
   is already built.
2. **Same objective, architecture-agnostic.** L2P's loss is `‖(ε−x0) − v_θ‖²` —
   the *same* rectified-flow target Anima trains on (`target = noise − latents`,
   `train.py:922`). The recipe freezes the core and touches only the I/O shells,
   so it doesn't care that Anima (2048-dim / 28-block) differs from the paper's
   source LDM (Z-Image, 3840 / 30).
3. **Slots into the existing distillation pattern.** "Freeze core, train shallow
   layers on self-synthetic data" is the same shape as `distill_spd` /
   `distill_turbo` / `distill_mod` and the teacher-synthetic pool infra. It would
   land as `scripts/distill_l2p.py` + `make exp-l2p`.

## Architecture mapping (exact module swaps)

The Anima forward (`library/anima/models.py::forward_mini_train_dit`) is:
`x_embedder` (patchify) → `prepare_embedded_sequence` (+RoPE) → `_run_blocks`
(the core) → `final_layer` (AdaLN) → `unpatchify`. L2P touches only the ends:

| Anima module | Today (latent) | L2P (pixel) |
|---|---|---|
| `x_embedder` (`PatchEmbed`) | Rearrange + `Linear(16·2·2 → 2048)` | fresh `Linear(3·16·16 → 2048)`, **trainable** |
| `_run_blocks` (28 blocks) | — | **frozen** (first *n* ∪ last *n* unfrozen — paper §3.3 "first and last n blocks"; Fig-9b shallow tuning, n per-end) |
| `final_layer` + `unpatchify` | AdaLN + `Linear → unpatch` to 16-ch latent | **DiP Detailer Head** (U-Net), bottleneck `(512+2048)→512` (paper's `512+3840`), → RGB |

RoPE is derived from the post-embed *token grid* (`pos_embedder(x_B_T_H_W_D)`),
not the patch size, so the input-shell swap is transparent to positional
encoding as long as the grid stays 64² (it does). Phase 0 stiffens the output
shell to a *pure token decoder* (no input skips) as a stricter feasibility bar;
Phase 1 builds the real DiP head.

## Gated plan

Full phase-by-phase gates live in **`bench/l2p/plan.md`**. Summary:

- **Phase 0 — shell reachability** (`bench/l2p/probe_shell_feasibility.py`): can
  the *frozen* core be reached from pixel space at Anima's scale? Overfit ≤64
  images, shells only. **The cheapest falsifier** — if loss plateaus at the noise
  floor, the paper's result doesn't transplant to our budget and we stop.
- **Phase 1 — minimal viable transfer:** real DiP detailer + last-5-block tuning
  + ~1–2k self-gen images; gate on CMMD/DPG within ~10% of source-Anima at 1K.
- **Phase 2 — native 4K:** patch 16→64, raise `discrete_flow_shift` (4K pixels
  under-corrupt otherwise, §3.4); gate on crisper-than-upsample + flat-cost.
- **Phase 3 — Tier-2 method:** `scripts/distill_l2p.py`, `configs/methods/l2p.toml`,
  `make exp-l2p`/`exp-test-l2p`, `docs/experimental/l2p.md`.

## Honest caveats

- **Not training-free, and not small.** Paper used 8 GPUs / 20k images. This is
  architectural surgery on the full DiT, well past the LoRA scale this repo runs
  on a single 5060 Ti. The self-gen data is cheap (teacher pools exist); the
  train itself is a real run. Phase 0/1 are the budget de-risking.
- **A separate model, not a drop-in.** A pixel-space Anima abandons the VAE and
  reshapes I/O. It's a parallel artifact — the latent caching / preprocess /
  inference pipeline assumes VAE latents, and L2P can't `merge_to_dit` back.
- **No 1K speed win.** The payoff is *native ultra-high-res*; at normal res the
  only benefit is dropping VAE decode.
- **Small quality tax.** L2P keeps ~93.6% of source GenEval (it wins on 4K
  detail/diversity). For an identity-fidelity pipeline that tradeoff matters.

## Open question — does the adapter library carry over?

The DiT core is frozen and bit-unchanged, so mid-stack identity/style LoRAs may
apply to an L2P-pixel Anima unchanged — which would hand the whole adapter
library pixel-space 4K for free. Early-block adapters keyed on latent input stats
probably won't transfer. Cheap to probe once a Phase-1 model exists. See the
bonus section of `bench/l2p/plan.md`.
