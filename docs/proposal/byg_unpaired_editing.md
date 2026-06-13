# byg_unpaired_editing — Bootstrap Your Generator: unpaired instruction editing for Anima

A *training-based* instruction-editing adapter for the Anima DiT that needs
**no paired (source, edited) data and no external reward model** — only
unpaired images plus VLM-generated (src-caption, tgt-caption, edit-instruction,
reverse-instruction) tuples. Adapted from **"Bootstrap Your Generator: Unpaired
Visual Editing with Flow Matching"** (Tewel, Atzmon, Chechik, Wolf; NVIDIA +
TAU, arXiv 2606.03911). The paper trains FLUX.1-dev (image) and Wan2.2 (video)
editors with **LoRA rank 64** — the same substrate and rank regime we already
run, which is the whole reason it's worth porting rather than admiring.

This is a different animal from everything we currently ship for editing:

- **DirectEdit** (`library/inference/editing/`) is *training-free* — inversion +
  edit-conditioning swap at inference. BYG *trains* an editing LoRA.
- **EasyControl / IP-Adapter** train *spatial/appearance conditioning* (frozen
  DiT, condition tokens), but they still expect the conditioning signal to be
  the answer — they don't learn an *instruction → edit* transformation.
- BYG learns the edit behavior itself, from unpaired data, and ships as a
  **plain LoRA** (zero inference overhead — same property as Turbo's output).

## The paper in three tricks

BYG plugs three specific holes in "unpaired flow-matching editing." Read these
as three independent levers; the ablation (paper Table 4) shows the method is a
balanced tripod, not one big idea.

1. **Bootstrapping the noisy input** (§4.1). Supervised editing trains on
   `y_t = noise(y_target)`, but unpaired data has no `y_target`. So a **frozen
   EMA copy** of the model rolls out *n* denoising steps from pure noise down to
   timestep *t*, producing a pseudo-noisy-target `ỹ_t`. The model trains on its
   own edits; the EMA copy stabilizes the loop (the trainable weights are too
   noisy step-to-step to roll out against themselves). Ablation: drop
   bootstrapping → forward process is driven by a *noised source* instead, train
   distribution mismatches inference, both edit-success and source-preservation
   degrade.

2. **Instruction following from the T2I prior** (§4.2) — essentially **DDS
   reborn**. Query the *frozen base model* with the source caption and target
   caption, and supervise only the **difference** (the edit direction)
   `v_tgt − v_src`, not `v_tgt` absolutely:

   ```
   L_dir = 1 − cos( v_fwd − v_src ,  v_tgt − v_src )       (Eq. 2)
   L_MSE = ‖v_fwd − v_tgt‖²                                 (magnitude anchor)
   L_prior = L_dir + α·L_MSE
   ```

   Matching `v_tgt` absolutely just regenerates toward the caption and drifts
   off-source; matching the *direction* isolates the text-induced velocity shift
   and leaves the rest of the image alone. Ablation: drop the directional term →
   stronger pull toward the target caption → more drift from source.

3. **Gradient routing via Straight-Through Estimation** (§4.3) — **the novel
   bit, and the part I'd lift even if we never build the rest.** The
   reverse/cycle pass needs *clean* inputs to behave like inference, but the only
   cheap differentiable estimate is the *blurry* one-step prediction
   `ŷ = ỹ_t − t·v_fwd`. So decouple "what the model sees" from "what it learns
   through" with a hybrid stop-gradient:

   ```
   ŷ_hyb = sg(ỹ_0) + (ŷ − sg(ŷ))          (Eq. 4)
   ```

   Forward value = the clean multi-step estimate `ỹ_0` (matches inference);
   gradient = flows through the one-step `ŷ` (cheap, differentiable). This is a
   general recipe for *any* loss that wants clean conditioning but needs
   gradients to reach a noisy state — see "Why this generalizes" below.

   Ablation: drop gradient routing → reverse pass conditions on noisy one-step
   preds → train/inference mismatch → source-preservation drops.

Plus two supporting losses: **cycle loss** (`L_cycle = ‖v_rev − (ε−x)‖²` —
applying the inverse instruction `c̄` to the edit must recover the source) and
**identity loss** (feed source as both input and condition with an
already-fulfilled instruction → must reconstruct exactly; this is what teaches
the model to *use* the condition at all). Ablation flags the load-bearing
warning: **drop the identity/regularization loss → the model collapses to
identity** (copies the input, "wins" source-preservation at edit-success 0.63).
That collapse mode is the thing to watch for on our backbone.

Full objective (paper Eq. 5):
`L = L_cycle + λ_prior·(L_prior^fwd + L_prior^rev) + λ_id·L_id`.
Paper hyperparams: `λ_prior=1.0, λ_id=0.2, λ_cycle=1.0`, `α` (MSE weight) tuned,
LoRA rank 64, lr 3e-4 (image) / 1e-4 (video), 10 integration steps for the
bootstrap rollout, first 200 steps identity-only for warmup, 15% (image) / 10%
(video) random identity steps as a regularizer.

## Why Anima is a clean fit

- **Same substrate.** Flow-matching DiT, LoRA rank 64. The losses are all in
  velocity space, which is exactly what `train.py` already computes.
- **Source conditioning is already solved here.** BYG conditions on the source
  by concatenating the source VAE latent along the token sequence dim
  (FLUX-Kontext style). That's **structurally what EasyControl already does**
  (`networks/methods/easycontrol.py`): two-stream block forward, target
  self-attention extended with the cond stream's K/V. We can reuse that
  conditioning plumbing wholesale and put the *training objective* on top of it
  — the hard architectural problem is mostly already paid for.
- **The VLM data pipeline is half-built.** BYG needs per-image
  `(src_caption, instruction, tgt_caption, reverse_instruction)`. We have
  **Anima Tagger** (`library/captioning/`) producing src captions, and the
  caption-index artifact (`caption_index.json`) for grouping. The
  edit-taxonomy + reverse-instruction generation (paper App. D) is the new
  piece, but it's an offline VLM pass, not training infra.
- **Video is in scope later.** The paper's video adaptation is "same objective,
  applied to Wan2.2 video latents via **Musubi Tuner**" — and our trainer is the
  same kohya/musubi lineage. Anima-video editing would be a natural v2.

## Architecture

Reuse EasyControl's two-stream conditioning; add the BYG training harness on top.

```
                 ┌─────────────────────────────────────────────────────────┐
                 │  frozen EMA copy  G_EMA  (no grad)                        │
   ε  ─────────► │  n-step rollout  t=1 → t  with instruction c             │  Bootstrapping
                 │  ⇒ pseudo-noisy-target  ỹ_t   (stop-grad)                │  (§4.1)
                 └─────────────────────────────────────────────────────────┘
                                        │ ỹ_t
                                        ▼
   source latent x ──(token-concat, EasyControl two-stream)──► G_edit (LoRA, train)
                                        │  v_fwd = G_edit(ỹ_t, t, c, x)
                                        ▼  ŷ = ỹ_t − t·v_fwd          (one-step)
                ┌───────────────────────┼───────────────────────────────────┐
                ▼                        ▼                                   ▼
        Prior loss (§4.2)        Cycle loss (§4.3)                    Identity loss
   query FROZEN base G_t2i    x_t = (1−t)x + tε                  feed (x, c̄_fulfilled, x)
   with p_src, p_tgt          ŷ_hyb = sg(ỹ_0)+(ŷ−sg(ŷ))         ⇒ must reconstruct x
   ⇒ v_src, v_tgt            v_rev = G_edit(x_t, t, c̄, ŷ_hyb)
   L_prior = L_dir+αL_MSE    L_cycle = ‖v_rev − (ε−x)‖²
```

Concrete component decisions:

- **`G_edit` = LoRA on the DiT**, same `apply_to` path as every other method.
  Source conditioning via EasyControl-style token concat — start by literally
  importing the two-stream `Block.forward` machinery so we don't re-derive the
  extended-self-attn / RoPE / `b_cond` details.
- **`G_t2i` (frozen base, for the prior)** is just the un-adapted DiT — a forward
  with the LoRA scaled to zero, or a cached reference module. Two extra frozen
  forwards per step (`v_src`, `v_tgt`).
- **`G_EMA` (frozen EMA of `G_edit`)** is the new heavy piece: an EMA copy of the
  trainable LoRA weights, used only for the bootstrap rollout. We don't currently
  carry an EMA mechanism in `train.py` — this is the main new training-infra
  surface (see Implementation step 3). The paper's video ablation (App. C.1)
  shows EMA is *optional* for video (improves source-pres, small editability
  cost); for images it's part of the default recipe.

### The 5D-latent boundary (load-bearing, do not skip)

Every velocity in the objective lives in the DiT's **5D `(B,C,T=1,H,W)`** space.
The bootstrap rollout, the one-step `ŷ`, the cycle's `x_t`, and the
stop-gradient blend in Eq. 4 all manipulate latents — and the cached `.npz`
source latent `x` is **4D `(B,C,H,W)`**. Per the repo invariant: `unsqueeze(2)`
going into the DiT, `squeeze(2)` coming out, **target dim 2 explicitly** (never
bare `squeeze()`). The `sg(ỹ_0) + (ŷ − sg(ŷ))` blend in particular must match
ndim before adding — this is the exact class of dim-2 bug that bit FreeText
repeatedly. Write a shape assertion at the blend site.

### compile-after-apply

The bootstrap rollout calls the *EMA* forward and the prior calls the *frozen
base* forward, both monkey-patched by the adapter. `compile_blocks()` must run
**after** `network.apply_to` + `load_weights` (the `build_anima` harness
ordering). If EMA is a separate module instance, it needs the same
apply→compile ordering, or run it eager.

## Mapping the four losses onto our code

| BYG term | Anima home | New work |
|---|---|---|
| `L_prior` (dir + MSE) | velocity MSE already in `train.py`; add cosine-dir term | two frozen base forwards (`v_src`,`v_tgt`); a directional-loss fn in `library/training/` loss registry |
| `L_cycle` | velocity MSE form | reverse forward with inverse instruction `c̄`; the Eq. 4 STE blend |
| `L_id` | velocity MSE on (x, c̄_fulfilled, x) | trivial once conditioning is wired; this is the anti-collapse anchor |
| bootstrap `ỹ_t` | sampler rollout exists in `library/inference/` | run it under `no_grad` against the EMA module inside the training step |

The prior's directional loss and the STE blend are the two genuinely new numeric
pieces; everything else is plumbing around existing velocity-MSE.

## Why this generalizes (the part worth keeping regardless)

The gradient-routing STE — `forward value = clean multi-step estimate, gradient
= cheap one-step pred` — is **not specific to editing**. It's a general fix for
the flow-matching train/inference mismatch that shows up anywhere we want a loss
on *clean* `x_0`-space while keeping gradients connected to the noisy state. We
have several such reverse/cycle-flavored objectives floating around (DirectEdit's
inversion loop, any future self-distillation). I'd prototype `ŷ_hyb = sg(ỹ_0) +
(ŷ − sg(ŷ))` as a standalone utility in `library/training/` first, validate it on
a toy reconstruction, and only then wire the full BYG objective. If BYG-the-method
stalls, this utility is still a keeper.

## Data construction

**No edited image, ever. No pair.** This is the part that's easy to misread, so
state it plainly: the training datum is **one image + one caption**. There is no
second (edited) image collected, generated, or stored anywhere in BYG — the
"target" is a *caption* (text the frozen base already understands), and the
supervision is the velocity *difference* the base predicts for src-caption vs
tgt-caption. That is the entire reason the method exists; if we ever find
ourselves producing edited images, we've left the method.

The five fields are **not hand-prepared and not a pair** — a single VLM call per
image emits all of them at once (paper App. D.1):

```
image + caption  ──one VLM call──►  {edit_type, src_caption, instruction,
                                      tgt_caption, reverse_instruction}
```

So the *only* residual cost over "I have captioned images" is **one offline VLM
annotation pass over the corpus**. Bounded, one-time, automatic — but not free:
a 30B-class VLM (paper uses Qwen3-VL-30B) over `post_image_dataset/` on a single
5060 Ti is real wall-clock, and that's the honest cost the pitch glosses. It is
*dramatically* cheaper than collecting before/after pairs (which don't exist),
but it isn't zero.

Each field is load-bearing — you can't drop any to save cost: `src_caption` +
`tgt_caption` drive the prior's `v_src`/`v_tgt`; `instruction` is the forward
conditioning; `reverse_instruction` is the cycle loss. Drop one and a loss term
dies.

### Two ways to generate the tuples, cheap-first

1. **Tag-swap synthesis (no VLM)** — the repo-specific shortcut the paper can't
   take. We already build `caption_index.json` (typed tags + groups) and have
   Anima Tagger. For *taggable* edit types — color, attribute, add/remove of a
   tagged concept — a tuple is mechanical and free:

   ```
   src_caption = full tags
   instruction = "change {red} to {blue}"   reverse_instruction = "change {blue} to {red}"
   tgt_caption = tags with red→blue
   edit_type   = color
   ```

   This costs nothing beyond tagging we already do, and respects the field
   constraints by construction (reverse is self-contained, tgt has no temporal
   phrasing). **Catch:** tag-swap tuples are low-diversity and — critically —
   **cannot express style edits** (`make it a watercolor`), which is exactly the
   category where the paper most beats supervised baselines. So tag-swap alone
   amputates the method's strongest use case.

2. **VLM pass (paper-faithful)** — needed for free-form and stylistic
   instructions. Select *exactly one* edit type from the taxonomy (color /
   material / shape / add / remove / replace / background / style / action-pose /
   text), emit JSON. Constraints that matter: `reverse_instruction` self-contained
   (no "restore"/"undo"); `tgt_caption` describes only the final image, no
   temporal/negative phrasing; stylized sources say so in `src_caption`.

   The cheap split: **tag-swap the taggable categories, VLM only the long tail
   (style, complex replace, scene rewrites)**. That shrinks the VLM pass to the
   minority of images/categories instead of the whole corpus.

**Source pool** — `post_image_dataset/` is our standing training corpus (the
project's primary dataset, not just an inversion set). No paired targets needed.

New script: `scripts/byg/build_edit_tuples.py` — emits a JSONL sidecar per image
via tag-swap by default, with a `--vlm` flag (and per-category routing) for the
style tail, mirroring `scripts/anima_tagger/cli.py`. The VLM pass — *if* enabled
on the full corpus — is the single biggest *non-training* lift; the tag-swap
default makes a first run essentially free.

## Cost & single-GPU feasibility

Per training step BYG does: 1 bootstrap rollout (n≈10 EMA forwards, no grad) +
1 trainable forward (fwd) + 1 trainable forward (rev) + 2 frozen base forwards
(prior, ×2 if symmetric) + 1 identity forward. Paper: ~3× supervised per-step
*time* (2.9s vs 0.97s on H100), but **converges fast — meaningful edits by ~1000
steps**, and **inference has zero overhead** (plain LoRA).

**Separate the two cost axes — they answer differently:**

### Per-step time: ~3× EasyControl

Driven by the `no_grad` work: the 10-step bootstrap rollout (10 EMA forwards) +
2 frozen-base prior forwards. These cost wall-clock, scale with rollout length,
and are the reason a step is ~3× slower. Shortening the rollout (fewer
integration steps) is the obvious time lever, at some pseudo-target quality cost
— a natural ablation.

### Peak VRAM: only ~+1–1.5 GB over EasyControl

The key fact: **EasyControl already forces gradient checkpointing** (its
two-stream doubles the token sequence; full activations don't fit without it).
Once grad_ckpt is on, peak is dominated by **stored checkpoint-boundary latents**
(one input per block), *not* the O(blocks × attention) activation term — that's
already been moved to sequential recompute in backward.

So the BYG-over-EasyControl marginal peak is small and computable. With Anima's
DiT (**28 blocks, dim 2048**) and EasyControl's two-stream boundary (target
~4200 + cond ~4200 ≈ 8400 tokens), one forward's boundary storage is:

```
28 blocks × 8400 tok × 2048 dim × 2 B (bf16)  ≈  0.96 GB  ≈ ~1 GB
```

| Source of marginal peak | Δ vs EasyControl |
|---|---|
| Reverse-pass boundary latents — coexist with forward (STE couples cycle grad back through `v_fwd`, so the forward graph can't free before the reverse backward) | **~+1 GB** |
| EMA copy (LoRA params only, ~44M, fp32) | ~+0.2 GB |
| Bootstrap rollout (10×) + prior forwards (2×) — `no_grad`, no boundaries stored | **~0** |
| Per-block recompute working set during backward — sequential, already in EC's budget | **~0** |
| **Total marginal** | **~+1.2 GB** |

Why so cheap: grad_ckpt already absorbed the expensive activation term. The STE
coupling forces forward + reverse to coexist, but under grad_ckpt that
coexistence costs only the *linear* boundary-latent term (~1 GB), and the
recompute working set is sequential so it never doubles.

**The knob that controls +1.2 vs +2.2 GB:** the identity loss is on an
*independent* graph. Backward it **separately** (`L_id.backward()`, then free)
and it adds nothing to peak. Sum it into one `.backward()` with the rest and you
pay another ~1 GB of boundary storage. So "stage the identity backward" is a
load-bearing implementation detail, not a nicety (see Implementation step 4).

**Bottom line:** if EasyControl currently fits with ≥ ~1.5 GB headroom at a given
resolution/token budget, BYG fits at the *same* budget with the *same* grad_ckpt
already on — it just runs ~3× slower per step. Mitigations in blessed order if
headroom is tight: **`compile_blocks()` first** (repo-default memory win) →
`custom_down_autograd` (the documented lever for the retained LoRA-input fp32
casts, which BYG doubles across its two grad forwards — what let SPD run full-res
on 16GB) → `blocks_to_swap` **with the offloader caveat below**.

> **Block-swap caveat.** The offloader desyncs on a *second* DiT forward
> (`project_blockswap_extra_forwards_gradcache`); BYG does *many* per step (10
> rollout + 2 prior + fwd + rev + id). Block-swap here must be audited before
> trusting it, or the `no_grad` EMA/prior forwards run on a non-swapped path.

These numbers are back-of-envelope (the real figure needs a measurement once
`networks/methods/byg.py` exists), but the structure is firm: BYG is a *time*
cost, not a *memory* cost, relative to EasyControl.

## Implementation steps

1. **Reuse EasyControl conditioning.** Factor the two-stream source-latent-concat
   forward out of `networks/methods/easycontrol.py` (or subclass it) so a new
   `networks/methods/byg.py` gets source conditioning for free. The novelty is
   the objective, not the conditioning.
2. **STE utility.** `library/training/` helper `ste_clean_blend(y0_clean,
   y_onestep)` → `sg(y0_clean) + (y_onestep − sg(y_onestep))`, with a dim-2
   ndim-match assertion. Unit-test it (gradient flows through `y_onestep`,
   value equals `y0_clean`). This is also the standalone keeper.
3. **EMA mechanism in `train.py`.** New training-infra surface: maintain an EMA
   copy of the trainable LoRA params, expose `byg_ema_decay`. Gate behind the
   method so no other method pays for it. (Cross-check the block-swap
   interaction — per `project_blockswap_extra_forwards_gradcache`, the offloader
   desyncs on a *second* DiT forward; BYG does **many** extra forwards per step.
   This is a real risk: the bootstrap rollout alone is n forwards. Audit the
   offloader path before trusting block-swap here, or run the EMA/base forwards
   on a non-swapped copy.)
4. **Loss terms + backward staging.** Directional cosine loss + symmetric prior
   in the loss registry; cycle loss with the STE blend; identity loss. Wire
   `λ_prior`, `λ_id`, `λ_cycle`, `α` as `network_args` / config knobs (register in
   the TOML allowlist per `project_network_kwarg_toml_allowlist`, or they'll be
   inert and fail the config test). **Backward the identity loss separately**
   (`(λ_id·L_id).backward()`, free, *then* the coupled forward+reverse+prior
   block) — the identity graph is independent, and separating it keeps peak VRAM
   at one extra forward's boundary storage (+~1.2 GB) instead of two (+~2.2 GB).
   See the Cost section. (`retain_graph` is not needed across the two backwards
   since they share no graph; just don't sum the losses before calling backward.)
5. **Training schedule.** First 200 steps identity-only; 15% random identity
   steps thereafter. Both are anti-collapse regularizers — the ablation says the
   model collapses to identity without the regularization signal, so treat these
   as load-bearing, not cosmetic.
6. **Config.** `configs/methods/byg.toml` (frozen DiT — force `blocks_to_swap`
   per its own needs, method-wins-on-overlap) + a clean `configs/gui-methods/byg.toml`.
   `make exp-byg` target in `scripts/experimental_tasks/`.
7. **Data script.** `scripts/byg/build_edit_tuples.py` (offline VLM → JSONL).
8. **Inference / test target.** Output is a plain editing LoRA conditioned on a
   source image + instruction. `make exp-test-byg REF_IMAGE=... PROMPT=...`
   mirroring `test-easycontrol`.
9. **Bench.** `bench/byg/` with the standard `_common.py` envelope (Tier-2
   requirement). Metrics below.

## ComfyUI workflow

At inference BYG is **structurally the EasyControl node** — frozen DiT + LoRA +
source-latent concat, output is a plain LoRA, zero sampler-level novelty. The
workflow reads like a Kontext edit:

```
UNETLoader (Anima DiT) ──► LoraLoader (BYG editing LoRA) ──┐
                                                           ▼
LoadImage (source) ──► VAEEncode ──► source latent ──► BYGEditCond (MODEL+VAE+IMAGE → MODEL')
                                                           │
CLIPTextEncode("change bg to a forest") ── positive ──┐    │
CLIPTextEncode("") ── negative ───────────────────────┼────┤
                                                       ▼    ▼
                                              KSampler (MODEL', pos, neg, EMPTY latent)
                                                       │
                                                       ▼
                                              VAEDecode ──► edited image
```

Same `MODEL+VAE+IMAGE→MODEL` shape as `~/ComfyUI-EasyControl-KSamplerCompat/`
(`project_easycontrol_ksampler_node`), same **empty-latent + KSampler** flow
(source is baked into the MODEL by the cond node, not fed as the init latent),
same **KV-cache-once** optimization (source conditioning is denoise-step- and
CFG-branch-invariant — source fixed, LoRA frozen).

Three differences from the EasyControl node:

1. **The text prompt *is* the edit instruction.** Source image = thing being
   edited; instruction drives the edit. Same plumbing, Kontext-style semantics.
2. **BYG is simpler — no `b_cond` gate, no cond-LoRA branch** (Kontext-native
   form). The cond node is a *thinner* EasyControl node: concat + KV cache, drop
   the gate and cond-LoRA exposure. The behavior lives in the LoRA weights,
   loaded by the standard / `comfyui-hydralora` Adapter loader.
3. **Positional-encoding alignment.** Per paper App. B.2, source tokens get the
   *same* RoPE positions as the target tokens (spatial-counterpart attention).
   The node must match positions, not offset them — verify against EasyControl's
   `cond_rope` handling.

Because the output is a plain LoRA, it **composes with the rest of the stack**:
Spectrum acceleration and DCW sampler correction operate at the sampler/block
level, orthogonal to a LoRA + concat, so a "BYG edit + Spectrum" workflow is
just the graph above with a Spectrum KSampler swapped in.

**This is the open architectural fork** (the one real design choice in the
proposal): the node's conditioning must mirror *training's* conditioning.
- *Reuse EasyControl two-stream plumbing* (Implementation step 1) → fastest to
  build, but carries a gate + cond-LoRA BYG doesn't need; the node is the
  EasyControl node minus those.
- *Go Kontext-native single-sequence concat* → closer to the paper, thinner node
  both sides, but new conditioning plumbing.

Either way the node is "EasyControl-alike." Decide this before writing
`networks/methods/byg.py`, since it fixes both the training forward and the node.

## Validation plan

| Question | How to answer |
|---|---|
| Does it edit at all (no collapse)? | Edit-success via VIEScore-style LLM judge (we have the Anima Tagger / a VLM) on a held-out instruction set. Watch for the identity-collapse mode: high source-pres + near-zero edit-success. |
| Source preserved? | DINO / DreamSim similarity to source on unedited regions. |
| Is the STE routing actually helping? | Ablate gradient routing (condition rev pass on noisy one-step pred). Should degrade source-pres — reproduce the paper's Table 4 direction on our backbone. |
| Is bootstrapping needed on Anima? | Ablate: drive forward from noised source instead. Paper says big degradation; confirm. |
| Does EMA matter for images here? | Ablate EMA (paper says it's part of image default but optional for video). |
| Object removal weakness reproduce? | Per-category breakdown; expect *remove* to be the weakest (target caption omits rather than negates — paper's acknowledged limitation). |
| Composes with our stack? | Does a BYG LoRA still work under `compile_blocks()`? Does inference compose with DCW/Spectrum (it's just a LoRA, so it should)? |

## Decisions to make

- **EMA or not for v1.** Paper default (image) uses it; video ablation shows it's
  optional. Skipping EMA halves the new training-infra surface and one frozen
  copy's memory. I lean **start without EMA** (bootstrap against a periodically-
  snapshotted copy of the trainable weights) to de-risk infra, then add EMA as an
  ablation if source-pres is weak. Counter-argument: the paper's EMA is what
  *stabilizes* the bootstrap loop, so skipping it may make v1 look worse than the
  method deserves. Genuinely a judgment call — flag for review.
- **Conditioning plumbing: reuse EasyControl two-stream vs Kontext-native concat.**
  The biggest architectural choice — it fixes both the training forward *and* the
  ComfyUI node (see "ComfyUI workflow"). EasyControl reuse is fastest to build but
  carries a `b_cond` gate + cond-LoRA BYG doesn't need; Kontext-native single-
  sequence concat is closer to the paper and thinner both sides but new plumbing.
  Decide before writing `networks/methods/byg.py`.
- **Conditioning: token-concat vs channel-concat.** Paper notes both work for
  transformers; token-concat is standard and matches EasyControl. Default
  token-concat. (Sub-decision under the fork above.)
- **Bootstrap rollout length.** Paper uses 10 integration steps. Shorter = cheaper
  but lower-quality pseudo-targets. Sweep on the single-GPU budget.
- **Prior symmetry.** Apply `L_prior` to both fwd and rev (paper) or fwd only
  (cheaper, one fewer frozen forward). Start symmetric, ablate.
- **VLM for tuple generation.** Anima Tagger (in-house, cheap, narrow vocab) vs a
  general VLM (Qwen-VL class, richer instructions). The paper's whole supervision
  signal rides on caption quality — this is worth an A/B before committing.
- **Source pool.** Whole `post_image_dataset/` vs a style-diverse curated subset.
  Long-tail style edits are where the paper most beats supervised baselines, so a
  diverse pool plays to the method's strength.

## Risks / honest caveats

- **It's a balanced tripod.** The ablation shows removing any single component
  degrades something, and removing regularization *collapses to identity*. This
  is not a method where you can ship two of the three tricks and get two-thirds
  of the result — budget for getting all four losses balanced, which is the real
  cost.
- **Many extra forwards per step** collides directly with our block-swap
  offloader desync warning (`project_blockswap_extra_forwards_gradcache`). This
  is the single most likely place v1 breaks subtly. Audit first.
- **Object removal is structurally weak** (target caption omits rather than
  describes absence) — the paper owns this; don't expect to beat it.
- **The directional loss is DDS** (they cite Hertz). Contribution #2 is
  integration, not new — relevant only in that we shouldn't oversell it as novel
  in any write-up.
- **Evidence is partly soft.** Win-rates use manual best-of-4 selection + LLM
  judge. The *ablation table* is the trustworthy evidence; lean on it.

## Relationship to existing methods

- **EasyControl** (`docs/experimental/easycontrol.md`) — supplies the source-
  conditioning plumbing BYG sits on. EasyControl learns *spatial conditioning*;
  BYG learns the *instruction→edit transformation* on top of that conditioning.
  Reuse, don't reinvent.
- **DirectEdit** (`docs/experimental/directedit_editing_v3.md`) — training-free
  inversion-based editing. BYG is the trained counterpart. They could even
  compose: a BYG-trained editing LoRA used *inside* DirectEdit's inversion loop.
- **IP-Adapter** (`docs/experimental/ip-adapter.md`) — parallel-KV image
  conditioning. Different conditioning topology; BYG's source-concat keeps the
  edit signal and source on the same stream so instructions can shift attention
  off the source where the edit demands it (the same argument
  `postfix_residual_for_directedit.md` makes against IP-Adapter's decoupling).
- **Turbo** (`docs/experimental/dpdmd.md`) — shares the "trains a thing, ships a
  plain LoRA, zero inference overhead" property. Good precedent that a heavy
  bespoke training method can land as a normal adapter here.
