# L2P — further wiring backlog (deltas from the reference `T2I-L2P` repo)

Captured 2026-05-26 after reading the authors' released code (`T2I-L2P/`:
`L2P_convert_weight.py`, `diffsynth/models/z_image_dit_L2P.py`,
`diffsynth/diffusion/{training_module,loss,flow_match}.py`, `train_run*.sh`)
against our Phase-0 probe (`probe_shell_feasibility.py`) + `plan.md`.

This is the "wire it later" list — concrete divergences worth porting, ordered
by how likely they are to move the Phase-0 plateau. Nothing here is wired yet.

## Decision already settled — LoRA shallow-tuning, NOT full-DiT

The released recipe (`train_run.sh: --trainable_models "dit"`) **full-tunes the
entire DiT** — all blocks + both Z-Image refiners + `cap_embedder` + `t_embedder`
+ pad tokens + `x_embedder` + `local_decoder`. This contradicts the *paper*,
which freezes the core and trains only the input proj + first-n/last-n blocks +
Detailer Head (§3.3; Fig-9b: 5-layer shallow > full).

**We deliberately keep the paper's frozen-core + shallow path (our `--lora_blocks`
surface), not the released full-tune.** It's the single-GPU-affordable recipe and
the paper's own ablation says it should win. Full-DiT is **out of scope** — do not
re-introduce it. (Note for the record: this means our Phase-0 plateau was measured
on the path the *paper* endorses but the *release* doesn't use; the items below are
the faithful-port fixes that could still break that plateau within the LoRA path.)

## 1. Noise-shift at 1K  [✅ WIRED 2026-05-26 — `--flow_shift`, default 3]

The reference applies a shifted FM schedule **already at 1K**, not just at 4K:
`set_timesteps_z_image` reparameterizes `σ ← shift·σ / (1 + (shift−1)·σ)` with
**`shift=3`**.

**Our whole stack is unshifted — we do *not* use shift=3.** `configs/base.toml`
sets `timestep_sampling = "sigmoid"` + `discrete_flow_shift = 1.0`, and in the
sigmoid branch (`library/runtime/noise.py:102`) the shift value isn't even applied
— it's plain `σ = sigmoid(sigmoid_scale·randn())` (the reparam only runs in the
`"shift"`/`"flux_shift"` branches, `noise.py:114`). `discrete_flow_shift=1.0` is an
identity anyway, so it's dead config under our default sampler. The probe
(`sigma_b = sigmoid(randn())`, line 505) matches base exactly → effective shift=1.

Paper §3.4 (line ~460) is explicit that pixel space *under-corrupts* →
"degenerates into trivial local reconstruction" unless the noise shift is raised.
Our `plan.md` defers shift to Phase 2 (4K) — **pull it forward to the probe.**

**Wired:** `--flow_shift` (default 3) applies `(σ·shift)/(1+(shift−1)σ)` to both
the training σ draw and the Euler-sample schedule (`_apply_flow_shift`, kept
aligned so the sampler doesn't walk a mismatched trajectory). `1.0` reproduces the
old unshifted probe. **Re-run `--lora_blocks 2 --dip_skip` (shift=3 now on by
default) before any further verdict** — the prior WEAK/plateau results predate it.

### Side note — their `max_timestep_boundary=0.8` is a framework default, NOT L2P

`FlowMatchSFTLoss` caps the training timestep at t ∈ [0, 0.8] via
`inputs.get("max_timestep_boundary", 0.8)`, and `train_L2P.py` never sets it → it
falls to the generic DiffSynth-Studio SFT default (the same loss is shared across
Flux/Qwen/Wan/Z-Image). So it's inherited boilerplate, not a paper result — treat
as **low confidence**. Mechanically it drops the top ~20% noise band (σ→1) where
the target `ε−x0 ≈ ε` is dominated by the unlearnable random component (noisy,
low-signal gradients), concentrating capacity on the low/mid-σ detail regime.
Optional `--max_t 0.8` to mirror it, but don't attribute it to L2P.

## 2. Detailer Head op-level fidelity  [medium — structure already matches]

Our `L2PDiPDecoder` matches their `MicroDiffusionModel` at the block-diagram level
(U-Net on noisy RGB, 4× down to the 64² grid, fuse DiT tokens at bottleneck,
skip-decode to RGB). Op-level divergences to align if we want a faithful port:

| | `MicroDiffusionModel` (theirs) | `L2PDiPDecoder` (ours) | port? |
|---|---|---|---|
| Downsample | `Conv2d(k3,p1)` + `MaxPool2d(2)` | strided `Conv2d(stride=2)` | low |
| Upsample | `Upsample(nearest)` + `Conv2d(k3,p1)` | `ConvTranspose2d(stride=2)` | low |
| Norm | none (SiLU only) | `GroupNorm` everywhere | low (ours is a stability add) |
| Bottleneck | `Conv2d(512+D, 512, **k=1**)` | `Conv2d(512+D, 512, k=3)` | low |
| **Token→grid fuse** | **`F.interpolate(c, p4_size)`** then cat | direct cat (sizes must match) | **yes — needed for Phase-2 4K** |
| `out_conv` init | plain | zero-init `proj_out` | keep ours (safe start under frozen core) |

**Wire (the one that matters):** `F.interpolate` the DiT feature map to the
bottleneck spatial size before concat. Our direct-cat only works at fixed
1024/patch16; the interpolation is what makes the decoder patch-size/resolution
agnostic, so it's a prerequisite for Phase-2 patch scaling (16²→64²). The other
rows are cosmetic — skip unless a faithful-port A/B is wanted.

## 3. Pixel-init convert + delta/merge scaffolding  [Phase 3 — not blocking]

Their inference path is a 4-step offline flow we don't mirror:
1. **convert** (`L2P_convert_weight.py`): build pixel DiT = random `x_embedder` +
   random `local_decoder` + **copied** latent backbone → save `*-pixel-init.safetensors`.
2. train a **delta** on top.
3. **merge** (`merge_weights.py`): `out = delta + pixel-init` → single-file weight.

Our Phase-3 `distill_l2p.py` sketch doesn't mention the pixel-init + two-file
delta/merge scheme. Adopt it when promoting past the probe — it's how they ship
inference weights, and it keeps the frozen backbone bit-identical to base Anima
(matters for the open question of whether the existing adapter library carries
over to the pixel model). For our LoRA path this is even simpler: the "delta" is
just the LoRA + shells, no merge of the frozen backbone needed.

## Confirmed NON-gaps (don't re-investigate)

- **FM convention** identical: `add_noise = (1-σ)x0 + σε`, `target = ε−x0`, plain
  MSE (their `training_weight` multiply is commented out, like ours). ✓
- **No refiner gap.** Z-Image has `noise_refiner`/`context_refiner` pre-blocks;
  **Anima has none** — one uniform `self.blocks` (28). Our `_shallow_block_indices`
  over `model.blocks` is the complete shallow surface. ✓
- **Input shell** `Linear(3·p·p → D)` large-patch tokenizer matches. ✓
