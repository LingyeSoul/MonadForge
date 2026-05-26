# bench/l2p — Latent-to-Pixel transfer for Anima

Preconditions + probes for porting **L2P** (Chen et al., arXiv:2605.12013) onto
Anima: turn the latent DiT into a pixel-space DiT (discard VAE, large-patch RGB
tokenization, freeze the core, re-train shallow shells + a detailer head) to
unlock **native 4K** at flat transformer cost.

- **Gated plan & go/no-go history:** `plan.md`
- **Proposal / design:** `docs/proposal/l2p_pixel_anima.md`

## Phase 0 — `probe_shell_feasibility.py`

The cheapest test that can falsify the whole approach: freeze the DiT, bolt on a
fresh RGB patch-embed + a pure token→pixel decoder, overfit ≤64 images with the
exact Anima FM objective (`noisy=(1-σ)x0+σε`, `target=ε−x0`). If shells-only
drives loss down and decodes recognizable structure, the frozen core is reachable
from pixel space.

`--dit` defaults to **Anima 1.0** (`models/diffusion_models/anima-base-v1.0.safetensors`);
pass `--dit` to override.

```bash
# smoke (run first — confirms plumbing, must not NaN, loss must move)
python bench/l2p/probe_shell_feasibility.py --num_images 8 --steps 200

# real Phase-0 overfit
python bench/l2p/probe_shell_feasibility.py \
    --image_dir post_image_dataset/resized \
    --te_cache_dir post_image_dataset/lora \
    --num_images 64 --steps 2000 --resolution 1024
```

Writes the standard `results/<ts>-shell-feasibility/result.json` envelope
(verdict, loss curve, trainable param count) + `sample_*.png` montages.

**Corpus note.** The paper trains on *source-LDM-generated* images (smooth
manifold → fast convergence; real data converges slower, Fig-9a). For a Phase-0
"does it learn at all" gate, resized dataset images are a fine stand-in. For
Phase 1, switch to a self-generated Anima pool.

**Flags worth knowing:** `--lora_blocks N` / `--train_blocks N` adapt the **first
N ∪ last N** DiT blocks (N per-end; paper §3.3 "first and last n blocks" / Fig-9b
shallow tuning — n=2 ≈ the paper's 5-layer default on Anima's 28 blocks; 0 = shells
only). LoRA leaves the activation graph identical to shells-only so it still fits
bs=1; `--train_blocks` full-tunes and auto-enables gradient checkpointing. `--dip_skip`
swaps the pure-token decoder for the real **DiP detailer** (U-Net encoder on the
noisy input → skip connections give pixels a path bypassing the frozen core; the
Phase-1 output shell, ~19M params); its full-res head activations still fit at
1024²/bs=1 without checkpointing (~15.3 GB with `--lora_blocks 2`). `--patch` is the pixel patch size
(16 → 64² grid at 1K), `--lr` defaults to the paper's `5e-5` AdamW (Table 3).

```bash
# DiP input-skip — the next probe after the lora_blocks=2 plateau (see plan.md)
python bench/l2p/probe_shell_feasibility.py --num_images 64 --steps 2000 --dip_skip
```
