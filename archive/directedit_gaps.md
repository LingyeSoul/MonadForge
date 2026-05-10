
## Bonus — Spectrum-integrated DirectEdit

Predicated on Gap 2's **Option B** landing. The two changes are coupled:
Option B is the prerequisite that lets Spectrum's hooking model work, and
Spectrum is the payoff that makes the larger-batch cost worthwhile.

### Why Option B unblocks Spectrum

Spectrum (`networks/spectrum.py`) installs a `register_forward_pre_hook` on
`final_layer` and assumes **one DiT call per denoising step** — that single
call's block-output trajectory is what the Chebyshev forecaster fits and
extrapolates from. Our current edit step does up to *three* model calls (src
capture + tar cond + tar uncond), and Spectrum has no defined behavior across
heterogeneous forwards on the same step (which trajectory does it cache? do
they get separate Chebyshev fits? when do cached steps "fire" relative to the
mode-toggle dance?). Option B collapses everything into a single 4-row
batched forward (`[neg_src, neg_tar, cond_src, cond_tar]`), so Spectrum sees
exactly one trajectory per step and forecasts all 4 rows simultaneously —
identical to how it already handles plain text-CFG (`anima(latents_2x, t,
[neg, pos])`).

The hook for V-injection has to become *row-indexed* under Option B (capture
from rows 0/2 = src, inject into rows 1/3 = tar inside the same forward,
instead of the current global `state.mode = CAPTURE` → `state.mode = INJECT`
toggle). That falls out naturally from author's `value[h//2:],
key[h//2:] = self.controller(...)` slicing
(`DirectEdit/controller/attn_norm_ctrl_sd35.py:362`).

### Block-skipping × K/V-injection conflict (and the t_inj < warmup fix)

On Spectrum's *cached* steps every block is skipped — only `t_embedder` +
`final_layer` + `unpatchify` run. K/V replacement lives inside each patched
`Attention.forward`, so any cached step within `[0, t_inj)` would silently
no-op the injection (no shape error, just wrong output — the same failure
mode as Gap 2's uncond leak).

The fix is alignment, not code: **constrain `t_inj < spectrum_warmup`** so
the entire injection window falls on real-forward (uncached) steps.
Spectrum's standard warmup is 7, and author's `attn_ratio = 0.05–0.30`
already places `t_inj ∈ [2, 9]` for `T = 28` — the windows naturally
overlap at the low end. Recommended Spectrum-mode default: **`t_inj = 6`**
(under warmup=7, > 0 so V-injection still contributes to identity
preservation).

This is also independent motivation to retire our current `t_inj = 24/T = 28
≈ 86%` default (already flagged out-of-scope above) — that value was
wrong-shaped vs the author and outright incompatible with Spectrum.

### Sketch

```python
# scripts/edit.py
parser.add_argument("--spectrum", action="store_true")
parser.add_argument("--spectrum_warmup", type=int, default=7)
...
if args.spectrum and args.t_inj >= args.spectrum_warmup:
    raise ValueError(
        f"t_inj={args.t_inj} must be < spectrum_warmup={args.spectrum_warmup} — "
        "K/V injection requires real (uncached) block forwards."
    )
```

```python
# library/inference/directedit.py::edit_forward (post-Option-B shape)
# One forward, 4 rows; row-indexed capture+inject inside the hook.
batch = torch.cat([neg_src, neg_tar, cond_src, cond_tar], dim=0)
v_4row = anima(batch, t_expand_4, embed_4row, padding_mask_4)
# split + per-branch CFG combine (src uses inv_cfg, tar uses recov_cfg)
```

### Memory

4-row batch × Spectrum's per-row block-feature cache ≈ ~4× standard Spectrum
VRAM footprint. Likely fine on the 5060 Ti at bench resolutions; tight on
8 GB cards (`presets.toml[low_vram]` users may need to fall back to
non-Spectrum edit). Track actual numbers in `bench/directedit_spectrum/`.

### Landing sequence

1. Gaps 1 + 2 (Option A) + 3 → `make exp-test-directedit-dry` clean.
2. Gap 2 Option B (batched forward) + lower default `t_inj` to 6.
3. `--spectrum` on `scripts/edit.py`, with the `t_inj < warmup` guard.

Steps 2 + 3 ship as one Tier-1.5 PR with `bench/directedit_spectrum/results/`
showing speedup parity with non-edit Spectrum (~3.75× per CLAUDE.md) and
no regression vs the step-1 baseline on dry mode + a real edit prompt.
