# scripts/calibration

Live calibration scripts — the ones you re-run to regenerate the **shipped**
artifacts under `networks/calibration/`. Promoted here from their original
benches (`bench/{pid,cns,channel_stats}`, archived under `_archive/bench/`)
once the gains were observed and the methods shipped. Self-contained: no
`bench/` dependency (no `bench._common` / `bench._anima`); they import the
public `anima_lora` façade + `library` / `networks` directly.

The frozen probe/analysis history (gamma_probe sweeps, sigma/turbo probes,
result envelopes) stays in `_archive/bench/`.

## CNS — colored-noise sampling γ matrix

Produces `networks/calibration/cns_gamma.npz` (consumed by `--cns auto`).

```bash
python scripts/calibration/cns_calibrate.py --cfg 4.0 --n_aspects 3   # compiled
```

- `cns_calibrate.py` — Phase-1 deploy calibrator (cfg=4.0, top-N aspect buckets);
  writes the npz (per-aspect σ50 staircase summary prints to stdout).
- `gamma_probe.py` — read-only Phase-0 staircase check + the shared γ/FFT helpers
  `cns_calibrate` imports. `--out_dir` for its standalone npz/heatmaps.

Phase log / precondition / composition tensions: `_archive/bench/cns/plan.md`
(premise corroborated by `project_sigma_signal_resolves_by_045`).

Consumer + math: `library/inference/corrections/cns.py`. User doc:
`docs/inference/cns.md`.

## channel_stats — per-channel LoRA gradient rebalance (SmoothQuant-style)

Produces the calibrations the `channel_scaling_alpha > 0` LoRA path absorbs:

```bash
# main stream → networks/calibration/channel_stats.safetensors
python scripts/calibration/analyze_lora_input_channels.py --per_artist \
    --dump_channel_stats networks/calibration/channel_stats.safetensors

# EasyControl cond stream → networks/calibration/cond_channel_stats.safetensors
python scripts/calibration/cond_stream_profile.py --per_artist \
    --dump_cond_stats networks/calibration/cond_channel_stats.safetensors
```

- `analyze_lora_input_channels.py` — per-input-channel `mean|x|` over real samples
  × 5 sigmas → dominance report + dumpable calibration.
- `cond_stream_profile.py` — the cond-stream counterpart (reuses the collector's
  dataset/dump helpers); cond calib does **not** transfer from the main file.

The DC-bias-vs-attention-sink decomposition and the GraLoRA alternative weighed
against: `_archive/bench/channel_stats/channel_dominance_analysis.md`.

Consumer: `networks/lora_anima/factory.py` (`_CHANNEL_STATS_PATH`) and
`networks/methods/easycontrol.py`. Regime analysis: memory
`project_per_channel_scaling_audit`. User doc:
`docs/optimizations/channel_scaling.md`.

## PiD — pixel-decoder color calibration

Fits a static PiD→native-VAE color transform on Anima's own latents (the cheap
decode-time equivalent of NVIDIA's retrained `_2606` checkpoint).

```bash
uv run python scripts/calibration/fit_color_calib.py --num_images 24 --steps 4
```

- `fit_color_calib.py` — writes `pid_color_calib.safetensors` to `--out_dir`
  (default `output/calibration/pid_color_calib/`); fit summary prints to stdout.
  Fit at the **exact** decode step count. The reference RGB is produced by the
  production decoder (`load_vae` + `decode_to_pixels`, `--vae` defaults to the
  configured Qwen-Image VAE).

The PiD node ships from the standalone `ComfyUI-Anima-PiD` repo. Findings:
memory `project_pid_color_drift_calib`.
