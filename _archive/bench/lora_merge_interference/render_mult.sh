#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PROMPTS=bench/lora_merge_interference/prompts.txt
OUT=output/tests/merge_interference
MERGED=output/ckpt/anima_artist123_merged.safetensors
BASE=(uv run python inference.py
  --dit models/diffusion_models/anima-base-v1.0.safetensors
  --text_encoder models/text_encoders/qwen_3_06b_base.safetensors
  --vae models/vae/qwen_image_vae.safetensors
  --vae_chunk_size 64 --vae_disable_cache --attn_mode flash
  --negative_prompt "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia"
  --image_size 1024 1024 --infer_steps 28 --flow_shift 3.0 --sampler euler --guidance_scale 4.0
  --from_file "$PROMPTS" --lora_weight "$MERGED")
for m in 0.5 0.33; do
  echo "=== merged @ multiplier $m ==="
  "${BASE[@]}" --lora_multiplier "$m" --save_path "$OUT/merged_m${m}"
done
echo "=== DONE ==="
