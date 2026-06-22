#!/usr/bin/env bash
# Render the 5-config x 4-prompt interference matrix.
# Configs: artist1/2/3 (own), merged (test subject), base (NOLORA reference).
# Each prompt carries a fixed seed (--d) so the same prompt is bit-identical
# noise across configs -> merged@Pk vs artistK@Pk isolates the adapter.
set -euo pipefail
cd "$(dirname "$0")/../.."

PROMPTS=bench/lora_merge_interference/prompts.txt
OUT=output/tests/merge_interference
CKPT=output/ckpt

BASE=(uv run python inference.py
  --dit models/diffusion_models/anima-base-v1.0.safetensors
  --text_encoder models/text_encoders/qwen_3_06b_base.safetensors
  --vae models/vae/qwen_image_vae.safetensors
  --vae_chunk_size 64 --vae_disable_cache
  --attn_mode flash --lora_multiplier 1.0
  --negative_prompt "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia"
  --image_size 1024 1024 --infer_steps 28 --flow_shift 3.0
  --sampler euler --guidance_scale 4.0
  --from_file "$PROMPTS")

run_cfg () {  # name  [lora_path]
  local name=$1; shift
  echo "=== rendering config: $name ==="
  local args=("${BASE[@]}" --save_path "$OUT/$name")
  if [ $# -gt 0 ]; then args+=(--lora_weight "$1"); fi
  "${args[@]}"
}

run_cfg base
run_cfg artist1 "$CKPT/anima_artist1.safetensors"
run_cfg artist2 "$CKPT/anima_artist2.safetensors"
run_cfg artist3 "$CKPT/anima_artist3.safetensors"
run_cfg merged  "$CKPT/anima_artist123_merged.safetensors"

echo "=== DONE -> $OUT/{base,artist1,artist2,artist3,merged}/ ==="