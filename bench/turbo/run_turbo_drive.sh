#!/usr/bin/env bash
# Cross-attn-drive(sigma): turbo student vs base/teacher, identical trajectory
# config (cfg4 / 28 steps / seed42 / euler / same real captions). Only the turbo
# LoRA is toggled, so any change in drive(sigma) is attributable to DP-DMD distill.
# Routed through tasks.py `test` so the DiT/TE/VAE paths get filled in.
set -euo pipefail
cd /home/sorryhyun/anima/anima_lora

PROMPTS=bench/turbo/real_prompts.txt
OUT=output/tests/turbo_drive
TURBO=output/ckpt/anima_turbo_R/anima_turbo_R_4500.safetensors
mkdir -p "$OUT"

echo "=== TEACHER (base, no LoRA) ==="
NOLORA=1 ANIMA_CFG_DELTA_LOG="$OUT/teacher.json" \
  python tasks.py test \
    --from_file "$PROMPTS" \
    --save_path "$OUT/teacher"

echo "=== STUDENT (turbo_R_4500 LoRA) ==="
NOLORA=1 ANIMA_CFG_DELTA_LOG="$OUT/student.json" \
  python tasks.py test \
    --from_file "$PROMPTS" \
    --lora_weight "$TURBO" \
    --save_path "$OUT/student"

echo "=== DONE: $OUT/{teacher,student}.json ==="
