#!/usr/bin/env bash
# sigma-cutoff text-knockout sweep: generate the same seeded prompts with text
# knocked out (continue unconditionally) below each cutoff, plus a full-prompt
# baseline. Compares with knockout_compare.py afterward.
set -euo pipefail

ROOT=output/tests/knockout
PROMPTS=bench/cross_attn_drive/knockout_prompts.txt
CUTOFFS=(0.95 0.80 0.60 0.40)

rm -rf "$ROOT"
mkdir -p "$ROOT"

echo "=== baseline (text on throughout) ==="
unset ANIMA_TEXT_KNOCKOUT_SIGMA || true
make test ARGS="--from_file $PROMPTS --save_path $ROOT/baseline"

for c in "${CUTOFFS[@]}"; do
  echo "=== knockout cutoff sigma=$c (text off below) ==="
  ANIMA_TEXT_KNOCKOUT_SIGMA=$c \
    make test ARGS="--from_file $PROMPTS --save_path $ROOT/cut_$c"
done

echo "=== all runs done -> $ROOT ==="
