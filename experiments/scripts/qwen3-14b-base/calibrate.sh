#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

SFT_CHECKPOINT="${1:-${ROOT}/checkpoints/qwen3-14b-base/sft/global_step_80/huggingface}"
SFT_CHECKPOINT="${SFT_CHECKPOINT%/}"
STEP_DIR="${SFT_CHECKPOINT%/huggingface}"
STEP=global_step_80
if [[ "${STEP_DIR##*/}" == global_step_* ]]; then
    STEP="${STEP_DIR##*/}"
fi

space-calibrate \
    --pre "${MODEL_PATH:-Qwen/Qwen3-14B-Base}" \
    --post "$SFT_CHECKPOINT" \
    --output "${2:-${ROOT}/checkpoints/qwen3-14b-base/space/${STEP}}" \
    --rho 0.5 --alpha 1.0
