#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

CHECKPOINT="${1:?Pass a checkpoint path or Hub model ID}"
CHECKPOINT="${CHECKPOINT%/}"
STEP_DIR="${CHECKPOINT%/huggingface}"
if [[ "${STEP_DIR##*/}" == global_step_* ]]; then
    OUTPUT_DIR="${2:-${STEP_DIR%/*}/eval_results/${STEP_DIR##*/}}"
else
    OUTPUT_DIR="${2:-${CHECKPOINT}/eval_results/global_step_390}"
fi
export MAX_GEN_TOKS=4096
export MAX_MODEL_LEN=8192

bash scripts/eval/math/eval_ck.sh "$CHECKPOINT" "$OUTPUT_DIR/math" \
    math_oai,aime24,aime25,amc23,minerva_math,olympiadbench 16

bash scripts/eval/lm_eval/eval_ck.sh "$CHECKPOINT" "$OUTPUT_DIR/mmlu_pro" mmlu_pro
bash scripts/eval/lm_eval/eval_gpqa_diamond_rethink.sh "$CHECKPOINT" "$OUTPUT_DIR/gpqa" 16
bash scripts/eval/code/eval_livecodebench.sh "$CHECKPOINT" "$OUTPUT_DIR/livecodebench"
bash scripts/eval/lm_eval/eval_ifeval.sh "$CHECKPOINT" "$OUTPUT_DIR/ifeval"
bash scripts/eval/lm_eval/eval_alpaca_rm.sh "$CHECKPOINT" "$OUTPUT_DIR/alpaca_rm"
bash scripts/eval/lm_eval/eval_halueval.sh "$CHECKPOINT" "$OUTPUT_DIR/halueval" halueval
