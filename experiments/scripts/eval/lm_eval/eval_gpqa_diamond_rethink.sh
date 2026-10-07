#!/usr/bin/env bash
set -euo pipefail

GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh" "$1" "$2"
TASK_NAME=gpqa_diamond_rethink
if [[ "${3:?Pass 3 or 16 repeats}" == 16 ]]; then
    TASK_NAME=gpqa_diamond_rethink_16
fi
python -m space.eval.lm_eval_models.cli run \
    --model vllm_seeded --model_args "${MODEL_ARGS}" \
    --include_path "${CUSTOM_TASKS_PATH}" \
    --tasks "${TASK_NAME}" --batch_size auto \
    --seed 0,1234,1234,1234 --apply_chat_template --log_samples \
    --gen_kwargs "max_gen_toks=${MAX_GEN_TOKS}" \
    --output_path "${OUTPUT_DIR}/gpqa_metrics.json" \
    "${@:4}" 2>&1 | tee "${OUTPUT_DIR}/eval.log"
