#!/usr/bin/env bash
set -euo pipefail
# Arguments: checkpoint, output directory, MMLU task, optional lm-eval flags.
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh" "$1" "$2"
TASK_NAME="${3:?Pass mmlu_pro or mmlu_pro_1k}"
python -m space.eval.lm_eval_models.cli run \
    --model vllm_seeded --model_args "${MODEL_ARGS}" \
    --include_path "${CUSTOM_TASKS_PATH}" \
    --tasks "${TASK_NAME}" --batch_size auto \
    --seed 0 --apply_chat_template --log_samples \
    --gen_kwargs "max_gen_toks=${MAX_GEN_TOKS}" \
    --output_path "${OUTPUT_DIR}/mmlu_pro_metrics.json" \
    "${@:4}" 2>&1 | tee "${OUTPUT_DIR}/eval.log"
