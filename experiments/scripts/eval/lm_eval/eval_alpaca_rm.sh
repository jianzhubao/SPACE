#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../common.sh" "$1" "$2"
TASK_NAME=alpaca_rm
mkdir -p "${OUTPUT_DIR}/generations"


python -m space.eval.lm_eval_models.cli run \
    --model vllm_seeded --model_args "${MODEL_ARGS}" \
    --include_path "${CUSTOM_TASKS_PATH}" \
    --tasks "${TASK_NAME}" --batch_size auto \
    --seed 0,1234,1234,1234 --apply_chat_template --predict_only --log_samples \
    --gen_kwargs "max_gen_toks=${MAX_GEN_TOKS}" \
    --output_path "${OUTPUT_DIR}/generations/generation_metrics.json" \
    "${@:3}" 2>&1 | tee "${OUTPUT_DIR}/eval.log"

python -m space.eval.alpaca_rm \
    --samples "${OUTPUT_DIR}"/generations/samples_alpaca_rm_*.jsonl \
    --output_dir "${OUTPUT_DIR}" --run_id eval \
    2>&1 | tee -a "${OUTPUT_DIR}/eval.log"
