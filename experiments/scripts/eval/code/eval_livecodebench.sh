#!/usr/bin/env bash
set -euo pipefail
# Arguments: checkpoint, output directory, optional generation flags.
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh" "$1" "$2"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH="${ROOT}/livecodebench${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}/livecodebench"

python -m space.eval.livecodebench generate \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --output_dir "${OUTPUT_DIR}" --run_id eval \
    --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
    --data_parallel_size "${DATA_PARALLEL_SIZE}" \
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
    --max_gen_toks "${MAX_GEN_TOKS}" --max_model_len "${MAX_MODEL_LEN}" \
    --n 3 --temperature 0.6 --top_p 0.95 --seed 1234 \
    "${@:3}" 2>&1 | tee "${OUTPUT_DIR}/eval.log"

python -m space.eval.livecodebench score \
    --samples "${OUTPUT_DIR}/livecodebench_v2_generations_eval.json" \
    --output_dir "${OUTPUT_DIR}" --run_id eval \
    --num_process_evaluate 16 --timeout 6 \
    2>&1 | tee -a "${OUTPUT_DIR}/eval.log"
