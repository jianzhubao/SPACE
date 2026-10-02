#!/usr/bin/env bash
# Arguments: checkpoint, output directory, comma-separated benchmarks, repeats.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../env.sh"
OUTPUT_DIR="${2:?Pass an output directory}"
mkdir -p "${OUTPUT_DIR}"

# Only this process uses ANTLR 4.11; training keeps its own ANTLR version.
PYTHONPATH="${SPACE_ENV}/eval-overrides:${ROOT}/math_evaluation/latex2sympy${PYTHONPATH:+:${PYTHONPATH}}" \
python "${ROOT}/math_evaluation/math_eval.py" \
    --model_name_or_path "${1:?Pass a checkpoint}" \
    --data_names "${3:?Pass benchmarks}" \
    --data_dir "${ROOT}/math_evaluation/data" \
    --output_dir "${OUTPUT_DIR}" \
    --prompt_type boxed-user --apply_chat_template \
    --temperature 1 --top_p 1 --seed 0 \
    --n_sampling "${4:?Pass the number of repeats}" \
    --max_tokens_per_call "${MAX_GEN_TOKS:-4096}" \
    --max_model_len "${MAX_MODEL_LEN:-8192}" \
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION:-0.8}" \
    --use_vllm "${@:5}" 2>&1 | tee "${OUTPUT_DIR}/eval.log"
