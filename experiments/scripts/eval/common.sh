#!/usr/bin/env bash

source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
MODEL_NAME_OR_PATH="${1:?Pass a checkpoint path or Hub model ID}"
if [[ -d "${MODEL_NAME_OR_PATH}" ]]; then
    MODEL_NAME_OR_PATH="$(realpath "${MODEL_NAME_OR_PATH}")"
fi
OUTPUT_DIR="$(realpath -m "${2:?Pass an output directory}")"
mkdir -p "${OUTPUT_DIR}"
IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-$(( ${#GPUS[@]} / TENSOR_PARALLEL_SIZE ))}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-4096}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MODEL_ARGS="pretrained=${MODEL_NAME_OR_PATH},tokenizer=${MODEL_NAME_OR_PATH},tensor_parallel_size=${TENSOR_PARALLEL_SIZE},data_parallel_size=${DATA_PARALLEL_SIZE},gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},max_model_len=${MAX_MODEL_LEN},seed=1234,enforce_eager=True"
CUSTOM_TASKS_PATH="${ROOT}/src/space/eval/lm_eval_tasks"
