#!/usr/bin/env bash
set -euo pipefail
# Arguments: checkpoint, output directory, optional lm-eval flags.
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh" "$1" "$2"
# NLTK's import check requires a working directory outside the virtualenv parent.
cd "${ROOT}/scripts"
python -P -m space.eval.lm_eval_models.cli run \
    --model vllm_seeded --model_args "${MODEL_ARGS}" \
    --include_path "${CUSTOM_TASKS_PATH}/ifeval" \
    --tasks ifeval --batch_size auto \
    --seed 0,1234,1234,1234 --apply_chat_template --log_samples \
    --gen_kwargs "max_gen_toks=${MAX_GEN_TOKS},do_sample=True,temperature=0.6,top_p=0.95" 'until=["<|im_end|>"]' \
    --output_path "${OUTPUT_DIR}/ifeval_metrics.json" \
    "${@:3}" 2>&1 | tee "${OUTPUT_DIR}/eval.log"
