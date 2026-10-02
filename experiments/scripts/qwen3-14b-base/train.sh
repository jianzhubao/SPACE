#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

# Paper recipe. Additional verl/Hydra overrides can be appended to this command.
IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES}"
torchrun --standalone --nnodes=1 --nproc_per_node="${#GPUS[@]}" \
    -m verl.trainer.sft_trainer \
    data.train_files="${ROOT}/data/math-cot/preprocessed/train.parquet" \
    data.val_files=null \
    data.messages_key=messages \
    data.custom_cls.path="${ROOT}/src/space/dataset/prompt_response_sft_dataset.py" \
    data.custom_cls.name=PromptResponseSFTDataset \
    data.train_batch_size=256 \
    data.max_length=20000 \
    data.truncation=right \
    data.use_dynamic_bsz=true \
    data.max_token_len_per_gpu=20000 \
    optim.lr=5e-5 \
    optim.lr_scheduler_type=cosine \
    optim.lr_warmup_steps_ratio=0.1 \
    engine=fsdp \
    engine.strategy=fsdp2 \
    engine.ulysses_sequence_parallel_size=1 \
    engine.model_dtype=bf16 \
    engine.dtype=bfloat16 \
    model.path="${MODEL_PATH:-Qwen/Qwen3-14B-Base}" \
    model.use_liger=true \
    model.use_remove_padding=true \
    trainer.default_local_dir="${ROOT}/checkpoints/qwen3-14b-base/sft" \
    trainer.project_name=SPACE \
    trainer.experiment_name=qwen3-14b-base-sft \
    'trainer.logger=[console]' \
    trainer.seed=42 \
    trainer.default_hdfs_dir=null \
    trainer.resume_mode=disable \
    trainer.save_freq=40 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    'checkpoint.save_contents=[hf_model]' \
    "$@"
