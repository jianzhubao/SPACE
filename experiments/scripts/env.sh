#!/usr/bin/env bash
# Shared paths only; experiment settings live in each stage script.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SPACE_ENV="${SPACE_ENV:-${ROOT}/.venv}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export PATH="${SPACE_ENV}/bin:${CUDA_HOME}/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
cd "${ROOT}"
