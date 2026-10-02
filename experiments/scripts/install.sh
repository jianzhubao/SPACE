#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
git -C "${ROOT}/.." submodule update --init --recursive
uv sync --locked
uv pip install --python .venv/bin/python --target .venv/eval-overrides --no-deps antlr4-python3-runtime==4.11.1
# IFEval's upstream instructions use these NLTK resources.
cd "${ROOT}/scripts"
"${ROOT}/.venv/bin/python" -P -m nltk.downloader punkt punkt_tab averaged_perceptron_tagger averaged_perceptron_tagger_eng
printf 'Experiment environment ready. Run the stage scripts in scripts/qwen2.5-7b/ or scripts/qwen3-14b-base/.\n'
