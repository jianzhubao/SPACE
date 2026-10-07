"""Run lm-eval with SPACE model adapters registered."""

from space.eval.lm_eval_models import vllm_seeded
from lm_eval.__main__ import cli_evaluate


if __name__ == "__main__":
    cli_evaluate()
