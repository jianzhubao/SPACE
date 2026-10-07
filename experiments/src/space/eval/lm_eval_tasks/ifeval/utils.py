"""Apply ReThink's thought cleanup before the standard IFEval checks."""

import os
import re

from lm_eval.tasks.ifeval.utils import process_results as upstream_process_results


def process_results(doc, results):
    response = results[0]
    if os.environ.get("IFEVAL_STRIP_THINK", "1") == "1":
        response = re.sub(r"<think>.*?</think>\s*", "", response, flags=re.DOTALL).strip()
        response = re.sub(r"^.*</think>", "", response, flags=re.DOTALL).lstrip()
        response = re.sub(r"<think>.*?<\|/think\|>\s*", "", response, flags=re.DOTALL).strip()
        response = re.sub(r"^.*<\|/think\|>", "", response, flags=re.DOTALL).lstrip()
    return upstream_process_results(doc, [response])
