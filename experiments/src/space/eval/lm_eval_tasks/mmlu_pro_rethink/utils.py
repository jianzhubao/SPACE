"""Prompt and boxed-answer helpers for the rethink-compatible MMLU-Pro task."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any


CHOICES = tuple("ABCDEFGHIJKLMNOP")
MMLU_PRO_CHOICES = CHOICES[:10]
_LATEX_WRAPPER = re.compile(
    r"^\\(?:text|textbf|mathrm|mathbf|operatorname)\s*\{(.*)\}$", re.DOTALL
)


def doc_to_text(doc: dict[str, Any]) -> str:
    """Render the target question exactly like the rethink repository."""
    prompt = "Question:\n"
    prompt += doc["question"] + "\n"
    prompt += "Options:\n"
    for index, option in enumerate(doc["options"]):
        prompt += "{}. {}\n".format(CHOICES[index], option)
    prompt += (
        r"Please reason step by step, and put your final answer within \boxed{}. "
        "Only include the letter choice (A, B, C, D, E, F, G, H, I or J) "
        "as your final answer."
    )
    return prompt


def _boxed_contents(text: str) -> list[str]:
    """Return the contents of all complete balanced ``\\boxed`` expressions."""
    contents = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        opening_brace = text.find("{", match.start())
        depth = 0
        for position in range(opening_brace, len(text)):
            if text[position] == "{":
                depth += 1
            elif text[position] == "}":
                depth -= 1
                if depth == 0:
                    contents.append(text[opening_brace + 1 : position])
                    break
    return contents


def _unwrap_latex(text: str) -> str:
    text = text.strip().strip("$").strip()
    while match := _LATEX_WRAPPER.fullmatch(text):
        text = match.group(1).strip().strip("$").strip()
    return text.strip("*_` ")


def _extract_boxed_choice(content: str, choices: Sequence[str]) -> str:
    content = _unwrap_latex(content)
    upper = "".join(choices)
    letters = re.escape(upper + upper.lower())
    prefix = (
        r"(?:(?:the\s+)?(?:final\s+|correct\s+|right\s+|best\s+)?"
        r"(?:answer|choice|option)\s*(?:is|:|=)?\s*)?"
    )

    # A single MMLU-Pro question must resolve to exactly one option.
    if re.match(
        rf"^\s*{prefix}[\(\[]?([{letters}])[\)\]]?\s*[,/&+]\s*"
        rf"[\(\[]?([{letters}])[\)\]]?(?![A-Za-z0-9])",
        content,
        re.IGNORECASE,
    ):
        return ""

    match = re.match(
        rf"^\s*{prefix}[\(\[]?([{letters}])[\)\]]?"
        rf"(?=$|[\s.,:;\-])",
        content,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).upper()

    # Also accept one unambiguous standalone uppercase label inside the box.
    candidates = set(
        re.findall(
            rf"(?<![A-Za-z0-9])([{re.escape(upper)}])(?![A-Za-z0-9])",
            content,
        )
    )
    return candidates.pop() if len(candidates) == 1 else ""


def _last_answer_is_choice(text: str, choices: Sequence[str]) -> str:
    """Support the legacy ``answer is A`` / ``answer is (A)`` fallback."""
    upper = "".join(choices)
    letters = re.escape(upper + upper.lower())
    matches = re.findall(
        rf"\banswer\s+is\s*[*_`$]*\s*[\(\[]?([{letters}])\s*[\)\]]?"
        rf"\s*[*_`$]*(?![A-Za-z0-9])",
        text,
        re.IGNORECASE,
    )
    return matches[-1].upper() if matches else ""


def get_multiple_choice_answer(
    prediction: str | None,
) -> str:
    """Extract the last parseable boxed choice, then legacy ``answer is``."""
    prediction = prediction or ""
    for boxed_content in reversed(_boxed_contents(prediction)):
        boxed_choice = _extract_boxed_choice(boxed_content, MMLU_PRO_CHOICES)
        if boxed_choice:
            return boxed_choice
    return _last_answer_is_choice(prediction, MMLU_PRO_CHOICES)


def extract_rethink_answers(
    responses: Iterable[Sequence[str]], docs: Sequence[dict[str, Any]]
) -> list[list[str]]:
    """Parse every generation while preserving lm-eval's response shape."""
    del docs
    return [
        [get_multiple_choice_answer(response) for response in question_responses]
        for question_responses in responses
    ]
