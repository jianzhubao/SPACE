"""Shared prompt, extraction, and aggregation helpers for GPQA subsets."""

from __future__ import annotations

import math
import random
import re
import statistics
from collections.abc import Iterable, Sequence
from typing import Any


NUM_REPEATS = 3
NUM_REPEATS_10 = 10
NUM_REPEATS_16 = 16
CHOICE_LETTERS = ("A", "B", "C", "D")

_STANDALONE_UPPER_CHOICE = re.compile(r"(?<![A-Za-z])([A-D])(?![A-Za-z])")
_EXPLICIT_ANSWER_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"(?:correct answer|right answer|answer)\s*(?:is|:)\s*([A-D])",
        r"(?:choose|select|pick)\s*(?:option)?\s*([A-D])",
        r"(?:option|choice)\s*([A-D])\s*(?:is correct|is right)",
        r"\b([A-D])\s*(?:is correct|is right|is the answer)",
    )
)


def process_doc(doc: dict[str, Any]) -> dict[str, Any]:
    """Shuffle choices with the Evalchemy GPQA convention."""
    choices = [
        {
            "text": doc["Correct Answer"],
            "is_correct": True,
        },
        {
            "text": doc["Incorrect Answer 1"],
            "is_correct": False,
        },
        {
            "text": doc["Incorrect Answer 2"],
            "is_correct": False,
        },
        {
            "text": doc["Incorrect Answer 3"],
            "is_correct": False,
        },
    ]
    question_hash = hash(doc["Question"]) % 10000
    random.Random(42 + question_hash).shuffle(choices)

    correct_index = next(
        index for index, choice in enumerate(choices) if choice["is_correct"]
    )
    choice_texts = [choice["text"] for choice in choices]

    return {
        "choice1": choice_texts[0],
        "choice2": choice_texts[1],
        "choice3": choice_texts[2],
        "choice4": choice_texts[3],
        "choices": choice_texts,
        # Evalchemy compares extracted bare letters rather than parenthesized labels.
        "answer": CHOICE_LETTERS[correct_index],
    }


def process_docs(dataset: Any) -> Any:
    """Apply GPQA choice preprocessing to a Hugging Face Dataset."""
    return dataset.map(process_doc)


def doc_to_text(doc: dict[str, Any]) -> str:
    """Render the Evalchemy GPQA prompt."""
    options = ", ".join(
        f"{letter}) {choice}"
        for letter, choice in zip(CHOICE_LETTERS, doc["choices"], strict=True)
    )
    return (
        f"Problem: {doc['Question']}\n"
        f"Options: {options}\n"
        "Please reason step by step and return your final answer within \\boxed{}. "
        "Only include the letter choice (A, B, C, or D) as your final answer."
    )


def last_boxed_content(text: str | None) -> str | None:
    """Return the content of the last balanced ``\\boxed{...}`` expression."""
    if not text:
        return None

    boxed_index = text.rfind("\\boxed{")
    if boxed_index < 0:
        return None

    left_brace_index = boxed_index + len("\\boxed")
    depth = 0
    for index in range(left_brace_index, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[left_brace_index + 1 : index]

    return None


def _extract_explicit_answer(text: str) -> str:
    for pattern in _EXPLICIT_ANSWER_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).upper()
    return ""


def _clean_single_choice(text: str) -> str:
    cleaned = text.rstrip(".").rstrip("/").strip()
    if len(cleaned) == 1 and cleaned.upper() in CHOICE_LETTERS:
        return cleaned.upper()
    return ""


def _extract_boxed_choice(content: str) -> str:
    """Extract the first standalone uppercase choice from boxed content."""
    match = _STANDALONE_UPPER_CHOICE.search(content)
    if match:
        return match.group(1)

    explicit_answer = _extract_explicit_answer(content)
    if explicit_answer:
        return explicit_answer

    # Accept a bare lowercase label without searching lowercase a-d in prose.
    return _clean_single_choice(content)


def get_multiple_choice_answer(prediction: str | None) -> str:
    """Extract a GPQA answer with last-box priority and Evalchemy fallbacks."""
    prediction = prediction or ""
    boxed_content = last_boxed_content(prediction)
    if boxed_content is not None:
        return _extract_boxed_choice(boxed_content)

    explicit_answer = _extract_explicit_answer(prediction)
    if explicit_answer:
        return explicit_answer
    return _clean_single_choice(prediction)


def extract_rethink_answers(
    responses: Iterable[Sequence[str]], docs: Sequence[dict[str, Any]]
) -> list[list[str]]:
    """Keep and parse all repeated generations for every question."""
    del docs
    return [
        [get_multiple_choice_answer(response) for response in question_responses]
        for question_responses in responses
    ]


def _unwrap_predictions(
    results: Sequence[Any], expected_num_repeats: int = NUM_REPEATS
) -> list[str]:
    if len(results) == 1 and isinstance(results[0], (list, tuple)):
        predictions = list(results[0])
    else:
        predictions = list(results)

    if len(predictions) != expected_num_repeats:
        raise ValueError(
            f"Expected {expected_num_repeats} GPQA predictions, got {len(predictions)}"
        )
    if not all(isinstance(prediction, str) for prediction in predictions):
        raise TypeError("GPQA predictions must be strings after filtering")
    return predictions


def _process_results(
    doc: dict[str, Any], results: Sequence[Any], num_repeats: int
) -> dict[str, Any]:
    predictions = _unwrap_predictions(results, expected_num_repeats=num_repeats)
    target = doc["answer"]
    correctness = tuple(int(prediction == target) for prediction in predictions)

    metrics = {
        "num_total": 1,
        "accuracy_avg": sum(correctness) / num_repeats,
        "accuracy_std_err": correctness,
        "solved_avg": correctness,
    }
    metrics.update(
        {
            f"accuracy_rep_{repeat_index + 1}": is_correct
            for repeat_index, is_correct in enumerate(correctness)
        }
    )
    return metrics


def process_results(doc: dict[str, Any], results: Sequence[Any]) -> dict[str, Any]:
    """Return per-question values for the standard three-repeat task."""
    return _process_results(doc, results, num_repeats=NUM_REPEATS)


def process_results_10(doc: dict[str, Any], results: Sequence[Any]) -> dict[str, Any]:
    """Return per-question values for the ten-repeat task."""
    return _process_results(doc, results, num_repeats=NUM_REPEATS_10)


def process_results_16(doc: dict[str, Any], results: Sequence[Any]) -> dict[str, Any]:
    """Return per-question values for the sixteen-repeat task."""
    return _process_results(doc, results, num_repeats=NUM_REPEATS_16)


def _correctness_rows(
    values: Iterable[Sequence[int]], expected_num_repeats: int = NUM_REPEATS
) -> list[tuple[int, ...]]:
    rows = [tuple(int(value) for value in row) for row in values]
    if not rows:
        raise ValueError("Cannot aggregate an empty GPQA result set")
    if any(len(row) != expected_num_repeats for row in rows):
        raise ValueError(
            f"Every GPQA result must contain {expected_num_repeats} repetitions"
        )
    return rows


def aggregate_num_total(values: Iterable[int]) -> int:
    return sum(values)


def _aggregate_accuracy_std_err(
    values: Iterable[Sequence[int]], num_repeats: int
) -> float:
    rows = _correctness_rows(values, expected_num_repeats=num_repeats)
    num_questions = len(rows)
    repetition_accuracies = [
        sum(row[repeat_index] for row in rows) / num_questions
        for repeat_index in range(num_repeats)
    ]
    return statistics.pstdev(repetition_accuracies) / math.sqrt(num_repeats)


def aggregate_accuracy_std_err(values: Iterable[Sequence[int]]) -> float:
    """Match Evalchemy's population std over three repetition accuracies."""
    return _aggregate_accuracy_std_err(values, num_repeats=NUM_REPEATS)


def aggregate_accuracy_std_err_10(values: Iterable[Sequence[int]]) -> float:
    """Match Evalchemy's population std over ten repetition accuracies."""
    return _aggregate_accuracy_std_err(values, num_repeats=NUM_REPEATS_10)


def aggregate_accuracy_std_err_16(values: Iterable[Sequence[int]]) -> float:
    """Match Evalchemy's population std over sixteen repetition accuracies."""
    return _aggregate_accuracy_std_err(values, num_repeats=NUM_REPEATS_16)


def _aggregate_solved_avg(values: Iterable[Sequence[int]], num_repeats: int) -> float:
    rows = _correctness_rows(values, expected_num_repeats=num_repeats)
    solved_per_repetition = [
        sum(row[repeat_index] for row in rows)
        for repeat_index in range(num_repeats)
    ]
    return sum(solved_per_repetition) / num_repeats


def aggregate_solved_avg(values: Iterable[Sequence[int]]) -> float:
    """Average the number of solved questions across three repetitions."""
    return _aggregate_solved_avg(values, num_repeats=NUM_REPEATS)


def aggregate_solved_avg_10(values: Iterable[Sequence[int]]) -> float:
    """Average the number of solved questions across ten repetitions."""
    return _aggregate_solved_avg(values, num_repeats=NUM_REPEATS_10)


def aggregate_solved_avg_16(values: Iterable[Sequence[int]]) -> float:
    """Average the number of solved questions across sixteen repetitions."""
    return _aggregate_solved_avg(values, num_repeats=NUM_REPEATS_16)
