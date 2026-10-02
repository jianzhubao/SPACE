"""ReThink HaluEval prompts and parsing with a whole-word label fallback."""

from __future__ import annotations

import re


# Preserve the reference wording and punctuation, including the extra quote.
_INSTRUCTIONS = (
    "I want you act as {judge}. Given {context}, your objective is to determine "
    "if the provided {item} contains non-factual or hallucinated information. "
    "You SHOULD give your judgement based on the following hallucination types "
    "and the world knowledge.\n"
    "You should try your best to determine if the {item} contains non-factual "
    "or hallucinated information. The answer you give MUST be \"Yes\" or \"No\"\"."
)


def doc_to_text_qa(doc: dict[str, str]) -> str:
    instruction = _INSTRUCTIONS.format(
        judge="an answer judge", context="a question and an answer", item="answer"
    )
    return (
        f"{instruction}\n\n#Knowledge: {doc['knowledge']}"
        f"\n#Question#: {doc['question']}\n#Answer#: {doc['answer']}"
        "\n#Your Judgement#:"
    )


def doc_to_text_dialogue(doc: dict[str, str]) -> str:
    instruction = _INSTRUCTIONS.format(
        judge="a response judge",
        context="a dialogue history and a response",
        item="response",
    )
    return (
        f"{instruction}\n\n#Knowledge: {doc['knowledge']}"
        f"\n#Dialogue History#: {doc['dialogue_history']}"
        f"\n#Response#: {doc['response']}\n#Your Judgement#:"
    )


def doc_to_text_summarization(doc: dict[str, str]) -> str:
    instruction = _INSTRUCTIONS.format(
        judge="a summary judge", context="a document and a summary", item="summary"
    )
    return (
        f"{instruction}\n\n#Document#: {doc['document']}"
        f"\n#Summary#: {doc['summary']}\n#Your Judgement#:"
    )


def extract_prediction(response: str) -> str | None:
    """Use ReThink parsing with whole-word labels in the first-line fallback."""
    # Match the reference cleanup, including closing tags without opening tags.
    response_clean = re.sub(r"<think>.*?</think>\s*", "", response, flags=re.DOTALL).strip()
    response_clean = re.sub(r"^.*</think>", "", response_clean, flags=re.DOTALL).lstrip()
    response_clean = re.sub(r"<think>.*?<\|/think\|>\s*", "", response_clean, flags=re.DOTALL).strip()
    response_clean = re.sub(r"^.*<\|/think\|>", "", response_clean, flags=re.DOTALL).lstrip()
    response_clean = response_clean.strip()
    # ReThink leaves unfinished thought blocks in the text being searched.
    matches = re.findall(r"\b(Yes|No)\b", response_clean, flags=re.IGNORECASE)
    if matches:
        return matches[-1].lower()

    # Keep ReThink's uncleaned first-line fallback, but reject partial words.
    first_line = response.strip().split("\n")[0].lower()
    has_yes = re.search(r"\byes\b", first_line) is not None
    has_no = re.search(r"\bno\b", first_line) is not None
    if has_yes == has_no:
        return None
    return "yes" if has_yes else "no"


def process_results(doc: dict[str, str], results: list[str]) -> dict[str, float]:
    prediction = extract_prediction(results[0])
    return {
        "acc": 100.0 if prediction == doc["hallucination"] else 0.0,
        "invalid_rate": 100.0 if prediction is None else 0.0,
    }
