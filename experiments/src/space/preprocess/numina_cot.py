"""Preprocess Numina-CoT messages for Qwen boxed-answer SFT."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

BOXED_INSTRUCTION = r"Please reason step by step, and put your final answer within \boxed{}."

DEFAULT_INPUT_PATH = Path("data/numina_cot/train.parquet")
DEFAULT_OUTPUT_PATH = Path("data/numina_cot/preprocessed/train.parquet")
DEFAULT_SAMPLE_OUTPUT_PATH = Path("data/numina_cot/preprocessed/train_sample_10.parquet")


def _starts_with_newline(content: str) -> bool:
    return content.startswith(("\n", "\r"))


def _ends_with_newline(content: str) -> bool:
    return content.endswith(("\n", "\r"))


def summarize_boundary_whitespace(messages_column: Iterable[Any]) -> dict[str, Any]:
    """Count leading/trailing newlines and strip-sensitive whitespace by role."""

    role_counts: dict[str, Counter[str]] = {}
    sample_count = 0
    samples_with_boundary_newline = 0
    samples_with_boundary_whitespace = 0

    for sample_index, messages in enumerate(messages_column):
        if not isinstance(messages, (list, tuple)) and not hasattr(messages, "tolist"):
            raise TypeError(f"messages at row {sample_index} must be list-like, got {type(messages).__name__}")
        if hasattr(messages, "tolist"):
            messages = messages.tolist()

        sample_count += 1
        sample_has_boundary_newline = False
        sample_has_boundary_whitespace = False

        for message_index, message in enumerate(messages):
            if not isinstance(message, Mapping):
                raise TypeError(
                    f"message at row {sample_index}, position {message_index} must be a mapping, "
                    f"got {type(message).__name__}"
                )
            role = message.get("role")
            content = message.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                raise TypeError(
                    f"message at row {sample_index}, position {message_index} must contain string role/content"
                )

            leading_newline = _starts_with_newline(content)
            trailing_newline = _ends_with_newline(content)
            boundary_newline = leading_newline or trailing_newline
            boundary_whitespace = content != content.strip()

            counts = role_counts.setdefault(role, Counter())
            counts["messages"] += 1
            counts["leading_newline"] += leading_newline
            counts["trailing_newline"] += trailing_newline
            counts["boundary_newline"] += boundary_newline
            counts["boundary_whitespace"] += boundary_whitespace

            sample_has_boundary_newline |= boundary_newline
            sample_has_boundary_whitespace |= boundary_whitespace

        samples_with_boundary_newline += sample_has_boundary_newline
        samples_with_boundary_whitespace += sample_has_boundary_whitespace

    return {
        "samples": sample_count,
        "roles": {role: dict(counts) for role, counts in sorted(role_counts.items())},
        "samples_with_boundary_newline": samples_with_boundary_newline,
        "samples_with_boundary_whitespace": samples_with_boundary_whitespace,
    }


def _normalize_message(message: Mapping[str, Any], row_index: int, message_index: int) -> dict[str, Any]:
    role = message.get("role")
    content = message.get("content")
    if not isinstance(role, str) or not isinstance(content, str):
        raise TypeError(f"message at row {row_index}, position {message_index} must contain string role/content")

    clean_content = content.strip()
    if not clean_content:
        raise ValueError(f"message content at row {row_index}, position {message_index} is empty after strip()")

    if role == "user" and not clean_content.endswith(BOXED_INSTRUCTION):
        clean_content = f"{clean_content}\n{BOXED_INSTRUCTION}"

    normalized = dict(message)
    normalized["content"] = clean_content
    return normalized


def _normalize_messages(messages: Any, row_index: int) -> list[dict[str, Any]]:
    if hasattr(messages, "tolist"):
        messages = messages.tolist()
    if not isinstance(messages, (list, tuple)):
        raise TypeError(f"messages at row {row_index} must be list-like, got {type(messages).__name__}")

    normalized = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise TypeError(
                f"message at row {row_index}, position {message_index} must be a mapping, "
                f"got {type(message).__name__}"
            )
        normalized.append(_normalize_message(message, row_index, message_index))
    roles = [message["role"] for message in normalized]
    if roles != ["user", "assistant"]:
        raise ValueError(f"messages at row {row_index} must have roles ['user', 'assistant'], got {roles!r}")
    return normalized


def _prune_extra_info(extra_info: Any, row_index: int) -> dict[str, Any] | None:
    if extra_info is None:
        return None
    if not isinstance(extra_info, Mapping):
        raise TypeError(f"extra_info at row {row_index} must be a mapping, got {type(extra_info).__name__}")
    return {key: value for key, value in extra_info.items() if key not in {"answer", "question"}}


def preprocess_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Strip messages, append the qwen-boxed instruction, and remove redundant fields."""

    if "messages" not in dataframe.columns:
        raise KeyError("input parquet must contain a messages column")

    processed = dataframe.drop(columns=["prompt"], errors="ignore").copy()
    processed["messages"] = [
        _normalize_messages(messages, row_index) for row_index, messages in enumerate(processed["messages"])
    ]

    if "extra_info" in processed.columns:
        processed["extra_info"] = [
            _prune_extra_info(extra_info, row_index)
            for row_index, extra_info in enumerate(processed["extra_info"])
        ]

    return processed


def _write_parquet_atomic(dataframe: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        dataframe.to_parquet(temporary_path, index=False, compression="snappy")
        temporary_path.chmod(0o664)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def preprocess_numina_cot(
    input_path: Path,
    output_path: Path,
    sample_output_path: Path,
    sample_size: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    dataframe = pd.read_parquet(input_path)
    if sample_size > len(dataframe):
        raise ValueError(f"sample_size {sample_size} exceeds dataset size {len(dataframe)}")

    before_stats = summarize_boundary_whitespace(dataframe["messages"])
    processed = preprocess_dataframe(dataframe)
    after_stats = summarize_boundary_whitespace(processed["messages"])

    sample = processed.sample(n=sample_size, random_state=seed).sort_index().reset_index(drop=True)
    _write_parquet_atomic(processed, output_path)
    _write_parquet_atomic(sample, sample_output_path)

    return {
        "input": str(input_path),
        "output": str(output_path),
        "sample_output": str(sample_output_path),
        "rows": len(processed),
        "sample_rows": len(sample),
        "columns": processed.columns.tolist(),
        "before": before_stats,
        "after": after_stats,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--sample-output", type=Path, default=DEFAULT_SAMPLE_OUTPUT_PATH)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = preprocess_numina_cot(
        input_path=args.input,
        output_path=args.output,
        sample_output_path=args.sample_output,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
