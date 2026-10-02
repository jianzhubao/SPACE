"""Preprocess Math-CoT-20k parquet for native multi-turn SFT."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import pandas as pd

from space.preprocess.numina_cot import BOXED_INSTRUCTION
from space.preprocess.numina_cot import _write_parquet_atomic

DEFAULT_INPUT_PATH = Path("data/math-cot/Math-CoT-20k/Math-CoT-20k.parquet")
DEFAULT_OUTPUT_PATH = Path("data/math-cot/preprocessed/train.parquet")
DEFAULT_SAMPLE_OUTPUT_PATH = Path(
    "data/math-cot/preprocessed/train_sample_10.parquet"
)
DEFAULT_DATA_SOURCE = "jasonrqh/Math-CoT-20k"
OUTPUT_COLUMNS = [
    "source",
    "messages",
    "data_source",
    "ability",
    "reward_model",
    "extra_info",
]
REQUIRED_INPUT_COLUMNS = {
    "data_source",
    "question",
    "answer",
    "message",
    "response",
    "response_length",
    "advantage",
}


def _require_text(example: Mapping[str, Any], key: str, row_index: int) -> str:
    value = example.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Row {row_index}: field {key!r} must be a non-empty string"
        )
    return value


def _source_user_message(
    example: Mapping[str, Any], row_index: int, question: str
) -> dict[str, str]:
    messages = example.get("message")
    if hasattr(messages, "tolist"):
        messages = messages.tolist()
    if not isinstance(messages, (list, tuple)):
        raise TypeError(
            f"Row {row_index}: field 'message' must be list-like, "
            f"got {type(messages).__name__}"
        )
    if len(messages) != 1 or not isinstance(messages[0], Mapping):
        raise ValueError(
            f"Row {row_index}: field 'message' must contain exactly one mapping"
        )

    source_message = messages[0]
    role = source_message.get("role")
    content = source_message.get("content")
    if role != "user" or not isinstance(content, str) or not content.strip():
        raise ValueError(
            f"Row {row_index}: source message must contain a non-empty user message"
        )

    expected_content = f"{question}\n{BOXED_INSTRUCTION}"
    if content != expected_content:
        raise ValueError(
            f"Row {row_index}: source message content does not match question plus "
            "the boxed-answer instruction"
        )
    return {"role": "user", "content": content}


def _require_response_length(example: Mapping[str, Any], row_index: int) -> int:
    value = example.get("response_length")
    if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
        raise ValueError(
            f"Row {row_index}: field 'response_length' must be a positive integer"
        )
    return int(value)


def _require_advantage(example: Mapping[str, Any], row_index: int) -> float:
    value = example.get("advantage")
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"Row {row_index}: field 'advantage' must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Row {row_index}: field 'advantage' must be finite")
    return value


def make_training_row(
    example: Mapping[str, Any],
    *,
    row_index: int,
    data_source: str = DEFAULT_DATA_SOURCE,
) -> dict[str, Any]:
    """Convert one source row to the messages schema consumed by verl SFT."""

    source = _require_text(example, "data_source", row_index)
    question = _require_text(example, "question", row_index)
    answer = _require_text(example, "answer", row_index)
    response = _require_text(example, "response", row_index)
    user_message = _source_user_message(example, row_index, question)
    response_length = _require_response_length(example, row_index)
    advantage = _require_advantage(example, row_index)

    return {
        "source": source,
        "messages": [
            user_message,
            {"role": "assistant", "content": response},
        ],
        "data_source": data_source,
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {
            "split": "train",
            "index": row_index,
            "response_length": response_length,
            "advantage": advantage,
        },
    }


def preprocess_dataframe(
    dataframe: pd.DataFrame,
    *,
    data_source: str = DEFAULT_DATA_SOURCE,
) -> pd.DataFrame:
    """Validate every source row and convert it without truncating the long CoT."""

    missing_columns = sorted(REQUIRED_INPUT_COLUMNS - set(dataframe.columns))
    if missing_columns:
        raise KeyError(f"input parquet is missing required columns: {missing_columns}")
    if not isinstance(data_source, str) or not data_source.strip():
        raise ValueError("data_source must be a non-empty string")

    rows = [
        make_training_row(example, row_index=row_index, data_source=data_source)
        for row_index, example in enumerate(dataframe.to_dict(orient="records"))
    ]
    return pd.DataFrame.from_records(rows, columns=OUTPUT_COLUMNS)


def preprocess_math_cot(
    input_path: Path,
    output_path: Path,
    sample_output_path: Path,
    *,
    sample_size: int = 10,
    seed: int = 42,
    data_source: str = DEFAULT_DATA_SOURCE,
) -> dict[str, Any]:
    """Create the full SFT parquet and a deterministic smoke-test sample."""

    dataframe = pd.read_parquet(input_path)
    if sample_size <= 0 or sample_size > len(dataframe):
        raise ValueError(
            f"sample_size must be between 1 and dataset size ({len(dataframe)}), "
            f"got {sample_size}"
        )

    processed = preprocess_dataframe(dataframe, data_source=data_source)
    sample = (
        processed.sample(n=sample_size, random_state=seed)
        .sort_index()
        .reset_index(drop=True)
    )
    _write_parquet_atomic(processed, output_path)
    _write_parquet_atomic(sample, sample_output_path)

    source_counts = processed["source"].value_counts().sort_index()
    response_lengths = dataframe["response_length"]
    return {
        "input": str(input_path),
        "output": str(output_path),
        "sample_output": str(sample_output_path),
        "input_rows": len(dataframe),
        "output_rows": len(processed),
        "sample_rows": len(sample),
        "seed": seed,
        "columns": processed.columns.tolist(),
        "source_counts": {
            str(source): int(count) for source, count in source_counts.items()
        },
        "response_length": {
            "mean": float(response_lengths.mean()),
            "p50": float(response_lengths.quantile(0.50)),
            "p90": float(response_lengths.quantile(0.90)),
            "p95": float(response_lengths.quantile(0.95)),
            "p99": float(response_lengths.quantile(0.99)),
            "max": int(response_lengths.max()),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--sample-output", type=Path, default=DEFAULT_SAMPLE_OUTPUT_PATH
    )
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-source", default=DEFAULT_DATA_SOURCE)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = preprocess_math_cot(
        input_path=args.input,
        output_path=args.output,
        sample_output_path=args.sample_output,
        sample_size=args.sample_size,
        seed=args.seed,
        data_source=args.data_source,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
