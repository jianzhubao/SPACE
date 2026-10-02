"""Sample MMLU-Pro using the category-balanced sampler from the local rethink fork.

This produces a local evaluation subset, not the paper authors' original 1k set.
The sampling algorithm is preserved from andataxx/rethink_sft_generalization,
evaluation/data/mmlu_pro/sample_1k_test.py (commit 5caeeb8).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "question_id", "question", "options", "answer", "answer_index",
    "cot_content", "category", "src",
}


def sample_uniform_by_category(df: pd.DataFrame, total: int, seed: int) -> pd.DataFrame:
    if total <= 0:
        raise ValueError("--num-samples must be positive")
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing source columns: {sorted(missing)}")
    if df["question_id"].isna().any() or df["question_id"].duplicated().any():
        raise ValueError("Source question IDs must be non-null and unique")
    if df["category"].isna().any():
        raise ValueError("Source categories must be non-null")

    categories = sorted(df["category"].unique())
    if not categories:
        raise ValueError("No categories found in source parquet")

    base_count = total // len(categories)
    remainder = total % len(categories)
    rng = random.Random(seed)
    extra_categories = set(rng.sample(categories, remainder))

    parts = []
    for index, category in enumerate(categories):
        sample_count = base_count + (1 if category in extra_categories else 0)
        group = df[df["category"] == category]
        if len(group) < sample_count:
            raise ValueError(
                f"Category {category!r} has only {len(group)} rows; need {sample_count}"
            )
        parts.append(group.sample(n=sample_count, random_state=seed + index))

    return pd.concat(parts, ignore_index=True).sample(
        frac=1, random_state=seed
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("data/mmlu_pro/source/test.parquet"),
        help="Official MMLU-Pro test parquet; relative paths use the working directory.",
    )
    parser.add_argument(
        "--validation-input", type=Path,
        default=Path("data/mmlu_pro/source/validation.parquet"),
        help="Official validation split, retained for lm-eval task initialization.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/mmlu_pro/1k_test.parquet"),
    )
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    validation = pd.read_parquet(args.validation_input)
    sampled = sample_uniform_by_category(df, args.num_samples, args.seed)
    validation_output = args.output.parent / "validation.parquet"
    if {args.output.resolve(), validation_output.resolve()} & {
        args.input.resolve(), args.validation_input.resolve()
    }:
        raise ValueError("Output files must not overwrite the source splits")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_parquet(args.output, index=False)
    validation.to_parquet(validation_output, index=False)

    counts = sampled["category"].value_counts().sort_index().to_dict()
    manifest = {
        "dataset": "TIGER-Lab/MMLU-Pro",
        "subset": "locally sampled; not the paper authors' original subset",
        "sampling": "uniform_by_category",
        "seed": args.seed,
        "source_test": str(args.input),
        "source_test_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "source_test_rows": len(df),
        "test_rows": len(sampled),
        "validation_rows": len(validation),
        "category_counts": counts,
        "question_ids": sampled["question_id"].tolist(),
        "test_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "pandas_version": pd.__version__,
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Wrote {len(sampled)} test questions to {args.output}")
    print(f"Seed: {args.seed}; category counts: {counts}")
    print(f"Validation: {validation_output} ({len(validation)} questions; zero-shot task)")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
