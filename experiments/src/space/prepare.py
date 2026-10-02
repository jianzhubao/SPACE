"""Prepare training data, math benchmarks, and evaluation subsets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
SOURCES = json.loads((ROOT / "configs/data_sources.json").read_text())


def download(url, path):
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        temporary = Path(f.name)
    try:
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _last_boxed(text):
    # Ground truth is metadata; SFT trains on the complete original solution.
    if "\\boxed " in text:
        return text.split("\\boxed ")[-1].split("$")[0]
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    depth = 1
    for end in range(start + 7, len(text)):
        depth += (text[end] == "{") - (text[end] == "}")
        if depth == 0:
            return text[start + 7:end]
    return None


def numina_rows(dataset):
    """DFT's prefix selection followed by SPACE's native message schema."""
    for index, row in enumerate(dataset):
        yield {
            "source": row["source"],
            "messages": [{"role": "user", "content": row["problem"]}, {"role": "assistant", "content": row["solution"]}],
            "data_source": SOURCES["numina_cot"]["repo_id"], "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": _last_boxed(row["solution"])},
            "extra_info": {"split": "train", "index": index},
        }


def prepare_train(model):
    import pandas as pd
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from space.preprocess.numina_cot import preprocess_dataframe, _write_parquet_atomic

    if model == "qwen2.5-7b":
        spec = SOURCES["numina_cot"]
        dataset = load_dataset(spec["repo_id"], revision=spec["revision"], split="train[:100000]")
        target = ROOT / "data/numina_cot/preprocessed"
        processed = preprocess_dataframe(pd.DataFrame(numina_rows(dataset)))
        _write_parquet_atomic(processed, target / "train.parquet")
        _write_parquet_atomic(processed.sample(n=10, random_state=42).sort_index(), target / "train_sample_10.parquet")
        rows = len(processed)
    else:
        from space.preprocess.math_cot import preprocess_math_cot
        spec = SOURCES["math_cot"]
        original = Path(hf_hub_download(
            spec["repo_id"], spec["filename"], repo_type="dataset", revision=spec["revision"],
        ))
        target = ROOT / "data/math-cot/preprocessed"
        info = preprocess_math_cot(original, target / "train.parquet", target / "train_sample_10.parquet")
        rows = info["output_rows"]
    print(f"Prepared {rows} training examples in {target}")


def prepare_math():
    for name, spec in SOURCES["math_evaluation"].items():
        download(spec["url"], ROOT / "math_evaluation/data" / name / "test.jsonl")


def prepare_subsets():
    from datasets import load_dataset

    spec = SOURCES["mmlu_pro"]
    data = load_dataset(spec["repo_id"], revision=spec["revision"])
    target = ROOT / "data/mmlu_pro"
    target.mkdir(parents=True, exist_ok=True)
    data["test"].shuffle(seed=42).select(range(1000)).to_parquet(target / "1k_test.parquet")
    data["validation"].to_parquet(target / "validation.parquet")

    spec = SOURCES["halueval"]
    target = ROOT / "data/halueval/3k"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("qa_samples", "dialogue_samples", "summarization_samples"):
        full = load_dataset(
            spec["repo_id"], name, split="data", revision=spec["revision"],
        )
        full.shuffle(seed=42).select(range(1000)).to_parquet(target / f"{name}.parquet")
    print("Prepared MMLU-Pro 1K and HaluEval 3K subsets with seed 42")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=["qwen2.5-7b", "qwen3-14b-base"])
    parser.add_argument("--part", choices=["train", "eval", "all"], default="all")
    args = parser.parse_args(argv)
    if args.part in {"train", "all"}:
        prepare_train(args.model)
    if args.part in {"eval", "all"}:
        prepare_math()
        if args.model == "qwen3-14b-base":
            prepare_subsets()
    print("Other OOD sources are fetched by their pinned evaluation tasks on first use.")


if __name__ == "__main__":
    main()
