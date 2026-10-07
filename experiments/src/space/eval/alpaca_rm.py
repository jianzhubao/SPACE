"""Score saved AlpacaEval generations with the ReThink reward-model protocol."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any


DEFAULT_REWARD_MODEL = "allenai/Llama-3.1-8B-Instruct-RM-RB2"
DEFAULT_REWARD_REVISION = "6779f9a302979f025eee2cb0d3edeafb4e100a7b"


def extract_answer(text: str) -> str:
    """Match ReThink: retain text after the last </think>, then strip."""
    return text.rsplit("</think>", 1)[-1].strip()


def load_samples(path: Path) -> list[dict[str, Any]]:
    """Read one response per instruction from a harness sample log."""
    records = []
    seen_ids = set()
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            sample = json.loads(line)
            responses = sample["resps"]
            if (
                not isinstance(responses, list)
                or len(responses) != 1
                or not isinstance(responses[0], list)
                or len(responses[0]) != 1
                or not isinstance(responses[0][0], str)
            ):
                raise ValueError("Alpaca RM expects exactly one response per document")
            doc_id = sample["doc_id"]
            if doc_id in seen_ids:
                raise ValueError(f"Duplicate document ID: {doc_id}")
            seen_ids.add(doc_id)
            doc = sample["doc"]
            output = responses[0][0]
            records.append({
                "doc_id": doc_id,
                "dataset": doc["dataset"],
                "instruction": doc["instruction"],
                "output": output,
                "scored_output": extract_answer(output),
            })
    if not records:
        raise ValueError("No AlpacaEval samples found")
    return records


def format_reward_input(tokenizer: Any, instruction: str, output: str) -> str:
    """Use the RM's template, not the generator's, and avoid duplicate BOS."""
    chat = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": output},
    ]
    text = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=False
    )
    return text.replace(tokenizer.bos_token, "")


def score_samples(
    records: list[dict[str, Any]], model: Any, tokenizer: Any
) -> list[float]:
    """Return raw classifier logits with batch size one and no truncation."""
    import torch
    from tqdm import tqdm

    model.eval()
    rewards = []
    with torch.inference_mode():
        for record in tqdm(records, desc="Scoring AlpacaEval"):
            text = format_reward_input(
                tokenizer, record["instruction"], record["scored_output"]
            )
            inputs = tokenizer(
                text, return_tensors="pt", add_special_tokens=True, truncation=False
            )
            if inputs["input_ids"].shape[-1] > model.config.max_position_embeddings:
                raise ValueError(
                    f"Document {record['doc_id']} exceeds the RM context limit; "
                    "refusing to truncate"
                )
            outputs = model(**inputs.to(model.device), use_cache=False)
            reward = outputs.logits.float().item()
            if not math.isfinite(reward):
                raise ValueError(f"Non-finite reward for document {record['doc_id']}")
            rewards.append(reward)
    return rewards


def summarize_rewards(rewards: list[float]) -> dict[str, float | int | None]:
    if not rewards:
        raise ValueError("Cannot summarize empty rewards")
    std = stdev(rewards) if len(rewards) > 1 else None
    return {
        "reward_mean": mean(rewards),
        "reward_std": std,
        "reward_stderr": std / math.sqrt(len(rewards)) if std is not None else None,
        "num_samples": len(rewards),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--reward_model", default=DEFAULT_REWARD_MODEL)
    parser.add_argument("--reward_revision", default=None)
    parser.add_argument("--device_map", default="auto")
    args = parser.parse_args(argv)

    records = load_samples(args.samples)
    revision = args.reward_revision
    if revision is None and args.reward_model == DEFAULT_REWARD_MODEL:
        revision = DEFAULT_REWARD_REVISION

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.reward_model, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model,
        revision=revision,
        dtype=torch.bfloat16,
        device_map=args.device_map,
    )
    rewards = score_samples(records, model, tokenizer)
    for record, reward in zip(records, rewards, strict=True):
        record["reward"] = reward

    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rewards_path = args.output_dir / f"alpaca_rm_rewards_{run_id}.jsonl"
    metrics_path = args.output_dir / f"alpaca_rm_metrics_{run_id}.json"
    summary = {
        "task": "alpaca_rm",
        "results": {"alpaca_rm": summarize_rewards(rewards)},
        "config": {
            "samples_path": str(args.samples.resolve()),
            "reward_model": args.reward_model,
            "reward_revision": revision,
            "dtype": "bfloat16",
            "device_map": args.device_map,
            "batch_size": 1,
            "truncation": False,
            "score_type": "raw_logit",
            "response_processing": "strip text after last </think>",
        },
        "rewards_path": str(rewards_path.resolve()),
    }
    if rewards_path.exists() or metrics_path.exists():
        raise FileExistsError(f"Results already exist for run ID {run_id}")
    with rewards_path.open("x", encoding="utf-8") as destination:
        for record in records:
            destination.write(
                json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
            )
    with metrics_path.open("x", encoding="utf-8") as destination:
        json.dump(summary, destination, indent=2, ensure_ascii=False, allow_nan=False)
        destination.write("\n")
    print(json.dumps(summary["results"], indent=2, allow_nan=False))
    print(f"Metrics: {metrics_path}")
    print(f"Per-sample rewards: {rewards_path}")


if __name__ == "__main__":
    main()
