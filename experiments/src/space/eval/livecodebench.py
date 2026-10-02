"""Generate LiveCodeBench v2 answers with vLLM and score them with the official evaluator."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import re
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any


TASK = "livecodebench_v2"
DATASET = "livecodebench/code_generation_lite"
DATASET_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
DATA_FILES = ("test.jsonl", "test2.jsonl")
NUM_QUESTIONS = 511
OFFICIAL_ROOT = Path(__file__).resolve().parents[3] / "livecodebench"


def official_commit() -> str:
    return subprocess.check_output(
        ["git", "-C", str(OFFICIAL_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def iter_dataset_rows(revision: str = DATASET_REVISION):
    """Read the two release_v2 JSONL files without executing a dataset script."""
    from huggingface_hub import hf_hub_download

    for filename in DATA_FILES:
        path = hf_hub_download(
            DATASET, filename, repo_type="dataset", revision=revision
        )
        with open(path, encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    yield json.loads(line)


def load_questions(limit: int | None = None) -> list[dict[str, Any]]:
    questions = [
        {
            "question_id": row["question_id"],
            "question_content": row["question_content"],
            "starter_code": row["starter_code"],
        }
        for row in iter_dataset_rows()
    ]
    if (
        len(questions) != NUM_QUESTIONS
        or len({q["question_id"] for q in questions}) != NUM_QUESTIONS
    ):
        raise ValueError(
            f"Expected {NUM_QUESTIONS} unique release_v2 questions, got {len(questions)}"
        )
    questions.sort(key=lambda question: str(question["question_id"]))
    for index, question in enumerate(questions):
        question["question_index"] = index
    return questions[:limit] if limit is not None else questions


def build_messages(question: dict[str, Any]) -> list[dict[str, str]]:
    """Use the official generic chat prompt, with the programmer instruction in system."""
    from lcb_runner.lm_styles import LMStyle
    from lcb_runner.prompts.code_generation import format_prompt_generation

    return format_prompt_generation(SimpleNamespace(**question), LMStyle.OpenAIChat)


def prepare_requests(questions, tokenizer, args):
    from vllm import SamplingParams, TokensPrompt

    prompts, params, messages_list = [], [], []
    stop_ids = [tokenizer.eos_token_id]
    # Exported tokenizers can retain chat tokens in the vocab but omit them from all_special_tokens.
    vocab = tokenizer.get_vocab()
    for token in ("<|im_end|>", "<|eot_id|>", "<end_of_turn>"):
        if token in vocab:
            stop_ids.append(vocab[token])
    stop_ids = list(dict.fromkeys(token_id for token_id in stop_ids if token_id is not None))

    for question in questions:
        messages = build_messages(question)
        token_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=False
        )
        if len(token_ids) + args.max_gen_toks > args.max_model_len:
            raise ValueError(
                f"Question {question['question_id']}: prompt ({len(token_ids)}) + "
                f"max_gen_toks ({args.max_gen_toks}) exceeds max_model_len ({args.max_model_len})"
            )
        prompts.append(TokensPrompt(prompt_token_ids=token_ids))
        params.append(
            SamplingParams(
                n=args.n,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_gen_toks,
                seed=args.seed + question["question_index"],
                stop_token_ids=stop_ids,
            )
        )
        messages_list.append(messages)
    return prompts, params, messages_list


def generate_questions(args, questions):
    from transformers import AutoTokenizer
    from vllm import LLM

    tokenizer_path = args.tokenizer_name_or_path or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    prompts, params, messages = prepare_requests(questions, tokenizer, args)
    model = LLM(
        model=args.model_name_or_path,
        tokenizer=tokenizer_path,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        seed=args.seed,
        trust_remote_code=True,
        language_model_only=os.environ.get("LANGUAGE_MODEL_ONLY", "false").lower() == "true",
    )
    try:
        outputs = model.generate(prompts, params)
    finally:
        # Close engine children before the outer multiprocessing worker exits.
        model.llm_engine.engine_core.shutdown()
    return [
        {
            "question_id": question["question_id"],
            "prompt": prompt,
            "seed": sampling.seed,
            "output_list": [completion.text for completion in output.outputs],
        }
        for question, prompt, sampling, output in zip(
            questions, messages, params, outputs, strict=True
        )
    ]


def split_questions(questions, worker_count):
    return [questions[index::worker_count] for index in range(worker_count)]


def _generation_worker(args, questions, devices, output_file):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    write_json(output_file, generate_questions(args, questions))


def run_generate(args) -> Path:
    questions = load_questions(args.limit)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        import torch

        devices = [str(index) for index in range(torch.cuda.device_count())]
    else:
        devices = [device.strip() for device in visible.split(",") if device.strip()]
    workers = min(args.data_parallel_size, len(questions))
    if (
        workers < 1
        or args.tensor_parallel_size < 1
        or workers * args.tensor_parallel_size > len(devices)
    ):
        raise ValueError(
            "TP × active DP workers must fit the visible GPUs, with at least one question"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Generating {len(questions)} questions × {args.n} answers; "
        f"TP={args.tensor_parallel_size}, DP={workers}",
        flush=True,
    )
    context = multiprocessing.get_context("spawn")
    with tempfile.TemporaryDirectory(prefix=".lcb-generate-", dir=args.output_dir) as temporary:
        processes = []
        for rank, shard in enumerate(split_questions(questions, workers)):
            start = rank * args.tensor_parallel_size
            process = context.Process(
                target=_generation_worker,
                args=(
                    args,
                    shard,
                    devices[start:start + args.tensor_parallel_size],
                    Path(temporary) / f"{rank}.json",
                ),
            )
            process.start()
            processes.append(process)
        for process in processes:
            process.join()
        if any(process.exitcode != 0 for process in processes):
            raise RuntimeError(f"Generation workers failed: {[p.exitcode for p in processes]}")
        records = []
        for rank in range(workers):
            records.extend(json.loads((Path(temporary) / f"{rank}.json").read_text()))
    records.sort(key=lambda record: str(record["question_id"]))

    config = {
        key: value for key, value in vars(args).items()
        if key not in ("command", "output_dir", "run_id")
    }
    config.update(
        dataset=DATASET,
        dataset_revision=DATASET_REVISION,
        release_version="release_v2",
        official_commit=official_commit(),
        prompt_style="generic",
        seed_policy="seed + index in complete dataset sorted by question_id",
    )
    path = args.output_dir / f"{TASK}_generations_{args.run_id}.json"
    write_json(path, {"config": config, "samples": records})
    print(f"Generations: {path}")
    return path


def load_evaluation_samples(question_ids, revision):
    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem

    wanted = set(question_ids)
    samples = {}
    for row in iter_dataset_rows(revision):
        if row["question_id"] in wanted:
            problem = CodeGenerationProblem(**row)
            samples[row["question_id"]] = {
                **problem.get_evaluation_sample(),
                "num_public_tests": len(problem.public_test_cases),
            }
    missing = wanted - samples.keys()
    if missing:
        raise ValueError(f"Question IDs missing from release_v2: {sorted(missing)}")
    return [samples[question_id] for question_id in question_ids]


def select_test_samples(samples, test_group):
    if test_group == "all":
        return samples
    selected = []
    for sample in samples:
        in_out = json.loads(sample["input_output"])
        count = sample["num_public_tests"]
        section = slice(None, count) if test_group == "public" else slice(count, None)
        in_out["inputs"] = in_out["inputs"][section]
        in_out["outputs"] = in_out["outputs"][section]
        selected.append({"input_output": json.dumps(in_out)})
    return selected


def extract_last_python_block(output: str) -> str:
    """Select the last complete Python fence, using evalchemy's matching rule."""
    matches = re.findall(r"```python\n(.*?)```", output, re.DOTALL)
    return matches[-1].strip() if matches else ""


def sample_pass_rates(grades, metric_name="pass@1"):
    """Compute accuracy across questions for each 1-based generation index."""
    return {
        f"{metric_name}_sample_{index + 1}": sum(row[index] for row in grades) / len(grades)
        for index in range(len(grades[0]))
    }


def score_generations(generations, num_process_evaluate=16, timeout=6, code_extraction="last_python"):
    from lcb_runner.evaluation import codegen_metrics, extract_instance_results
    from lcb_runner.lm_styles import LMStyle
    from lcb_runner.utils.extraction_utils import extract_code

    records = generations["samples"]
    question_ids = [record["question_id"] for record in records]
    if not records or len(set(question_ids)) != len(question_ids):
        raise ValueError("Generations must contain nonempty, unique question IDs")
    n = generations["config"]["n"]
    if n < 1 or any(
        len(record["output_list"]) != n
        or not all(isinstance(output, str) for output in record["output_list"])
        for record in records
    ):
        raise ValueError(f"Expected exactly {n} answer strings per question")
    samples = load_evaluation_samples(question_ids, generations["config"]["dataset_revision"])
    extract = {
        "last_python": extract_last_python_block,
        "official": lambda output: extract_code(output, LMStyle.OpenAIChat),
    }[code_extraction]
    code_lists = [
        [extract(output) for output in record["output_list"]]
        for record in records
    ]
    details = [
        {"question_id": question_id, "code_list": code_lists[index]}
        for index, question_id in enumerate(question_ids)
    ]
    scores = {
        "num_questions": len(records),
        "num_generations": len(records) * n,
    }
    # Score each group independently: upstream stops at the first failed test.
    for test_group in ("all", "public", "private"):
        print(f"Scoring {test_group} tests", flush=True)
        metrics, results, metadata = codegen_metrics(
            select_test_samples(samples, test_group),
            code_lists,
            k_list=sorted({1, n}) if test_group == "all" else [1],
            num_process_evaluate=num_process_evaluate,
            timeout=timeout,
        )
        grades = extract_instance_results(results)
        if test_group == "all":
            scores.update({f"pass@{k}": float(metrics[f"pass@{k}"]) for k in sorted({1, n})})
        else:
            scores[f"{test_group}_pass@1"] = float(metrics["pass@1"])
        metric_name = "pass@1" if test_group == "all" else f"{test_group}_pass@1"
        scores.update(sample_pass_rates(grades, metric_name))
        for index, detail in enumerate(details):
            evaluation = {
                "graded_list": grades[index],
                "test_results": results[index],
                "metadata": [json.loads(entry) for entry in metadata[index]],
            }
            if test_group == "all":
                detail.update(evaluation)
            else:
                detail[test_group] = evaluation
    return scores, details


def run_score(args) -> Path:
    generations = json.loads(args.samples.read_text(encoding="utf-8"))
    metrics, details = score_generations(
        generations, args.num_process_evaluate, args.timeout, args.code_extraction
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    details_path = args.output_dir / f"{TASK}_eval_{args.run_id}.json"
    metrics_path = args.output_dir / f"{TASK}_metrics_{args.run_id}.json"
    write_json(details_path, details)
    write_json(
        metrics_path,
        {
            "task": TASK,
            "results": {TASK: metrics},
            "config": {
                "generation": generations["config"],
                "official_commit": official_commit(),
                "num_process_evaluate": args.num_process_evaluate,
                "timeout": args.timeout,
                "code_extraction": args.code_extraction,
            },
            "samples_path": str(args.samples.resolve()),
            "details_path": str(details_path.resolve()),
        },
    )
    print(json.dumps(metrics, indent=2))
    print(f"Metrics: {metrics_path}")
    return metrics_path


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="Generate and save model answers")
    score = commands.add_parser("score", help="Score saved answers with the official evaluator")
    for command in (generate, score):
        command.add_argument("--output_dir", type=Path, required=True)
        command.add_argument("--run_id", default=datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    generate.add_argument("--model_name_or_path", required=True)
    generate.add_argument("--tokenizer_name_or_path")
    generate.add_argument("--tensor_parallel_size", type=int, default=1)
    generate.add_argument("--data_parallel_size", type=int, default=1)
    generate.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    generate.add_argument("--max_gen_toks", type=int, default=4096)
    generate.add_argument("--max_model_len", type=int, default=32768)
    generate.add_argument("--n", type=int, default=3)
    generate.add_argument("--temperature", type=float, default=0.6)
    generate.add_argument("--top_p", type=float, default=0.95)
    generate.add_argument("--seed", type=int, default=1234)
    generate.add_argument("--limit", type=int)
    score.add_argument("--samples", type=Path, required=True)
    score.add_argument("--num_process_evaluate", type=int, default=16)
    score.add_argument("--timeout", type=int, default=6)
    score.add_argument(
        "--code_extraction", choices=("last_python", "official"), default="last_python",
        help="Default: last_python (last complete Python block, stripping only surrounding whitespace); "
        "official uses the upstream extraction rule.",
    )
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if args.command == "generate":
        run_generate(args)
    else:
        run_score(args)


if __name__ == "__main__":
    main()
