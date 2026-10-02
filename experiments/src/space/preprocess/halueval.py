"""Create a fixed HaluEval 3k set: 1,000 random rows from each sample subset."""

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset


DATASET = "pminervini/HaluEval"
REVISION = "12a856119f03975a94509091e8cada3e6be6ead7"
SUBSETS = ("qa_samples", "dialogue_samples", "summarization_samples")
SEED = 42
SAMPLES_PER_SUBSET = 1000


def main() -> None:
    output_dir = Path("data/halueval/3k")
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    manifest = {
        "dataset": DATASET,
        "revision": REVISION,
        "split": "data",
        "sampling": "random.Random.sample without replacement, in subset order",
        "seed": SEED,
        "samples_per_subset": SAMPLES_PER_SUBSET,
        "subsets": {},
    }
    for subset in SUBSETS:
        source = load_dataset(DATASET, subset, revision=REVISION, split="data")
        indices = rng.sample(range(len(source)), SAMPLES_PER_SUBSET)
        sampled = source.select(indices)
        output = output_dir / f"{subset}.parquet"
        sampled.to_parquet(output)
        counts = dict(Counter(sampled["hallucination"]))
        manifest["subsets"][subset] = {
            "source_rows": len(source),
            "rows": len(sampled),
            "source_row_indices": indices,
            "label_counts": counts,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        }
        print(f"Wrote {len(sampled)} rows to {output}; labels: {counts}")
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
