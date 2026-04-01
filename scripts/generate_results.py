#!/usr/bin/env python3
"""
Generate results.csv by running verification on all output files.
"""

import os
import sys
import csv
import re
from pathlib import Path

# Add inference to path for verify_solutions
sys.path.insert(0, str(Path(__file__).parent / "inference"))

from verify_solutions import verify_file

# Output directories to scan
OUTPUT_BASE = "/scratch/dkhasha1/tli104/outputs"
OUTPUT_DIRS = [
    "imobench_accumulate",
    "answerbench",
    "hmmt_feb_2025_boxed",
    "hmmt_feb_2025",
]

# Default model (as specified in context_refinement_vllm.py)
DEFAULT_MODEL = "Qwen/Qwen3-4B"


def parse_filename(filename: str, dirname: str) -> dict:
    """Parse filename to extract parameters."""
    # Pattern: output_t{tokens}_r{rounds}[_acc].jsonl
    match = re.match(r'output_t(\d+)_r(\d+)(_acc)?\.jsonl', filename)
    if not match:
        return None

    num_tokens = int(match.group(1))
    rounds = int(match.group(2))
    accumulate = match.group(3) is not None

    # Extract dataset from directory name
    dataset = dirname.replace("_accumulate", "").replace("_boxed", "")

    return {
        "num_tokens": num_tokens,
        "rounds": rounds,
        "accumulate": accumulate,
        "dataset": dataset,
    }


def main():
    results = []

    for dir_name in OUTPUT_DIRS:
        dir_path = Path(OUTPUT_BASE) / dir_name
        if not dir_path.exists():
            print(f"Skipping {dir_path} (not found)")
            continue

        jsonl_files = sorted(dir_path.glob("*.jsonl"))
        print(f"\nProcessing {dir_name}: {len(jsonl_files)} files")

        for file_path in jsonl_files:
            params = parse_filename(file_path.name, dir_name)
            if params is None:
                print(f"  Skipping {file_path.name} (could not parse filename)")
                continue

            print(f"  Evaluating {file_path.name}...", end=" ")

            try:
                stats = verify_file(str(file_path), verbose=False)
                total = stats["total"]
                correct = stats["correct"]
                accuracy = (correct / total * 100) if total > 0 else 0

                print(f"{correct}/{total} = {accuracy:.2f}%")

                # Build sampling params string
                sampling_params = f"tokens={params['num_tokens']}, rounds={params['rounds']}"
                if params["accumulate"]:
                    sampling_params += ", accumulate=True"

                results.append({
                    "model": DEFAULT_MODEL,
                    "dataset": params["dataset"],
                    "sampling_params": sampling_params,
                    "performance": f"{accuracy:.2f}%",
                    "correct": correct,
                    "total": total,
                    "num_tokens": params["num_tokens"],
                    "rounds": params["rounds"],
                    "accumulate": params["accumulate"],
                })
            except Exception as e:
                print(f"ERROR: {e}")

    # Sort results
    results.sort(key=lambda x: (x["dataset"], x["num_tokens"], x["rounds"]))

    # Write to CSV
    csv_path = Path(__file__).parent / "results.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "dataset", "sampling_params", "performance"
        ])
        writer.writeheader()
        for row in results:
            writer.writerow({
                "model": row["model"],
                "dataset": row["dataset"],
                "sampling_params": row["sampling_params"],
                "performance": row["performance"],
            })

    print(f"\n{'='*80}")
    print(f"Results saved to: {csv_path}")
    print(f"{'='*80}")

    # Print summary table
    print(f"\n{'='*100}")
    print(f"{'Model':<20} {'Dataset':<20} {'Sampling Params':<40} {'Performance':<10}")
    print(f"{'='*100}")
    for row in results:
        print(f"{row['model']:<20} {row['dataset']:<20} {row['sampling_params']:<40} {row['performance']:<10}")
    print(f"{'='*100}")

    return results


if __name__ == "__main__":
    main()
