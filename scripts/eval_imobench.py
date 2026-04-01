#!/usr/bin/env python3
"""
Evaluate all IMOBench outputs and generate results CSV.

Usage:
    python eval_imobench.py
"""

import csv
import re
import sys
from pathlib import Path

# Add inference directory to path
sys.path.insert(0, str(Path(__file__).parent / "inference"))
from verify_solutions import verify_file

# Output directories to scan
OUTPUT_DIRS = [
    "/scratch/dkhasha1/tli104/outputs/imobench_accumulate",
    "/scratch/dkhasha1/tli104/outputs/imobench_qwen3_instruct",
    "/scratch/dkhasha1/tli104/outputs/imobench_glm47_flash",
    "/scratch/dkhasha1/tli104/outputs/imobench_inference",
]

# Model name mapping from directory names
DIR_TO_MODEL = {
    "imobench_accumulate": "Qwen3-4B",
    "imobench_qwen3_instruct": "Qwen3-4B-Instruct",
    "imobench_glm47_flash": "GLM-4-9B-Chat-Flash",
    "imobench_inference": "Qwen3-4B",  # Default, may be overridden by filename
}


def parse_filename(filename: str, dir_name: str) -> dict:
    """
    Parse output filename to extract parameters.

    Examples:
        output_t4096_r1_acc.jsonl -> tokens=4096, rounds=1
        output_t8192_n16_r3.jsonl -> tokens=8192, rounds=3, n_samples=16
        output_t16384_n16_r3_Qwen3-4B-Instruct-2507.jsonl -> with model name
    """
    result = {
        "model": DIR_TO_MODEL.get(dir_name, "Unknown"),
        "max_tokens": None,
        "rounds": None,
        "n_samples": None,
    }

    # Extract tokens (t followed by digits)
    tokens_match = re.search(r't(\d+)', filename)
    if tokens_match:
        result["max_tokens"] = int(tokens_match.group(1))

    # Extract rounds (r followed by digits)
    rounds_match = re.search(r'r(\d+)', filename)
    if rounds_match:
        result["rounds"] = int(rounds_match.group(1))

    # Extract n_samples (n followed by digits)
    n_match = re.search(r'n(\d+)', filename)
    if n_match:
        result["n_samples"] = int(n_match.group(1))

    # Check for model name in filename (e.g., Qwen3-4B-Instruct-2507)
    if "Qwen3-4B-Instruct" in filename:
        result["model"] = "Qwen3-4B-Instruct"
    elif "GLM" in filename:
        result["model"] = "GLM-4-9B-Chat-Flash"

    return result


def main():
    results = []

    # Find all JSONL files
    all_files = []
    for output_dir in OUTPUT_DIRS:
        dir_path = Path(output_dir)
        if dir_path.exists():
            for jsonl_file in dir_path.glob("*.jsonl"):
                all_files.append((jsonl_file, dir_path.name))

    if not all_files:
        print("No JSONL files found!")
        return

    print(f"Found {len(all_files)} files to evaluate")
    print("=" * 80)

    for file_path, dir_name in sorted(all_files):
        print(f"\nEvaluating: {file_path.name}")

        # Parse filename for parameters
        params = parse_filename(file_path.name, dir_name)

        # Run verification
        try:
            stats = verify_file(str(file_path), verbose=False, timeout=60)
            pass_at_1 = stats.get("pass_at_k", {}).get(1, 0.0)
            total = stats.get("total", 0)
            timeout_count = stats.get("timeout", 0)
        except Exception as e:
            print(f"  Error: {e}")
            pass_at_1 = 0.0
            total = 0
            timeout_count = 0

        # Calculate max refinement tokens (total - base tokens for rounds > 1)
        # Refinement tokens = max_tokens * (rounds - 1) for accumulate mode
        max_tokens = params["max_tokens"] or 0
        rounds = params["rounds"] or 1
        max_refinement_tokens = max_tokens * (rounds - 1) if rounds > 1 else 0

        result = {
            "model": params["model"],
            "max_tokens": max_tokens,
            "max_refinement_tokens": max_refinement_tokens,
            "rounds": rounds,
            "pass_at_1": pass_at_1,
            "total_problems": total,
            "timeouts": timeout_count,
            "filename": file_path.name,
        }
        results.append(result)

        print(f"  Model: {result['model']}, Tokens: {max_tokens}, Rounds: {rounds}, pass@1: {pass_at_1:.2f}%")

    # Sort results
    results.sort(key=lambda x: (x["model"], x["max_tokens"], x["rounds"]))

    # Write CSV
    output_csv = Path(__file__).parent / "imobench_results.csv"
    fieldnames = ["model", "max_tokens", "max_refinement_tokens", "rounds", "pass_at_1", "total_problems", "timeouts", "filename"]

    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 80)
    print(f"Results saved to: {output_csv}")
    print("=" * 80)

    # Print summary table
    print(f"\n{'Model':<25} {'Tokens':<10} {'Refine':<10} {'Rounds':<8} {'pass@1':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r['model']:<25} {r['max_tokens']:<10} {r['max_refinement_tokens']:<10} {r['rounds']:<8} {r['pass_at_1']:<10.2f}")


if __name__ == "__main__":
    main()
