#!/usr/bin/env python3
"""Analyze prefix recovery results across all models.

Loads recovery output files and computes recovery rates, with
per-category breakdowns if category data is available.

Usage:
    python -m inference.analyze_prefix_recovery \
        --files recovery_qwen3_4b.jsonl recovery_qwen3_30b.jsonl \
               recovery_gemini3flash.jsonl recovery_minimax.jsonl \
               recovery_deepseekv3.jsonl \
        --baseline_files baseline_qwen3_4b.jsonl baseline_qwen3_30b.jsonl \
                        baseline_gemini3flash.jsonl baseline_minimax.jsonl \
                        baseline_deepseekv3.jsonl
"""

import argparse
from collections import defaultdict

from inference.data_utils import load_jsonl


def analyze_one(recovery_file: str, baseline_file: str = None):
    """Analyze a single model's recovery results."""
    data = load_jsonl(recovery_file)
    if not data:
        return None

    model = data[0].get("model", "unknown")
    total_incorrect = len(data)
    num_recovered = sum(1 for item in data if item.get("recovered", False))

    # Per-category breakdown
    cat_stats = defaultdict(lambda: {"total": 0, "recovered": 0})
    for item in data:
        cat = item.get("category", "unknown")
        cat_stats[cat]["total"] += 1
        if item.get("recovered", False):
            cat_stats[cat]["recovered"] += 1

    # Baseline total (if provided)
    baseline_total = None
    if baseline_file:
        baseline_data = load_jsonl(baseline_file)
        baseline_total = len(baseline_data)

    return {
        "model": model,
        "file": recovery_file,
        "baseline_total": baseline_total,
        "total_incorrect": total_incorrect,
        "num_recovered": num_recovered,
        "recovery_rate": num_recovered / total_incorrect if total_incorrect > 0 else 0,
        "categories": dict(cat_stats),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze prefix recovery results across models"
    )
    parser.add_argument("--files", nargs="+", required=True,
                        help="Recovery output JSONL files")
    parser.add_argument("--baseline_files", nargs="*", default=None,
                        help="Corresponding baseline JSONL files (for total counts)")
    args = parser.parse_args()

    baseline_files = args.baseline_files or [None] * len(args.files)
    if len(baseline_files) < len(args.files):
        baseline_files.extend([None] * (len(args.files) - len(baseline_files)))

    results = []
    for rf, bf in zip(args.files, baseline_files):
        r = analyze_one(rf, bf)
        if r:
            results.append(r)

    if not results:
        print("No results to analyze.")
        return

    # Print summary table
    print(f"\n{'='*90}")
    print(f"Prefix Recovery Experiment — Summary")
    print(f"{'='*90}")
    print(f"{'Model':<40} {'Baseline':>8} {'Incorrect':>9} {'Recovered':>9} {'Rate':>8}")
    print(f"{'-'*40} {'-'*8} {'-'*9} {'-'*9} {'-'*8}")

    for r in sorted(results, key=lambda x: x["recovery_rate"]):
        bt = str(r["baseline_total"]) if r["baseline_total"] else "?"
        print(f"{r['model']:<40} {bt:>8} {r['total_incorrect']:>9} "
              f"{r['num_recovered']:>9} {r['recovery_rate']:>7.1%}")

    # Per-category breakdown
    all_cats = set()
    for r in results:
        all_cats.update(r["categories"].keys())
    all_cats.discard("unknown")

    if all_cats:
        print(f"\n{'='*90}")
        print(f"Per-Category Recovery Rates")
        print(f"{'='*90}")

        # Short model names
        short_names = []
        for r in results:
            name = r["model"].split("/")[-1]
            if len(name) > 20:
                name = name[:17] + "..."
            short_names.append(name)

        header = f"{'Category':<20}"
        for name in short_names:
            header += f" {name:>20}"
        print(header)
        print("-" * len(header))

        for cat in sorted(all_cats):
            row = f"{cat:<20}"
            for r in results:
                cs = r["categories"].get(cat, {"total": 0, "recovered": 0})
                if cs["total"] > 0:
                    rate = cs["recovered"] / cs["total"]
                    row += f" {cs['recovered']}/{cs['total']} ({rate:.0%}):>20"
                else:
                    row += f" {'—':>20}"
            print(row)

    print()


if __name__ == "__main__":
    main()
