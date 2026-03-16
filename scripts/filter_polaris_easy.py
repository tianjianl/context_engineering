#!/usr/bin/env python3
"""
Filter easy problems from Polaris-53K baseline generation outputs.

Reads all chunk output files, verifies each sample against ground truth,
and removes problems where >= 7/8 samples are correct (too easy).
Outputs a filtered dataset JSONL and filtering statistics.
"""

import argparse
import json
import os
import sys
from collections import defaultdict, Counter
from pathlib import Path

# Add inference directory to path for verify_solutions imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "inference"))
from verify_solutions import verify_batch


INPUT_DIR = "/scratch/dkhasha1/tli104/outputs/polaris_53k_baseline"
DATASET_DIR = "/scratch/dkhasha1/tli104/datasets/polaris_53k"
NUM_CHUNKS = 10
EASY_THRESHOLD = 7  # Remove problems with >= this many correct out of 8


def main():
    parser = argparse.ArgumentParser(description="Filter easy problems from Polaris-53K baseline outputs")
    parser.add_argument("--input_dir", type=str, default=INPUT_DIR, help="Directory with chunk output files")
    parser.add_argument("--dataset_dir", type=str, default=DATASET_DIR, help="Directory with original dataset")
    parser.add_argument("--threshold", type=int, default=EASY_THRESHOLD, help="Remove problems with >= this many correct (default: 7)")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of verification workers")
    args = parser.parse_args()

    # Load all chunk outputs
    all_samples = []
    for chunk_idx in range(NUM_CHUNKS):
        pattern = f"baseline_qwen3-4b-instruct-2507_polaris53k_t16384_n8_temp1.0_topp1.0_chunk{chunk_idx:02d}.jsonl"
        chunk_path = os.path.join(args.input_dir, pattern)
        if not os.path.exists(chunk_path):
            print(f"WARNING: Missing chunk file: {chunk_path}")
            continue
        with open(chunk_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_samples.append(json.loads(line))
    print(f"Loaded {len(all_samples)} total samples from output files")

    # Group by problem_id
    problems = defaultdict(list)
    for sample in all_samples:
        problems[sample["problem_id"]].append(sample)
    print(f"Found {len(problems)} unique problems")

    # Verify all samples
    print("Verifying all samples...")
    verify_items = [(s["answer"], s["generation"]) for s in all_samples]

    BATCH_SIZE = 10000
    all_results = []
    for batch_start in range(0, len(verify_items), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(verify_items))
        batch = verify_items[batch_start:batch_end]
        print(f"  Verifying batch {batch_start}-{batch_end} ({len(batch)} samples)...")
        results = verify_batch(batch, num_workers=args.num_workers)
        all_results.extend(results)
    print(f"Verification complete: {len(all_results)} results")

    # Map results back to samples
    for i, sample in enumerate(all_samples):
        is_correct, status, parsed = all_results[i]
        sample["_correct"] = is_correct

    # Count correct per problem
    correct_counts = {}
    for pid, samples in problems.items():
        correct_counts[pid] = sum(1 for s in samples if s["_correct"])

    # Distribution of correct counts
    dist = Counter(correct_counts.values())
    print("\nCorrect count distribution:")
    for k in sorted(dist.keys()):
        print(f"  {k}/8 correct: {dist[k]} problems")

    # Filter: remove problems with >= threshold correct
    kept_pids = set()
    removed_pids = set()
    for pid, count in correct_counts.items():
        if count >= args.threshold:
            removed_pids.add(pid)
        else:
            kept_pids.add(pid)

    print(f"\nFiltering threshold: >= {args.threshold}/8 correct")
    print(f"Removed: {len(removed_pids)} problems (too easy)")
    print(f"Kept: {len(kept_pids)} problems")

    # Load original dataset and write filtered version
    full_dataset_path = os.path.join(args.dataset_dir, "polaris_53k.jsonl")
    filtered_path = os.path.join(args.dataset_dir, "polaris_53k_filtered.jsonl")
    kept_count = 0
    with open(full_dataset_path, "r") as fin, open(filtered_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            pid = item["problem_id"]
            if pid in kept_pids:
                # Add correctness info
                item["num_correct"] = correct_counts.get(pid, 0)
                item["num_samples"] = 8
                fout.write(json.dumps(item) + "\n")
                kept_count += 1
    print(f"Wrote {kept_count} problems to {filtered_path}")

    # Save stats
    stats = {
        "total_problems": len(problems),
        "total_samples": len(all_samples),
        "threshold": args.threshold,
        "removed": len(removed_pids),
        "kept": len(kept_pids),
        "correct_count_distribution": {str(k): dist[k] for k in sorted(dist.keys())},
    }
    stats_path = os.path.join(args.input_dir, "filtering_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved to {stats_path}")


if __name__ == "__main__":
    main()
