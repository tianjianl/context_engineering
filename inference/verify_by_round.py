#!/usr/bin/env python3
"""
Verify solutions by round for refinement experiments.
"""

import argparse
import json
from collections import defaultdict

from inference.data_utils import load_jsonl
from inference.verify_utils import verify_batch, compute_pass_at_k, DEFAULT_VERIFY_TIMEOUT


def main():
    parser = argparse.ArgumentParser(description="Verify solutions by round")
    parser.add_argument("input", type=str, help="Input JSONL file")
    parser.add_argument("--timeout", type=float, default=DEFAULT_VERIFY_TIMEOUT)
    args = parser.parse_args()

    # Load data
    data = load_jsonl(args.input)
    print(f"Loaded {len(data)} problems")

    # Determine number of rounds (scan all samples)
    num_rounds = 0
    for item in data:
        for sample in item.get("samples", []):
            num_rounds = max(num_rounds, len(sample.get("rounds", [])))

    print(f"Number of rounds: {num_rounds}")

    # Collect verification tasks for each round
    round_tasks = defaultdict(list)
    round_task_map = defaultdict(list)  # (problem_idx, sample_idx)

    for prob_idx, item in enumerate(data):
        gold_answer = item.get("answer", "")
        samples = item.get("samples", [])

        for sample_idx, sample in enumerate(samples):
            rounds = sample.get("rounds", [])
            for round_idx, round_data in enumerate(rounds):
                generation = round_data.get("current_round_generation", "")
                if generation and gold_answer:
                    round_tasks[round_idx].append((gold_answer, generation))
                    round_task_map[round_idx].append((prob_idx, sample_idx))

    # Verify each round
    round_results = {}
    for round_idx in range(num_rounds):
        tasks = round_tasks[round_idx]
        print(f"Verifying round {round_idx + 1}: {len(tasks)} samples...", end=" ", flush=True)
        results = verify_batch(tasks, timeout=args.timeout)
        round_results[round_idx] = results
        print("done")

    # Compute per-round statistics
    num_samples_per_problem = len(data[0].get("samples", [])) if data else 0
    k_values = [k for k in [1, 2, 4, 8, 16] if k <= num_samples_per_problem]

    print(f"\n{'Round':<8}", end="")
    for k in k_values:
        print(f"{'pass@'+str(k):<10}", end="")
    print()
    print("-" * (8 + 10 * len(k_values)))

    for round_idx in range(num_rounds):
        results = round_results[round_idx]
        task_map = round_task_map[round_idx]

        # Group results by problem
        problem_correct_counts = defaultdict(int)
        problem_total_counts = defaultdict(int)

        for (prob_idx, sample_idx), (is_correct, status, _) in zip(task_map, results):
            problem_total_counts[prob_idx] += 1
            if is_correct:
                problem_correct_counts[prob_idx] += 1

        # Compute pass@k for each problem and average
        pass_at_k_sums = {k: 0.0 for k in k_values}
        pass_at_k_counts = {k: 0 for k in k_values}  # Count problems with enough samples

        for prob_idx in range(len(data)):
            n = problem_total_counts[prob_idx]
            c = problem_correct_counts[prob_idx]

            for k in k_values:
                if n >= k:
                    pass_at_k_sums[k] += compute_pass_at_k(n, c, k)
                    pass_at_k_counts[k] += 1

        # Average across problems with enough samples
        pass_at_k_avg = {k: (pass_at_k_sums[k] / pass_at_k_counts[k] * 100) if pass_at_k_counts[k] > 0 else 0.0 for k in k_values}

        print(f"R{round_idx + 1:<7}", end="")
        for k in k_values:
            print(f"{pass_at_k_avg[k]:<10.2f}", end="")
        # Show sample counts for this round
        total_samples = sum(problem_total_counts.values())
        print(f"  (n={total_samples})")


if __name__ == "__main__":
    main()
