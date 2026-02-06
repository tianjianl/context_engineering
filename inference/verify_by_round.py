#!/usr/bin/env python3
"""
Verify solutions by round for refinement experiments.
"""

import argparse
import json
import sys
import warnings
import multiprocessing
import os
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

warnings.filterwarnings("ignore")

try:
    import logging
    logging.getLogger().setLevel(logging.ERROR)
    from math_verify import parse, verify
except ImportError:
    print("Error: math-verify not installed")
    exit(1)

DEFAULT_TIMEOUT = 5


def _verify_single(args: Tuple[str, str]) -> Tuple[bool, str]:
    """Worker function for single verification."""
    gold_answer, generated_text = args
    try:
        gold_text = f"${gold_answer}$" if not gold_answer.startswith('$') else gold_answer
        gold_parsed = parse(gold_text)
        if not gold_parsed:
            return (False, "gold_parse_failed")

        answer_parsed = parse(generated_text)
        if not answer_parsed:
            return (False, "answer_parse_failed")

        is_correct = verify(gold_parsed, answer_parsed)
        return (is_correct, "verified" if is_correct else "incorrect")
    except Exception as e:
        return (False, f"error: {str(e)[:50]}")


def _pool_init():
    """Initialize pool worker - suppress output."""
    warnings.filterwarnings("ignore")
    import logging
    logging.disable(logging.CRITICAL)
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')


def verify_batch(items: List[Tuple[str, str]], timeout: float = DEFAULT_TIMEOUT, num_workers: int = 8) -> List[Tuple[bool, str]]:
    """Verify a batch of (gold_answer, generated_text) pairs in parallel."""
    if not items:
        return []

    results = [(False, "timeout")] * len(items)

    with multiprocessing.Pool(num_workers, initializer=_pool_init, maxtasksperchild=10) as pool:
        async_results = []
        for i, item in enumerate(items):
            ar = pool.apply_async(_verify_single, (item,))
            async_results.append((i, ar))

        for i, ar in async_results:
            try:
                results[i] = ar.get(timeout=timeout)
            except multiprocessing.TimeoutError:
                results[i] = (False, "timeout")
            except Exception as e:
                results[i] = (False, f"error: {str(e)[:30]}")

    return results


def compute_pass_at_k(n: int, c: int, k: int) -> float:
    """Compute unbiased pass@k estimate."""
    if n < k:
        return 1.0 if c > 0 else 0.0
    if c == 0:
        return 0.0
    if c >= n:
        return 1.0

    result = 1.0
    for i in range(k):
        result *= (n - c - i) / (n - i)
    return 1.0 - result


def main():
    parser = argparse.ArgumentParser(description="Verify solutions by round")
    parser.add_argument("input", type=str, help="Input JSONL file")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    args = parser.parse_args()

    # Load data
    data = []
    with open(args.input, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    print(f"Loaded {len(data)} problems")

    # Determine number of rounds
    num_rounds = 0
    for item in data:
        samples = item.get("samples", [])
        if samples:
            rounds = samples[0].get("rounds", [])
            num_rounds = max(num_rounds, len(rounds))
            break

    print(f"Number of rounds: {num_rounds}")

    # Collect verification tasks for each round
    round_tasks = {r: [] for r in range(num_rounds)}
    round_task_map = {r: [] for r in range(num_rounds)}  # (problem_idx, sample_idx)

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

        for (prob_idx, sample_idx), (is_correct, status) in zip(task_map, results):
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
