#!/usr/bin/env python3
"""
Cumulative verification by round for iterative refinement.

For each round R, evaluates every sample using its latest generation
up to round R. Samples that finished before R keep their last round's result.
This shows how overall accuracy evolves as rounds progress.
"""

import argparse
import json
import sys
import warnings
import multiprocessing
import os
from collections import defaultdict
from typing import List, Dict, Tuple

warnings.filterwarnings("ignore")

try:
    import logging
    logging.getLogger().setLevel(logging.ERROR)
    from math_verify import parse, verify
except ImportError:
    print("Error: math-verify not installed")
    exit(1)

DEFAULT_TIMEOUT = 5


def _verify_single(args):
    gold_answer, generated_text = args
    try:
        gold_text = f"${gold_answer}$" if not gold_answer.startswith('$') else gold_answer
        gold_parsed = parse(gold_text)
        if not gold_parsed:
            return False
        answer_parsed = parse(generated_text)
        if not answer_parsed:
            return False
        return verify(gold_parsed, answer_parsed)
    except:
        return False


def _pool_init():
    warnings.filterwarnings("ignore")
    import logging
    logging.disable(logging.CRITICAL)
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')


def verify_batch(items, timeout=DEFAULT_TIMEOUT, num_workers=None):
    if not items:
        return []
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)
    results = [False] * len(items)
    with multiprocessing.Pool(num_workers, initializer=_pool_init, maxtasksperchild=10) as pool:
        async_results = [(i, pool.apply_async(_verify_single, (item,))) for i, item in enumerate(items)]
        for i, ar in async_results:
            try:
                results[i] = ar.get(timeout=timeout)
            except:
                results[i] = False
    return results


def compute_pass_at_k(n, c, k):
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
    parser = argparse.ArgumentParser(description="Cumulative verification by round")
    parser.add_argument("input", type=str, help="Input JSONL file")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    args = parser.parse_args()

    data = []
    with open(args.input, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    print(f"Loaded {len(data)} problems")

    # Determine max rounds
    max_rounds = 0
    for item in data:
        for sample in item.get("samples", []):
            max_rounds = max(max_rounds, len(sample.get("rounds", [])))
    print(f"Max rounds: {max_rounds}")

    # For each round, collect the "latest generation up to that round" for every sample.
    # First, verify ALL round generations in one big batch to avoid redundant work.
    all_tasks = []  # (gold, text)
    task_index = []  # (prob_idx, sample_idx, round_idx)

    for prob_idx, item in enumerate(data):
        gold = item.get("answer", "")
        for sample_idx, sample in enumerate(item.get("samples", [])):
            for round_idx, rd in enumerate(sample.get("rounds", [])):
                gen = rd.get("current_round_generation", "")
                if gen and gold:
                    all_tasks.append((gold, gen))
                    task_index.append((prob_idx, sample_idx, round_idx))

    print(f"Verifying {len(all_tasks)} total (problem, sample, round) generations...", end=" ", flush=True)
    all_results = verify_batch(all_tasks, timeout=args.timeout)
    print("done")

    # Build lookup: correct[prob_idx][sample_idx][round_idx] = bool
    correct = defaultdict(lambda: defaultdict(dict))
    for (prob_idx, sample_idx, round_idx), is_correct in zip(task_index, all_results):
        correct[prob_idx][sample_idx][round_idx] = is_correct

    # For each sample, determine its last round index
    last_round = {}  # (prob_idx, sample_idx) -> last_round_idx
    for prob_idx, item in enumerate(data):
        for sample_idx, sample in enumerate(item.get("samples", [])):
            num_rounds = len(sample.get("rounds", []))
            last_round[(prob_idx, sample_idx)] = num_rounds - 1 if num_rounds > 0 else -1

    num_problems = len(data)
    num_samples = len(data[0].get("samples", [])) if data else 0
    k_values = [k for k in [1, 2, 4, 8, 16] if k <= num_samples]

    # Cumulative evaluation: at round R, each sample uses min(R, its_last_round)
    print(f"\n{'Round':<8}", end="")
    for k in k_values:
        print(f"{'pass@'+str(k):<10}", end="")
    print()
    print("-" * (8 + 10 * len(k_values)))

    for eval_round in range(max_rounds):
        # For each problem, count correct samples
        problem_correct = defaultdict(int)
        problem_total = defaultdict(int)

        for prob_idx in range(num_problems):
            for sample_idx in range(num_samples):
                lr = last_round.get((prob_idx, sample_idx), -1)
                if lr < 0:
                    continue
                # Use the latest round up to eval_round
                use_round = min(eval_round, lr)
                is_correct = correct[prob_idx][sample_idx].get(use_round, False)
                problem_total[prob_idx] += 1
                if is_correct:
                    problem_correct[prob_idx] += 1

        # Compute pass@k
        pass_at_k = {}
        for k in k_values:
            total = count = 0
            for prob_idx in range(num_problems):
                n = problem_total[prob_idx]
                c = problem_correct[prob_idx]
                if n >= k:
                    total += compute_pass_at_k(n, c, k)
                    count += 1
            pass_at_k[k] = (total / count * 100) if count > 0 else 0.0

        print(f"R{eval_round + 1:<7}", end="")
        for k in k_values:
            print(f"{pass_at_k[k]:<10.2f}", end="")
        print()


if __name__ == "__main__":
    main()
