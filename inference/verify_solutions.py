#!/usr/bin/env python3
"""
Solution Verification Script using Math-Verify

This script reads output JSONL files from context refinement and verifies
the solutions against ground truth answers using the math-verify package.
"""

import argparse
import json
import math
import os
import sys
import warnings
import multiprocessing
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

# Suppress warnings
warnings.filterwarnings("ignore")

# Default timeout for verification operations (in seconds)
DEFAULT_VERIFY_TIMEOUT = 5

try:
    # Suppress math-verify import warnings
    import logging
    logging.getLogger().setLevel(logging.ERROR)
    from math_verify import parse, verify
    MATH_VERIFY_AVAILABLE = True
except ImportError:
    MATH_VERIFY_AVAILABLE = False
    print("Error: math-verify not installed. Install with: pip install 'math-verify[antlr4_13_2]'")
    exit(1)


def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def _verify_single(args: Tuple[str, str]) -> Tuple[bool, str, Optional[str]]:
    """Worker function for single verification."""
    gold_answer, generated_text = args
    try:
        # Parse gold answer
        gold_text = f"${gold_answer}$" if not gold_answer.startswith('$') else gold_answer
        gold_parsed = parse(gold_text)
        if not gold_parsed:
            return (False, "gold_parse_failed", None)

        # Parse generated text
        answer_parsed = parse(generated_text)
        if not answer_parsed:
            return (False, "answer_parse_failed", None)

        parsed_answer_str = str(answer_parsed) if answer_parsed else None
        is_correct = verify(gold_parsed, answer_parsed)
        return (is_correct, "verified" if is_correct else "incorrect", parsed_answer_str)

    except Exception as e:
        return (False, f"error: {str(e)[:50]}", None)


def _pool_init():
    """Initialize pool worker - suppress output."""
    import warnings
    import logging
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')


def verify_batch(items: List[Tuple[str, str]], timeout: float = DEFAULT_VERIFY_TIMEOUT, num_workers: int = 4) -> List[Tuple[bool, str, Optional[str]]]:
    """
    Verify a batch of (gold_answer, generated_text) pairs in parallel.
    Uses multiprocessing.Pool with maxtasksperchild to handle stuck workers.
    """
    if not items:
        return []

    results = [(False, "timeout", None)] * len(items)

    # Use pool with task limit to restart hung workers
    with multiprocessing.Pool(num_workers, initializer=_pool_init, maxtasksperchild=10) as pool:
        async_results = []
        for i, item in enumerate(items):
            ar = pool.apply_async(_verify_single, (item,))
            async_results.append((i, ar))

        for i, ar in async_results:
            try:
                results[i] = ar.get(timeout=timeout)
            except multiprocessing.TimeoutError:
                results[i] = (False, "timeout", None)
            except Exception as e:
                results[i] = (False, f"error: {str(e)[:30]}", None)

    return results


def verify_answer(gold_answer: str, generated_text: str, verbose: bool = False, timeout: float = DEFAULT_VERIFY_TIMEOUT) -> Tuple[bool, str, Optional[str]]:
    """
    Verify if the generated text contains the correct answer.
    Single-item wrapper around verify_batch.
    """
    results = verify_batch([(gold_answer, generated_text)], timeout=timeout, num_workers=1)
    return results[0] if results else (False, "error", None)


def compute_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Compute unbiased pass@k estimate.

    Args:
        n: Total number of samples
        c: Number of correct samples
        k: Number of samples to consider

    Returns:
        Probability that at least one of k samples is correct.
        Uses the formula: 1 - C(n-c, k) / C(n, k)
    """
    if n < k:
        return 1.0 if c > 0 else 0.0
    if c == 0:
        return 0.0
    if c >= n:
        return 1.0

    # Compute using log to avoid overflow
    # pass@k = 1 - prod_{i=0}^{k-1} (n-c-i)/(n-i)
    result = 1.0
    for i in range(k):
        result *= (n - c - i) / (n - i)
    return 1.0 - result


def get_text_from_sample(sample: Dict) -> Tuple[str, str]:
    """Extract generated text from a sample dict. Returns (text, source_name)."""
    # Check current_round_generation FIRST - this contains the actual solution with \boxed{}
    # The refined context fields do NOT contain the final answer by design
    text_sources = []

    rounds = sample.get("rounds", [])
    if rounds:
        last_round = rounds[-1]
        text_sources.append(
            ("last_round_generation", last_round.get("current_round_generation", ""))
        )

    # Fall back to these only if no rounds exist
    text_sources.extend([
        ("full_assistant_message", sample.get("full_assistant_message", "")),
        ("final_refined_context", sample.get("final_refined_context", "")),
        ("generation", sample.get("generation", "")),
    ])

    for source_name, text in text_sources:
        if text and text.strip():
            return text, source_name

    return "", None


def verify_file(file_path: str, verbose: bool = False, timeout: float = DEFAULT_VERIFY_TIMEOUT, quiet: bool = False, num_workers: int = None) -> Dict:
    """
    Verify all solutions in a JSONL file using batch processing.
    Supports multi-sample format: averages correctness across samples per question.
    Returns statistics dictionary with pass@k metrics.
    """
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)

    data = load_jsonl(file_path)
    total_items = len(data)

    # Determine k values for pass@k based on sample counts
    max_samples = 1
    for item in data:
        samples = item.get("samples", None)
        if samples:
            max_samples = max(max_samples, len(samples))

    k_values = [k for k in [1, 2, 4, 8, 16, 32, 64] if k <= max_samples]

    stats = {
        "total": len(data),
        "correct": 0,
        "incorrect": 0,
        "parse_failed": 0,
        "timeout": 0,
        "avg_accuracy": 0.0,
        "pass_at_k": {k: 0.0 for k in k_values},
        "status_counts": defaultdict(int),
        "by_problem_type": defaultdict(lambda: {"correct": 0, "total": 0, "avg_accuracy": 0.0, "pass_at_k": {k: 0.0 for k in k_values}}),
        "details": []
    }

    # Phase 1: Collect all verification tasks
    if not quiet:
        print("  Collecting samples...", end="", flush=True)

    verify_tasks = []  # List of (gold, text) pairs
    task_map = []  # Maps task index to (item_idx, sample_idx, source)

    for idx, item in enumerate(data):
        gold_answer = item.get("answer", "")
        samples = item.get("samples", None)

        if samples is not None:
            for s_idx, sample in enumerate(samples):
                generated_text, source_used = get_text_from_sample(sample)
                if generated_text and gold_answer:
                    verify_tasks.append((gold_answer, generated_text))
                    task_map.append((idx, s_idx, source_used, True))  # True = needs verification
                else:
                    task_map.append((idx, s_idx, source_used, False))  # False = skip
        else:
            # Single-sample format
            generated_text, source_used = get_text_from_sample(item)
            if generated_text and gold_answer:
                verify_tasks.append((gold_answer, generated_text))
                task_map.append((idx, -1, source_used, True))
            else:
                task_map.append((idx, -1, source_used, False))

    if not quiet:
        print(f" {len(verify_tasks)} tasks")

    # Phase 2: Batch verify all tasks
    if not quiet:
        print(f"  Verifying {len(verify_tasks)} samples...", end="", flush=True)

    if verify_tasks:
        verify_results = verify_batch(verify_tasks, timeout=timeout, num_workers=num_workers)
    else:
        verify_results = []

    if not quiet:
        print(" done")

    # Phase 3: Process results
    result_iter = iter(verify_results)
    total_avg_correct = 0.0
    pass_at_k_sums = {k: 0.0 for k in k_values}

    for idx, item in enumerate(data):
        gold_answer = item.get("answer", "")
        problem_idx = item.get("problem_idx", idx)
        problem_types = item.get("problem_type", ["Unknown"])
        samples = item.get("samples", None)

        if samples is not None:
            num_samples = len(samples)
            sample_results = []
            correct_count = 0
            parsed_solutions = []

            for s_idx, sample in enumerate(samples):
                generated_text, source_used = get_text_from_sample(sample)

                if not generated_text:
                    is_correct, status, parsed_answer = False, "no_text", None
                elif not gold_answer:
                    is_correct, status, parsed_answer = False, "no_gold", None
                else:
                    is_correct, status, parsed_answer = next(result_iter)

                if is_correct:
                    correct_count += 1

                sample_results.append({
                    "sample_idx": s_idx,
                    "is_correct": is_correct,
                    "status": status,
                    "source": source_used,
                    "parsed_answer": parsed_answer
                })
                parsed_solutions.append(parsed_answer)

            avg_correct = correct_count / num_samples if num_samples > 0 else 0.0
            total_avg_correct += avg_correct

            question_pass_at_k = {}
            for k in k_values:
                if k <= num_samples:
                    question_pass_at_k[k] = compute_pass_at_k(num_samples, correct_count, k)
                    pass_at_k_sums[k] += question_pass_at_k[k]

            is_question_correct = avg_correct > 0.5
            if is_question_correct:
                stats["correct"] += 1
            else:
                stats["incorrect"] += 1

            for ptype in problem_types:
                stats["by_problem_type"][ptype]["total"] += 1
                stats["by_problem_type"][ptype]["avg_accuracy"] += avg_correct
                for k in k_values:
                    if k <= num_samples:
                        stats["by_problem_type"][ptype]["pass_at_k"][k] += question_pass_at_k.get(k, 0.0)
                if is_question_correct:
                    stats["by_problem_type"][ptype]["correct"] += 1

            detail = {
                "problem_idx": problem_idx,
                "gold_answer": gold_answer,
                "num_samples": num_samples,
                "correct_count": correct_count,
                "avg_accuracy": avg_correct,
                "pass_at_k": question_pass_at_k,
                "is_majority_correct": is_question_correct,
                "sample_results": sample_results,
                "parsed_solutions": parsed_solutions,
                "problem_types": problem_types
            }
            stats["details"].append(detail)

        else:
            # Single-sample format
            generated_text, source_used = get_text_from_sample(item)

            if not generated_text:
                is_correct, status, parsed_answer = False, "no_text", None
            elif not gold_answer:
                is_correct, status, parsed_answer = False, "no_gold", None
            else:
                is_correct, status, parsed_answer = next(result_iter)

            if is_correct:
                stats["correct"] += 1
                total_avg_correct += 1.0
                for k in k_values:
                    pass_at_k_sums[k] += 1.0
            elif status == "timeout":
                stats["timeout"] += 1
            elif status in ["gold_parse_failed", "answer_parse_failed", "no_text", "no_gold"]:
                stats["parse_failed"] += 1
            else:
                stats["incorrect"] += 1

            stats["status_counts"][status] += 1

            for ptype in problem_types:
                stats["by_problem_type"][ptype]["total"] += 1
                if is_correct:
                    stats["by_problem_type"][ptype]["correct"] += 1
                    stats["by_problem_type"][ptype]["avg_accuracy"] += 1.0
                    for k in k_values:
                        stats["by_problem_type"][ptype]["pass_at_k"][k] += 1.0

            detail = {
                "problem_idx": problem_idx,
                "gold_answer": gold_answer,
                "is_correct": is_correct,
                "status": status,
                "source": source_used,
                "parsed_answer": parsed_answer,
                "problem_types": problem_types
            }
            stats["details"].append(detail)

    # Calculate overall average accuracy
    stats["avg_accuracy"] = (total_avg_correct / len(data) * 100) if len(data) > 0 else 0.0

    # Calculate overall pass@k
    for k in k_values:
        stats["pass_at_k"][k] = (pass_at_k_sums[k] / len(data) * 100) if len(data) > 0 else 0.0

    # Normalize by_problem_type avg_accuracy and pass_at_k
    for ptype in stats["by_problem_type"]:
        total = stats["by_problem_type"][ptype]["total"]
        if total > 0:
            stats["by_problem_type"][ptype]["avg_accuracy"] = (
                stats["by_problem_type"][ptype]["avg_accuracy"] / total * 100
            )
            for k in k_values:
                stats["by_problem_type"][ptype]["pass_at_k"][k] = (
                    stats["by_problem_type"][ptype]["pass_at_k"][k] / total * 100
                )

    return stats


def print_summary(stats: Dict, file_name: str = "", minimal: bool = True):
    """Print a summary of verification results."""
    total = stats["total"]
    pass_at_k = stats.get("pass_at_k", {})
    timeout_count = stats.get("timeout", 0)

    if minimal:
        # One-line summary
        pass1 = pass_at_k.get(1, 0.0)
        timeout_str = f" ({timeout_count} timeouts)" if timeout_count > 0 else ""
        print(f"{file_name}: {total} problems, pass@1={pass1:.2f}%{timeout_str}")
    else:
        print(f"\n{'='*60}")
        if file_name:
            print(f"Results for: {file_name}")
        print(f"{'='*60}")
        print(f"Total problems: {total}")

        if pass_at_k and 1 in pass_at_k:
            print(f"pass@1: {pass_at_k[1]:.2f}%")

        if timeout_count > 0:
            print(f"Timeouts: {timeout_count}")

        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Verify math solutions using math-verify package"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL file or directory containing JSONL files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for detailed results (optional)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed results for each problem"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.jsonl",
        help="Glob pattern for files when input is a directory (default: *.jsonl)"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_VERIFY_TIMEOUT,
        help=f"Timeout in seconds for each verification (default: {DEFAULT_VERIFY_TIMEOUT})"
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output (only show final results)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: number of CPUs - 1)"
    )

    args = parser.parse_args()

    num_workers = args.num_workers if args.num_workers else max(1, multiprocessing.cpu_count() - 1)
    print(f"Using {num_workers} workers (detected {multiprocessing.cpu_count()} CPUs)")

    input_path = Path(args.input)
    all_results = {}

    if input_path.is_file():
        # Single file
        print(f"Verifying: {input_path.name}")
        stats = verify_file(str(input_path), args.verbose, args.timeout, args.quiet, num_workers=num_workers)
        print_summary(stats, input_path.name)
        all_results[input_path.name] = stats

    elif input_path.is_dir():
        # Directory of files
        files = sorted(input_path.glob(args.pattern))

        if not files:
            print(f"No files matching '{args.pattern}' found in {input_path}")
            return

        print(f"Found {len(files)} files")

        for file_path in files:
            print(f"\nVerifying: {file_path.name}")
            stats = verify_file(str(file_path), args.verbose, args.timeout, args.quiet, num_workers=num_workers)
            print_summary(stats, file_path.name)
            all_results[file_path.name] = stats

        # Print comparison table only if multiple files
        if len(files) > 1:
            print(f"\n{'File':<45} {'N':<6} {'pass@1':<8}")
            print("-" * 60)
            for fname, fstats in sorted(all_results.items()):
                pass_at_k = fstats.get("pass_at_k", {})
                pass_at_1 = pass_at_k.get(1, 0.0)
                print(f"{fname:<45} {fstats['total']:<6} {pass_at_1:.2f}%")

    else:
        print(f"Error: {input_path} does not exist")
        return

    # Save detailed results if requested
    if args.output:
        # Convert defaultdicts to regular dicts for JSON serialization
        def convert_defaultdict(d):
            if isinstance(d, defaultdict):
                d = {k: convert_defaultdict(v) for k, v in d.items()}
            elif isinstance(d, dict):
                d = {k: convert_defaultdict(v) for k, v in d.items()}
            elif isinstance(d, list):
                d = [convert_defaultdict(v) for v in d]
            return d

        output_data = convert_defaultdict(all_results)

        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
