#!/usr/bin/env python3
"""
Solution Verification Script using Math-Verify

This script reads output JSONL files from context refinement and verifies
the solutions against ground truth answers using the math-verify package.
"""

import argparse
import json
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

try:
    from math_verify import parse, verify
    from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig
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


def verify_answer(gold_answer: str, generated_text: str, verbose: bool = False) -> Tuple[bool, str, Optional[str]]:
    """
    Verify if the generated text contains the correct answer.
    Uses math-verify to parse and compare.
    Returns (is_correct, status_message, parsed_answer_str).
    """
    try:
        # Parse gold answer - wrap in $ if not already LaTeX formatted
        if not gold_answer.startswith('$'):
            gold_text = f"${gold_answer}$"
        else:
            gold_text = gold_answer

        gold_parsed = parse(gold_text)

        if not gold_parsed:
            if verbose:
                print(f"    Could not parse gold answer: {gold_answer}")
            return False, "gold_parse_failed", None

        # Parse the generated text - math-verify will extract the answer
        answer_parsed = parse(generated_text)

        if not answer_parsed:
            if verbose:
                print(f"    Could not extract answer from generated text")
            return False, "answer_parse_failed", None

        # Convert parsed answer to string representation
        parsed_answer_str = str(answer_parsed) if answer_parsed else None

        # Verify - order matters! gold first, then answer
        is_correct = verify(gold_parsed, answer_parsed)

        if is_correct:
            return True, "verified", parsed_answer_str
        else:
            return False, "incorrect", parsed_answer_str

    except Exception as e:
        if verbose:
            print(f"    Verification error: {e}")
        return False, f"error: {str(e)[:50]}", None


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
    ])

    for source_name, text in text_sources:
        if text and text.strip():
            return text, source_name

    return "", None


def verify_file(file_path: str, verbose: bool = False) -> Dict:
    """
    Verify all solutions in a JSONL file.
    Supports multi-sample format: averages correctness across samples per question.
    Returns statistics dictionary with pass@k metrics.
    """
    data = load_jsonl(file_path)

    # Determine k values for pass@k based on sample counts
    max_samples = 1
    for item in data:
        samples = item.get("samples", None)
        if samples:
            max_samples = max(max_samples, len(samples))

    # Common k values to compute
    k_values = [k for k in [1, 2, 4, 8, 16, 32, 64] if k <= max_samples]

    stats = {
        "total": len(data),
        "correct": 0,
        "incorrect": 0,
        "parse_failed": 0,
        "avg_accuracy": 0.0,
        "pass_at_k": {k: 0.0 for k in k_values},
        "status_counts": defaultdict(int),
        "by_problem_type": defaultdict(lambda: {"correct": 0, "total": 0, "avg_accuracy": 0.0, "pass_at_k": {k: 0.0 for k in k_values}}),
        "details": []
    }

    total_avg_correct = 0.0
    pass_at_k_sums = {k: 0.0 for k in k_values}

    for idx, item in enumerate(data):
        gold_answer = item.get("answer", "")
        problem_idx = item.get("problem_idx", idx)
        problem_types = item.get("problem_type", ["Unknown"])

        # Check if this is multi-sample format
        samples = item.get("samples", None)

        if samples is not None:
            # Multi-sample format: verify each sample and average
            num_samples = len(samples)
            sample_results = []
            correct_count = 0
            parsed_solutions = []

            for s_idx, sample in enumerate(samples):
                generated_text, source_used = get_text_from_sample(sample)

                if not generated_text:
                    is_correct = False
                    status = "no_text"
                    parsed_answer = None
                elif not gold_answer:
                    is_correct = False
                    status = "no_gold"
                    parsed_answer = None
                else:
                    is_correct, status, parsed_answer = verify_answer(gold_answer, generated_text, verbose)

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

            # Calculate average accuracy for this question
            avg_correct = correct_count / num_samples if num_samples > 0 else 0.0
            total_avg_correct += avg_correct

            # Calculate pass@k for this question
            question_pass_at_k = {}
            for k in k_values:
                if k <= num_samples:
                    question_pass_at_k[k] = compute_pass_at_k(num_samples, correct_count, k)
                    pass_at_k_sums[k] += question_pass_at_k[k]

            # For stats, count as correct if majority is correct (>50%)
            is_question_correct = avg_correct > 0.5

            if is_question_correct:
                stats["correct"] += 1
            else:
                stats["incorrect"] += 1

            # Track by problem type
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

            if verbose:
                status_str = f"{correct_count}/{num_samples} correct ({avg_correct*100:.1f}%)"
                pass_at_1 = question_pass_at_k.get(1, 0.0) * 100
                print(f"[{problem_idx}] {status_str} pass@1={pass_at_1:.1f}% | Gold: {gold_answer}")
                print(f"    Parsed solutions: {parsed_solutions[:5]}{'...' if len(parsed_solutions) > 5 else ''}")

        else:
            # Single-sample format (legacy): use old behavior
            generated_text = ""
            source_used = None

            text_sources = [
                ("full_assistant_message", item.get("full_assistant_message", "")),
                ("final_refined_context", item.get("final_refined_context", "")),
            ]

            rounds = item.get("rounds", [])
            if rounds:
                last_round = rounds[-1]
                text_sources.append(
                    ("last_round_generation", last_round.get("current_round_generation", ""))
                )

            for source_name, text in text_sources:
                if text and text.strip():
                    generated_text = text
                    source_used = source_name
                    break

            if not generated_text:
                is_correct = False
                status = "no_text"
                parsed_answer = None
            elif not gold_answer:
                is_correct = False
                status = "no_gold"
                parsed_answer = None
            else:
                is_correct, status, parsed_answer = verify_answer(gold_answer, generated_text, verbose)

            if is_correct:
                stats["correct"] += 1
                total_avg_correct += 1.0
                # For single sample, pass@k = 1 if correct, 0 if not
                for k in k_values:
                    pass_at_k_sums[k] += 1.0
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

            if verbose:
                status_str = "CORRECT" if is_correct else "WRONG"
                print(f"[{problem_idx}] {status_str} | Gold: {gold_answer} | Parsed: {parsed_answer}")

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


def print_summary(stats: Dict, file_name: str = ""):
    """Print a summary of verification results."""
    total = stats["total"]
    correct = stats["correct"]
    accuracy = (correct / total * 100) if total > 0 else 0
    avg_accuracy = stats.get("avg_accuracy", accuracy)
    pass_at_k = stats.get("pass_at_k", {})

    print(f"\n{'='*60}")
    if file_name:
        print(f"Results for: {file_name}")
    print(f"{'='*60}")
    print(f"Total problems:      {total}")
    print(f"Correct (majority):  {correct}")
    print(f"Incorrect:           {stats['incorrect']}")
    print(f"Parse failed:        {stats['parse_failed']}")
    print(f"Majority accuracy:   {accuracy:.2f}%")
    print(f"Average accuracy:    {avg_accuracy:.2f}%")

    if pass_at_k:
        print(f"\nPass@k metrics:")
        for k in sorted(pass_at_k.keys()):
            print(f"  pass@{k}: {pass_at_k[k]:.2f}%")

    if stats["status_counts"]:
        print(f"\nStatus breakdown:")
        for status, count in sorted(stats["status_counts"].items()):
            print(f"  {status}: {count}")

    if stats["by_problem_type"]:
        print(f"\nAccuracy by problem type:")
        for ptype, pstats in sorted(stats["by_problem_type"].items()):
            pt_acc = (pstats["correct"] / pstats["total"] * 100) if pstats["total"] > 0 else 0
            pt_avg = pstats.get("avg_accuracy", pt_acc)
            pt_pass_at_k = pstats.get("pass_at_k", {})
            pass_at_1 = pt_pass_at_k.get(1, pt_acc) if pt_pass_at_k else pt_acc
            print(f"  {ptype}: {pstats['correct']}/{pstats['total']} (avg: {pt_avg:.1f}%, pass@1: {pass_at_1:.1f}%)")

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

    args = parser.parse_args()

    input_path = Path(args.input)
    all_results = {}

    if input_path.is_file():
        # Single file
        stats = verify_file(str(input_path), args.verbose)
        print_summary(stats, input_path.name)
        all_results[input_path.name] = stats

    elif input_path.is_dir():
        # Directory of files
        files = sorted(input_path.glob(args.pattern))

        if not files:
            print(f"No files matching '{args.pattern}' found in {input_path}")
            return

        print(f"Found {len(files)} files to process")

        aggregate_stats = {
            "total": 0,
            "correct": 0,
            "incorrect": 0,
            "parse_failed": 0,
            "status_counts": defaultdict(int),
            "by_problem_type": defaultdict(lambda: {"correct": 0, "total": 0})
        }

        for file_path in files:
            stats = verify_file(str(file_path), args.verbose)
            print_summary(stats, file_path.name)
            all_results[file_path.name] = stats

            # Aggregate
            aggregate_stats["total"] += stats["total"]
            aggregate_stats["correct"] += stats["correct"]
            aggregate_stats["incorrect"] += stats["incorrect"]
            aggregate_stats["parse_failed"] += stats["parse_failed"]
            for status, count in stats["status_counts"].items():
                aggregate_stats["status_counts"][status] += count
            for ptype, pstats in stats["by_problem_type"].items():
                aggregate_stats["by_problem_type"][ptype]["total"] += pstats["total"]
                aggregate_stats["by_problem_type"][ptype]["correct"] += pstats["correct"]

        # Print aggregate summary
        print("\n" + "#"*60)
        print("AGGREGATE RESULTS")
        print("#"*60)
        print_summary(aggregate_stats, "All Files")

        # Print comparison table
        print("\nComparison Table:")
        print("-"*60)
        print(f"{'File':<30} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")
        print("-"*60)
        for fname, fstats in sorted(all_results.items()):
            acc = (fstats['correct'] / fstats['total'] * 100) if fstats['total'] > 0 else 0
            print(f"{fname:<30} {fstats['correct']:<10} {fstats['total']:<10} {acc:.2f}%")
        print("-"*60)

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
