#!/usr/bin/env python3
"""
Solution Verification Script using Math-Verify

This script reads output JSONL files from context refinement and verifies
the solutions against ground truth answers using the math-verify package.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple
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


def verify_answer(gold_answer: str, generated_text: str, verbose: bool = False) -> Tuple[bool, str]:
    """
    Verify if the generated text contains the correct answer.
    Uses math-verify to parse and compare.
    Returns (is_correct, status_message).
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
            return False, "gold_parse_failed"

        # Parse the generated text - math-verify will extract the answer
        answer_parsed = parse(generated_text)

        if not answer_parsed:
            if verbose:
                print(f"    Could not extract answer from generated text")
            return False, "answer_parse_failed"

        # Verify - order matters! gold first, then answer
        is_correct = verify(gold_parsed, answer_parsed)

        if is_correct:
            return True, "verified"
        else:
            return False, "incorrect"

    except Exception as e:
        if verbose:
            print(f"    Verification error: {e}")
        return False, f"error: {str(e)[:50]}"


def verify_file(file_path: str, verbose: bool = False) -> Dict:
    """
    Verify all solutions in a JSONL file.
    Returns statistics dictionary.
    """
    data = load_jsonl(file_path)

    stats = {
        "total": len(data),
        "correct": 0,
        "incorrect": 0,
        "parse_failed": 0,
        "status_counts": defaultdict(int),
        "by_problem_type": defaultdict(lambda: {"correct": 0, "total": 0}),
        "details": []
    }

    for idx, item in enumerate(data):
        gold_answer = item.get("answer", "")
        problem_idx = item.get("problem_idx", idx)
        problem_types = item.get("problem_type", ["Unknown"])

        # Get generated text - try multiple sources
        generated_text = ""
        source_used = None

        # Priority order for text sources
        text_sources = [
            ("full_assistant_message", item.get("full_assistant_message", "")),
            ("final_refined_context", item.get("final_refined_context", "")),
        ]

        # Also check the last round's generation
        rounds = item.get("rounds", [])
        if rounds:
            last_round = rounds[-1]
            text_sources.append(
                ("last_round_generation", last_round.get("current_round_generation", ""))
            )

        # Use the first non-empty source
        for source_name, text in text_sources:
            if text and text.strip():
                generated_text = text
                source_used = source_name
                break

        if not generated_text:
            is_correct = False
            status = "no_text"
        elif not gold_answer:
            is_correct = False
            status = "no_gold"
        else:
            is_correct, status = verify_answer(gold_answer, generated_text, verbose)

        # Update stats
        if is_correct:
            stats["correct"] += 1
        elif status in ["gold_parse_failed", "answer_parse_failed", "no_text", "no_gold"]:
            stats["parse_failed"] += 1
        else:
            stats["incorrect"] += 1

        stats["status_counts"][status] += 1

        # Track by problem type
        for ptype in problem_types:
            stats["by_problem_type"][ptype]["total"] += 1
            if is_correct:
                stats["by_problem_type"][ptype]["correct"] += 1

        detail = {
            "problem_idx": problem_idx,
            "gold_answer": gold_answer,
            "is_correct": is_correct,
            "status": status,
            "source": source_used,
            "problem_types": problem_types
        }
        stats["details"].append(detail)

        if verbose:
            status_str = "CORRECT" if is_correct else "WRONG"
            print(f"[{problem_idx}] {status_str} | Gold: {gold_answer} | Status: {status}")

    return stats


def print_summary(stats: Dict, file_name: str = ""):
    """Print a summary of verification results."""
    total = stats["total"]
    correct = stats["correct"]
    accuracy = (correct / total * 100) if total > 0 else 0

    print(f"\n{'='*60}")
    if file_name:
        print(f"Results for: {file_name}")
    print(f"{'='*60}")
    print(f"Total problems:      {total}")
    print(f"Correct:             {correct}")
    print(f"Incorrect:           {stats['incorrect']}")
    print(f"Parse failed:        {stats['parse_failed']}")
    print(f"Accuracy:            {accuracy:.2f}%")

    print(f"\nStatus breakdown:")
    for status, count in sorted(stats["status_counts"].items()):
        print(f"  {status}: {count}")

    if stats["by_problem_type"]:
        print(f"\nAccuracy by problem type:")
        for ptype, pstats in sorted(stats["by_problem_type"].items()):
            pt_acc = (pstats["correct"] / pstats["total"] * 100) if pstats["total"] > 0 else 0
            print(f"  {ptype}: {pstats['correct']}/{pstats['total']} ({pt_acc:.1f}%)")

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
