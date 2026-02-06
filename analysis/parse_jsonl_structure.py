#!/usr/bin/env python3
"""
Parse and display the structure of a JSONL file from hmmt25_inference outputs.
"""

import json
import sys
from collections import defaultdict
from typing import Any


def analyze_value(value: Any, indent: int = 0, max_str_len: int = 80) -> str:
    """Analyze and format a value for display."""
    prefix = "  " * indent

    if isinstance(value, str):
        truncated = value[:max_str_len] + "..." if len(value) > max_str_len else value
        truncated = truncated.replace('\n', '\\n')
        return f'{prefix}str (len={len(value)}): "{truncated}"'
    elif isinstance(value, int):
        return f"{prefix}int: {value}"
    elif isinstance(value, float):
        return f"{prefix}float: {value}"
    elif isinstance(value, bool):
        return f"{prefix}bool: {value}"
    elif value is None:
        return f"{prefix}null"
    elif isinstance(value, list):
        return f"{prefix}list (len={len(value)})"
    elif isinstance(value, dict):
        return f"{prefix}dict (keys={list(value.keys())})"
    else:
        return f"{prefix}{type(value).__name__}: {value}"


def display_structure(data: dict, indent: int = 0):
    """Recursively display the structure of a dictionary."""
    prefix = "  " * indent

    for key, value in data.items():
        if isinstance(value, dict):
            print(f"{prefix}{key}: dict")
            display_structure(value, indent + 1)
        elif isinstance(value, list) and len(value) > 0:
            print(f"{prefix}{key}: list (len={len(value)})")
            # Show structure of first item
            if isinstance(value[0], dict):
                print(f"{prefix}  [0]: dict")
                display_structure(value[0], indent + 2)
            else:
                print(f"{prefix}  [0]: {analyze_value(value[0])}")
        else:
            print(f"{prefix}{key}: {analyze_value(value)}")


def main():
    if len(sys.argv) < 2:
        filepath = "/scratch/dkhasha1/tli104/outputs/hmmt25_inference/output_t16384_n16_4b_instruct_2507.jsonl"
    else:
        filepath = sys.argv[1]

    print(f"Parsing: {filepath}")
    print("=" * 80)

    # Read and parse the file
    records = []
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num}: {e}")

    print(f"\nTotal records: {len(records)}")
    print("=" * 80)

    # Show structure of first record
    if records:
        print("\n### Structure of a single record (line 1):\n")
        first_record = records[0]
        display_structure(first_record)

    # Aggregate statistics
    print("\n" + "=" * 80)
    print("### Aggregate Statistics:\n")

    # Collect stats
    num_samples_list = []
    num_rounds_list = []
    gen_lengths = []
    refined_lengths = []

    for record in records:
        num_samples_list.append(record.get('num_samples', 0))

        for sample in record.get('samples', []):
            rounds = sample.get('rounds', [])
            num_rounds_list.append(len(rounds))

            for r in rounds:
                gen = r.get('current_round_generation', '')
                refined = r.get('refined_context', '')
                gen_lengths.append(len(gen))
                refined_lengths.append(len(refined))

    print(f"Number of problems: {len(records)}")
    print(f"Samples per problem: {num_samples_list[0] if num_samples_list else 'N/A'}")
    print(f"Rounds per sample: {num_rounds_list[0] if num_rounds_list else 'N/A'}")
    print(f"Total samples: {sum(num_samples_list)}")

    if gen_lengths:
        print(f"\nGeneration lengths:")
        print(f"  Min: {min(gen_lengths)}, Max: {max(gen_lengths)}, Avg: {sum(gen_lengths)/len(gen_lengths):.1f}")

    if refined_lengths:
        print(f"\nRefined context lengths:")
        print(f"  Min: {min(refined_lengths)}, Max: {max(refined_lengths)}, Avg: {sum(refined_lengths)/len(refined_lengths):.1f}")

    # Show example content
    print("\n" + "=" * 80)
    print("### Example Content (first problem, first sample):\n")

    if records:
        first = records[0]
        print(f"problem_idx: {first.get('problem_idx', 'N/A')}")
        print(f"answer: {first.get('answer', 'N/A')}")
        print(f"problem_type: {first.get('problem_type', 'N/A')}")

        prompt = first.get('original_prompt', '')
        print(f"\noriginal_prompt (first 300 chars):")
        print(f"  {prompt[:300]}...")

        original_problem = first.get('original_problem', '')
        if original_problem:
            print(f"\noriginal_problem (first 300 chars):")
            print(f"  {original_problem[:300]}...")

        samples = first.get('samples', [])
        if samples:
            sample = samples[0]
            print(f"\n--- Sample 0 ---")
            print(f"final_refined_context length: {len(sample.get('final_refined_context', ''))}")
            print(f"full_assistant_message length: {len(sample.get('full_assistant_message', ''))}")

            rounds = sample.get('rounds', [])
            if rounds:
                r = rounds[0]
                print(f"\n  Round 1:")
                print(f"    current_round_generation length: {len(r.get('current_round_generation', ''))}")
                print(f"    refined_context length: {len(r.get('refined_context', ''))}")


if __name__ == "__main__":
    main()
