"""
Prepare Polaris problems for RC user rollouts (SFT data generation).

Converts Polaris JSONL format to RC JSON format and samples a subset
of hard problems for generating multi-step rollouts.
"""

import argparse
import json
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Polaris JSONL file (e.g. polaris_filtered_removed_all_correct.jsonl)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON file in RC format")
    parser.add_argument("--num_problems", type=int, default=2000,
                        help="Number of problems to sample (default: 2000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude_json", type=str, default=None,
                        help="JSON file of previously sampled problems to exclude (avoids overlap)")
    args = parser.parse_args()

    # Load Polaris JSONL
    problems = []
    with open(args.input) as f:
        for line in f:
            item = json.loads(line)
            problems.append(item)

    print(f"Loaded {len(problems)} problems from {args.input}")

    # Filter out proof problems (no numeric/symbolic answer to verify)
    filtered = []
    for p in problems:
        answer = p.get("answer", "").strip()
        # Skip if answer looks like a proof stub or is empty
        if not answer or len(answer) > 200:
            continue
        filtered.append(p)

    print(f"After filtering proofs: {len(filtered)} problems")

    # Exclude previously sampled problems
    if args.exclude_json:
        with open(args.exclude_json) as f:
            exclude_ids = {p["id"] for p in json.load(f)}
        before = len(filtered)
        filtered = [p for p in filtered if p["problem_id"] not in exclude_ids]
        print(f"Excluded {before - len(filtered)} previously sampled problems, {len(filtered)} remaining")

    # Sample
    random.seed(args.seed)
    if len(filtered) > args.num_problems:
        sampled = random.sample(filtered, args.num_problems)
    else:
        sampled = filtered
    print(f"Sampled {len(sampled)} problems")

    # Convert to RC format: {"problem": ..., "answer": ..., "id": ...}
    rc_data = []
    for p in sampled:
        rc_data.append({
            "problem": p["prompt"],
            "answer": p["answer"],
            "id": p["problem_id"],
        })

    with open(args.output, "w") as f:
        json.dump(rc_data, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(rc_data)} problems to {args.output}")


if __name__ == "__main__":
    main()
