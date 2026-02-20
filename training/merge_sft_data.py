#!/usr/bin/env python3
"""
Merge SFT data from correct and reconstructed trajectories.

Takes multiple JSON files produced by build_sft_data.py, merges them,
deduplicates, shuffles, and splits into train/val sets.

Usage:
    python training/merge_sft_data.py \
        --input_files datasets/tool_sft/correct_sft.json datasets/tool_sft/reconstructed_sft.json \
        --output_dir /scratch/dkhasha1/tli104/datasets/tool_sft/ \
        --val_ratio 0.1
"""

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Dict, List


def conversation_hash(example: Dict) -> str:
    """Hash an example for deduplication."""
    parts = []
    for turn in example.get("conversations", []):
        parts.append(turn.get("from", ""))
        parts.append(turn.get("value", "")[:500])  # First 500 chars of each turn
    content = "|||".join(parts)
    return hashlib.sha256(content.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Merge, deduplicate, shuffle, and split SFT data"
    )
    parser.add_argument(
        "--input_files", type=str, nargs="+", required=True,
        help="Input JSON files from build_sft_data.py"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for train/val JSON files"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.1,
        help="Fraction for validation set (default: 0.1)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    all_examples = []
    for input_file in args.input_files:
        print(f"Loading: {input_file}")
        with open(input_file, 'r') as f:
            data = json.load(f)
        print(f"  Loaded {len(data)} examples")
        all_examples.extend(data)

    print(f"\nTotal before dedup: {len(all_examples)}")

    # Deduplicate
    seen = set()
    deduped = []
    for ex in all_examples:
        h = conversation_hash(ex)
        if h not in seen:
            seen.add(h)
            deduped.append(ex)

    print(f"After dedup: {len(deduped)} (removed {len(all_examples) - len(deduped)})")

    # Shuffle
    random.seed(args.seed)
    random.shuffle(deduped)

    # Split
    val_size = int(len(deduped) * args.val_ratio)
    train_size = len(deduped) - val_size
    train_data = deduped[:train_size]
    val_data = deduped[train_size:]

    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "tool_sft_train.json"
    val_path = output_dir / "tool_sft_val.json"

    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False)
    print(f"Saved train: {train_path}")

    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False)
    print(f"Saved val: {val_path}")

    # Stats
    print(f"\n{'='*60}")
    print(f"Merge Summary")
    print(f"{'='*60}")
    for split_name, split_data in [("train", train_data), ("val", val_data)]:
        tool_calls = {}
        for ex in split_data:
            n = sum(1 for t in ex["conversations"] if t["from"] == "gpt" and "<tool_call>" in t["value"])
            tool_calls[n] = tool_calls.get(n, 0) + 1
        print(f"\n{split_name} ({len(split_data)} examples):")
        print(f"  Tool call distribution:")
        for k in sorted(tool_calls.keys()):
            print(f"    {k} calls: {tool_calls[k]}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
