#!/usr/bin/env python3
"""
Extract truncated trajectories from Gemini annotations (CPU only).

Matches annotation positions to trajectory text, truncates at the annotated
position, and saves the truncated data. No GPU inference needed — this is
the CPU-only preprocessing step before running continuation inference.

Output format: JSONL where each line contains the truncated trajectory
ready for continuation inference.
"""

import argparse
import difflib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional


# Categories worth reconstructing (skip called_too_early)
RECONSTRUCT_CATEGORIES = {"never_called", "called_too_late", "should_call_again", "wrong_moment"}


def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def find_position_in_text(text: str, position_quote: str) -> Optional[int]:
    """Find the approximate character offset of a position quote in text.

    Strategy:
    1. Try exact substring match (fast)
    2. Try case-insensitive match
    3. Try matching the longest distinctive phrase (3+ word chunks)
    4. Fall back to keyword-based search with scoring

    Returns the character offset where the match ends (truncation point).
    """
    if not position_quote or not text:
        return None

    quote = position_quote.strip()
    if not quote:
        return None

    # 1. Exact substring match
    idx = text.find(quote)
    if idx >= 0:
        return idx + len(quote)

    # 2. Case-insensitive match
    text_lower = text.lower()
    quote_lower = quote.lower()
    idx = text_lower.find(quote_lower)
    if idx >= 0:
        return idx + len(quote)

    # 3. Try progressively shorter substrings from the quote
    # Start from the full quote, remove chars from the end/start
    words = quote.split()
    if len(words) >= 3:
        # Try consecutive word chunks of decreasing length
        for chunk_size in range(len(words), max(2, len(words) // 2) - 1, -1):
            for start_word in range(len(words) - chunk_size + 1):
                chunk = " ".join(words[start_word:start_word + chunk_size])
                idx = text_lower.find(chunk.lower())
                if idx >= 0:
                    return idx + len(chunk)

    # 4. Keyword scoring: find the region with highest keyword overlap
    # Extract distinctive keywords (4+ chars, not common math words)
    common = {"that", "this", "with", "from", "then", "have", "been", "will",
              "each", "more", "also", "when", "which", "their", "there"}
    keywords = [w.strip(".,;:!?()[]{}$\\") for w in words
                if len(w.strip(".,;:!?()[]{}$\\")) >= 4
                and w.strip(".,;:!?()[]{}$\\").lower() not in common]

    if not keywords:
        return None

    # Score each position by counting keyword hits in a window
    window = len(quote) * 2
    best_score = 0
    best_pos = None

    for kw in keywords:
        kw_lower = kw.lower()
        start = 0
        while True:
            idx = text_lower.find(kw_lower, start)
            if idx < 0:
                break
            # Score: count how many other keywords appear nearby
            region_start = max(0, idx - window // 2)
            region_end = min(len(text), idx + window // 2)
            region = text_lower[region_start:region_end]
            score = sum(1 for k in keywords if k.lower() in region)
            if score > best_score:
                best_score = score
                best_pos = idx + len(kw)
            start = idx + 1

    if best_score >= max(2, len(keywords) // 3):
        return best_pos

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Extract truncated trajectories from Gemini annotations"
    )
    parser.add_argument(
        "--annotation_files", type=str, nargs="+", required=True,
        help="Gemini annotation JSONL files"
    )
    parser.add_argument(
        "--trajectory_files", type=str, nargs="+", required=True,
        help="Original tool refinement JSONL files (same order as annotation_files)"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output JSONL file with truncated trajectories"
    )

    args = parser.parse_args()

    if len(args.annotation_files) != len(args.trajectory_files):
        print(f"Error: {len(args.annotation_files)} annotation files but "
              f"{len(args.trajectory_files)} trajectory files")
        sys.exit(1)

    # Load trajectory files
    trajectory_maps = []
    for traj_file in args.trajectory_files:
        data = load_jsonl(traj_file)
        trajectory_maps.append(data)
        print(f"  Loaded {len(data)} problems from {traj_file}")

    tasks = []
    skipped = {"bad_category": 0, "no_annotation": 0, "no_match": 0,
               "no_trajectory": 0, "no_position": 0}

    for ann_file_idx, ann_file in enumerate(args.annotation_files):
        print(f"\nProcessing annotations: {ann_file}")
        annotations = load_jsonl(ann_file)

        if ann_file_idx >= len(trajectory_maps):
            print(f"  Warning: no matching trajectory file for annotation {ann_file_idx}")
            continue
        traj_data = trajectory_maps[ann_file_idx]

        for ann in annotations:
            ann_result = ann.get("annotation")
            if ann_result is None:
                skipped["no_annotation"] += 1
                continue

            line_idx = ann.get("line_idx", -1)
            sample_idx = ann.get("sample_idx", -1)
            answer = ann.get("answer", "")

            if line_idx < 0 or line_idx >= len(traj_data):
                skipped["no_trajectory"] += 1
                continue

            traj_item = traj_data[line_idx]
            samples = traj_item.get("samples", [])
            if sample_idx < 0 or sample_idx >= len(samples):
                skipped["no_trajectory"] += 1
                continue

            sample = samples[sample_idx]
            rounds = sample.get("rounds", [])
            original_prompt = traj_item.get("original_prompt", "")

            for sub_ann in ann_result.get("annotations", []):
                cat = sub_ann.get("category", "")
                if cat not in RECONSTRUCT_CATEGORIES:
                    skipped["bad_category"] += 1
                    continue

                position_quote = sub_ann.get("position", "")
                try:
                    ann_round = int(sub_ann.get("round", "0"))
                except (ValueError, TypeError):
                    ann_round = 0

                if ann_round < 0 or ann_round >= len(rounds):
                    skipped["no_position"] += 1
                    continue

                gen_text = rounds[ann_round].get("current_round_generation", "")
                offset = find_position_in_text(gen_text, position_quote)
                if offset is None:
                    skipped["no_position"] += 1
                    continue

                tasks.append({
                    "original_prompt": original_prompt,
                    "answer": answer,
                    "problem_id": traj_item.get("problem_id", ""),
                    "category": traj_item.get("category", ""),
                    "subcategory": traj_item.get("subcategory", ""),
                    "source": traj_item.get("source", ""),
                    "pre_rounds": rounds[:ann_round],
                    "truncated_generation": gen_text[:offset],
                    "truncate_round": ann_round,
                    "annotation_category": cat,
                    "annotation_position": position_quote,
                    "annotation_round": ann_round,
                    "annotation_reason": sub_ann.get("reason", ""),
                    "source_line_idx": line_idx,
                    "source_sample_idx": sample_idx,
                })

    print(f"\nSkip stats: {skipped}")

    # Deduplicate: first annotation per (line_idx, sample_idx, round)
    seen = set()
    deduped = []
    for t in tasks:
        key = (t["source_line_idx"], t["source_sample_idx"], t["truncate_round"])
        if key not in seen:
            seen.add(key)
            deduped.append(t)

    print(f"Total tasks before dedup: {len(tasks)}")
    print(f"After dedup: {len(deduped)}")

    # Category distribution
    cat_counts = {}
    for t in deduped:
        cat_counts[t["annotation_category"]] = cat_counts.get(t["annotation_category"], 0) + 1
    print(f"Category distribution: {cat_counts}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in deduped:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"\nSaved {len(deduped)} truncated trajectories to {args.output}")


if __name__ == "__main__":
    main()
