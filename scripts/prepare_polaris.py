#!/usr/bin/env python3
"""
Download and prepare the Polaris-Dataset-53K for baseline inference.

Downloads from HuggingFace, converts to JSONL format compatible with
baseline_vllm.py --dataset hmmt, and splits into 10 chunks.
"""

import json
import os
import math
from datasets import load_dataset

OUTPUT_DIR = "/scratch/dkhasha1/tli104/datasets/polaris_53k"
NUM_CHUNKS = 10


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Downloading Polaris-Dataset-53K from HuggingFace...")
    ds = load_dataset("POLARIS-Project/Polaris-Dataset-53K", split="train")
    print(f"Downloaded {len(ds)} problems")

    # Print column names for reference
    print(f"Columns: {ds.column_names}")
    print(f"First example keys: {list(ds[0].keys())}")

    # Convert to JSONL format
    items = []
    for idx, row in enumerate(ds):
        item = {
            "problem_id": f"polaris_{idx}",
            "prompt": row["problem"],
            "answer": str(row["answer"]),
            "problem_idx": idx,
        }
        # Preserve optional metadata fields if present
        for key in ["difficulty", "category", "subcategory", "source"]:
            if key in row and row[key] is not None:
                item[key] = row[key]
        items.append(item)

    # Write full dataset
    full_path = os.path.join(OUTPUT_DIR, "polaris_53k.jsonl")
    with open(full_path, "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")
    print(f"Wrote {len(items)} problems to {full_path}")

    # Split into chunks
    chunk_size = math.ceil(len(items) / NUM_CHUNKS)
    for chunk_idx in range(NUM_CHUNKS):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, len(items))
        chunk = items[start:end]
        chunk_path = os.path.join(OUTPUT_DIR, f"polaris_53k_chunk{chunk_idx:02d}.jsonl")
        with open(chunk_path, "w") as f:
            for item in chunk:
                f.write(json.dumps(item) + "\n")
        print(f"Chunk {chunk_idx:02d}: {len(chunk)} problems ({start}-{end-1}) -> {chunk_path}")

    print(f"\nDone! {len(items)} problems split into {NUM_CHUNKS} chunks of ~{chunk_size} each")


if __name__ == "__main__":
    main()
