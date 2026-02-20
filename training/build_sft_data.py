#!/usr/bin/env python3
"""
Build SFT data from tool refinement trajectories for LlamaFactory.

Converts correct tool refinement trajectories into LlamaFactory sharegpt format
with function_call/observation roles for teaching tool-calling timing.

Usage:
    python training/build_sft_data.py \
        --input_files outputs/.../tool_refinement_*.jsonl \
        --verify_files outputs/.../tool_refinement_*_verified.json \
        --output /scratch/dkhasha1/tli104/datasets/tool_sft/correct_sft.json \
        --max_tokens 32768
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Reuse constants from tool_refinement.py
LLM_REFINE_TOOL_JSON = json.dumps([{
    "type": "function",
    "function": {
        "name": "llm_refine",
        "description": (
            "Summarize your progress and continue with a fresh start. "
            "Call after each major reasoning step."
        ),
        "parameters": {"type": "object", "properties": {}}
    }
}])

SYSTEM_PROMPT = (
    "You are a mathematical reasoning assistant. You have a tool `llm_refine` that "
    "summarizes your work so far. After calling it, you continue solving with a fresh "
    "perspective while retaining key insights. Call it after each major reasoning step. "
    "Present your final answer using \\boxed{} notation."
)

LLM_REFINE_USER_HINT = (
    "\n\nYou have `llm_refine` — call it after each major reasoning step "
    "to checkpoint your progress."
)

TOOL_CALL_CONTENT = json.dumps({"name": "llm_refine", "arguments": {}})

CONTINUATION_INSTRUCTIONS = (
    "\n\nContinue solving the problem, improving upon the "
    "summary above. You may verify previous conclusions, try a "
    "different approach, or build on the progress so far.\n"
    "Return your final answer in \\boxed{}."
)


def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_verification(verify_path: str) -> Dict[int, List[bool]]:
    """Load verification JSON → {problem_idx: [bool per sample]}."""
    with open(verify_path, 'r') as f:
        data = json.load(f)

    # verify_solutions.py outputs {filename: {details: [...]}}
    if isinstance(data, dict):
        for key, val in data.items():
            if isinstance(val, dict) and "details" in val:
                data = val
                break

    results = {}
    details = data.get("details", [])
    for idx, detail in enumerate(details):
        sample_results = detail.get("sample_results", [])
        if sample_results:
            results[idx] = [sr.get("is_correct", False) for sr in sample_results]
        else:
            results[idx] = [detail.get("is_correct", False)]

    return results


def trajectory_hash(problem: str, rounds: List[Dict]) -> str:
    """Create a hash for deduplication based on problem + full trajectory."""
    parts = [problem]
    for r in rounds:
        parts.append(r.get("current_round_generation", ""))
        parts.append(str(r.get("called_tool", False)))
    content = "|||".join(parts)
    return hashlib.sha256(content.encode()).hexdigest()


def convert_sample_to_sharegpt(
    original_prompt: str,
    sample: Dict,
) -> Optional[Dict]:
    """Convert a single correct sample to LlamaFactory sharegpt format.

    Returns None if the sample has no rounds or is invalid.
    """
    rounds = sample.get("rounds", [])
    if not rounds:
        return None

    conversations = []

    # User message: problem + tool hint
    conversations.append({
        "from": "human",
        "value": original_prompt + LLM_REFINE_USER_HINT,
    })

    for r in rounds:
        generation = r.get("current_round_generation", "")
        called_tool = r.get("called_tool", False)
        refined_context = r.get("refined_context", "") or ""

        if called_tool:
            # Merge reasoning + tool call into single assistant turn
            # so model learns <tool_call> comes BEFORE <|im_end|> (EOS)
            tool_call_xml = f'\n<tool_call>\n{TOOL_CALL_CONTENT}\n</tool_call>'
            conversations.append({
                "from": "gpt",
                "value": generation + tool_call_xml,
            })
            conversations.append({
                "from": "observation",
                "value": refined_context + CONTINUATION_INSTRUCTIONS,
            })
        else:
            # Final round: assistant generates without tool call
            conversations.append({
                "from": "gpt",
                "value": generation,
            })

    return {
        "conversations": conversations,
        "tools": LLM_REFINE_TOOL_JSON,
        "system": SYSTEM_PROMPT,
    }


def estimate_tokens(example: Dict) -> int:
    """Rough token estimate: ~4 chars per token for English/math text."""
    total_chars = len(example.get("system", ""))
    total_chars += len(example.get("tools", ""))
    for turn in example.get("conversations", []):
        total_chars += len(turn.get("value", ""))
        total_chars += 20  # role/formatting overhead
    return total_chars // 4


def main():
    parser = argparse.ArgumentParser(
        description="Build SFT data from tool refinement trajectories"
    )
    parser.add_argument(
        "--input_files", type=str, nargs="+", required=True,
        help="Tool refinement JSONL output files"
    )
    parser.add_argument(
        "--verify_files", type=str, nargs="+", required=True,
        help="Verification JSON files (same order as input_files)"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=32768,
        help="Max token estimate for filtering (default: 32768)"
    )
    parser.add_argument(
        "--require_tool_call", action="store_true",
        help="Only include samples that made at least one tool call"
    )

    args = parser.parse_args()

    if len(args.input_files) != len(args.verify_files):
        print(f"Error: {len(args.input_files)} input files but "
              f"{len(args.verify_files)} verify files")
        sys.exit(1)

    all_examples = []
    seen_hashes = set()
    stats = {
        "total_correct": 0,
        "no_tool_call_skipped": 0,
        "too_long_skipped": 0,
        "duplicate_skipped": 0,
        "invalid_skipped": 0,
        "kept": 0,
    }

    for input_file, verify_file in zip(args.input_files, args.verify_files):
        print(f"\nProcessing: {input_file}")
        print(f"  Verify:   {verify_file}")

        data = load_jsonl(input_file)
        verification = load_verification(verify_file)
        print(f"  Loaded {len(data)} problems, {len(verification)} verified")

        file_correct = 0
        file_kept = 0

        for prob_idx, item in enumerate(data):
            original_prompt = item.get("original_prompt", "")
            samples = item.get("samples", [])
            correctness = verification.get(prob_idx, [])

            for s_idx, sample in enumerate(samples):
                is_correct = correctness[s_idx] if s_idx < len(correctness) else False
                if not is_correct:
                    continue

                stats["total_correct"] += 1
                file_correct += 1

                # Optionally filter: require at least one tool call
                if args.require_tool_call and sample.get("num_tool_calls", 0) == 0:
                    stats["no_tool_call_skipped"] += 1
                    continue

                # Convert
                example = convert_sample_to_sharegpt(original_prompt, sample)
                if example is None:
                    stats["invalid_skipped"] += 1
                    continue

                # Token length filter
                est_tokens = estimate_tokens(example)
                if est_tokens > args.max_tokens:
                    stats["too_long_skipped"] += 1
                    continue

                # Deduplication
                h = trajectory_hash(original_prompt, sample.get("rounds", []))
                if h in seen_hashes:
                    stats["duplicate_skipped"] += 1
                    continue
                seen_hashes.add(h)

                all_examples.append(example)
                stats["kept"] += 1
                file_kept += 1

        print(f"  Correct samples: {file_correct}, kept: {file_kept}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_examples, f, ensure_ascii=False, indent=None)

    # Report
    print(f"\n{'='*60}")
    print(f"SFT Data Build Summary")
    print(f"{'='*60}")
    print(f"Total correct samples found: {stats['total_correct']}")
    print(f"No tool call (skipped):      {stats['no_tool_call_skipped']}")
    print(f"Too long (skipped):          {stats['too_long_skipped']}")
    print(f"Duplicate (skipped):         {stats['duplicate_skipped']}")
    print(f"Invalid (skipped):           {stats['invalid_skipped']}")
    print(f"Final examples:              {stats['kept']}")
    print(f"Output: {args.output}")
    print(f"{'='*60}")

    # Also print tool call distribution
    tool_call_counts = {}
    for ex in all_examples:
        n_calls = sum(
            1 for t in ex["conversations"]
            if t["from"] == "gpt" and "<tool_call>" in t["value"]
        )
        tool_call_counts[n_calls] = tool_call_counts.get(n_calls, 0) + 1
    print(f"\nTool call distribution:")
    for k in sorted(tool_call_counts.keys()):
        print(f"  {k} calls: {tool_call_counts[k]} examples")


if __name__ == "__main__":
    main()
