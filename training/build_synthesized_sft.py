#!/usr/bin/env python3
"""
Build SFT data from synthesized (tool-call-injected) trajectories.

Converts synthesized trajectories (where tool calls were injected at
Gemini-annotated positions) into LlamaFactory sharegpt format.

For the injected tool call that lacks a real refined_context, we use a
brief synthetic observation derived from the generation text. The remaining
rounds (with real observations) are included as-is.

Usage:
    python training/build_synthesized_sft.py \
        --input /scratch/dkhasha1/tli104/datasets/tool_sft/tool_call_truncated_synthesized.jsonl \
        --output /scratch/dkhasha1/tli104/datasets/tool_sft/synthesized_sft.json \
        --max_tokens 32768
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional


# Same constants as build_sft_data.py
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


def make_synthetic_observation(generation_text: str, max_chars: int = 800) -> str:
    """Create a brief synthetic observation from the generation text.

    Mimics what llm_refine would produce: a summary of the reasoning so far.
    Extracts key sentences from the generation, focusing on results and conclusions.
    """
    if not generation_text or len(generation_text) < 50:
        return "Progress has been captured. Continue solving the problem."

    # Split into sentences (rough)
    sentences = re.split(r'(?<=[.!?])\s+', generation_text.strip())
    # Filter out very short sentences and problem restatements
    meaningful = []
    for s in sentences:
        s = s.strip()
        if len(s) < 20:
            continue
        # Skip if it looks like a section header
        if s.startswith('#') or s.startswith('---'):
            continue
        meaningful.append(s)

    if not meaningful:
        return "Progress has been captured. Continue solving the problem."

    # Take a mix: first sentence (context) + last few sentences (latest progress)
    selected = []
    total_len = 0

    # Add first meaningful sentence for context
    selected.append(meaningful[0])
    total_len += len(meaningful[0])

    # Add sentences from the end (most recent progress)
    for s in reversed(meaningful[1:]):
        if total_len + len(s) > max_chars:
            break
        selected.insert(1, s)  # Insert after the first sentence
        total_len += len(s)

    return " ".join(selected)


def trajectory_hash(problem: str, rounds: List[Dict]) -> str:
    """Create a hash for deduplication."""
    parts = [problem]
    for r in rounds:
        parts.append(r.get("current_round_generation", ""))
        parts.append(str(r.get("called_tool", False)))
    content = "|||".join(parts)
    return hashlib.sha256(content.encode()).hexdigest()


def convert_synthesized_to_sharegpt(record: Dict) -> Optional[Dict]:
    """Convert a synthesized trajectory to LlamaFactory sharegpt format."""
    synth_rounds = record.get("synthesized_rounds", [])
    remain_rounds = record.get("remaining_rounds", [])
    original_prompt = record.get("original_prompt", "")

    if not synth_rounds:
        return None

    conversations = []

    # User message: problem + tool hint
    conversations.append({
        "from": "human",
        "value": original_prompt + LLM_REFINE_USER_HINT,
    })

    all_rounds = synth_rounds + remain_rounds

    for i, r in enumerate(all_rounds):
        generation = r.get("current_round_generation", "")
        called_tool = r.get("called_tool", False)
        refined_context = r.get("refined_context") or ""

        if called_tool:
            # Merge reasoning + tool call into single assistant turn
            # so model learns <tool_call> comes BEFORE <|im_end|> (EOS)
            gpt_text = generation if generation else "Let me refine my approach."
            tool_call_xml = f'\n<tool_call>\n{TOOL_CALL_CONTENT}\n</tool_call>'
            conversations.append({
                "from": "gpt",
                "value": gpt_text + tool_call_xml,
            })

            # For the injected round (no real observation), synthesize one
            if not refined_context:
                refined_context = make_synthetic_observation(generation)

            conversations.append({
                "from": "observation",
                "value": refined_context + CONTINUATION_INSTRUCTIONS,
            })
        else:
            # Final round: assistant generates without tool call
            if generation:
                conversations.append({
                    "from": "gpt",
                    "value": generation,
                })

    # Validate: must have at least user + one gpt turn
    gpt_turns = [t for t in conversations if t["from"] == "gpt"]
    if not gpt_turns:
        return None

    # Must end with a gpt turn (model output)
    if conversations[-1]["from"] != "gpt":
        # If ends with observation, that's fine if there's remaining text
        # But if conversation ends at function_call or observation with no
        # continuation, skip it
        return None

    return {
        "conversations": conversations,
        "tools": LLM_REFINE_TOOL_JSON,
        "system": SYSTEM_PROMPT,
    }


def estimate_tokens(example: Dict) -> int:
    """Rough token estimate: ~4 chars per token."""
    total_chars = len(example.get("system", ""))
    total_chars += len(example.get("tools", ""))
    for turn in example.get("conversations", []):
        total_chars += len(turn.get("value", ""))
        total_chars += 20
    return total_chars // 4


def main():
    parser = argparse.ArgumentParser(
        description="Build SFT data from synthesized trajectories"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Synthesized trajectories JSONL file"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=32768,
        help="Max token estimate for filtering (default: 32768)"
    )

    args = parser.parse_args()

    # Load synthesized trajectories
    records = []
    with open(args.input, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} synthesized trajectories")

    all_examples = []
    seen_hashes = set()
    stats = {
        "total": len(records),
        "invalid": 0,
        "too_long": 0,
        "duplicate": 0,
        "kept": 0,
    }

    for record in records:
        example = convert_synthesized_to_sharegpt(record)
        if example is None:
            stats["invalid"] += 1
            continue

        # Token length filter
        est_tokens = estimate_tokens(example)
        if est_tokens > args.max_tokens:
            stats["too_long"] += 1
            continue

        # Deduplication
        all_rounds = record.get("synthesized_rounds", []) + record.get("remaining_rounds", [])
        h = trajectory_hash(record.get("original_prompt", ""), all_rounds)
        if h in seen_hashes:
            stats["duplicate"] += 1
            continue
        seen_hashes.add(h)

        all_examples.append(example)
        stats["kept"] += 1

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_examples, f, ensure_ascii=False, indent=None)

    # Report
    print(f"\n{'='*60}")
    print(f"Synthesized SFT Data Build Summary")
    print(f"{'='*60}")
    print(f"Total records:      {stats['total']}")
    print(f"Invalid (skipped):  {stats['invalid']}")
    print(f"Too long (skipped): {stats['too_long']}")
    print(f"Duplicate (skipped):{stats['duplicate']}")
    print(f"Final examples:     {stats['kept']}")
    print(f"Output: {args.output}")
    print(f"{'='*60}")

    # Tool call distribution
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
