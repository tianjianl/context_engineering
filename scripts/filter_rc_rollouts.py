"""
Rejection-sample RC rollouts to build high-quality tool-calling SFT data.

Grades every round's \\boxed{} answer against ground truth using math-verify.
Keeps trajectories where refinement helped (round 1 wrong → later correct)
or round 1 was already correct (no-tool examples).

Optionally filters by actual token count with --tokenizer_path to enforce
a hard max sequence length for training (replaces the old filter_sft_32k.py).

Usage:
  python scripts/filter_rc_rollouts.py \
      --rollout_dirs /scratch/.../rc_sft_rollouts /scratch/.../rc_sft_rollouts_b2 \
      --output_dir /scratch/.../datasets/rc_sft_filtered \
      --no_tool_ratio 0.5 \
      --tokenizer_path /scratch/.../models/Qwen3-4B-Instruct-2507 \
      --max_tokens 32768
"""

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Import prompts from the RL pipeline
sys.path.insert(0, str(Path(__file__).parent.parent / "training" / "tool_refinement_rl"))
from prompts import SYSTEM_PROMPT, LLM_REFINE_TOOL, CONTINUATION_INSTRUCTIONS

# Import math verification
sys.path.insert(0, str(Path(__file__).parent.parent / "inference"))
from verify_utils import verify_batch

TOOL_DEF_STR = json.dumps([LLM_REFINE_TOOL])

ROLE_MAP = {"human": "user", "gpt": "assistant", "observation": "tool"}


def count_tokens(example: dict, tokenizer) -> int:
    """Count tokens for a sharegpt example using actual tokenizer."""
    messages = []
    if example.get("system"):
        messages.append({"role": "system", "content": example["system"]})
    for turn in example["conversations"]:
        role = ROLE_MAP.get(turn["from"], turn["from"])
        messages.append({"role": role, "content": turn["value"] or ""})
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return len(tokenizer.encode(text, add_special_tokens=False))


def extract_boxed(text: str) -> str | None:
    """Extract the last \\boxed{...} answer from text."""
    matches = []
    i = 0
    while i < len(text):
        idx = text.find("\\boxed{", i)
        if idx == -1:
            break
        depth = 0
        j = idx + 7
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                if depth == 0:
                    matches.append(text[idx + 7 : j])
                    break
                depth -= 1
            j += 1
        i = j + 1 if j < len(text) else len(text)
    return matches[-1].strip() if matches else None


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> tags, keep content."""
    text = re.sub(r"<think>\s*", "", text)
    text = re.sub(r"\s*</think>\s*", "\n\n", text)
    return text.strip()


def grade_rollouts(rollouts: list[dict], num_workers: int = 8) -> list[dict]:
    """Grade each round of each rollout against ground truth.

    Adds 'round_correctness' field: list of bool per reasoning round.
    """
    # Collect all (gold, generated) pairs for batch verification
    verify_pairs = []
    pair_map = []  # (rollout_idx, round_idx)

    for ri, rollout in enumerate(rollouts):
        label = rollout["label"]
        for si, reasoning in enumerate(rollout.get("reasoning_store", [])):
            boxed = extract_boxed(reasoning)
            if boxed is not None:
                verify_pairs.append((label, f"\\boxed{{{boxed}}}"))
                pair_map.append((ri, si, True))
            else:
                pair_map.append((ri, si, False))

    # Batch verify
    if verify_pairs:
        results = verify_batch(verify_pairs, timeout=5, num_workers=num_workers)
    else:
        results = []

    # Map results back
    result_iter = iter(results)
    for ri, rollout in enumerate(rollouts):
        rollout["round_correctness"] = []

    for ri, si, has_boxed in pair_map:
        if has_boxed:
            is_correct, status, _ = next(result_iter)
            rollouts[ri]["round_correctness"].append(is_correct)
        else:
            rollouts[ri]["round_correctness"].append(None)  # no answer

    return rollouts


def classify_trajectory(rollout: dict) -> str:
    """Classify a graded trajectory.

    Returns:
      "refinement_helped" - round 1 wrong, later round correct
      "round1_correct" - round 1 already correct
      "never_correct" - no round ever correct
      "regressed" - round 1 correct, later wrong (not useful for tool SFT)
      "no_round1_answer" - round 1 had no boxed answer, later correct
    """
    rc = rollout.get("round_correctness", [])
    if not rc:
        return "never_correct"

    r1 = rc[0]  # None if no answer, True/False otherwise
    later_correct = any(c is True for c in rc[1:])

    if r1 is True:
        return "round1_correct"
    elif r1 is False and later_correct:
        return "refinement_helped"
    elif r1 is None and later_correct:
        return "no_round1_answer"
    else:
        return "never_correct"


def first_correct_round(rollout: dict) -> int | None:
    """Return 0-based index of first correct round, or None."""
    for i, c in enumerate(rollout.get("round_correctness", [])):
        if c is True:
            return i
    return None


def build_tool_example(rollout: dict) -> dict | None:
    """Build a tool-calling SFT example from a trajectory where refinement helped."""
    reasoning_store = rollout.get("reasoning_store", [])
    summarization_store = rollout.get("summarization_store", [])
    rc = rollout.get("round_correctness", [])

    # Find first correct round (must be > 0)
    correct_idx = first_correct_round(rollout)
    if correct_idx is None or correct_idx < 1:
        return None
    if correct_idx >= len(reasoning_store):
        return None

    # Build multi-turn conversation
    conversations = [{"from": "human", "value": rollout["problem"]}]

    for r in range(correct_idx):
        reasoning = strip_think_tags(reasoning_store[r])
        # Minimum reasoning quality check
        if len(reasoning) < 200:
            return None

        # Add reasoning + tool call
        reasoning_with_tool = (
            reasoning + '\n<tool_call>\n{"name": "llm_refine", "arguments": {}}\n</tool_call>'
        )
        conversations.append({"from": "gpt", "value": reasoning_with_tool})

        # Add summary as observation
        if r < len(summarization_store):
            summary = strip_think_tags(summarization_store[r])
            # NOTE: Do NOT wrap in <tool_response> tags here — LLaMA-Factory's
            # qwen3 format_observation template already adds them.
            obs = f"{summary}{CONTINUATION_INSTRUCTIONS}"
            conversations.append({"from": "observation", "value": obs})
        else:
            return None  # missing summary

    # Final correct round
    correct_reasoning = strip_think_tags(reasoning_store[correct_idx])
    if len(correct_reasoning) < 200:
        return None
    conversations.append({"from": "gpt", "value": correct_reasoning})

    return {
        "conversations": conversations,
        "system": SYSTEM_PROMPT,
        "tools": TOOL_DEF_STR,
    }


def build_no_tool_example(rollout: dict) -> dict | None:
    """Build a no-tool example from a trajectory where round 1 was correct."""
    reasoning_store = rollout.get("reasoning_store", [])
    if not reasoning_store:
        return None

    reasoning = strip_think_tags(reasoning_store[0])
    if len(reasoning) < 500:
        return None
    if not extract_boxed(reasoning):
        return None

    return {
        "conversations": [
            {"from": "human", "value": rollout["problem"]},
            {"from": "gpt", "value": reasoning},
        ],
        "system": SYSTEM_PROMPT,
        "tools": TOOL_DEF_STR,
    }


def load_rollouts(rollout_dirs: list[str]) -> list[dict]:
    """Load all rollouts from shard files across directories."""
    all_rollouts = []
    for d in rollout_dirs:
        for shard_file in sorted(Path(d).glob("shard_*.json")):
            if "_annotated" in shard_file.name or "_timing" in shard_file.name:
                continue
            with open(shard_file) as f:
                data = json.load(f)
            print(f"  Loaded {len(data)} rollouts from {shard_file}")
            all_rollouts.extend(data)
    return all_rollouts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout_dirs", nargs="+", required=True,
                        help="Directories with shard_*.json rollout files")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--no_tool_ratio", type=float, default=0.5,
                        help="Ratio of no-tool to tool examples (default: 0.5)")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="HuggingFace tokenizer path for accurate token counting")
    parser.add_argument("--max_tokens", type=int, default=32768,
                        help="Max tokens per example (default: 32768). "
                             "Uses tokenizer if --tokenizer_path is set, otherwise char count.")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Workers for math-verify (default: 8)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--best_per_problem", action="store_true",
                        help="Keep only the best trajectory per problem (earliest correct round)")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer if specified
    tokenizer = None
    if args.tokenizer_path:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_path, trust_remote_code=True
        )
        print(f"Loaded tokenizer from {args.tokenizer_path}")
        print(f"Max tokens filter: {args.max_tokens}")
    else:
        print(f"No tokenizer specified, using char-count filter: {args.max_tokens}")

    # Load all rollouts
    print("Loading rollouts...")
    all_rollouts = load_rollouts(args.rollout_dirs)
    print(f"Total rollouts: {len(all_rollouts)}")

    # Count unique problems
    problem_ids = set(r["problem_id"] for r in all_rollouts)
    print(f"Unique problems: {len(problem_ids)}")

    # Grade all rollouts
    print("\nGrading rollouts with math-verify...")
    all_rollouts = grade_rollouts(all_rollouts, num_workers=args.num_workers)

    # Classify
    category_counts = Counter()
    for r in all_rollouts:
        cat = classify_trajectory(r)
        r["_category"] = cat
        category_counts[cat] += 1

    print(f"\n{'='*60}")
    print("TRAJECTORY CLASSIFICATION")
    print(f"{'='*60}")
    for cat, count in category_counts.most_common():
        print(f"  {cat:25s}: {count:6d} ({100*count/len(all_rollouts):.1f}%)")

    # Group by problem_id
    by_problem = defaultdict(list)
    for r in all_rollouts:
        by_problem[r["problem_id"]].append(r)

    # Build tool examples
    tool_examples = []
    tool_problems = set()

    for pid, rollouts in by_problem.items():
        # Get all "refinement_helped" or "no_round1_answer" trajectories
        candidates = [r for r in rollouts if r["_category"] in ("refinement_helped", "no_round1_answer")]
        if not candidates:
            continue

        if args.best_per_problem:
            # Pick trajectory with earliest correct round
            candidates.sort(key=lambda r: first_correct_round(r) or 999)
            candidates = candidates[:1]

        for r in candidates:
            ex = build_tool_example(r)
            if ex:
                if tokenizer:
                    length = count_tokens(ex, tokenizer)
                else:
                    length = sum(len(c["value"]) for c in ex["conversations"])
                if length <= args.max_tokens:
                    tool_examples.append(ex)
                    tool_problems.add(pid)
                    if args.best_per_problem:
                        break  # one per problem

    # Build no-tool examples
    no_tool_examples = []
    for pid, rollouts in by_problem.items():
        if pid in tool_problems:
            continue  # don't use the same problem for both
        r1_correct = [r for r in rollouts if r["_category"] == "round1_correct"]
        if r1_correct:
            ex = build_no_tool_example(r1_correct[0])
            if ex:
                no_tool_examples.append(ex)

    print(f"\n{'='*60}")
    print("SFT EXAMPLE COUNTS")
    print(f"{'='*60}")
    print(f"  Tool examples (raw):    {len(tool_examples)}")
    print(f"  No-tool examples (raw): {len(no_tool_examples)}")

    # Cap no-tool examples
    random.shuffle(no_tool_examples)
    target_no_tool = int(len(tool_examples) * args.no_tool_ratio)
    no_tool_selected = no_tool_examples[:target_no_tool]

    # Combine and shuffle
    all_examples = tool_examples + no_tool_selected
    random.shuffle(all_examples)

    n_tool = len(tool_examples)
    n_notool = len(no_tool_selected)
    total = len(all_examples)

    print(f"\n{'='*60}")
    print("FINAL DATASET")
    print(f"{'='*60}")
    print(f"  Tool examples:    {n_tool} ({100*n_tool/total:.1f}%)" if total else "  No examples")
    print(f"  No-tool examples: {n_notool} ({100*n_notool/total:.1f}%)" if total else "")
    print(f"  Total:            {total}")

    if not all_examples:
        print("No examples to write!")
        return

    # Length stats
    tool_lens = [sum(len(c["value"]) for c in ex["conversations"]) for ex in tool_examples]
    notool_lens = [sum(len(c["value"]) for c in ex["conversations"]) for ex in no_tool_selected]
    if tool_lens:
        print(f"\n  Tool lengths:    mean={sum(tool_lens)//len(tool_lens)}, "
              f"median={sorted(tool_lens)[len(tool_lens)//2]}, max={max(tool_lens)}")
    if notool_lens:
        print(f"  No-tool lengths: mean={sum(notool_lens)//len(notool_lens)}, "
              f"median={sorted(notool_lens)[len(notool_lens)//2]}, max={max(notool_lens)}")

    # Split train/val (90/10)
    val_size = max(1, total // 10)
    val_examples = all_examples[:val_size]
    train_examples = all_examples[val_size:]

    train_path = os.path.join(args.output_dir, "train.json")
    val_path = os.path.join(args.output_dir, "val.json")

    with open(train_path, "w") as f:
        json.dump(train_examples, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {train_path} ({len(train_examples)} train)")

    with open(val_path, "w") as f:
        json.dump(val_examples, f, indent=2, ensure_ascii=False)
    print(f"Wrote {val_path} ({len(val_examples)} val)")


if __name__ == "__main__":
    main()
