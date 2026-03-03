#!/usr/bin/env python3
"""Batch Questions Experiment — feed N questions in a single prompt.

Instead of multi-turn conversations, this experiment feeds all questions into
a single prompt separated by delimiters, asking the model to solve them in
order. This tests whether model accuracy degrades for later questions within
a single long generation.

Models (via OpenRouter):
    google/gemini-3-flash-preview
    minimax/minimax-m2.5
    moonshotai/kimi-k2.5

Usage:
    source ~/.bashrc   # sets OPENROUTER_API_KEY

    python -m context_rot_prelim.batch_questions_inference \
        --model google/gemini-3-flash-preview \
        --input_file /scratch/dkhasha1/tli104/datasets/hmmt_nov_2025/hmmt_nov_2025.jsonl \
        --num_questions 10 \
        --seeds 42 123 456 \
        --max_tokens 320000 \
        --output_file /scratch/dkhasha1/tli104/outputs/context_rot/batch_gemini-3-flash_hmmt_nov.jsonl
"""

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inference.data_utils import strip_thinking
from inference.verify_utils import verify_batch

SCRATCH = "/scratch/dkhasha1/tli104"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Batch Questions Experiment via OpenRouter API")
    p.add_argument("--model", required=True,
                   help="OpenRouter model ID (e.g. google/gemini-3-flash-preview)")
    p.add_argument("--input_file", required=True,
                   help="Path to HMMT Nov 2025 JSONL file")
    p.add_argument("--num_questions", type=int, default=10,
                   help="Number of questions per batch (default: 10)")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456],
                   help="Random seeds for question ordering")
    p.add_argument("--max_tokens", type=int, default=320000,
                   help="Max generation tokens (default: 320000)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--output_file", required=True)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_problems(input_file: str) -> List[Dict]:
    problems = []
    with open(input_file) as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    for p in problems:
        if "original_problem" in p:
            p["problem_text"] = p["original_problem"]
        else:
            text = p.get("prompt", "")
            if "Problem:" in text:
                text = text.split("Problem:", 1)[1]
            if text.strip().endswith("Solution:"):
                text = text.rsplit("Solution:", 1)[0]
            p["problem_text"] = text.strip()
        if "problem_id" not in p:
            p["problem_id"] = f"hmmt_nov_{p.get('problem_idx', 0)}"
    return problems


def select_questions(problems: List[Dict], num: int,
                     seed: int) -> List[Dict]:
    rng = random.Random(seed)
    order = list(problems)
    rng.shuffle(order)
    return order[:num]


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

BATCH_PROMPT_TEMPLATE = """\
Solve the following {n} math problems in order. For each problem, show your \
reasoning step by step and provide your final answer in \\boxed{{}}.

Separate your solutions using the delimiters shown below. For each problem, \
start your solution with the corresponding delimiter (e.g., \
"=== Solution 1 ===") before writing your work.

{problems_block}

Now solve each problem in order. Remember to use "=== Solution N ===" \
delimiters before each solution and put your final answer in \\boxed{{}}."""


def build_batch_prompt(questions: List[Dict]) -> str:
    parts = []
    for i, q in enumerate(questions, 1):
        parts.append(f"=== Problem {i} ===\n{q['problem_text']}")
    problems_block = "\n\n".join(parts)
    return BATCH_PROMPT_TEMPLATE.format(n=len(questions),
                                        problems_block=problems_block)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def extract_boxed(text: str) -> Optional[str]:
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    start = idx + len("\\boxed{")
    depth, pos = 1, start
    while pos < len(text) and depth > 0:
        if text[pos] == '{':
            depth += 1
        elif text[pos] == '}':
            depth -= 1
        pos += 1
    return text[start:pos - 1].strip() if depth == 0 else None


def parse_batch_response(response: str, num_questions: int
                         ) -> List[Tuple[str, Optional[str]]]:
    """Split a batch response into per-question (section_text, boxed_answer).

    Splits on "=== Solution N ===" delimiters. Falls back to "=== Problem N ==="
    if solution delimiters aren't found (some models echo the problem headers).
    """
    cleaned = strip_thinking(response)

    # Try splitting on solution delimiters first
    pattern = r"===\s*Solution\s+(\d+)\s*==="
    splits = list(re.finditer(pattern, cleaned, re.IGNORECASE))

    # Fallback: some models may use "Problem N" headers instead
    if len(splits) < 2:
        pattern = r"===\s*Problem\s+(\d+)\s*==="
        splits = list(re.finditer(pattern, cleaned, re.IGNORECASE))

    results: List[Tuple[str, Optional[str]]] = []

    if len(splits) >= 2:
        # Extract text between consecutive delimiters
        sections = {}
        for i, match in enumerate(splits):
            q_num = int(match.group(1))
            start = match.end()
            end = splits[i + 1].start() if i + 1 < len(splits) else len(cleaned)
            sections[q_num] = cleaned[start:end].strip()

        for q in range(1, num_questions + 1):
            section = sections.get(q, "")
            boxed = extract_boxed(section) if section else None
            results.append((section, boxed))
    else:
        # Last resort: try to find all \boxed{} answers in order
        # Split by any numbered header pattern
        boxed_pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
        all_boxed = re.findall(boxed_pattern, cleaned)
        for q in range(num_questions):
            if q < len(all_boxed):
                results.append(("", all_boxed[q].strip()))
            else:
                results.append(("", None))

    return results


# ---------------------------------------------------------------------------
# API call with retries
# ---------------------------------------------------------------------------

async def call_api(client: AsyncOpenAI, model: str,
                   messages: List[Dict], max_tokens: int,
                   temperature: float,
                   max_retries: int = 8) -> Dict:
    for attempt in range(max_retries):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            choice = resp.choices[0]
            return {
                "content": choice.message.content or "",
                "finish_reason": choice.finish_reason,
                "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
            }
        except Exception as e:
            wait = min(2 ** attempt + random.random(), 60)
            if attempt < max_retries - 1:
                print(f"    API error (attempt {attempt+1}/{max_retries}): "
                      f"{type(e).__name__}: {e}, retrying in {wait:.1f}s...")
                await asyncio.sleep(wait)
            else:
                print(f"    API error (final): {e}")
                return {
                    "content": "",
                    "finish_reason": "error",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "error": str(e),
                }
    return {"content": "", "finish_reason": "error",
            "prompt_tokens": 0, "completion_tokens": 0}


# ---------------------------------------------------------------------------
# Verification helper
# ---------------------------------------------------------------------------

def _verify_answers(items: List[Tuple[str, str]]):
    if not items:
        return []
    try:
        return verify_batch(items, timeout=10.0)
    except Exception:
        from inference.verify_utils import _verify_single
        return [_verify_single(item) for item in items]


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

async def run_batch(client: AsyncOpenAI, model: str,
                    questions: List[Dict], seed: int,
                    args) -> List[Dict]:
    """Run a single batch of questions and return per-question results."""
    prompt = build_batch_prompt(questions)
    messages = [{"role": "user", "content": prompt}]

    print(f"\n  Seed {seed}: sending {len(questions)} questions "
          f"(prompt ~{len(prompt)} chars)...")
    t0 = time.time()

    r = await call_api(client, model, messages, args.max_tokens,
                       args.temperature)

    elapsed = time.time() - t0
    print(f"    Response: {r['completion_tokens']} tokens, "
          f"finish_reason={r['finish_reason']}, {elapsed:.0f}s")

    # Parse per-question answers
    parsed_sections = parse_batch_response(r["content"], len(questions))

    # Verify each answer
    verify_items = []
    for i, (q, (section, boxed)) in enumerate(zip(questions, parsed_sections)):
        gt = q.get("answer", "")
        # Use the section text for verification if boxed not found
        text_for_verify = section if section else r["content"]
        verify_items.append((gt, text_for_verify))

    vfy = _verify_answers(verify_items)

    results = []
    for i, (q, (section, boxed)) in enumerate(zip(questions, parsed_sections)):
        is_ok, _st, parsed_ans = vfy[i] if i < len(vfy) else (False, "error", None)
        results.append({
            "mode": "batch",
            "seed": seed,
            "question_position": i + 1,  # 1-indexed position in batch
            "num_questions": len(questions),
            "problem_id": q["problem_id"],
            "problem_idx": q.get("problem_idx", i),
            "ground_truth": q.get("answer", ""),
            "extracted_answer": boxed,
            "parsed_answer": parsed_ans,
            "is_correct": is_ok,
            "section_text": section[:2000] if section else "",
            "prompt_tokens": r["prompt_tokens"],
            "completion_tokens": r["completion_tokens"],
            "finish_reason": r["finish_reason"],
            "model": model,
            "full_response_length": len(r["content"]),
        })

    correct = sum(res["is_correct"] for res in results)
    print(f"    Accuracy: {correct}/{len(results)} = "
          f"{correct/len(results):.1%}")

    # Per-position breakdown
    for res in results:
        mark = "+" if res["is_correct"] else "-"
        print(f"      Q{res['question_position']}: [{mark}] "
              f"extracted={res['extracted_answer']}  "
              f"gt={res['ground_truth']}")

    return results


async def async_main():
    args = parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set. Run: source ~/.bashrc")
        sys.exit(1)

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    print(f"{'='*70}")
    print(f"Batch Questions Experiment (OpenRouter API)")
    print(f"{'='*70}")
    print(f"Model        : {args.model}")
    print(f"Input        : {args.input_file}")
    print(f"Questions    : {args.num_questions}")
    print(f"Seeds        : {args.seeds}")
    print(f"Max tokens   : {args.max_tokens}")
    print(f"Temperature  : {args.temperature}")

    problems = load_problems(args.input_file)
    print(f"Total probs  : {len(problems)}")
    print(f"{'='*70}")

    # Also run baseline (each question individually) for comparison
    all_results = []

    # Load checkpoint if exists
    ckpt_path = args.output_file + ".ckpt"
    completed_phases = set()
    if Path(ckpt_path).exists():
        with open(ckpt_path) as f:
            state = json.load(f)
        all_results = state.get("results", [])
        completed_phases = set(state.get("completed_phases", []))
        print(f"  Loaded checkpoint: {len(all_results)} results, "
              f"phases={sorted(completed_phases)}")

    def save_ckpt():
        with open(ckpt_path, "w") as f:
            json.dump({"results": all_results,
                       "completed_phases": sorted(completed_phases)}, f)

    # Run baseline (individual questions) for the first seed's selection
    first_seed_questions = select_questions(problems, args.num_questions,
                                            args.seeds[0])
    if "baseline" not in completed_phases:
        print(f"\n--- Baseline (individual questions) ---")
        for i, q in enumerate(first_seed_questions):
            prompt = (
                "Solve the following math problem. Show your reasoning "
                "step by step and provide your final answer in \\boxed{}."
                f"\n\nProblem: {q['problem_text']}"
            )
            messages = [{"role": "user", "content": prompt}]
            r = await call_api(client, args.model, messages,
                               args.max_tokens, args.temperature)
            content = strip_thinking(r["content"])
            boxed = extract_boxed(content)
            vfy = _verify_answers([(q.get("answer", ""), content)])
            is_ok, _st, parsed = vfy[0] if vfy else (False, "error", None)
            mark = "+" if is_ok else "-"
            print(f"  Q{i+1}: [{mark}] extracted={boxed}  gt={q.get('answer','')}")
            all_results.append({
                "mode": "baseline",
                "seed": args.seeds[0],
                "question_position": i + 1,
                "num_questions": args.num_questions,
                "problem_id": q["problem_id"],
                "problem_idx": q.get("problem_idx", i),
                "ground_truth": q.get("answer", ""),
                "extracted_answer": boxed,
                "parsed_answer": parsed,
                "is_correct": is_ok,
                "section_text": content[:2000],
                "prompt_tokens": r["prompt_tokens"],
                "completion_tokens": r["completion_tokens"],
                "finish_reason": r["finish_reason"],
                "model": args.model,
                "full_response_length": len(r["content"]),
            })
        completed_phases.add("baseline")
        save_ckpt()

        bl = [r for r in all_results if r["mode"] == "baseline"]
        c = sum(r["is_correct"] for r in bl)
        print(f"  Baseline: {c}/{len(bl)} = {c/len(bl):.1%}\n")
    else:
        print("  Baseline: already completed (skipping)")

    # Run batch for each seed
    for seed in args.seeds:
        phase = f"batch_{seed}"
        if phase in completed_phases:
            print(f"  Batch seed={seed}: already completed (skipping)")
            continue

        questions = select_questions(problems, args.num_questions, seed)
        batch_results = await run_batch(client, args.model, questions,
                                        seed, args)
        all_results.extend(batch_results)
        completed_phases.add(phase)
        save_ckpt()

    # Save final results
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    # Cleanup checkpoint
    if Path(ckpt_path).exists():
        Path(ckpt_path).unlink()

    # Summary
    print(f"\n{'='*70}")
    print(f"Summary")
    print(f"{'='*70}")

    bl = [r for r in all_results if r["mode"] == "baseline"]
    if bl:
        c = sum(r["is_correct"] for r in bl)
        print(f"Baseline       : {c}/{len(bl)} = {c/len(bl):.1%}")

    batch = [r for r in all_results if r["mode"] == "batch"]
    if batch:
        c = sum(r["is_correct"] for r in batch)
        print(f"Batch (all)    : {c}/{len(batch)} = {c/len(batch):.1%}")

        # Per-position accuracy across all seeds
        from collections import defaultdict
        by_pos = defaultdict(list)
        for r in batch:
            by_pos[r["question_position"]].append(r["is_correct"])
        print(f"\nPer-position accuracy (across all seeds):")
        for pos in sorted(by_pos):
            arr = by_pos[pos]
            c = sum(arr)
            print(f"  Position {pos:2d}: {c}/{len(arr)} = {c/len(arr):.1%}")

    print(f"\nSaved {len(all_results)} results -> {args.output_file}")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
