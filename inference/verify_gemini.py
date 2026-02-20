#!/usr/bin/env python3
"""
Verify math solutions using Gemini as a judge.

Reads inference output JSONL files (from tool_refinement.py, baseline_vllm.py, etc.)
and grades each solution using Gemini with categorical correctness labels.

Requires: pip install google-genai
Set GEMINI_API_KEY environment variable before running.
"""

import argparse
import json
import os
import sys
import time
import asyncio
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

try:
    from google import genai
    from google.genai import errors
except ImportError:
    print("Error: google-genai not installed. Install with: pip install google-genai")
    sys.exit(1)


GRADING_PROMPT = """\
Carefully analyze the given problem statement and the proposed solution, and then write \
out your analysis regarding the correctness of the proposed solution.
Keep your analysis concise (under 500 words). Focus on identifying the key errors or confirming correctness.
After the analysis, you must provide a score based on the following criteria:
• incorrect: The solution is completely incorrect or irrelevant.
• partial: The solution is partially correct but has significant errors or omissions.
• almost: The solution is almost correct but contains minor errors or inaccuracies.
• correct: The solution is fully correct and complete.
The very last part of your response must be only one of the following words: incorrect, \
partial, almost, or correct.
Problem:{problem} Solution:{solution}"""

VALID_LABELS = {"incorrect", "partial", "almost", "correct"}


def load_jsonl(file_path: str) -> list:
    """Load data from a JSONL file."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def get_solution_text(sample: dict) -> str:
    """Extract the generated solution text from a sample."""
    rounds = sample.get("rounds", [])
    if rounds:
        last_round = rounds[-1]
        text = last_round.get("current_round_generation", "")
        if text:
            refined = sample.get("final_refined_context", "")
            if refined:
                return refined + "\n\n" + text
            return text

    for key in ["full_assistant_message", "final_refined_context", "generation"]:
        text = sample.get(key, "")
        if text and text.strip():
            return text

    return ""


def get_solution_for_round(sample: dict, round_idx: int) -> str:
    """Extract the solution text up to a specific round (0-indexed cumulative)."""
    rounds = sample.get("rounds", [])
    if not rounds or round_idx >= len(rounds):
        return get_solution_text(sample)

    target_round = rounds[round_idx]
    text = target_round.get("current_round_generation", "")
    refined = target_round.get("refined_context", "")
    if refined and text:
        return refined + "\n\n" + text
    if text:
        return text
    if refined:
        return refined
    return get_solution_text(sample)


def extract_label(response_text: str) -> str:
    """Extract the categorical label from the last word of the response."""
    text = response_text.strip().rstrip(".").strip().lower()
    # Check last word
    last_word = text.split()[-1] if text.split() else ""
    last_word = last_word.strip(".,;:!?\"'()")
    if last_word in VALID_LABELS:
        return last_word
    # Fallback: search for label near the end
    for label in VALID_LABELS:
        if label in text[-100:]:
            return label
    return "unknown"


async def grade_solution_async(
    client, model: str, problem: str, solution: str, max_retries: int = 5
) -> dict:
    """Grade a single solution using Gemini (async)."""
    prompt = GRADING_PROMPT.format(problem=problem, solution=solution)

    for attempt in range(max_retries):
        try:
            response = await client.aio.models.generate_content(
                model=model,
                contents=prompt,
                config={
                    "max_output_tokens": 8192,
                    "thinking_config": {"thinking_budget": 0},
                },
            )
            text = response.text.strip()
            label = extract_label(text)

            usage_info = {}
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage_info = {
                    "prompt_tokens": getattr(
                        response.usage_metadata, "prompt_token_count", None
                    ),
                    "completion_tokens": getattr(
                        response.usage_metadata, "candidates_token_count", None
                    ),
                    "total_tokens": getattr(
                        response.usage_metadata, "total_token_count", None
                    ),
                }

            return {
                "response": text,
                "label": label,
                "usage": usage_info,
                "status": "success",
            }
        except errors.ServerError as e:
            if attempt < max_retries - 1:
                wait = 2**attempt
                print(f"    Server error (attempt {attempt + 1}): {e}")
                await asyncio.sleep(wait)
            else:
                return {"response": "", "label": "error", "status": "failed", "error": str(e)}
        except errors.ClientError as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"    Client error (attempt {attempt + 1}): {e}")
                await asyncio.sleep(wait)
            else:
                return {"response": "", "label": "error", "status": "failed", "error": str(e)}
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2**attempt
                print(f"    Error (attempt {attempt + 1}): {str(e)[:100]}")
                await asyncio.sleep(wait)
            else:
                return {"response": "", "label": "error", "status": "failed", "error": str(e)}

    return {"response": "", "label": "error", "status": "failed", "error": "max retries"}


def load_completed(output_file: Path) -> dict:
    """Load already-completed gradings for resume. Returns {(problem_idx, sample_idx, round): result}."""
    completed = {}
    if output_file.exists():
        with open(output_file, "r") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    key = (data["problem_idx"], data["sample_idx"], data.get("round", -1))
                    completed[key] = data
                except (json.JSONDecodeError, KeyError):
                    continue
    return completed


async def main_async(args):
    # Check API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    # Load inference data
    print(f"Loading inference results from {args.input}...")
    inference_data = load_jsonl(args.input)
    print(f"  Loaded {len(inference_data)} problems")

    # Output path
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_gemini_graded.jsonl")

    output_path = Path(args.output)

    # Load completed for resume
    completed = load_completed(output_path)
    print(f"  Found {len(completed)} already completed gradings")

    # Build tasks: (problem_idx, sample_idx, round, problem_text, solution_text)
    tasks = []
    for item in inference_data:
        problem_idx = item.get("problem_idx", item.get("problem_id", ""))
        problem_text = item.get("original_prompt", item.get("prompt", ""))
        samples = item.get("samples", None)

        if samples is None:
            # Single solution format
            sol = get_solution_text(item)
            if sol.strip():
                key = (problem_idx, 0, -1)
                if key not in completed:
                    tasks.append((problem_idx, 0, -1, problem_text, sol))
            continue

        max_s = len(samples) if args.max_samples == -1 else min(args.max_samples, len(samples))

        for s_idx in range(max_s):
            sample = samples[s_idx]

            if args.by_round:
                # Grade each round cumulatively
                num_rounds = len(sample.get("rounds", []))
                for r in range(num_rounds):
                    key = (problem_idx, s_idx, r)
                    if key not in completed:
                        sol = get_solution_for_round(sample, r)
                        if sol.strip():
                            tasks.append((problem_idx, s_idx, r, problem_text, sol))
            else:
                # Grade final solution only
                sol = get_solution_text(sample)
                if sol.strip():
                    key = (problem_idx, s_idx, -1)
                    if key not in completed:
                        tasks.append((problem_idx, s_idx, -1, problem_text, sol))

    print(f"\nGrading {len(tasks)} solutions using {args.model}")
    print(f"  Max concurrent: {args.max_concurrent}")
    print(f"  Output: {args.output}")

    if not tasks:
        print("No tasks to grade. Done.")
        return

    # Process with async semaphore
    semaphore = asyncio.Semaphore(args.max_concurrent)
    graded_count = 0

    async def grade_with_semaphore(task):
        nonlocal graded_count
        problem_idx, s_idx, round_idx, problem_text, solution = task
        async with semaphore:
            result = await grade_solution_async(
                client, args.model, problem_text, solution, max_retries=args.max_retries
            )
            result["problem_idx"] = problem_idx
            result["sample_idx"] = s_idx
            result["round"] = round_idx
            graded_count += 1
            label = result["label"]
            print(f"  [{graded_count}/{len(tasks)}] {problem_idx} s{s_idx}"
                  f"{f' r{round_idx}' if round_idx >= 0 else ''}: {label}")
            return result

    # Run all tasks
    batch_size = args.batch_size
    all_results = []

    with open(args.output, "a") as f:
        for batch_start in range(0, len(tasks), batch_size):
            batch = tasks[batch_start : batch_start + batch_size]
            coros = [grade_with_semaphore(t) for t in batch]
            results = await asyncio.gather(*coros)

            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                all_results.append(result)
            f.flush()

            if args.delay_between_batches > 0 and batch_start + batch_size < len(tasks):
                await asyncio.sleep(args.delay_between_batches)

    # Combine with previously completed
    for data in completed.values():
        all_results.append(data)

    # Print summary
    print_summary(all_results, inference_data, args)


def print_summary(all_results: list, inference_data: list, args):
    """Print grading summary with pass@k style metrics."""
    print(f"\n{'=' * 60}")
    print("Gemini Grading Summary")
    print(f"{'=' * 60}")

    # Count labels
    label_counts = defaultdict(int)
    for r in all_results:
        label_counts[r.get("label", "unknown")] += 1

    total = sum(label_counts.values())
    print(f"\nTotal graded: {total}")
    for label in ["correct", "almost", "partial", "incorrect", "error", "unknown"]:
        count = label_counts.get(label, 0)
        if count > 0:
            print(f"  {label}: {count} ({count / total * 100:.1f}%)")

    # Per-problem accuracy (correct = correct, almost counts as partial credit)
    problem_results = defaultdict(list)
    for r in all_results:
        pid = r.get("problem_idx", "")
        problem_results[pid].append(r.get("label", "unknown"))

    # pass@k where "correct" counts as pass
    num_problems = len(problem_results)
    if num_problems > 0:
        pass_at_1_sum = 0
        for pid, labels in problem_results.items():
            n = len(labels)
            c = sum(1 for l in labels if l == "correct")
            # pass@1 = 1 - C(n-c, 1)/C(n, 1) = c/n
            pass_at_1_sum += c / n if n > 0 else 0

        pass_at_1 = pass_at_1_sum / num_problems * 100

        print(f"\npass@1 (strict=correct only): {pass_at_1:.2f}%")

        # Also compute with "almost" counting
        pass_at_1_lenient_sum = 0
        for pid, labels in problem_results.items():
            n = len(labels)
            c = sum(1 for l in labels if l in ("correct", "almost"))
            pass_at_1_lenient_sum += c / n if n > 0 else 0

        pass_at_1_lenient = pass_at_1_lenient_sum / num_problems * 100
        print(f"pass@1 (lenient=correct+almost): {pass_at_1_lenient:.2f}%")

    # Per-category if available
    cat_results = defaultdict(lambda: defaultdict(list))
    pid_to_cat = {}
    for item in inference_data:
        pid = item.get("problem_idx", item.get("problem_id", ""))
        cat = item.get("category", item.get("subcategory", ""))
        if cat:
            pid_to_cat[pid] = cat

    for r in all_results:
        pid = r.get("problem_idx", "")
        cat = pid_to_cat.get(pid, "unknown")
        cat_results[cat][pid].append(r.get("label", "unknown"))

    if len(cat_results) > 1:
        print(f"\nBy category:")
        for cat in sorted(cat_results.keys()):
            problems = cat_results[cat]
            n_problems = len(problems)
            strict_sum = sum(
                sum(1 for l in labels if l == "correct") / len(labels)
                for labels in problems.values()
            )
            strict_pct = strict_sum / n_problems * 100 if n_problems > 0 else 0
            print(f"  {cat}: {strict_pct:.1f}% strict ({n_problems} problems)")

    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Verify math solutions using Gemini as a judge"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Inference output JSONL file to grade"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSONL file (default: <input>_gemini_graded.jsonl)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help="Gemini model to use (default: gemini-2.5-flash)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1,
        help="Max samples per problem to grade (default: 1, -1 for all)",
    )
    parser.add_argument(
        "--by-round",
        action="store_true",
        help="Grade each round cumulatively (instead of just the final solution)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Max concurrent API requests (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for processing (default: 50)",
    )
    parser.add_argument(
        "--delay-between-batches",
        type=float,
        default=0.5,
        help="Delay between batches in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Max retries per API call (default: 5)",
    )

    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
