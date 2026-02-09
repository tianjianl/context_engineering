#!/usr/bin/env python3
"""
Grade mathematical proof solutions using Gemini 3 Preview.

Reads inference output JSONL files and grades each solution against the
marking scheme and reference solution using Gemini as an expert grader.

Requires: pip install google-genai
Set GEMINI_API_KEY environment variable before running.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from google import genai
except ImportError:
    print("Error: google-genai not installed. Install with: pip install google-genai")
    sys.exit(1)


GRADING_PROMPT = """\
You are an expert mathematical olympiad grader. Grade the following solution according to the marking scheme provided.

## Problem
{problem}

## Reference Solution
{reference_solution}

## Marking Scheme
{marking_scheme}

## Student Solution to Grade
{student_solution}

## Instructions
1. Carefully read the marking scheme and understand each checkpoint.
2. Compare the student solution against each checkpoint in the marking scheme.
3. Award points according to the rubric. Be strict but fair.
4. For each checkpoint, state whether it was met and the points awarded.

Output your grading in the following JSON format (and nothing else):
```json
{{
  "checkpoints": [
    {{"checkpoint": "<description>", "max_points": <int>, "awarded": <int>, "justification": "<brief reason>"}},
    ...
  ],
  "total_score": <int>,
  "max_score": <int>,
  "overall_comment": "<brief overall assessment>"
}}
```"""


def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_problem_metadata(metadata_file: str) -> Dict[str, Dict]:
    """Load problem metadata (marking_scheme, reference_solution) keyed by problem_idx."""
    metadata = {}
    data = load_jsonl(metadata_file)
    for item in data:
        pid = item.get("problem_idx", "")
        metadata[pid] = {
            "marking_scheme": item.get("marking_scheme", ""),
            "reference_solution": item.get("reference_solution", ""),
            "prompt": item.get("prompt", ""),
        }
    return metadata


def get_solution_text(sample: Dict) -> str:
    """Extract the generated solution text from a sample."""
    # Check rounds first
    rounds = sample.get("rounds", [])
    if rounds:
        last_round = rounds[-1]
        text = last_round.get("current_round_generation", "")
        if text:
            # Prepend refined context if available
            refined = sample.get("final_refined_context", "")
            if refined:
                return refined + "\n\n" + text
            return text

    # Fall back to other fields
    for key in ["full_assistant_message", "final_refined_context", "generation"]:
        text = sample.get(key, "")
        if text and text.strip():
            return text

    return ""


def grade_solution(client, model: str, problem: str, reference_solution: str,
                   marking_scheme: str, student_solution: str,
                   max_retries: int = 3) -> Optional[Dict]:
    """Grade a single solution using Gemini."""
    prompt = GRADING_PROMPT.format(
        problem=problem,
        reference_solution=reference_solution,
        marking_scheme=marking_scheme,
        student_solution=student_solution,
    )

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
            )
            text = response.text.strip()

            # Extract JSON from response (handle markdown code blocks)
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            result = json.loads(text)
            return result

        except json.JSONDecodeError as e:
            print(f"    JSON parse error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                wait = 2 ** (attempt + 2)
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"    API error (attempt {attempt + 1}): {error_str[:100]}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Grade proof solutions using Gemini 3 Preview"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Inference output JSONL file to grade"
    )
    parser.add_argument(
        "--metadata", type=str, required=True,
        help="Problem metadata JSONL file with marking_scheme and reference_solution "
             "(e.g., the original proofbench_hf.jsonl)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSONL file for grading results (default: <input>_graded.jsonl)"
    )
    parser.add_argument(
        "--model", type=str, default="gemini-3-pro-preview",
        help="Gemini model to use (default: gemini-3-pro-preview)"
    )
    parser.add_argument(
        "--max-samples", type=int, default=1,
        help="Max samples per problem to grade (default: 1, use -1 for all)"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="Number of parallel API workers (default: 4)"
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="Delay between API calls in seconds (default: 0.5)"
    )

    args = parser.parse_args()

    # Check API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    # Load data
    print(f"Loading inference results from {args.input}...")
    inference_data = load_jsonl(args.input)
    print(f"  Loaded {len(inference_data)} problems")

    print(f"Loading problem metadata from {args.metadata}...")
    metadata = load_problem_metadata(args.metadata)
    print(f"  Loaded metadata for {len(metadata)} problems")

    # Set output path
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_graded.jsonl")

    # Collect grading tasks
    tasks = []  # (problem_idx, sample_idx, problem, ref_sol, marking, student_sol)
    for item in inference_data:
        problem_idx = item.get("problem_idx", "")
        meta = metadata.get(problem_idx, {})

        if not meta.get("marking_scheme"):
            continue

        problem = meta.get("prompt", item.get("original_prompt", ""))
        reference_solution = meta.get("reference_solution", "")
        marking_scheme = meta.get("marking_scheme", "")

        samples = item.get("samples", None)
        if samples is not None:
            max_s = len(samples) if args.max_samples == -1 else min(args.max_samples, len(samples))
            for s_idx in range(max_s):
                solution = get_solution_text(samples[s_idx])
                if solution.strip():
                    tasks.append((problem_idx, s_idx, problem, reference_solution,
                                  marking_scheme, solution))
        else:
            solution = get_solution_text(item)
            if solution.strip():
                tasks.append((problem_idx, 0, problem, reference_solution,
                              marking_scheme, solution))

    print(f"\nGrading {len(tasks)} solutions using {args.model} with {args.num_workers} workers...")

    # Grade with thread pool
    results = {}  # (problem_idx, sample_idx) -> grading_result
    completed = 0

    def grade_task(task):
        problem_idx, s_idx, problem, ref_sol, marking, student_sol = task
        result = grade_solution(client, args.model, problem, ref_sol, marking, student_sol)
        return problem_idx, s_idx, result

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {}
        for i, task in enumerate(tasks):
            future = executor.submit(grade_task, task)
            futures[future] = task
            # Stagger submissions to avoid burst
            if i < args.num_workers:
                time.sleep(args.delay)

        for future in as_completed(futures):
            problem_idx, s_idx, result = future.result()
            results[(problem_idx, s_idx)] = result
            completed += 1

            if result:
                score = result.get("total_score", "?")
                max_score = result.get("max_score", "?")
                print(f"  [{completed}/{len(tasks)}] {problem_idx} sample {s_idx}: "
                      f"{score}/{max_score}")
            else:
                print(f"  [{completed}/{len(tasks)}] {problem_idx} sample {s_idx}: "
                      f"FAILED")

    # Build output
    print(f"\nWriting results to {args.output}...")
    graded_output = []
    problem_scores = defaultdict(list)

    for item in inference_data:
        problem_idx = item.get("problem_idx", "")
        meta = metadata.get(problem_idx, {})

        graded_item = {
            "problem_idx": problem_idx,
            "problem": meta.get("prompt", item.get("original_prompt", "")),
            "problem_type": item.get("problem_type", []),
            "graded_samples": [],
        }

        samples = item.get("samples", None)
        if samples is not None:
            max_s = len(samples) if args.max_samples == -1 else min(args.max_samples, len(samples))
            for s_idx in range(max_s):
                grading = results.get((problem_idx, s_idx))
                graded_item["graded_samples"].append({
                    "sample_idx": s_idx,
                    "grading": grading,
                })
                if grading:
                    problem_scores[problem_idx].append(
                        grading.get("total_score", 0) / max(grading.get("max_score", 7), 1)
                    )
        else:
            grading = results.get((problem_idx, 0))
            graded_item["graded_samples"].append({
                "sample_idx": 0,
                "grading": grading,
            })
            if grading:
                problem_scores[problem_idx].append(
                    grading.get("total_score", 0) / max(grading.get("max_score", 7), 1)
                )

        graded_output.append(graded_item)

    with open(args.output, 'w', encoding='utf-8') as f:
        for item in graded_output:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # Print summary
    total_problems = len(problem_scores)
    if total_problems > 0:
        avg_scores = [sum(scores) / len(scores) for scores in problem_scores.values()]
        overall_avg = sum(avg_scores) / len(avg_scores) * 100

        # By contest type
        contest_scores = defaultdict(list)
        for item in graded_output:
            pid = item["problem_idx"]
            ptypes = item.get("problem_type", [])
            if pid in problem_scores:
                avg = sum(problem_scores[pid]) / len(problem_scores[pid])
                for ptype in ptypes:
                    contest_scores[ptype].append(avg)

        print(f"\n{'='*60}")
        print(f"Grading Summary")
        print(f"{'='*60}")
        print(f"Problems graded: {total_problems}")
        print(f"Overall average score: {overall_avg:.1f}%")
        print(f"\nBy contest:")
        for contest, scores in sorted(contest_scores.items()):
            avg = sum(scores) / len(scores) * 100
            print(f"  {contest}: {avg:.1f}% ({len(scores)} problems)")
        print(f"{'='*60}")

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
