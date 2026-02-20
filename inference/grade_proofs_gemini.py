#!/usr/bin/env python3
"""
Grade mathematical proof solutions using Gemini API.

Reads inference output JSONL files and grades each solution against the
marking scheme and reference solution using Gemini as an expert grader.
Uses an XML-based grading prompt (grading_prompt.txt) with scores 0-7.

Supports two output formats:
  - Refinement format: one line per problem with nested "samples" list
  - Baseline format: one line per sample with "generation" field

Requires: pip install google-genai
Set GEMINI_API_KEY environment variable before running.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from google import genai
except ImportError:
    print("Error: google-genai not installed. Install with: pip install google-genai")
    sys.exit(1)


def load_grading_prompt() -> str:
    """Load the grading prompt template from grading_prompt.txt."""
    prompt_path = Path(__file__).parent / "grading_prompt.txt"
    if not prompt_path.exists():
        print(f"Error: Grading prompt not found at {prompt_path}")
        sys.exit(1)
    return prompt_path.read_text(encoding="utf-8")


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
    """Load problem metadata keyed by prompt text (for baseline matching)."""
    metadata = {}
    data = load_jsonl(metadata_file)
    for item in data:
        prompt = item.get("prompt", "").strip()
        if prompt and prompt not in metadata:
            metadata[prompt] = {
                "marking_scheme": item.get("marking_scheme", ""),
                "reference_solution": item.get("reference_solution", ""),
                "problem_idx": item.get("problem_idx", ""),
                "problem_type": item.get("problem_type", []),
            }
    return metadata


def get_solution_text(sample: Dict) -> str:
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


def parse_xml_response(text: str) -> Optional[Dict]:
    """Parse the XML grading response into a dict with score, assessment, errors."""
    score_match = re.search(r'<score>\s*(\d+)\s*</score>', text)
    if not score_match:
        return None
    score = int(score_match.group(1))
    score = max(0, min(7, score))

    assessment_match = re.search(r'<assessment>(.*?)</assessment>', text, re.DOTALL)
    assessment = assessment_match.group(1).strip() if assessment_match else ""

    errors_match = re.search(r'<errors>(.*?)</errors>', text, re.DOTALL)
    errors = errors_match.group(1).strip() if errors_match else ""

    return {
        "score": score,
        "max_score": 7,
        "assessment": assessment,
        "errors": errors,
    }


def grade_solution(client, model: str, prompt_template: str,
                   problem: str, human_solution: str,
                   marking_scheme: str, solution: str,
                   max_retries: int = 5) -> Optional[Dict]:
    """Grade a single solution using Gemini."""
    prompt = prompt_template.format(
        problem=problem,
        human_solution=human_solution,
        marking_scheme=marking_scheme,
        solution=solution,
    )

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
            )
            text = response.text.strip()
            result = parse_xml_response(text)
            if result is not None:
                return result
            print(f"    XML parse failed (attempt {attempt + 1}), raw: {text[:200]}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                wait = 2 ** (attempt + 2)
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"    API error (attempt {attempt + 1}): {error_str[:200]}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

    return None


def detect_format(data: List[Dict]) -> str:
    """Detect whether the input is refinement format or baseline format."""
    if not data:
        return "unknown"
    first = data[0]
    if "samples" in first:
        return "refinement"
    if "generation" in first:
        return "baseline"
    return "unknown"


def collect_tasks_refinement(inference_data: List[Dict], max_samples: int,
                             existing: set) -> Tuple[List[tuple], int]:
    """Collect grading tasks from refinement-format output (RC, CTR).
    Uses embedded metadata from each line. Keys by (line_idx, sample_idx)."""
    tasks = []
    skipped = 0
    for line_idx, item in enumerate(inference_data):
        marking_scheme = item.get("marking_scheme", "")
        if not marking_scheme.strip():
            continue

        problem = item.get("original_prompt", "")
        reference_solution = item.get("reference_solution", "")
        problem_idx = item.get("problem_idx", "")
        problem_type = item.get("problem_type", [])

        samples = item.get("samples", [])
        max_s = len(samples) if max_samples == -1 else min(max_samples, len(samples))
        for s_idx in range(max_s):
            if (line_idx, s_idx) in existing:
                skipped += 1
                continue
            solution = get_solution_text(samples[s_idx])
            if solution.strip():
                tasks.append((line_idx, s_idx, problem_idx, problem_type,
                              problem, reference_solution, marking_scheme, solution))

    return tasks, skipped


def collect_tasks_baseline(inference_data: List[Dict], metadata: Dict[str, Dict],
                           max_samples: int, existing: set) -> Tuple[List[tuple], int]:
    """Collect grading tasks from baseline-format output.
    Groups by problem_id, matches metadata by prompt text."""
    # Group samples by problem_id
    problems = defaultdict(list)
    for item in inference_data:
        pid = item.get("problem_id", "")
        problems[pid].append(item)

    tasks = []
    skipped = 0
    for line_idx, (pid, samples) in enumerate(sorted(problems.items())):
        prompt = samples[0].get("prompt", "").strip()
        meta = metadata.get(prompt, {})
        marking_scheme = meta.get("marking_scheme", "")
        if not marking_scheme.strip():
            continue

        reference_solution = meta.get("reference_solution", "")
        problem_idx = meta.get("problem_idx", pid)
        problem_type = meta.get("problem_type", [])

        max_s = len(samples) if max_samples == -1 else min(max_samples, len(samples))
        for s_idx in range(max_s):
            if (line_idx, s_idx) in existing:
                skipped += 1
                continue
            solution = samples[s_idx].get("generation", "")
            if solution.strip():
                tasks.append((line_idx, s_idx, problem_idx, problem_type,
                              prompt, reference_solution, marking_scheme, solution))

    return tasks, skipped


def load_existing_results(output_path: str) -> Tuple[Dict, set]:
    """Load already-graded results for resume support.
    Returns (full_data_by_line, set_of_graded_keys)."""
    existing_keys = set()
    existing_data = {}
    if not os.path.exists(output_path):
        return existing_data, existing_keys
    try:
        data = load_jsonl(output_path)
        for item in data:
            line_idx = item.get("line_idx", -1)
            for gs in item.get("graded_samples", []):
                s_idx = gs.get("sample_idx", 0)
                grading = gs.get("grading")
                if grading is not None:
                    existing_keys.add((line_idx, s_idx))
                    existing_data[(line_idx, s_idx)] = grading
        print(f"  Resuming: found {len(existing_keys)} existing graded samples")
    except Exception as e:
        print(f"  Warning: could not load existing results: {e}")
    return existing_data, existing_keys


def main():
    parser = argparse.ArgumentParser(
        description="Grade proof solutions using Gemini"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Inference output JSONL file to grade"
    )
    parser.add_argument(
        "--metadata", type=str, default=None,
        help="Problem metadata JSONL (required for baseline format, optional for refinement)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSONL file for grading results (default: <input>_graded.jsonl)"
    )
    parser.add_argument(
        "--model", type=str, default="gemini-3-flash-preview",
        help="Gemini model to use (default: gemini-3-flash-preview)"
    )
    parser.add_argument(
        "--max-samples", type=int, default=1,
        help="Max samples per problem to grade (default: 1, use -1 for all)"
    )
    parser.add_argument(
        "--num-workers", type=int, default=8,
        help="Number of parallel API workers (default: 8)"
    )
    parser.add_argument(
        "--delay", type=float, default=0.2,
        help="Delay between API calls in seconds (default: 0.2)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing output file, skipping already-graded samples"
    )

    args = parser.parse_args()

    # Check API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    # Load grading prompt template
    prompt_template = load_grading_prompt()
    print("Loaded grading prompt template")

    # Load data
    print(f"Loading inference results from {args.input}...")
    inference_data = load_jsonl(args.input)
    fmt = detect_format(inference_data)
    print(f"  Loaded {len(inference_data)} entries (format: {fmt})")

    # Set output path
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_graded.jsonl")

    # Load existing results for resume
    existing_data, existing_keys = {}, set()
    if args.resume:
        existing_data, existing_keys = load_existing_results(args.output)

    # Collect grading tasks
    if fmt == "refinement":
        tasks, skipped = collect_tasks_refinement(
            inference_data, args.max_samples, existing_keys)
    elif fmt == "baseline":
        if not args.metadata:
            print("Error: --metadata required for baseline format")
            sys.exit(1)
        print(f"Loading problem metadata from {args.metadata}...")
        metadata = load_problem_metadata(args.metadata)
        print(f"  Loaded metadata for {len(metadata)} unique problems")
        tasks, skipped = collect_tasks_baseline(
            inference_data, metadata, args.max_samples, existing_keys)
    else:
        print("Error: Could not detect input format")
        sys.exit(1)

    if skipped:
        print(f"  Skipped {skipped} already-graded samples")
    print(f"\nGrading {len(tasks)} solutions using {args.model} with {args.num_workers} workers...")

    # Grade with thread pool
    results = dict(existing_data)
    completed = 0
    failed = 0

    def grade_task(task):
        line_idx, s_idx, problem_idx, problem_type, problem, ref_sol, marking, student_sol = task
        result = grade_solution(client, args.model, prompt_template,
                                problem, ref_sol, marking, student_sol)
        return line_idx, s_idx, problem_idx, result

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {}
        for i, task in enumerate(tasks):
            future = executor.submit(grade_task, task)
            futures[future] = task
            if i < args.num_workers:
                time.sleep(args.delay)

        for future in as_completed(futures):
            line_idx, s_idx, problem_idx, result = future.result()
            results[(line_idx, s_idx)] = result
            completed += 1

            if result:
                score = result.get("score", "?")
                print(f"  [{completed}/{len(tasks)}] {problem_idx} (line {line_idx}) "
                      f"sample {s_idx}: {score}/7")
            else:
                failed += 1
                print(f"  [{completed}/{len(tasks)}] {problem_idx} (line {line_idx}) "
                      f"sample {s_idx}: FAILED")

    if failed:
        print(f"\n  Warning: {failed}/{len(tasks)} samples failed to grade")

    # Build output - one line per input line
    print(f"\nWriting results to {args.output}...")
    graded_output = []
    problem_scores = defaultdict(list)

    if fmt == "refinement":
        for line_idx, item in enumerate(inference_data):
            problem_idx = item.get("problem_idx", "")
            problem_type = item.get("problem_type", [])
            problem = item.get("original_prompt", "")

            graded_item = {
                "line_idx": line_idx,
                "problem_idx": problem_idx,
                "problem": problem,
                "problem_type": problem_type,
                "graded_samples": [],
            }

            samples = item.get("samples", [])
            max_s = len(samples) if args.max_samples == -1 else min(args.max_samples, len(samples))
            for s_idx in range(max_s):
                grading = results.get((line_idx, s_idx))
                graded_item["graded_samples"].append({
                    "sample_idx": s_idx,
                    "grading": grading,
                })
                if grading:
                    problem_scores[(line_idx, problem_idx)].append(
                        grading.get("score", 0) / 7.0
                    )

            graded_output.append(graded_item)

    elif fmt == "baseline":
        problems = defaultdict(list)
        for item in inference_data:
            problems[item.get("problem_id", "")].append(item)

        for line_idx, (pid, samples) in enumerate(sorted(problems.items())):
            prompt = samples[0].get("prompt", "").strip()
            meta = metadata.get(prompt, {})
            problem_idx = meta.get("problem_idx", pid)
            problem_type = meta.get("problem_type", [])

            graded_item = {
                "line_idx": line_idx,
                "problem_idx": problem_idx,
                "problem": prompt,
                "problem_type": problem_type,
                "graded_samples": [],
            }

            max_s = len(samples) if args.max_samples == -1 else min(args.max_samples, len(samples))
            for s_idx in range(max_s):
                grading = results.get((line_idx, s_idx))
                graded_item["graded_samples"].append({
                    "sample_idx": s_idx,
                    "grading": grading,
                })
                if grading:
                    problem_scores[(line_idx, problem_idx)].append(
                        grading.get("score", 0) / 7.0
                    )

            graded_output.append(graded_item)

    with open(args.output, 'w', encoding='utf-8') as f:
        for item in graded_output:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # Print summary
    total_items = len(problem_scores)
    if total_items > 0:
        avg_scores = [sum(scores) / len(scores) for scores in problem_scores.values()]
        overall_avg = sum(avg_scores) / len(avg_scores) * 100

        # By contest type
        contest_scores = defaultdict(list)
        for item in graded_output:
            key = (item["line_idx"], item["problem_idx"])
            ptypes = item.get("problem_type", [])
            if key in problem_scores:
                avg = sum(problem_scores[key]) / len(problem_scores[key])
                for ptype in ptypes:
                    contest_scores[ptype].append(avg)

        print(f"\n{'='*60}")
        print(f"Grading Summary")
        print(f"{'='*60}")
        print(f"Model: {args.model}")
        print(f"Items graded: {total_items}")
        print(f"Overall average score: {overall_avg:.1f}%")
        print(f"\nBy contest:")
        for contest, scores in sorted(contest_scores.items()):
            avg = sum(scores) / len(scores) * 100
            print(f"  {contest}: {avg:.1f}% ({len(scores)} items)")
        print(f"{'='*60}")

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
