#!/usr/bin/env python3
"""
Annotate incorrect tool-calling trajectories using Gemini.

For each incorrect sample from tool refinement runs, sends the full trajectory
to Gemini-3-Flash asking it to identify WHERE the model should have called
`llm_refine` (or called it differently). Outputs annotated JSONL.

Requires: pip install google-genai
Set GEMINI_API_KEY environment variable before running.
"""

import argparse
import json
import os
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


ANNOTATION_PROMPT = """\
You are an expert at analyzing mathematical reasoning trajectories from an LLM \
that has access to a tool called `llm_refine`.

## What `llm_refine` does
When the model calls `llm_refine`, the current generation is sent to a summarizer \
that extracts key progress, results, and insights. The model then continues reasoning \
from a fresh context with only the problem statement and this summary. This is useful when:
- The model is going down a wrong path and needs to restart with a fresh perspective
- The model has completed a major step and should checkpoint before continuing
- The context is getting long and the model needs to compress its progress

## Your task
Analyze this INCORRECT trajectory where the model failed to reach the correct answer. \
Identify specific points where calling `llm_refine` (or calling it differently) could \
have helped the model reach the correct answer.

## Problem
{problem}

## Ground truth answer
{answer}

## Trajectory
The model made {num_tool_calls} tool call(s) across {num_rounds} round(s).
Done reason: {done_reason}

{trajectory_text}

## Instructions
Analyze this trajectory and identify WHERE `llm_refine` should have been called \
(or where the existing call was poorly timed). For each annotation:

1. Quote the specific text segment (10-30 words) where the tool should have been called
2. Explain WHY calling `llm_refine` at that point would help
3. Categorize the issue

Respond in this exact XML format:
<analysis>
<summary>Brief overall assessment of what went wrong (1-2 sentences)</summary>
<annotations>
<annotation>
<position>Exact quote from the trajectory text (10-30 words)</position>
<round>Which round number this occurs in (0-indexed)</round>
<reason>Why llm_refine should be called here</reason>
<category>One of: never_called, called_too_late, called_too_early, wrong_moment, should_call_again</category>
</annotation>
</annotations>
</analysis>

You may include 1-5 annotations. Focus on the most impactful missed opportunities.
"""


def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def format_trajectory(sample: Dict) -> str:
    """Format a sample's trajectory into readable text for Gemini."""
    parts = []
    rounds = sample.get("rounds", [])
    for i, r in enumerate(rounds):
        gen = r.get("current_round_generation", "")
        called_tool = r.get("called_tool", False)
        finish = r.get("finish_reason", "")
        refined = r.get("refined_context", "")

        parts.append(f"### Round {i}")
        if gen:
            # Truncate very long generations to avoid token limits
            if len(gen) > 8000:
                gen = gen[:4000] + "\n[... truncated ...]\n" + gen[-4000:]
            parts.append(gen)

        if called_tool:
            parts.append(f"\n[MODEL CALLED llm_refine]")
            if refined:
                if len(refined) > 2000:
                    refined = refined[:2000] + "\n[... truncated ...]"
                parts.append(f"[REFINED CONTEXT RETURNED:]\n{refined}")

        parts.append(f"[Finish reason: {finish}]")
        parts.append("")

    # Add final context if present
    final = sample.get("full_assistant_message", "")
    if final and len(rounds) > 1:
        parts.append("### Full assistant message (final)")
        if len(final) > 4000:
            final = final[:2000] + "\n[... truncated ...]\n" + final[-2000:]
        parts.append(final)

    return "\n".join(parts)


def parse_annotations(text: str) -> Optional[Dict]:
    """Parse Gemini's XML annotation response."""
    import re

    summary_match = re.search(r'<summary>(.*?)</summary>', text, re.DOTALL)
    summary = summary_match.group(1).strip() if summary_match else ""

    annotations = []
    for ann_match in re.finditer(r'<annotation>(.*?)</annotation>', text, re.DOTALL):
        ann_text = ann_match.group(1)

        pos_match = re.search(r'<position>(.*?)</position>', ann_text, re.DOTALL)
        round_match = re.search(r'<round>(.*?)</round>', ann_text, re.DOTALL)
        reason_match = re.search(r'<reason>(.*?)</reason>', ann_text, re.DOTALL)
        cat_match = re.search(r'<category>(.*?)</category>', ann_text, re.DOTALL)

        annotations.append({
            "position": pos_match.group(1).strip() if pos_match else "",
            "round": round_match.group(1).strip() if round_match else "",
            "reason": reason_match.group(1).strip() if reason_match else "",
            "category": cat_match.group(1).strip() if cat_match else "",
        })

    if not annotations:
        return None

    return {
        "summary": summary,
        "annotations": annotations,
    }


def annotate_trajectory(client, model: str, problem: str, answer: str,
                        sample: Dict, max_retries: int = 10) -> Optional[Dict]:
    """Send a single trajectory to Gemini for annotation."""
    trajectory_text = format_trajectory(sample)
    num_rounds = len(sample.get("rounds", []))
    num_tool_calls = sample.get("num_tool_calls", 0)
    done_reason = sample.get("done_reason", "unknown")

    prompt = ANNOTATION_PROMPT.format(
        problem=problem,
        answer=answer,
        num_tool_calls=num_tool_calls,
        num_rounds=num_rounds,
        done_reason=done_reason,
        trajectory_text=trajectory_text,
    )

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
            )
            text = response.text.strip()
            result = parse_annotations(text)
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


def load_verification(verify_path: str) -> Dict:
    """Load verification JSON and build a map of (problem_idx -> sample_results).

    Returns dict mapping problem index (line number in JSONL) to list of
    per-sample correctness booleans.
    """
    with open(verify_path, 'r') as f:
        data = json.load(f)

    # verify_solutions.py outputs {filename: {details: [...]}}
    # Get the first (only) file's results
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
            # Single sample format
            results[idx] = [detail.get("is_correct", False)]

    return results


def load_existing_annotations(output_path: str) -> set:
    """Load existing annotations for resume support. Returns set of (line_idx, sample_idx) keys."""
    existing = set()
    if not os.path.exists(output_path):
        return existing
    try:
        with open(output_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    li = item.get("line_idx", -1)
                    si = item.get("sample_idx", -1)
                    if item.get("annotation") is not None:
                        existing.add((li, si))
        print(f"  Resuming: found {len(existing)} existing annotations")
    except Exception as e:
        print(f"  Warning: could not load existing annotations: {e}")
    return existing


def main():
    parser = argparse.ArgumentParser(
        description="Annotate incorrect tool-calling trajectories using Gemini"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Tool refinement output JSONL file"
    )
    parser.add_argument(
        "--verify", type=str, required=True,
        help="Verification JSON output from verify_solutions.py"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSONL file (default: <input>_annotated.jsonl)"
    )
    parser.add_argument(
        "--model", type=str, default="gemini-3-flash-preview",
        help="Gemini model to use (default: gemini-3-flash-preview)"
    )
    parser.add_argument(
        "--mode", type=str, default="all_incorrect",
        choices=["all_incorrect", "no_correct_sample"],
        help="Which samples to annotate: "
             "'all_incorrect' = every incorrect sample, "
             "'no_correct_sample' = only problems where NO sample got it right "
             "(default: all_incorrect)"
    )
    parser.add_argument(
        "--max-samples", type=int, default=-1,
        help="Max incorrect samples per problem to annotate (-1 for all, default: -1)"
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
        help="Resume from existing output file"
    )

    args = parser.parse_args()

    # Check API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    # Load data
    print(f"Loading trajectories from {args.input}...")
    trajectory_data = load_jsonl(args.input)
    print(f"  Loaded {len(trajectory_data)} problems")

    print(f"Loading verification from {args.verify}...")
    verify_results = load_verification(args.verify)
    print(f"  Loaded verification for {len(verify_results)} problems")

    # Set output path
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_annotated.jsonl")

    # Load existing annotations for resume
    existing = set()
    if args.resume:
        existing = load_existing_annotations(args.output)

    # Collect annotation tasks
    tasks = []
    total_incorrect = 0
    problems_with_no_correct = 0

    for line_idx, item in enumerate(trajectory_data):
        problem = item.get("original_prompt", "")
        answer = item.get("answer", "")
        samples = item.get("samples", [])
        correctness = verify_results.get(line_idx, [])

        # Check if any sample is correct
        any_correct = any(correctness) if correctness else False
        if not any_correct:
            problems_with_no_correct += 1

        # Filter based on mode
        if args.mode == "no_correct_sample" and any_correct:
            continue

        incorrect_count = 0
        for s_idx, sample in enumerate(samples):
            is_correct = correctness[s_idx] if s_idx < len(correctness) else False
            if is_correct:
                continue

            total_incorrect += 1

            if (line_idx, s_idx) in existing:
                continue

            if args.max_samples != -1 and incorrect_count >= args.max_samples:
                continue

            tasks.append((line_idx, s_idx, problem, answer, sample))
            incorrect_count += 1

    print(f"\nStats:")
    print(f"  Total problems: {len(trajectory_data)}")
    print(f"  Problems with no correct sample: {problems_with_no_correct}")
    print(f"  Total incorrect samples: {total_incorrect}")
    print(f"  Tasks to annotate: {len(tasks)}")
    if existing:
        print(f"  Skipped (already annotated): {len(existing)}")

    if not tasks:
        print("No tasks to annotate. Exiting.")
        return

    print(f"\nAnnotating {len(tasks)} trajectories using {args.model} "
          f"with {args.num_workers} workers...")

    # Annotate with thread pool
    completed = 0
    failed = 0
    results = []

    def annotate_task(task):
        line_idx, s_idx, problem, answer, sample = task
        result = annotate_trajectory(client, args.model, problem, answer, sample)
        return line_idx, s_idx, result

    # Open output file in append mode for incremental writing
    output_file = open(args.output, 'a', encoding='utf-8')

    try:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {}
            for i, task in enumerate(tasks):
                future = executor.submit(annotate_task, task)
                futures[future] = task
                if i < args.num_workers:
                    time.sleep(args.delay)

            for future in as_completed(futures):
                line_idx, s_idx, result = future.result()
                completed += 1

                task = futures[future]
                _, _, problem, answer, sample = task

                # Write result immediately
                output_item = {
                    "line_idx": line_idx,
                    "sample_idx": s_idx,
                    "problem": problem[:200],  # Truncated for readability
                    "answer": answer,
                    "num_tool_calls": sample.get("num_tool_calls", 0),
                    "done_reason": sample.get("done_reason", ""),
                    "annotation": result,
                }
                output_file.write(json.dumps(output_item, ensure_ascii=False) + '\n')
                output_file.flush()

                if result:
                    n_ann = len(result.get("annotations", []))
                    cats = [a["category"] for a in result.get("annotations", [])]
                    print(f"  [{completed}/{len(tasks)}] line={line_idx} sample={s_idx}: "
                          f"{n_ann} annotations, categories={cats}")
                else:
                    failed += 1
                    print(f"  [{completed}/{len(tasks)}] line={line_idx} sample={s_idx}: FAILED")

    finally:
        output_file.close()

    # Print summary
    print(f"\n{'='*60}")
    print(f"Annotation Summary")
    print(f"{'='*60}")
    print(f"Total annotated: {completed}")
    print(f"Failed: {failed}")
    print(f"Output: {args.output}")

    # Category distribution
    if os.path.exists(args.output):
        category_counts = defaultdict(int)
        total_annotations = 0
        with open(args.output, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                ann = item.get("annotation")
                if ann:
                    for a in ann.get("annotations", []):
                        category_counts[a.get("category", "unknown")] += 1
                        total_annotations += 1

        print(f"\nCategory distribution ({total_annotations} total annotations):")
        for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
            pct = count / total_annotations * 100 if total_annotations > 0 else 0
            print(f"  {cat}: {count} ({pct:.1f}%)")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
