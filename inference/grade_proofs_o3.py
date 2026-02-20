#!/usr/bin/env python3
"""
Grade mathematical proof solutions using OpenAI o3.

Reads inference output JSONL files and grades each solution using o3.
Supports two grading schemes, auto-detected per problem:

  - Rubric scheme (ProofBench-HF): requires marking_scheme + reference_solution.
    Uses grading_prompt.txt. Outputs score 0-7 in XML.

  - Categorical scheme (ProofBench-60): requires grading_guidelines + solution.
    Outputs label in {incorrect, partial, almost, correct} in XML.

Supports two input formats:
  - Refinement format: one line per problem with nested "samples" list
  - Baseline format: one line per sample with "generation" field

Requires: pip install openai
Set OPENAI_API_KEY environment variable before running.
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
    from openai import OpenAI
    import openai
except ImportError:
    print("Error: openai not installed. Install with: pip install openai")
    sys.exit(1)


# ── Categorical grading prompt (ProofBench-60 style) ────────────────────────

CATEGORICAL_PROMPT = """\
You are an **expert math proof grader**. You are judging the correctness of an \
LLM-generated proof for a math problem.

### Input

* **Problem Statement**: The math problem being solved.
* **Reference Solution**: A correct solution provided for reference. Not the only valid path.
* **Grading Guidelines**: Describes what partial and near-complete proofs look like for this problem.
* **Proof Solution**: The proof to evaluate.

### Task

Assign exactly one of the following labels:

- **incorrect**: The proof is completely wrong, irrelevant, or makes no meaningful progress.
- **partial**: The proof makes significant progress (matches one or more "Partial" checkpoints) \
but is not nearly complete.
- **almost**: The proof is essentially correct but has a minor gap, omission, or unverified step \
(matches "Almost" criteria).
- **correct**: The proof is fully correct and complete.

Analyze carefully. Then respond with **only** well-formed XML:

<label>correct</label>
<assessment>Detailed step-by-step analysis referencing specific claims. \
Explain which checkpoints were met and why you chose this label.</assessment>
<errors>Specific issues found, or empty if correct.</errors>

----------------------------------------------------------

**Problem Statement**

{problem}

**Reference Solution**

{reference_solution}

**Grading Guidelines**

{grading_guidelines}

**Proof Solution**

{solution}
"""


def load_rubric_prompt() -> str:
    """Load the 0-7 rubric grading prompt template from grading_prompt.txt."""
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
                "marking_scheme":      item.get("marking_scheme", ""),
                "reference_solution":  item.get("reference_solution", ""),
                "grading_guidelines":  item.get("grading_guidelines", ""),
                "solution":            item.get("solution", ""),
                "problem_idx":         item.get("problem_idx", ""),
                "problem_type":        item.get("problem_type", []),
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


# ── Scheme detection ─────────────────────────────────────────────────────────

def detect_scheme(marking_scheme: str, reference_solution: str,
                  grading_guidelines: str, solution_ref: str) -> str:
    """Return 'rubric' (0-7) or 'categorical' based on available fields."""
    if marking_scheme.strip():
        return "rubric"
    if grading_guidelines.strip():
        return "categorical"
    return "unknown"


# ── Response parsing ─────────────────────────────────────────────────────────

def parse_rubric_response(text: str) -> Optional[Dict]:
    """Parse 0-7 XML grading response."""
    score_match = re.search(r'<score>\s*(\d+)\s*</score>', text)
    if not score_match:
        return None
    score = max(0, min(7, int(score_match.group(1))))

    assessment_match = re.search(r'<assessment>(.*?)</assessment>', text, re.DOTALL)
    assessment = assessment_match.group(1).strip() if assessment_match else ""

    errors_match = re.search(r'<errors>(.*?)</errors>', text, re.DOTALL)
    errors = errors_match.group(1).strip() if errors_match else ""

    return {
        "scheme": "rubric",
        "score": score,
        "max_score": 7,
        "assessment": assessment,
        "errors": errors,
    }


VALID_LABELS = {"incorrect", "partial", "almost", "correct"}

def parse_categorical_response(text: str) -> Optional[Dict]:
    """Parse categorical XML grading response."""
    label_match = re.search(r'<label>\s*(\w+)\s*</label>', text)
    if not label_match:
        return None
    label = label_match.group(1).strip().lower()
    if label not in VALID_LABELS:
        return None

    assessment_match = re.search(r'<assessment>(.*?)</assessment>', text, re.DOTALL)
    assessment = assessment_match.group(1).strip() if assessment_match else ""

    errors_match = re.search(r'<errors>(.*?)</errors>', text, re.DOTALL)
    errors = errors_match.group(1).strip() if errors_match else ""

    return {
        "scheme": "categorical",
        "label": label,
        "assessment": assessment,
        "errors": errors,
    }


# ── Core grading function ─────────────────────────────────────────────────────

def grade_solution(client: OpenAI, model: str, rubric_prompt_template: str,
                   problem: str, reference_solution: str,
                   marking_scheme: str, grading_guidelines: str,
                   solution_ref: str, student_solution: str,
                   reasoning_effort: str = "medium",
                   max_retries: int = 5) -> Optional[Dict]:
    """Grade a single solution. Auto-selects rubric or categorical scheme."""
    scheme = detect_scheme(marking_scheme, reference_solution,
                           grading_guidelines, solution_ref)

    if scheme == "rubric":
        prompt = rubric_prompt_template.format(
            problem=problem,
            human_solution=reference_solution,
            marking_scheme=marking_scheme,
            solution=student_solution,
        )
    elif scheme == "categorical":
        prompt = CATEGORICAL_PROMPT.format(
            problem=problem,
            reference_solution=solution_ref,
            grading_guidelines=grading_guidelines,
            solution=student_solution,
        )
    else:
        return None  # no usable grading info

    for attempt in range(max_retries):
        try:
            kwargs = dict(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=8192,
            )
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort

            response = client.chat.completions.create(**kwargs)
            text = response.choices[0].message.content.strip()

            result = (parse_rubric_response(text) if scheme == "rubric"
                      else parse_categorical_response(text))

            if result is not None:
                usage = response.usage
                if usage:
                    result["usage"] = {
                        "prompt_tokens":     usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                        "total_tokens":      usage.total_tokens,
                    }
                return result

            print(f"    XML parse failed (attempt {attempt + 1}), raw: {text[:200]}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

        except openai.RateLimitError:
            wait = 2 ** (attempt + 2)
            print(f"    Rate limited (attempt {attempt + 1}), waiting {wait}s...")
            time.sleep(wait)
        except openai.APIStatusError as e:
            print(f"    API error (attempt {attempt + 1}): {e.status_code} {str(e)[:200]}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
        except Exception as e:
            print(f"    Error (attempt {attempt + 1}): {str(e)[:200]}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    return None


# ── Format detection ──────────────────────────────────────────────────────────

def detect_format(data: List[Dict]) -> str:
    if not data:
        return "unknown"
    first = data[0]
    if "samples" in first:
        return "refinement"
    if "generation" in first:
        return "baseline"
    return "unknown"


# ── Task collection ───────────────────────────────────────────────────────────

def collect_tasks_refinement(inference_data: List[Dict], max_samples: int,
                             existing: set) -> Tuple[List[tuple], int]:
    """Collect tasks from refinement-format output.
    Supports both ProofBench-HF (marking_scheme) and ProofBench-60 (grading_guidelines)."""
    tasks = []
    skipped = 0
    for line_idx, item in enumerate(inference_data):
        marking_scheme     = item.get("marking_scheme", "")
        reference_solution = item.get("reference_solution", "")
        grading_guidelines = item.get("grading_guidelines", "")
        solution_ref       = item.get("solution", "")

        scheme = detect_scheme(marking_scheme, reference_solution,
                               grading_guidelines, solution_ref)
        if scheme == "unknown":
            continue

        problem      = item.get("original_prompt", "")
        problem_idx  = item.get("problem_idx", "")
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
                              problem, reference_solution, marking_scheme,
                              grading_guidelines, solution_ref, solution))

    return tasks, skipped


def collect_tasks_baseline(inference_data: List[Dict], metadata: Dict[str, Dict],
                           max_samples: int, existing: set) -> Tuple[List[tuple], int]:
    """Collect tasks from baseline-format output."""
    problems = defaultdict(list)
    for item in inference_data:
        pid = item.get("problem_id", "")
        problems[pid].append(item)

    tasks = []
    skipped = 0
    for line_idx, (pid, samples) in enumerate(sorted(problems.items())):
        prompt = samples[0].get("prompt", "").strip()
        meta   = metadata.get(prompt, {})

        marking_scheme     = meta.get("marking_scheme", "")
        reference_solution = meta.get("reference_solution", "")
        grading_guidelines = meta.get("grading_guidelines", "")
        solution_ref       = meta.get("solution", "")

        scheme = detect_scheme(marking_scheme, reference_solution,
                               grading_guidelines, solution_ref)
        if scheme == "unknown":
            continue

        problem_idx  = meta.get("problem_idx", pid)
        problem_type = meta.get("problem_type", [])

        max_s = len(samples) if max_samples == -1 else min(max_samples, len(samples))
        for s_idx in range(max_s):
            if (line_idx, s_idx) in existing:
                skipped += 1
                continue
            solution = samples[s_idx].get("generation", "")
            if solution.strip():
                tasks.append((line_idx, s_idx, problem_idx, problem_type,
                              prompt, reference_solution, marking_scheme,
                              grading_guidelines, solution_ref, solution))

    return tasks, skipped


# ── Resume support ────────────────────────────────────────────────────────────

def load_existing_results(output_path: str) -> Tuple[Dict, set]:
    existing_keys = set()
    existing_data = {}
    if not os.path.exists(output_path):
        return existing_data, existing_keys
    try:
        data = load_jsonl(output_path)
        for item in data:
            line_idx = item.get("line_idx", -1)
            for gs in item.get("graded_samples", []):
                s_idx   = gs.get("sample_idx", 0)
                grading = gs.get("grading")
                if grading is not None:
                    existing_keys.add((line_idx, s_idx))
                    existing_data[(line_idx, s_idx)] = grading
        print(f"  Resuming: found {len(existing_keys)} existing graded samples")
    except Exception as e:
        print(f"  Warning: could not load existing results: {e}")
    return existing_data, existing_keys


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Grade proof solutions using OpenAI o3 (rubric or categorical)"
    )
    parser.add_argument("--input",   type=str, required=True,
                        help="Inference output JSONL file to grade")
    parser.add_argument("--metadata", type=str, default=None,
                        help="Problem metadata JSONL (required for baseline format)")
    parser.add_argument("--output",  type=str, default=None,
                        help="Output JSONL (default: <input>_o3_graded.jsonl)")
    parser.add_argument("--model",   type=str, default="o3",
                        help="OpenAI model (default: o3)")
    parser.add_argument("--reasoning-effort", type=str, default="medium",
                        choices=["low", "medium", "high"],
                        help="Reasoning effort for o3/o3-mini (default: medium)")
    parser.add_argument("--max-samples", type=int, default=1,
                        help="Max samples per problem (default: 1, -1 = all)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Parallel API workers (default: 4)")
    parser.add_argument("--delay",   type=float, default=0.5,
                        help="Delay between submissions (default: 0.5s)")
    parser.add_argument("--resume",  action="store_true",
                        help="Resume from existing output, skip already-graded")

    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    rubric_prompt_template = load_rubric_prompt()
    print("Loaded rubric grading prompt template")

    print(f"Loading inference results from {args.input}...")
    inference_data = load_jsonl(args.input)
    fmt = detect_format(inference_data)
    print(f"  Loaded {len(inference_data)} entries (format: {fmt})")

    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_o3_graded.jsonl")

    existing_data, existing_keys = {}, set()
    if args.resume:
        existing_data, existing_keys = load_existing_results(args.output)

    metadata = {}
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

    # Report scheme breakdown
    rubric_count = sum(
        1 for t in tasks
        if detect_scheme(t[6], t[5], t[7], t[8]) == "rubric"
    )
    categorical_count = len(tasks) - rubric_count
    print(f"\nGrading {len(tasks)} solutions using {args.model} "
          f"(reasoning_effort={args.reasoning_effort}, {args.num_workers} workers)")
    print(f"  Rubric (0-7):    {rubric_count}")
    print(f"  Categorical:     {categorical_count}")

    results = dict(existing_data)
    completed = 0
    failed = 0

    def grade_task(task):
        (line_idx, s_idx, problem_idx, problem_type,
         problem, reference_solution, marking_scheme,
         grading_guidelines, solution_ref, student_sol) = task
        result = grade_solution(
            client, args.model, rubric_prompt_template,
            problem, reference_solution, marking_scheme,
            grading_guidelines, solution_ref, student_sol,
            reasoning_effort=args.reasoning_effort,
        )
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
                scheme = result.get("scheme", "?")
                display = (f"{result['score']}/7" if scheme == "rubric"
                           else result.get("label", "?"))
                print(f"  [{completed}/{len(tasks)}] {problem_idx} "
                      f"(line {line_idx}) s{s_idx}: {display}")
            else:
                failed += 1
                print(f"  [{completed}/{len(tasks)}] {problem_idx} "
                      f"(line {line_idx}) s{s_idx}: FAILED")

    if failed:
        print(f"\n  Warning: {failed}/{len(tasks)} samples failed to grade")

    # ── Build output ──────────────────────────────────────────────────────────
    print(f"\nWriting results to {args.output}...")
    graded_output = []
    problem_scores = defaultdict(list)

    def _build_graded_item(line_idx, problem_idx, problem_type, problem, max_s):
        graded_item = {
            "line_idx":     line_idx,
            "problem_idx":  problem_idx,
            "problem":      problem,
            "problem_type": problem_type,
            "graded_samples": [],
        }
        for s_idx in range(max_s):
            grading = results.get((line_idx, s_idx))
            graded_item["graded_samples"].append({
                "sample_idx": s_idx,
                "grading":    grading,
            })
            if grading:
                scheme = grading.get("scheme", "rubric")
                if scheme == "rubric":
                    problem_scores[(line_idx, problem_idx)].append(
                        grading.get("score", 0) / 7.0)
                else:
                    label_to_score = {"incorrect": 0.0, "partial": 0.33,
                                      "almost": 0.75, "correct": 1.0}
                    problem_scores[(line_idx, problem_idx)].append(
                        label_to_score.get(grading.get("label", "incorrect"), 0.0))
        return graded_item

    if fmt == "refinement":
        for line_idx, item in enumerate(inference_data):
            samples = item.get("samples", [])
            max_s = len(samples) if args.max_samples == -1 else min(args.max_samples, len(samples))
            graded_output.append(_build_graded_item(
                line_idx,
                item.get("problem_idx", ""),
                item.get("problem_type", []),
                item.get("original_prompt", ""),
                max_s,
            ))

    elif fmt == "baseline":
        problems_grouped = defaultdict(list)
        for item in inference_data:
            problems_grouped[item.get("problem_id", "")].append(item)

        for line_idx, (pid, samples) in enumerate(sorted(problems_grouped.items())):
            prompt = samples[0].get("prompt", "").strip()
            meta   = metadata.get(prompt, {})
            max_s  = len(samples) if args.max_samples == -1 else min(args.max_samples, len(samples))
            graded_output.append(_build_graded_item(
                line_idx,
                meta.get("problem_idx", pid),
                meta.get("problem_type", []),
                prompt,
                max_s,
            ))

    with open(args.output, 'w', encoding='utf-8') as f:
        for item in graded_output:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # ── Summary ───────────────────────────────────────────────────────────────
    total_items = len(problem_scores)
    if total_items > 0:
        avg_scores = [sum(s) / len(s) for s in problem_scores.values()]
        overall_avg = sum(avg_scores) / len(avg_scores) * 100

        contest_scores = defaultdict(list)
        for item in graded_output:
            key = (item["line_idx"], item["problem_idx"])
            if key in problem_scores:
                avg = sum(problem_scores[key]) / len(problem_scores[key])
                for ptype in item.get("problem_type", []):
                    contest_scores[ptype].append(avg)

        print(f"\n{'='*60}")
        print(f"Grading Summary")
        print(f"{'='*60}")
        print(f"Model: {args.model} (reasoning_effort={args.reasoning_effort})")
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
