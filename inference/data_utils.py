"""Shared data loading, text processing, and format detection utilities."""

import csv
import json
import os
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Dataset URLs
IMOBENCH_URL = "https://raw.githubusercontent.com/google-deepmind/superhuman/main/imobench/answerbench.csv"
IMOBENCH_V2_URL = "https://raw.githubusercontent.com/google-deepmind/superhuman/main/imobench/answerbench_v2.csv"


# ── Shared prompts ──────────────────────────────────────────────────────────

RC_USER_REASONING_PROMPT = """You are given a maths problem. You may also be given a summary of a previous attempt to solve it. This previous attempt may or may not be correct.

### PROBLEM
{problem}

### SUMMARY OF PREVIOUS ATTEMPT
{summary}

### INSTRUCTIONS
If no summary of a previous attempt is provided, solve the problem from scratch.

If a summary of a previous attempt is provided, your task is to improve upon this attempt. You should rely on this summary to guide your thinking.
Some strategies you could use include:
- Verifying the previous solution.
- Proving the result in a different way.
- Finding alternative problem-solving strategies.
- Continuing from where the previous solution left off, assuming that the previous solution is incomplete.

Reason step-by-step and return your final answer in \\boxed{{}}."""

MATH_PROMPT_TEMPLATE = (
    "Solve the following math problem. Show your reasoning step by step "
    "and provide your final answer in \\boxed{{}}.\n\n"
    "Problem: {problem}"
)

CRITIQUE_PROMPT = (
    "Review your solution step by step. Check each calculation and logical step."
)

REVISE_PROMPT = (
    "Based on your review, write your final solution. "
    "Provide your answer in \\boxed{}."
)


# ── JSONL I/O ────────────────────────────────────────────────────────────────

def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], file_path: str, mode: str = 'w') -> None:
    """Write a list of dicts to a JSONL file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, mode, encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


# ── Filtering by correctness ─────────────────────────────────────────────────

def _filter_by_correctness(data: List[Dict], keep_correct: bool) -> List[Dict]:
    """Filter items by whether their baseline generation is correct/incorrect."""
    from inference.verify_utils import verify_batch
    items_to_verify = []
    for item in data:
        gold = item.get("answer", "")
        gen = item.get("generation", "")
        text = strip_thinking(gen) if gen else ""
        items_to_verify.append((gold, text))

    results = verify_batch(items_to_verify)
    return [
        item for item, (is_correct, _, _) in zip(data, results)
        if is_correct == keep_correct
    ]


def filter_incorrect(data: List[Dict]) -> List[Dict]:
    """Return only items whose baseline generation is incorrect."""
    return _filter_by_correctness(data, keep_correct=False)


def filter_correct(data: List[Dict]) -> List[Dict]:
    """Return only items whose baseline generation is correct."""
    return _filter_by_correctness(data, keep_correct=True)


# ── IMOBench / dataset loading ───────────────────────────────────────────────

_IMOBENCH_VARIANTS = {
    "imobench":    (IMOBENCH_URL,    "answerbench.csv",    "IMOBench"),
    "imobench_v2": (IMOBENCH_V2_URL, "answerbench_v2.csv", "IMOBench v2"),
}


def download_imobench(cache_dir: str, variant: str = "imobench") -> str:
    """Download an IMOBench CSV variant if not already cached."""
    url, filename, label = _IMOBENCH_VARIANTS[variant]
    cache_path = Path(cache_dir) / filename
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    if not cache_path.exists():
        print(f"Downloading {label} from {url}...")
        urllib.request.urlretrieve(url, cache_path)
        print(f"Saved to {cache_path}")
    else:
        print(f"Loading {label} from {cache_path}")

    return str(cache_path)


def load_imobench(csv_path: str) -> List[Dict]:
    """Load data from IMOBench (AnswerBench) CSV file."""
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                "problem_id": row.get("Problem ID", ""),
                "prompt": row.get("Problem", ""),
                "answer": row.get("Short Answer", ""),
                "category": row.get("Category", ""),
                "subcategory": row.get("Subcategory", ""),
                "source": row.get("Source", "")
            })
    return data


def load_constory(parquet_path: str) -> List[Dict]:
    """Load ConStory-Bench prompts from a parquet file.

    Returns list of dicts with keys: id, language, task_type, prompt.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("pyarrow is required for parquet files: pip install pyarrow")
    table = pq.read_table(parquet_path)
    col_names = set(table.column_names)
    if "prompt" not in col_names:
        raise ValueError(
            f"Parquet file {parquet_path} missing required 'prompt' column. "
            f"Found columns: {sorted(col_names)}"
        )
    columns = table.to_pydict()
    num_rows = table.num_rows
    return [{col: columns[col][i] for col in columns} for i in range(num_rows)]


def load_dataset(dataset: str, input_file: Optional[str] = None,
                 cache_dir: str = "/scratch/dkhasha1/tli104/imobench") -> List[Dict]:
    """Unified dataset loading dispatch.

    Args:
        dataset: 'imobench', 'hmmt', or 'constory'
        input_file: Path to input file (required for hmmt/constory, optional for imobench)
        cache_dir: Directory to cache downloaded datasets
    """
    if dataset in ("imobench", "imobench_v2"):
        if input_file:
            if input_file.endswith('.csv'):
                return load_imobench(input_file)
            return load_jsonl(input_file)
        csv_path = download_imobench(cache_dir, variant=dataset)
        return load_imobench(csv_path)
    elif dataset == "hmmt":
        return load_jsonl(input_file)
    elif dataset == "constory":
        if not input_file:
            raise ValueError("--input_file is required for constory dataset")
        if input_file.endswith('.parquet'):
            return load_constory(input_file)
        return load_jsonl(input_file)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# ── Text processing ──────────────────────────────────────────────────────────

def strip_thinking(text: str) -> str:
    """Strip <think>...</think> section from model output.

    If </think> is not found (model ran out of tokens while thinking),
    returns empty string since there's no actual content.
    """
    if '<think>' not in text:
        return text

    think_end = text.find('</think>')
    if think_end == -1:
        # Model ran out of tokens while thinking - no actual content
        return ""

    # Return content after </think>
    return text[think_end + 8:].strip()


# ── Solution text extraction ─────────────────────────────────────────────────

def _is_tool_call_only(text: str) -> bool:
    """Return True if text is just a tool call with no substantive reasoning."""
    stripped = text.strip()
    return stripped.startswith("<tool_call>") and len(stripped) < 500


def get_text_from_sample(sample: Dict) -> Tuple[str, Optional[str]]:
    """Extract generated text from a sample dict. Returns (text, source_name).

    Used by verify_solutions.py — prioritizes last_round_generation for
    answer extraction (since refined context intentionally omits \\boxed{}).
    Skips sources that are just a tool call (no \\boxed{} answer possible).
    """
    text_sources = []

    rounds = sample.get("rounds", [])
    if rounds:
        last_round = rounds[-1]
        text_sources.append(
            ("last_round_generation", last_round.get("current_round_generation", ""))
        )

    text_sources.extend([
        ("full_assistant_message", sample.get("full_assistant_message", "")),
        ("final_refined_context", sample.get("final_refined_context", "")),
        ("generation", sample.get("generation", "")),
    ])

    for source_name, text in text_sources:
        if text and text.strip() and not _is_tool_call_only(text):
            return text, source_name

    return "", None


def get_solution_text(sample: Dict) -> str:
    """Extract the generated solution text from a sample.

    Used by grading scripts — combines refined context with final generation
    to give the grader the full reasoning chain.
    """
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


# ── Format detection and metadata (grading scripts) ─────────────────────────

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


def load_existing_results(output_path: str) -> Tuple[Dict, set]:
    """Load already-graded results for resume support.

    Returns (full_data_by_key, set_of_graded_keys) where keys are (line_idx, sample_idx).
    """
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


# ── Refinement prompt ────────────────────────────────────────────────────────

def apply_chat_template(tokenizer, prompt: str, add_generation_prompt: bool = False) -> str:
    """Apply chat template to format a user prompt for instruct models."""
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt
    )


def apply_chat_template_with_prefix(tokenizer, prompt: str, assistant_prefix: str) -> str:
    """Apply chat template and append an assistant prefix for continued generation.

    Builds a prompt with the user message + start of assistant turn + prefix.
    Does NOT close the assistant turn so the model continues generating.
    """
    base = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True
    )
    return base + assistant_prefix


def create_refinement_prompt(original_prompt: str, partial_generation: str,
                             preserve_answer: bool = True) -> str:
    """Create the context refinement prompt.

    Args:
        original_prompt: The original problem statement
        partial_generation: The generation to refine/compress
        preserve_answer: If True, preserve any final answer found. If False, strip answers.
    """
    if preserve_answer:
        return f"""Context Refinement Prompt:

Original Prompt:
{original_prompt}

Partial Generation:
{partial_generation}

Your task is to create a compressed summary for another model to continue solving from.

RULES:
1. If a final answer (e.g., \\boxed{{}}) was found, PRESERVE IT at the end of your summary
2. Keep key insights, important calculations, and the reasoning path
3. Remove redundant text, false starts, and unnecessary repetition
4. If the answer seems wrong or unverified, note that verification is needed
5. Be concise but preserve all critical mathematical steps

Output format:
- Key insights and progress made
- Important intermediate results
- If found: "Final Answer: [the answer]" or the \\boxed{{}} expression
- If not solved: what still needs to be done"""
    else:
        return f"""Context Refinement Prompt:

Original Prompt:
{original_prompt}

Partial Generation:
{partial_generation}

Your task is to create a WORK-IN-PROGRESS summary for another model to continue solving from.

CRITICAL RULES:
1. NEVER include any final answer or \\boxed{{}} in your output
2. NEVER conclude or claim the problem is solved
3. Remove any "Final Answer" sections completely
4. Keep only intermediate calculations, key insights, and partial progress
5. End your summary at a natural continuation point where more work is needed
6. If the generation reached a wrong answer, note the approach taken but indicate it needs verification

Output a concise summary of the progress made so far, ending with what still needs to be done. Do NOT provide any final answer."""
