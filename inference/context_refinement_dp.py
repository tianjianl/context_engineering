#!/usr/bin/env python3
"""
Context Refinement Script using vLLM with Data Parallelism

This script uses multiple GPUs with data parallelism (each GPU runs its own
vLLM instance with tensor_parallel_size=1).
"""

import argparse
import json
import csv
import os
import torch
import multiprocessing as mp
from typing import List, Dict
from pathlib import Path
import urllib.request


# Dataset URLs
IMOBENCH_URL = "https://raw.githubusercontent.com/google-deepmind/superhuman/main/imobench/answerbench.csv"


def download_imobench(cache_dir: str) -> str:
    """Download IMOBench (AnswerBench) CSV if not already cached."""
    cache_path = Path(cache_dir) / "answerbench.csv"
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    if not cache_path.exists():
        print(f"Downloading IMOBench from {IMOBENCH_URL}...")
        urllib.request.urlretrieve(IMOBENCH_URL, cache_path)
        print(f"Saved to {cache_path}")
    else:
        print(f"Loading IMOBench from {cache_path}")

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


def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


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


def create_refinement_prompt(original_prompt: str, partial_generation: str, preserve_answer: bool = True) -> str:
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


def create_rc_refinement_prompt(problem: str, existing_summary: str, reasoning: str) -> str:
    """Create an RC-style refinement prompt (incremental summarization).

    Instead of compressing the full context, this merges an existing summary
    with only the latest reasoning chunk, producing a replacement summary.
    """
    return f"""You are given a maths problem and a candidate solution to it. You may also be given a summary of a previous candidate solution to the problem. If this is provided, you may assume that the current candidate solution was generated conditioned on the summary of the previous candidate solution.
Your task is to write a summary of the current candidate solution.

The new summary you generate should possess the following characteristics:
- It should provide a detailed overview of what occurred in the current candidate solution. This may include a summary of the high-level problem-solving strategy, a description of theorems used, verification attempts, calculations and logical deductions etc.
- It should summarize the current candidate solution in light of any previous summaries, if provided. We should be able to understand the relationship between the previous solution and the current solution by reading the summary. Make sure any important information contained in the existing summary is retained in the new one.
- It should be no more than two paragraph long and written in paragraph form, without headers or subheaders.
- It should be written in the first person, as if though it is being written by the person solving the problem.
- The candidate solution may not be complete. In this case, the summary should still attempt to summarize the partial solution.

IMPORTANT: Do not under any circumstances add any additional reasoning not contained in the latest reasoning step. Your task is only to summarize what is given to you.

### PROBLEM
{problem}

### EXISTING SUMMARY
{existing_summary}

### LATEST CANDIDATE SOLUTION
{reasoning}"""


def save_intermediate_results(gpu_id: int, valid_items: List[Dict], original_prompts: List[str],
                               all_samples_rounds_data, all_samples_prefixes,
                               num_samples: int, round_num: int, output_file: str):
    """Save intermediate results for this GPU after a round completes."""
    results = []
    for q_idx, (item, orig_prompt) in enumerate(zip(valid_items, original_prompts)):
        samples = []
        for s_idx in range(num_samples):
            rounds_data = all_samples_rounds_data[q_idx][s_idx]
            full_message = all_samples_prefixes[q_idx][s_idx]
            sample_result = {
                "sample_idx": s_idx,
                "rounds": rounds_data,
                "final_refined_context": rounds_data[-1]["refined_context"],
                "full_assistant_message": full_message
            }
            samples.append(sample_result)

        result = {
            "original_prompt": orig_prompt,
            "num_samples": num_samples,
            "samples": samples,
            **{k: v for k, v in item.items() if k != "prompt"}
        }
        results.append(result)

    # Save to intermediate file (one per GPU, overwritten each round)
    output_path = Path(output_file)
    intermediate_file = output_path.parent / f"{output_path.stem}_intermediate_gpu{gpu_id}.jsonl"
    intermediate_file.parent.mkdir(parents=True, exist_ok=True)
    with open(intermediate_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    print(f"[GPU {gpu_id}] Saved intermediate results after round {round_num + 1} to {intermediate_file}")


def worker_process(gpu_id: int, data_shard: List[Dict], args, result_queue: mp.Queue):
    """Worker process that runs on a single GPU."""
    # Set GPU visibility before importing vLLM
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Import vLLM after setting GPU
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"[GPU {gpu_id}] Starting worker with {len(data_shard)} problems")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Initialize vLLM with tensor_parallel_size=1
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # Sampling parameters
    initial_sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.num_tokens,
        n=args.num_samples
    )

    refinement_sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_refinement_tokens
    )

    def apply_chat_template(prompt: str, add_generation_prompt: bool = False) -> str:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt
        )

    def apply_chat_template_with_prefix(prompt: str, assistant_prefix: str) -> str:
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant_prefix}
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

    results = []
    num_samples = args.num_samples

    valid_items = [item for item in data_shard if "prompt" in item and item["prompt"]]
    original_prompts = [item["prompt"] for item in valid_items]

    # Track data for each sample: [question_idx][sample_idx]
    all_samples_rounds_data = [[[] for _ in range(num_samples)] for _ in range(len(valid_items))]
    all_samples_prefixes = [["" for _ in range(num_samples)] for _ in range(len(valid_items))]

    for round_num in range(args.rounds):
        print(f"[GPU {gpu_id}] Round {round_num + 1}/{args.rounds}")

        if round_num == 0:
            generation_prompts = [apply_chat_template(p, add_generation_prompt=True) for p in original_prompts]
            initial_outputs = llm.generate(generation_prompts, initial_sampling_params)

            current_round_generations_raw = [
                [output.outputs[s].text for s in range(num_samples)]
                for output in initial_outputs
            ]
        else:
            flat_prompts = []
            for q_idx, orig in enumerate(original_prompts):
                for s_idx in range(num_samples):
                    prefix = all_samples_prefixes[q_idx][s_idx]
                    flat_prompts.append(apply_chat_template_with_prefix(orig, prefix))

            subsequent_sampling_params = SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_tokens=args.num_tokens,
                n=1
            )
            initial_outputs = llm.generate(flat_prompts, subsequent_sampling_params)

            current_round_generations_raw = []
            idx = 0
            for q_idx in range(len(valid_items)):
                sample_gens = []
                for s_idx in range(num_samples):
                    sample_gens.append(initial_outputs[idx].outputs[0].text)
                    idx += 1
                current_round_generations_raw.append(sample_gens)

        # Process generations: strip thinking if requested
        # We keep both raw and stripped versions
        current_round_generations = []  # Stripped version for use in refinement/accumulation
        current_round_generations_raw_stored = current_round_generations_raw  # Keep raw for storage

        for q_idx in range(len(valid_items)):
            sample_gens = []
            for s_idx in range(num_samples):
                raw_gen = current_round_generations_raw[q_idx][s_idx]
                if args.strip_thinking_from_generation:
                    stripped_gen = strip_thinking(raw_gen)
                    if not stripped_gen and '<think>' in raw_gen:
                        # Generation was all thinking - model ran out of tokens
                        # Use empty string to avoid propagating incomplete thinking
                        print(f"[GPU {gpu_id}] Warning: Generation all thinking for q{q_idx}/s{s_idx}/r{round_num+1}, need more tokens")
                        sample_gens.append("")  # Empty - will need to regenerate
                    else:
                        sample_gens.append(stripped_gen)
                else:
                    sample_gens.append(raw_gen)
            current_round_generations.append(sample_gens)

        # Refinement
        flat_refinement_prompts = []
        for q_idx, orig in enumerate(original_prompts):
            for s_idx in range(num_samples):
                current_gen = current_round_generations[q_idx][s_idx]
                if args.rc:
                    existing_summary = all_samples_prefixes[q_idx][s_idx]
                    refinement_prompt_raw = create_rc_refinement_prompt(
                        orig, existing_summary, current_gen
                    )
                else:
                    if args.accumulate:
                        if round_num == 0:
                            context_to_refine = current_gen
                        else:
                            context_to_refine = all_samples_prefixes[q_idx][s_idx] + current_gen
                    else:
                        context_to_refine = current_gen
                    refinement_prompt_raw = create_refinement_prompt(
                        orig, context_to_refine, preserve_answer=args.preserve_answer
                    )
                # Add /no_think suffix to disable thinking if requested
                if args.disable_thinking_for_refinement:
                    refinement_prompt_raw = refinement_prompt_raw + "\n\n/no_think"
                flat_refinement_prompts.append(apply_chat_template(refinement_prompt_raw, add_generation_prompt=True))

        refinement_outputs = llm.generate(flat_refinement_prompts, refinement_sampling_params)

        idx = 0
        for q_idx in range(len(valid_items)):
            for s_idx in range(num_samples):
                current_gen = current_round_generations[q_idx][s_idx]  # Stripped version
                current_gen_raw = current_round_generations_raw_stored[q_idx][s_idx]  # Raw version
                refined_ctx_raw = refinement_outputs[idx].outputs[0].text

                # Strip thinking from refinement if requested
                if args.strip_thinking_from_refinement:
                    refined_ctx = strip_thinking(refined_ctx_raw)
                    # If stripping resulted in empty string, model didn't finish thinking
                    # Fall back to using the stripped generation as context
                    if not refined_ctx:
                        print(f"[GPU {gpu_id}] Warning: Refinement incomplete for q{q_idx}/s{s_idx}/r{round_num+1}, using generation")
                        refined_ctx = current_gen  # Use stripped generation, not raw
                else:
                    refined_ctx = refined_ctx_raw

                round_data = {
                    "round": round_num + 1,
                    "current_round_generation": current_gen,  # Stripped for evaluation
                    "current_round_generation_raw": current_gen_raw,  # Raw for debugging
                    "refined_context": refined_ctx,
                    "refined_context_raw": refined_ctx_raw  # Raw for debugging
                }
                all_samples_rounds_data[q_idx][s_idx].append(round_data)

                # Update prefix for next round
                if args.rc:
                    # RC mode: summary replaces previous context (fixed-size window)
                    all_samples_prefixes[q_idx][s_idx] = refined_ctx
                else:
                    # Accumulation mode: append to growing prefix
                    if args.accumulate_raw:
                        content_to_add = current_gen
                    else:
                        content_to_add = refined_ctx
                    if round_num == 0:
                        all_samples_prefixes[q_idx][s_idx] = content_to_add
                    else:
                        all_samples_prefixes[q_idx][s_idx] = all_samples_prefixes[q_idx][s_idx] + content_to_add

                idx += 1

        # Save intermediate results after each round
        save_intermediate_results(
            gpu_id, valid_items, original_prompts,
            all_samples_rounds_data, all_samples_prefixes,
            num_samples, round_num, args.output_file
        )

    # Build results
    for q_idx, (item, orig_prompt) in enumerate(zip(valid_items, original_prompts)):
        samples = []
        for s_idx in range(num_samples):
            rounds_data = all_samples_rounds_data[q_idx][s_idx]
            full_message = all_samples_prefixes[q_idx][s_idx]
            sample_result = {
                "sample_idx": s_idx,
                "rounds": rounds_data,
                "final_refined_context": rounds_data[-1]["refined_context"],
                "full_assistant_message": full_message
            }
            samples.append(sample_result)

        result = {
            "original_prompt": orig_prompt,
            "num_samples": num_samples,
            "samples": samples,
            **{k: v for k, v in item.items() if k != "prompt"}
        }
        results.append(result)

    print(f"[GPU {gpu_id}] Completed processing {len(results)} problems")
    result_queue.put((gpu_id, results))


def main():
    parser = argparse.ArgumentParser(
        description="Generate and refine contexts using vLLM with Data Parallelism"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["hmmt", "imobench"],
        required=True,
        help="Dataset to use: 'hmmt' or 'imobench'"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Input JSONL file (required for hmmt, optional for imobench)"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/scratch/dkhasha1/tli104/imobench",
        help="Directory to cache downloaded datasets"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Model name or path to use for generation"
    )
    parser.add_argument(
        "--num_tokens",
        type=int,
        required=True,
        help="Number of tokens to generate in the initial generation"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="output_refined.jsonl",
        help="Output JSONL file for results"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Number of refinement rounds to perform (default: 1)"
    )
    parser.add_argument(
        "--accumulate",
        action="store_true",
        help="Accumulate context across rounds"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to generate per question (default: 1)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter (default: 0.9)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=-1,
        help="Top-k sampling parameter (default: -1, disabled)"
    )
    parser.add_argument(
        "--max_refinement_tokens",
        type=int,
        default=None,
        help="Maximum tokens for refinement output (default: 16384 for thinking models)"
    )
    parser.add_argument(
        "--preserve_answer",
        action="store_true",
        default=True,
        help="Preserve final answers in refinement (default: True)"
    )
    parser.add_argument(
        "--strip_answer",
        action="store_true",
        help="Strip final answers from refinement (opposite of --preserve_answer)"
    )
    parser.add_argument(
        "--disable_thinking_for_refinement",
        action="store_true",
        help="Disable thinking mode for refinement prompts (adds /no_think suffix)"
    )
    parser.add_argument(
        "--accumulate_raw",
        action="store_true",
        help="Accumulate raw generations instead of refined context for continuation (simple continuation baseline)"
    )
    parser.add_argument(
        "--rc",
        action="store_true",
        help="Use RC-style incremental summarization: each round merges existing summary with latest reasoning into a fixed-size replacement summary (instead of appending refined context)"
    )
    parser.add_argument(
        "--strip_thinking_from_refinement",
        action="store_true",
        default=True,
        help="Strip <think>...</think> from refinement output (default: True)"
    )
    parser.add_argument(
        "--keep_thinking_in_refinement",
        action="store_true",
        help="Keep <think>...</think> in refinement output (opposite of --strip_thinking_from_refinement)"
    )
    parser.add_argument(
        "--strip_thinking_from_generation",
        action="store_true",
        default=True,
        help="Strip <think>...</think> from main generation output (default: True)"
    )
    parser.add_argument(
        "--keep_thinking_in_generation",
        action="store_true",
        help="Keep <think>...</think> in main generation output"
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Number of GPUs to use for data parallelism (default: all available)"
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=None,
        help="Maximum model context length"
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.95,
        help="GPU memory utilization (default: 0.95)"
    )

    args = parser.parse_args()

    # Default refinement tokens to 16k for thinking models to ensure completion
    if args.max_refinement_tokens is None:
        args.max_refinement_tokens = 16384

    # Handle preserve_answer / strip_answer logic
    if args.strip_answer:
        args.preserve_answer = False

    # Handle strip_thinking logic
    if args.keep_thinking_in_refinement:
        args.strip_thinking_from_refinement = False
    if args.keep_thinking_in_generation:
        args.strip_thinking_from_generation = False

    # Default max_model_len to handle generation + refinement (16k refinement tokens)
    if args.max_model_len is None:
        if args.accumulate_raw:
            # Need more context when accumulating raw generations
            # rounds * num_tokens + some buffer for input
            args.max_model_len = min(args.num_tokens * (args.rounds + 2), 131072)
        else:
            # Allow for input + generation + 16k refinement
            args.max_model_len = min(args.num_tokens + args.max_refinement_tokens + 4096, 131072)

    # Validate arguments
    if args.dataset == "hmmt" and args.input_file is None:
        parser.error("--input_file is required when using --dataset hmmt")

    # Detect GPUs
    num_gpus = torch.cuda.device_count()
    if args.num_gpus is None:
        args.num_gpus = num_gpus
    else:
        args.num_gpus = min(args.num_gpus, num_gpus)

    print(f"\n{'='*80}")
    print(f"Data Parallel Inference")
    print(f"{'='*80}")
    print(f"Available GPUs: {num_gpus}")
    print(f"Using GPUs: {args.num_gpus}")
    print(f"Model: {args.model}")
    print(f"Max Model Length: {args.max_model_len}")
    print(f"Num Tokens: {args.num_tokens}")
    print(f"Max Refinement Tokens: {args.max_refinement_tokens}")
    print(f"Rounds: {args.rounds}")
    print(f"Accumulate: {args.accumulate}")
    print(f"Accumulate Raw: {args.accumulate_raw}")
    print(f"RC Mode: {args.rc}")
    print(f"Preserve Answer: {args.preserve_answer}")
    print(f"Disable Thinking for Refinement: {args.disable_thinking_for_refinement}")
    print(f"Strip Thinking from Generation: {args.strip_thinking_from_generation}")
    print(f"Strip Thinking from Refinement: {args.strip_thinking_from_refinement}")
    print(f"{'='*80}\n")

    # Load data
    print(f"Loading Dataset: {args.dataset.upper()}")
    if args.dataset == "imobench":
        if args.input_file:
            data = load_jsonl(args.input_file)
        else:
            csv_path = download_imobench(args.cache_dir)
            data = load_imobench(csv_path)
    elif args.dataset == "hmmt":
        data = load_jsonl(args.input_file)

    print(f"Loaded {len(data)} problems")

    # Shard data across GPUs
    shards = [[] for _ in range(args.num_gpus)]
    for i, item in enumerate(data):
        shards[i % args.num_gpus].append(item)

    print(f"Data sharding: {[len(s) for s in shards]}")

    # Start worker processes
    mp.set_start_method('spawn', force=True)
    result_queue = mp.Queue()
    processes = []

    for gpu_id in range(args.num_gpus):
        p = mp.Process(
            target=worker_process,
            args=(gpu_id, shards[gpu_id], args, result_queue)
        )
        p.start()
        processes.append(p)

    # Collect results
    all_results = {}
    for _ in range(args.num_gpus):
        gpu_id, results = result_queue.get()
        all_results[gpu_id] = results
        print(f"[Main] Received {len(results)} results from GPU {gpu_id}")

    # Wait for all processes
    for p in processes:
        p.join()

    # Merge results maintaining original order
    merged_results = []
    indices = [0] * args.num_gpus
    for i in range(len(data)):
        gpu_id = i % args.num_gpus
        if indices[gpu_id] < len(all_results[gpu_id]):
            merged_results.append(all_results[gpu_id][indices[gpu_id]])
            indices[gpu_id] += 1

    # Save results
    print(f"\nSaving {len(merged_results)} results to {args.output_file}...")
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for result in merged_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(f"\n{'='*80}")
    print(f"Done! Processed {len(merged_results)} problems with {args.num_gpus} GPUs.")
    print(f"Results saved to {args.output_file}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
