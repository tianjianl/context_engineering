#!/usr/bin/env python3
"""
Context Refinement Script using vLLM

This script supports multiple benchmarks (HMMT, IMOBench) and performs
multi-round generation with context refinement.
"""

import argparse
import json
import csv
from typing import List, Dict
from pathlib import Path
import torch
from vllm import LLM, SamplingParams
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


def create_refinement_prompt(original_prompt: str, partial_generation: str) -> str:
    """Create the context refinement prompt."""
    return f"""Context Refinement Prompt:

Original Prompt:
{original_prompt}

Partial Generation:
{partial_generation}

Refine and summarize the partial generation for continuation. Correct errors, remove redundancy, and keep only accurate, relevant information. Do not try to solve the question. Do not add new facts. Only output the refined context only."""


def process_debug_mode(llm, data, initial_sampling_params, refinement_sampling_params, args):
    """Process prompts one at a time with verbose debug output."""
    results = []

    print("\n" + "="*80)
    print(f"DEBUG MODE: Processing prompts with verbose output ({args.rounds} rounds)")
    print("="*80)

    for idx, item in enumerate(data):
        if "prompt" not in item:
            print(f"Warning: Item {idx} missing 'prompt' field, skipping...")
            continue

        original_prompt = item["prompt"]

        print(f"\n{'='*80}")
        print(f"[{idx + 1}/{len(data)}] Processing prompt")
        print(f"{'='*80}")
        print(f"\nORIGINAL PROMPT:")
        print(f"{'-'*80}")
        print(original_prompt)
        print(f"{'-'*80}")

        # Track all rounds
        rounds_data = []
        assistant_message_prefix = ""

        for round_num in range(args.rounds):
            print(f"\n{'#'*80}")
            print(f"ROUND {round_num + 1}/{args.rounds}")
            print(f"{'#'*80}")

            if round_num == 0:
                prompt_text = original_prompt
            else:
                prompt_text = f"{original_prompt}\n\nAssistant: {assistant_message_prefix}"

            print(f"\nGenerating {args.num_tokens} tokens...")
            if round_num > 0:
                print(f"Assistant message prefix length: {len(assistant_message_prefix)} chars")

            initial_outputs = llm.generate([prompt_text], initial_sampling_params)
            current_round_generation = initial_outputs[0].outputs[0].text

            print(f"\nCURRENT ROUND GENERATION ({args.num_tokens} tokens):")
            print(f"{'-'*80}")
            print(current_round_generation)
            print(f"{'-'*80}")

            if args.accumulate:
                if round_num == 0:
                    context_to_refine = current_round_generation
                else:
                    context_to_refine = assistant_message_prefix + current_round_generation
                print(f"\nRefining ACCUMULATED context (prefix + current generation)...")
            else:
                context_to_refine = current_round_generation
                print(f"\nRefining CURRENT ROUND generation only...")

            refinement_prompt = create_refinement_prompt(original_prompt, context_to_refine)

            print(f"\nREFINEMENT PROMPT:")
            print(f"{'-'*80}")
            print(refinement_prompt)
            print(f"{'-'*80}")

            refinement_outputs = llm.generate([refinement_prompt], refinement_sampling_params)
            refined_context = refinement_outputs[0].outputs[0].text

            print(f"\nREFINED CONTEXT:")
            print(f"{'-'*80}")
            print(refined_context)
            print(f"{'-'*80}")

            round_data = {
                "round": round_num + 1,
                "current_round_generation": current_round_generation,
                "refined_context": refined_context
            }
            rounds_data.append(round_data)

            if round_num == 0:
                assistant_message_prefix = refined_context
            else:
                assistant_message_prefix = assistant_message_prefix + refined_context

        result = {
            "original_prompt": original_prompt,
            "rounds": rounds_data,
            "final_refined_context": rounds_data[-1]["refined_context"],
            "full_assistant_message": assistant_message_prefix,
            **{k: v for k, v in item.items() if k != "prompt"}
        }
        results.append(result)

    return results


def process_batch_mode(llm, data, initial_sampling_params, refinement_sampling_params, args):
    """Process prompts in batches for efficiency with multiple samples per question."""
    results = []
    num_samples = args.num_samples

    print("\n" + "="*80)
    print(f"BATCH MODE: Processing all prompts in batches ({args.rounds} rounds, {num_samples} samples each)")
    print("="*80)

    valid_items = [item for item in data if "prompt" in item and item["prompt"]]
    if len(valid_items) < len(data):
        print(f"Warning: Skipped {len(data) - len(valid_items)} items missing 'prompt' field")

    original_prompts = [item["prompt"] for item in valid_items]

    # Track data for each sample: [question_idx][sample_idx]
    all_samples_rounds_data = [[[] for _ in range(num_samples)] for _ in range(len(valid_items))]
    all_samples_prefixes = [["" for _ in range(num_samples)] for _ in range(len(valid_items))]

    for round_num in range(args.rounds):
        print(f"\n{'='*80}")
        print(f"ROUND {round_num + 1}/{args.rounds}")
        print(f"{'='*80}")

        if round_num == 0:
            # First round: generate n samples per prompt
            generation_prompts = original_prompts
            print(f"  Stage 1: Generating {args.num_tokens} tokens x {num_samples} samples for {len(generation_prompts)} prompts...")
            initial_outputs = llm.generate(generation_prompts, initial_sampling_params)

            # Extract all samples: initial_outputs[question_idx].outputs[sample_idx]
            current_round_generations = [
                [output.outputs[s].text for s in range(num_samples)]
                for output in initial_outputs
            ]
        else:
            # Subsequent rounds: need to generate separately for each sample path
            # Flatten all prompts (question x sample combinations)
            flat_prompts = []
            for q_idx, orig in enumerate(original_prompts):
                for s_idx in range(num_samples):
                    prefix = all_samples_prefixes[q_idx][s_idx]
                    flat_prompts.append(f"{orig}\n\nAssistant: {prefix}")

            print(f"  Stage 1: Generating {args.num_tokens} tokens for {len(flat_prompts)} prompt-sample combinations...")
            # Use n=1 for subsequent rounds since we have separate prompts per sample
            subsequent_sampling_params = SamplingParams(
                temperature=initial_sampling_params.temperature,
                top_p=initial_sampling_params.top_p,
                top_k=initial_sampling_params.top_k,
                max_tokens=initial_sampling_params.max_tokens,
                n=1
            )
            initial_outputs = llm.generate(flat_prompts, subsequent_sampling_params)

            # Reshape back to [question][sample]
            current_round_generations = []
            idx = 0
            for q_idx in range(len(valid_items)):
                sample_gens = []
                for s_idx in range(num_samples):
                    sample_gens.append(initial_outputs[idx].outputs[0].text)
                    idx += 1
                current_round_generations.append(sample_gens)

        print(f"    ✓ Completed initial generation")

        # Refinement: flatten all contexts and refine in batch
        flat_refinement_prompts = []
        for q_idx, orig in enumerate(original_prompts):
            for s_idx in range(num_samples):
                current_gen = current_round_generations[q_idx][s_idx]
                if args.accumulate:
                    if round_num == 0:
                        context_to_refine = current_gen
                    else:
                        context_to_refine = all_samples_prefixes[q_idx][s_idx] + current_gen
                else:
                    context_to_refine = current_gen
                flat_refinement_prompts.append(create_refinement_prompt(orig, context_to_refine))

        if args.accumulate:
            print(f"  Stage 2: Refining ACCUMULATED contexts...")
        else:
            print(f"  Stage 2: Refining CURRENT ROUND generations...")

        refinement_outputs = llm.generate(flat_refinement_prompts, refinement_sampling_params)
        print(f"    ✓ Completed refinement")

        # Reshape and store results
        idx = 0
        for q_idx in range(len(valid_items)):
            for s_idx in range(num_samples):
                current_gen = current_round_generations[q_idx][s_idx]
                refined_ctx = refinement_outputs[idx].outputs[0].text

                round_data = {
                    "round": round_num + 1,
                    "current_round_generation": current_gen,
                    "refined_context": refined_ctx
                }
                all_samples_rounds_data[q_idx][s_idx].append(round_data)

                # Update prefix for next round
                if round_num == 0:
                    all_samples_prefixes[q_idx][s_idx] = refined_ctx
                else:
                    all_samples_prefixes[q_idx][s_idx] = all_samples_prefixes[q_idx][s_idx] + refined_ctx

                idx += 1

    # Build results with all samples
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

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate and refine contexts using vLLM for multiple benchmarks"
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
        default="Qwen/Qwen3-4B",
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
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose output (slower, processes one at a time)"
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
        default=512,
        help="Maximum tokens for refinement output (default: 512)"
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=None,
        help="Number of GPUs to use for tensor parallelism"
    )

    args = parser.parse_args()

    # Validate arguments based on dataset
    if args.dataset == "hmmt" and args.input_file is None:
        parser.error("--input_file is required when using --dataset hmmt")

    # Detect available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"\n{'='*80}")
    print(f"GPU Detection")
    print(f"{'='*80}")
    print(f"Detected {num_gpus} GPU(s)")

    if args.tensor_parallel_size is None:
        tensor_parallel_size = num_gpus if num_gpus > 0 else 1
        print(f"Auto-setting tensor_parallel_size = {tensor_parallel_size}")
    else:
        tensor_parallel_size = args.tensor_parallel_size
        print(f"Using user-specified tensor_parallel_size = {tensor_parallel_size}")

    print(f"{'='*80}\n")

    # Load input data based on dataset
    print(f"\n{'='*80}")
    print(f"Loading Dataset: {args.dataset.upper()}")
    print(f"{'='*80}")

    if args.dataset == "imobench":
        if args.input_file:
            print(f"Loading IMOBench from custom file: {args.input_file}...")
            data = load_jsonl(args.input_file)
        else:
            csv_path = download_imobench(args.cache_dir)
            print(f"Loading IMOBench from {csv_path}...")
            data = load_imobench(csv_path)
    elif args.dataset == "hmmt":
        print(f"Loading HMMT from {args.input_file}...")
        data = load_jsonl(args.input_file)

    print(f"Loaded {len(data)} problems")

    # Initialize vLLM
    print(f"Loading model {args.model} with tensor_parallel_size={tensor_parallel_size}...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True
    )

    # Sampling parameters
    initial_sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.num_tokens,
        n=args.num_samples  # Generate multiple samples per question
    )

    refinement_sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_refinement_tokens
    )

    # Process based on mode
    if args.debug:
        results = process_debug_mode(
            llm, data, initial_sampling_params, refinement_sampling_params, args
        )
    else:
        results = process_batch_mode(
            llm, data, initial_sampling_params, refinement_sampling_params, args
        )

    # Save results
    print(f"\nSaving results to {args.output_file}...")
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(f"\n{'='*80}")
    print(f"Done! Processed {len(results)} problems.")
    print(f"Results saved to {args.output_file}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
