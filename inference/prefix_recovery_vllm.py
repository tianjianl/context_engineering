#!/usr/bin/env python3
"""Prefix Recovery Experiment — vLLM (local models).

Tests whether LMs can recover after producing an incorrect solution.
Loads baseline outputs, filters for incorrect solutions, then feeds those
as assistant prefixes and lets the model continue generating.

Usage:
    python -m inference.prefix_recovery_vllm \
        --model Qwen/Qwen3-4B-Instruct-2507 \
        --baseline_file .../baseline_qwen3_4b_imobench.jsonl \
        --output_file .../recovery_qwen3_4b_imobench.jsonl \
        --num_tokens 16384 --temperature 0.9
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from inference.data_utils import (
    apply_chat_template_with_prefix,
    load_jsonl,
    save_jsonl,
    strip_thinking,
)
from inference.verify_utils import verify_batch


def filter_incorrect(data: List[Dict]) -> List[Dict]:
    """Return only items whose baseline generation is incorrect."""
    items_to_verify = []
    for item in data:
        gold = item.get("answer", "")
        gen = item.get("generation", "")
        text = strip_thinking(gen) if gen else ""
        items_to_verify.append((gold, text))

    results = verify_batch(items_to_verify)
    incorrect = []
    for item, (is_correct, status, _) in zip(data, results):
        if not is_correct:
            incorrect.append(item)
    return incorrect


def main():
    parser = argparse.ArgumentParser(
        description="Prefix recovery experiment for local vLLM models"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Model name or path")
    parser.add_argument("--baseline_file", type=str, required=True,
                        help="Baseline JSONL output file to load")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output JSONL file for recovery results")
    parser.add_argument("--num_tokens", type=int, default=16384,
                        help="Max tokens to generate in continuation")
    parser.add_argument("--temperature", type=float, default=0.9,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling")
    parser.add_argument("--top_k", type=int, default=-1,
                        help="Top-k sampling (-1 = disabled)")
    parser.add_argument("--max_model_len", type=int, default=None,
                        help="Max model context length")
    parser.add_argument("--tensor_parallel_size", type=int, default=None,
                        help="Tensor parallel size (default: auto)")
    args = parser.parse_args()

    # Load baseline
    print(f"Loading baseline from {args.baseline_file}")
    baseline_data = load_jsonl(args.baseline_file)
    print(f"Loaded {len(baseline_data)} baseline results")

    # Filter for incorrect
    print("Verifying baseline solutions...")
    incorrect = filter_incorrect(baseline_data)
    print(f"Found {len(incorrect)} incorrect solutions out of {len(baseline_data)} "
          f"({len(incorrect)/len(baseline_data):.1%})")

    if not incorrect:
        print("No incorrect solutions to recover from. Exiting.")
        return

    # Setup vLLM
    num_gpus = torch.cuda.device_count()
    tp = args.tensor_parallel_size if args.tensor_parallel_size else (num_gpus if num_gpus > 0 else 1)
    print(f"Detected {num_gpus} GPU(s), using tensor_parallel_size={tp}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    llm_kwargs = dict(
        model=args.model,
        tensor_parallel_size=tp,
        trust_remote_code=True,
    )
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.num_tokens,
    )

    # Build prefixed prompts — model continues from its own incorrect generation
    print("Building prefix prompts...")
    prefixed_prompts = []
    for item in incorrect:
        prompt = item["prompt"]
        prefix = item["generation"]  # full generation including <think> and \boxed{}
        prefixed_prompts.append(
            apply_chat_template_with_prefix(tokenizer, prompt, prefix)
        )

    # Generate continuations
    print(f"Generating continuations for {len(prefixed_prompts)} problems...")
    outputs = llm.generate(prefixed_prompts, sampling_params)
    print("Generation complete!")

    # Verify continuations
    print("Verifying continuations...")
    verify_items = []
    for item, output in zip(incorrect, outputs):
        gold = item.get("answer", "")
        # The continuation is the NEW text after the prefix
        continuation = output.outputs[0].text
        # For verification, check the full text (prefix + continuation)
        full_text = item["generation"] + continuation
        text_for_verify = strip_thinking(full_text) if full_text else ""
        verify_items.append((gold, text_for_verify))

    verify_results = verify_batch(verify_items)

    # Build output
    results = []
    num_recovered = 0
    for item, output, (is_correct, status, _) in zip(incorrect, outputs, verify_results):
        continuation = output.outputs[0].text
        recovered = bool(is_correct)
        if recovered:
            num_recovered += 1
        result = {
            "problem_id": item.get("problem_id", ""),
            "prompt": item["prompt"],
            "answer": item.get("answer", ""),
            "incorrect_prefix": item["generation"],
            "continuation": continuation,
            "recovered": recovered,
            "model": args.model,
        }
        for key in ["category", "subcategory", "source"]:
            if key in item:
                result[key] = item[key]
        results.append(result)

    # Save
    print(f"\nSaving {len(results)} results to {args.output_file}")
    save_jsonl(results, args.output_file)

    # Summary
    print(f"\n{'='*60}")
    print(f"Prefix Recovery Results")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Baseline total: {len(baseline_data)}")
    print(f"Baseline incorrect: {len(incorrect)} ({len(incorrect)/len(baseline_data):.1%})")
    print(f"Recovered: {num_recovered}/{len(incorrect)} ({num_recovered/len(incorrect):.1%})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
