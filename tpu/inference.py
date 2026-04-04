#!/usr/bin/env python3
"""
Baseline inference on TPU using vLLM.

Usage:
    python -m tpu.inference \
        --dataset imobench_v2 \
        --model Qwen/Qwen3-4B \
        --num_tokens 16384 \
        --output_file results/tpu_qwen3.5_4b_imobench_v2.jsonl
"""

import argparse
import json
from pathlib import Path

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from inference.data_utils import load_dataset, apply_chat_template, save_jsonl


def main():
    parser = argparse.ArgumentParser(description="TPU baseline inference with vLLM")
    parser.add_argument("--dataset", type=str, default="imobench_v2",
                        choices=["imobench", "imobench_v2"])
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default="./data_cache",
                        help="Local directory for dataset caching (no scratch on TPU VM)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--num_tokens", type=int, default=16384)
    parser.add_argument("--output_file", type=str, default="results/tpu_output.jsonl")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--tensor_parallel_size", type=int, default=8,
                        help="Number of TPU chips for tensor parallelism (default: 8)")
    parser.add_argument("--max_model_len", type=int, default=None)
    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    data = load_dataset(args.dataset, args.input_file, args.cache_dir)
    print(f"Loaded {len(data)} problems")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Initialize vLLM for TPU
    llm_kwargs = dict(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
    )
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len

    print(f"Initializing vLLM on TPU (tp={args.tensor_parallel_size})...")
    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.num_tokens,
        n=args.num_samples,
    )

    # Format prompts
    valid_items = [item for item in data if item.get("prompt")]
    prompts = [
        apply_chat_template(tokenizer, item["prompt"], add_generation_prompt=True)
        for item in valid_items
    ]
    print(f"Generating {len(prompts)} prompts (n={args.num_samples}, max_tokens={args.num_tokens})...")

    # Generate
    outputs = llm.generate(prompts, sampling_params)
    print("Generation complete!")

    # Collect results
    results = []
    for idx, (item, output) in enumerate(zip(valid_items, outputs)):
        base = {
            "problem_id": item.get("problem_id", f"problem_{idx}"),
            "prompt": item["prompt"],
            "answer": item.get("answer", ""),
            "model": args.model,
            "num_tokens": args.num_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        }
        for key in ("category", "subcategory", "source"):
            if key in item:
                base[key] = item[key]

        if args.num_samples == 1:
            results.append({**base, "generation": output.outputs[0].text})
        else:
            for s_idx in range(args.num_samples):
                results.append({
                    **base,
                    "sample_id": s_idx,
                    "generation": output.outputs[s_idx].text,
                })

    # Save
    save_jsonl(results, args.output_file)
    print(f"Saved {len(results)} results to {args.output_file}")


if __name__ == "__main__":
    main()
