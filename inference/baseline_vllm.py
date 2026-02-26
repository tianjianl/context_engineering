#!/usr/bin/env python3
"""
Baseline Inference Script using vLLM (No Refinement)

This script performs standard generation without context refinement.
Supports both tensor parallelism and data parallelism.
"""

import argparse
import json
import os
from typing import List, Dict
from pathlib import Path
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import multiprocessing as mp

from inference.data_utils import load_jsonl, load_dataset, apply_chat_template
from inference.args_utils import add_common_args, validate_args


def data_parallel_worker(
    gpu_id: int,
    data_shard: List[Dict],
    args,
    result_queue: mp.Queue,
):
    """Worker process for data parallel inference using dp_utils signature."""
    tp = args.tensor_parallel_size
    if tp > 1:
        gpu_start = gpu_id * tp
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_start + i) for i in range(tp))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[Worker {gpu_id}] Starting on GPU(s), processing {len(data_shard)} items")

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
        n=args.num_samples,
    )

    valid_items = [item for item in data_shard if "prompt" in item and item["prompt"]]
    generation_prompts = [
        apply_chat_template(tokenizer, item["prompt"], add_generation_prompt=True)
        for item in valid_items
    ]

    print(f"[Worker {gpu_id}] Generating responses for {len(generation_prompts)} prompts...")
    outputs = llm.generate(generation_prompts, sampling_params)
    print(f"[Worker {gpu_id}] Generation complete!")

    results = []
    for idx, (item, output) in enumerate(zip(valid_items, outputs)):
        if args.num_samples == 1:
            result = {
                "problem_id": item.get("problem_id", f"problem_{idx}"),
                "prompt": item["prompt"],
                "answer": item.get("answer", ""),
                "generation": output.outputs[0].text,
                "model": args.model,
                "num_tokens": args.num_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
            }
            for key in ["category", "subcategory", "source"]:
                if key in item:
                    result[key] = item[key]
            results.append(result)
        else:
            for sample_idx in range(args.num_samples):
                result = {
                    "problem_id": item.get("problem_id", f"problem_{idx}"),
                    "sample_id": sample_idx,
                    "prompt": item["prompt"],
                    "answer": item.get("answer", ""),
                    "generation": output.outputs[sample_idx].text,
                    "model": args.model,
                    "num_tokens": args.num_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                }
                for key in ["category", "subcategory", "source"]:
                    if key in item:
                        result[key] = item[key]
                results.append(result)

    print(f"[Worker {gpu_id}] Completed processing {len(results)} results")
    result_queue.put((gpu_id, results))


def main():
    parser = argparse.ArgumentParser(
        description="Baseline inference using vLLM (no refinement)"
    )
    add_common_args(parser)
    parser.add_argument(
        "--tensor_parallel_size", type=int, default=None,
        help="Number of GPUs to use for tensor parallelism (per worker in DP mode)"
    )
    parser.add_argument(
        "--data_parallel", action="store_true",
        help="Enable data parallelism across GPUs"
    )
    parser.add_argument(
        "--num_dp_workers", type=int, default=None,
        help="Number of data parallel workers (default: auto-detect based on GPUs)"
    )
    parser.add_argument(
        "--max_model_len", type=int, default=None,
        help="Maximum model context length (default: use model's default)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing output file, skipping already-completed problem_ids"
    )

    args = parser.parse_args()
    validate_args(args, parser)

    # Detect available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"\n{'='*80}")
    print(f"GPU Detection")
    print(f"{'='*80}")
    print(f"Detected {num_gpus} GPU(s)")

    # Handle data parallel vs tensor parallel mode
    if args.data_parallel:
        tp_size_per_worker = args.tensor_parallel_size if args.tensor_parallel_size else 1
        if args.num_dp_workers is None:
            num_dp_workers = num_gpus // tp_size_per_worker
        else:
            num_dp_workers = args.num_dp_workers

        if num_dp_workers < 1:
            num_dp_workers = 1

        # Store for worker access
        args.tensor_parallel_size = tp_size_per_worker

        print(f"Mode: Data Parallel")
        print(f"Data parallel workers: {num_dp_workers}")
        print(f"Tensor parallel size per worker: {tp_size_per_worker}")
    else:
        if args.tensor_parallel_size is None:
            tensor_parallel_size = num_gpus if num_gpus > 0 else 1
            print(f"Mode: Tensor Parallel")
            print(f"Auto-setting tensor_parallel_size = {tensor_parallel_size}")
        else:
            tensor_parallel_size = args.tensor_parallel_size
            print(f"Mode: Tensor Parallel")
            print(f"Using user-specified tensor_parallel_size = {tensor_parallel_size}")

    print(f"{'='*80}\n")

    # Load input data
    print(f"\n{'='*80}")
    print(f"Loading Dataset: {args.dataset.upper()}")
    print(f"{'='*80}")

    data = load_dataset(args.dataset, args.input_file, args.cache_dir)
    print(f"Loaded {len(data)} problems")

    # Resume: filter out already-completed problem_ids
    completed_problem_ids = set()
    if args.resume and Path(args.output_file).exists():
        existing = load_jsonl(args.output_file)
        for rec in existing:
            completed_problem_ids.add(rec.get("problem_id", ""))
        print(f"Resume: found {len(completed_problem_ids)} completed problem_ids in {args.output_file}")
        original_count = len(data)
        data = [item for item in data if item.get("problem_id", "") not in completed_problem_ids]
        print(f"Resume: {original_count} -> {len(data)} problems remaining")
        if len(data) == 0:
            print("All problems already completed. Exiting.")
            return

    print(f"\n{'='*80}")
    print(f"BASELINE INFERENCE (No Refinement)")
    print(f"{'='*80}")
    print(f"Model: {args.model}")
    print(f"Max tokens: {args.num_tokens}")
    print(f"Samples per question: {args.num_samples}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p: {args.top_p}")
    if args.data_parallel:
        print(f"Mode: Data Parallel ({num_dp_workers} workers, TP={tp_size_per_worker} per worker)")
    else:
        print(f"Mode: Tensor Parallel (TP={tensor_parallel_size})")
    print(f"{'='*80}\n")

    # Filter valid items
    valid_items = [item for item in data if "prompt" in item and item["prompt"]]
    if len(valid_items) < len(data):
        print(f"Warning: Skipped {len(data) - len(valid_items)} items missing 'prompt' field")

    if args.data_parallel:
        # Data parallel mode using dp_utils
        from inference.dp_utils import shard_data, run_data_parallel

        shards = shard_data(valid_items, num_dp_workers)
        print(f"Data distribution: {[len(s) for s in shards]} items per worker")

        print(f"Running data parallel inference with {num_dp_workers} workers...")
        # Set num_gpus on args for dp_utils compatibility
        args.num_gpus = num_dp_workers
        merged_results = run_data_parallel(data_parallel_worker, shards, num_dp_workers, args)
        print("Data parallel generation complete!")

        results = merged_results
    else:
        # Tensor parallel only mode
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

        llm_kwargs = dict(
            model=args.model,
            tensor_parallel_size=tensor_parallel_size,
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
            n=args.num_samples
        )

        generation_prompts = [
            apply_chat_template(tokenizer, item["prompt"], add_generation_prompt=True)
            for item in valid_items
        ]

        print(f"Generating responses for {len(generation_prompts)} prompts...")
        outputs = llm.generate(generation_prompts, sampling_params)
        print("Generation complete!")

        results = []
        for idx, (item, output) in enumerate(zip(valid_items, outputs)):
            if args.num_samples == 1:
                result = {
                    "problem_id": item.get("problem_id", f"problem_{idx}"),
                    "prompt": item["prompt"],
                    "answer": item.get("answer", ""),
                    "generation": output.outputs[0].text,
                    "model": args.model,
                    "num_tokens": args.num_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                }
                for key in ["category", "subcategory", "source"]:
                    if key in item:
                        result[key] = item[key]
                results.append(result)
            else:
                for sample_idx in range(args.num_samples):
                    result = {
                        "problem_id": item.get("problem_id", f"problem_{idx}"),
                        "sample_id": sample_idx,
                        "prompt": item["prompt"],
                        "answer": item.get("answer", ""),
                        "generation": output.outputs[sample_idx].text,
                        "model": args.model,
                        "num_tokens": args.num_tokens,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                    }
                    for key in ["category", "subcategory", "source"]:
                        if key in item:
                            result[key] = item[key]
                    results.append(result)

    # Save results (append if resuming, overwrite otherwise)
    print(f"\nSaving results to {args.output_file}...")
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    write_mode = 'a' if args.resume and completed_problem_ids else 'w'
    with open(args.output_file, write_mode, encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    total_saved = len(results) + (len(completed_problem_ids) * args.num_samples if args.resume and completed_problem_ids else 0)
    print(f"Done! Saved {len(results)} new results to {args.output_file} (total ~{total_saved} lines)")


if __name__ == "__main__":
    main()
