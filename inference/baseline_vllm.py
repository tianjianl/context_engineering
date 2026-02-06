#!/usr/bin/env python3
"""
Baseline Inference Script using vLLM (No Refinement)

This script performs standard generation without context refinement.
Supports both tensor parallelism and data parallelism.
"""

import argparse
import json
import csv
import os
from typing import List, Dict, Tuple
from pathlib import Path
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import urllib.request
import multiprocessing as mp
from functools import partial


# Dataset URLs
IMOBENCH_URL = "https://raw.githubusercontent.com/google-deepmind/superhuman/main/imobench/answerbench.csv"


def apply_chat_template(tokenizer, prompt: str, add_generation_prompt: bool = False) -> str:
    """Apply chat template to format prompt for instruct models."""
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt
    )


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


def data_parallel_worker(
    worker_id: int,
    gpu_ids: List[int],
    model_name: str,
    data_chunk: List[Dict],
    chunk_indices: List[int],
    sampling_params_dict: Dict,
    tensor_parallel_size: int,
    output_queue: mp.Queue,
    max_model_len: int = None,
):
    """
    Worker function for data parallel inference.
    Each worker runs on assigned GPU(s) and processes its data chunk.
    Results are put into output_queue.
    """
    # Set CUDA_VISIBLE_DEVICES for this worker
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

    print(f"[Worker {worker_id}] Starting on GPU(s): {gpu_ids}, processing {len(data_chunk)} items")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Initialize vLLM for this worker
    llm_kwargs = dict(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
    )
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len
    llm = LLM(**llm_kwargs)

    # Create sampling params
    sampling_params = SamplingParams(**sampling_params_dict)

    # Apply chat template to prompts
    generation_prompts = [
        apply_chat_template(tokenizer, item["prompt"], add_generation_prompt=True)
        for item in data_chunk
    ]

    # Generate
    print(f"[Worker {worker_id}] Generating responses for {len(generation_prompts)} prompts...")
    outputs = llm.generate(generation_prompts, sampling_params)
    print(f"[Worker {worker_id}] Generation complete!")

    # Collect results with original indices
    # Convert outputs to serializable format
    results = []
    for orig_idx, item, output in zip(chunk_indices, data_chunk, outputs):
        output_texts = [o.text for o in output.outputs]
        results.append((orig_idx, item, output_texts))

    # Put results in queue
    output_queue.put((worker_id, results))
    print(f"[Worker {worker_id}] Results sent to queue")


def run_data_parallel(
    data: List[Dict],
    model_name: str,
    sampling_params_dict: Dict,
    num_dp_workers: int,
    tp_size_per_worker: int,
    max_model_len: int = None,
) -> List[Tuple[int, Dict, List[str]]]:
    """
    Run inference with data parallelism across multiple GPUs.

    Args:
        data: List of input items
        model_name: Model name or path
        sampling_params_dict: Sampling parameters as dict
        num_dp_workers: Number of data parallel workers
        tp_size_per_worker: Tensor parallel size per worker

    Returns:
        List of (original_index, item, output_texts) tuples
    """
    num_gpus = torch.cuda.device_count()
    gpus_per_worker = tp_size_per_worker

    # Assign GPUs to workers
    gpu_assignments = []
    for i in range(num_dp_workers):
        start_gpu = i * gpus_per_worker
        end_gpu = start_gpu + gpus_per_worker
        gpu_assignments.append(list(range(start_gpu, end_gpu)))

    print(f"\n{'='*80}")
    print(f"Data Parallel Configuration")
    print(f"{'='*80}")
    print(f"Total GPUs: {num_gpus}")
    print(f"Data parallel workers: {num_dp_workers}")
    print(f"Tensor parallel size per worker: {tp_size_per_worker}")
    print(f"GPU assignments: {gpu_assignments}")
    print(f"{'='*80}\n")

    # Split data across workers
    chunk_size = (len(data) + num_dp_workers - 1) // num_dp_workers
    data_chunks = []
    index_chunks = []
    for i in range(num_dp_workers):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, len(data))
        data_chunks.append(data[start_idx:end_idx])
        index_chunks.append(list(range(start_idx, end_idx)))

    print(f"Data distribution: {[len(chunk) for chunk in data_chunks]} items per worker")

    # Use multiprocessing spawn to avoid CUDA issues
    ctx = mp.get_context("spawn")

    # Create output queue for results
    output_queue = ctx.Queue()

    # Create and start worker processes (non-daemonic)
    processes = []
    for worker_id in range(num_dp_workers):
        p = ctx.Process(
            target=data_parallel_worker,
            args=(
                worker_id,
                gpu_assignments[worker_id],
                model_name,
                data_chunks[worker_id],
                index_chunks[worker_id],
                sampling_params_dict,
                tp_size_per_worker,
                output_queue,
                max_model_len,
            ),
        )
        processes.append(p)
        p.start()

    # Collect results from all workers
    all_results = {}
    for _ in range(num_dp_workers):
        worker_id, results = output_queue.get()
        all_results[worker_id] = results
        print(f"[Main] Received results from worker {worker_id}")

    # Wait for all processes to finish
    for p in processes:
        p.join()

    # Flatten and sort results by original index
    flattened_results = []
    for worker_id in range(num_dp_workers):
        flattened_results.extend(all_results[worker_id])

    # Sort by original index to maintain order
    flattened_results.sort(key=lambda x: x[0])

    return flattened_results


def main():
    parser = argparse.ArgumentParser(
        description="Baseline inference using vLLM (no refinement)"
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
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="output_baseline.jsonl",
        help="Output JSONL file for results"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=16,
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
        "--tensor_parallel_size",
        type=int,
        default=None,
        help="Number of GPUs to use for tensor parallelism (per worker in DP mode)"
    )
    parser.add_argument(
        "--data_parallel",
        action="store_true",
        help="Enable data parallelism across GPUs"
    )
    parser.add_argument(
        "--num_dp_workers",
        type=int,
        default=None,
        help="Number of data parallel workers (default: auto-detect based on GPUs)"
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=None,
        help="Maximum model context length (default: use model's default)"
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

    # Handle data parallel vs tensor parallel mode
    if args.data_parallel:
        # Data parallel mode
        tp_size_per_worker = args.tensor_parallel_size if args.tensor_parallel_size else 1
        if args.num_dp_workers is None:
            num_dp_workers = num_gpus // tp_size_per_worker
        else:
            num_dp_workers = args.num_dp_workers

        if num_dp_workers < 1:
            num_dp_workers = 1

        print(f"Mode: Data Parallel")
        print(f"Data parallel workers: {num_dp_workers}")
        print(f"Tensor parallel size per worker: {tp_size_per_worker}")
    else:
        # Tensor parallel only mode
        if args.tensor_parallel_size is None:
            tensor_parallel_size = num_gpus if num_gpus > 0 else 1
            print(f"Mode: Tensor Parallel")
            print(f"Auto-setting tensor_parallel_size = {tensor_parallel_size}")
        else:
            tensor_parallel_size = args.tensor_parallel_size
            print(f"Mode: Tensor Parallel")
            print(f"Using user-specified tensor_parallel_size = {tensor_parallel_size}")

    print(f"{'='*80}\n")

    # Load input data based on dataset
    print(f"\n{'='*80}")
    print(f"Loading Dataset: {args.dataset.upper()}")
    print(f"{'='*80}")

    if args.dataset == "imobench":
        if args.input_file:
            print(f"Loading IMOBench from custom file: {args.input_file}...")
            if args.input_file.endswith('.csv'):
                data = load_imobench(args.input_file)
            else:
                data = load_jsonl(args.input_file)
        else:
            csv_path = download_imobench(args.cache_dir)
            print(f"Loading IMOBench from {csv_path}...")
            data = load_imobench(csv_path)
    elif args.dataset == "hmmt":
        print(f"Loading HMMT from {args.input_file}...")
        data = load_jsonl(args.input_file)

    print(f"Loaded {len(data)} problems")

    # Initialize tokenizer and model only for non-data-parallel mode
    if not args.data_parallel:
        # Initialize tokenizer for chat template
        print(f"Loading tokenizer for {args.model}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

        # Initialize vLLM
        print(f"Loading model {args.model} with tensor_parallel_size={tensor_parallel_size}...")
        llm_kwargs = dict(
            model=args.model,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
        )
        if args.max_model_len is not None:
            llm_kwargs["max_model_len"] = args.max_model_len
        llm = LLM(**llm_kwargs)

        # Sampling parameters
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.num_tokens,
            n=args.num_samples
        )

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

    # Sampling params as dict for passing to workers
    sampling_params_dict = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.num_tokens,
        "n": args.num_samples,
    }

    if args.data_parallel:
        # Data parallel mode
        print(f"Running data parallel inference with {num_dp_workers} workers...")
        dp_results = run_data_parallel(
            data=valid_items,
            model_name=args.model,
            sampling_params_dict=sampling_params_dict,
            num_dp_workers=num_dp_workers,
            tp_size_per_worker=tp_size_per_worker,
            max_model_len=args.max_model_len,
        )
        print("Data parallel generation complete!")

        # Build results from data parallel outputs
        # dp_results contains (orig_idx, item, output_texts) where output_texts is List[str]
        results = []
        for orig_idx, item, output_texts in dp_results:
            if args.num_samples == 1:
                result = {
                    "problem_id": item.get("problem_id", f"problem_{orig_idx}"),
                    "prompt": item["prompt"],
                    "answer": item.get("answer", ""),
                    "generation": output_texts[0],
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
                        "problem_id": item.get("problem_id", f"problem_{orig_idx}"),
                        "sample_id": sample_idx,
                        "prompt": item["prompt"],
                        "answer": item.get("answer", ""),
                        "generation": output_texts[sample_idx],
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
        # Original tensor parallel mode
        # Apply chat template to all prompts
        generation_prompts = [
            apply_chat_template(tokenizer, item["prompt"], add_generation_prompt=True)
            for item in valid_items
        ]

        print(f"Generating responses for {len(generation_prompts)} prompts...")
        outputs = llm.generate(generation_prompts, sampling_params)
        print("Generation complete!")

        # Build results
        results = []
        for idx, (item, output) in enumerate(zip(valid_items, outputs)):
            if args.num_samples == 1:
                # Single sample
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
                # Add optional fields if present
                for key in ["category", "subcategory", "source"]:
                    if key in item:
                        result[key] = item[key]
                results.append(result)
            else:
                # Multiple samples
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

    # Save results
    print(f"\nSaving results to {args.output_file}...")
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(f"Done! Saved {len(results)} results to {args.output_file}")


if __name__ == "__main__":
    main()
