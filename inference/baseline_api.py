#!/usr/bin/env python3
"""Baseline inference via OpenRouter API.

Single-pass generation matching the output format of baseline_vllm.py
so that prefix_recovery_api.py can consume the results.

Usage:
    export OPENROUTER_API_KEY=...
    python -m inference.baseline_api \
        --model google/gemini-3-flash-preview \
        --dataset imobench \
        --max_tokens 16384 --temperature 0.9 \
        --output_file .../baseline_gemini3flash_imobench.jsonl
"""

import argparse
import asyncio
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List

from openai import AsyncOpenAI

from inference.data_utils import load_dataset, load_jsonl, save_jsonl, strip_thinking
from inference.verify_utils import verify_batch


PROMPT_TEMPLATE = (
    "Solve the following math problem. Show your reasoning step by step "
    "and provide your final answer in \\boxed{{}}.\n\n"
    "Problem: {problem}"
)


async def call_api(client: AsyncOpenAI, model: str,
                   messages: List[Dict], max_tokens: int,
                   temperature: float,
                   semaphore: asyncio.Semaphore,
                   max_retries: int = 8) -> Dict:
    """Call OpenRouter API with exponential backoff."""
    for attempt in range(max_retries):
        async with semaphore:
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                choice = resp.choices[0]
                return {
                    "content": choice.message.content or "",
                    "finish_reason": choice.finish_reason,
                    "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                    "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
                }
            except Exception as e:
                wait = min(2 ** attempt + random.random(), 60)
                if attempt < max_retries - 1:
                    print(f"  API error (attempt {attempt+1}/{max_retries}): "
                          f"{type(e).__name__}: {e}, retrying in {wait:.1f}s...")
                    await asyncio.sleep(wait)
                else:
                    print(f"  API error (final): {e}")
                    return {
                        "content": "",
                        "finish_reason": "error",
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "error": str(e),
                    }
    return {"content": "", "finish_reason": "error",
            "prompt_tokens": 0, "completion_tokens": 0}


async def async_main(args):
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not set")
        return

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    data = load_dataset(args.dataset, args.input_file,
                        "/scratch/dkhasha1/tli104/imobench")
    print(f"Loaded {len(data)} problems")

    # Resume support
    completed_ids = set()
    existing_results = []
    if args.resume and Path(args.output_file).exists():
        existing_results = load_jsonl(args.output_file)
        completed_ids = {r.get("problem_id", "") for r in existing_results}
        print(f"Resume: found {len(completed_ids)} completed, "
              f"{len(data) - len(completed_ids)} remaining")
        data = [item for item in data if item.get("problem_id", "") not in completed_ids]
        if not data:
            print("All problems completed. Exiting.")
            return

    # Run baseline
    semaphore = asyncio.Semaphore(args.concurrency)
    t0 = time.time()

    async def do_one(idx, item):
        prompt_text = PROMPT_TEMPLATE.format(problem=item["prompt"])
        messages = [{"role": "user", "content": prompt_text}]
        r = await call_api(client, args.model, messages,
                           args.max_tokens, args.temperature, semaphore)
        return idx, item, r

    print(f"Running baseline for {args.model} on {len(data)} problems "
          f"(concurrency={args.concurrency})...")
    tasks = [do_one(i, item) for i, item in enumerate(data)]
    api_results = await asyncio.gather(*tasks)
    api_results.sort(key=lambda x: x[0])

    elapsed = time.time() - t0
    print(f"API calls complete in {elapsed:.0f}s")

    # Build results in baseline_vllm format
    results = []
    for idx, item, r in api_results:
        result = {
            "problem_id": item.get("problem_id", f"problem_{idx}"),
            "prompt": item["prompt"],
            "answer": item.get("answer", ""),
            "generation": r["content"],
            "model": args.model,
            "num_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": 1.0,
        }
        for key in ["category", "subcategory", "source"]:
            if key in item:
                result[key] = item[key]
        results.append(result)

    # Verify and report accuracy
    verify_items = [(item.get("answer", ""), strip_thinking(r["content"]))
                    for _, item, r in api_results]
    vfy = verify_batch(verify_items)
    correct = sum(1 for is_ok, _, _ in vfy if is_ok)
    print(f"Accuracy: {correct}/{len(results)} = {correct/len(results):.1%}")

    # Save
    all_results = existing_results + results
    save_jsonl(all_results, args.output_file)
    print(f"Saved {len(all_results)} results to {args.output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Baseline inference via OpenRouter API"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="OpenRouter model ID")
    parser.add_argument("--dataset", type=str, choices=["hmmt", "imobench"],
                        required=True, help="Dataset to use")
    parser.add_argument("--input_file", type=str, default=None,
                        help="Input file (required for hmmt)")
    parser.add_argument("--max_tokens", type=int, default=16384,
                        help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.9,
                        help="Sampling temperature")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output JSONL file")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Max concurrent API calls")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file")
    args = parser.parse_args()

    if args.dataset == "hmmt" and args.input_file is None:
        parser.error("--input_file required for hmmt dataset")

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
