#!/usr/bin/env python3
"""Prefix Revision Experiment — 2-turn critique-then-revise via OpenRouter.

Tests whether API models can self-correct by reviewing their own solution
and then rewriting it. No hint is given that the solution is incorrect.

Turn 1: "Review your solution step by step. Check each calculation and logical step."
Turn 2: "Based on your review, write your final solution. Provide your answer in \\boxed{}."

Usage:
    export OPENROUTER_API_KEY=...
    python -m inference.prefix_revision_api \
        --model google/gemini-3-flash-preview \
        --baseline_file .../baseline_gemini3flash_imobench.jsonl \
        --output_file .../revision_gemini3flash_imobench.jsonl \
        --max_tokens 16384 --temperature 0.9
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

from inference.data_utils import load_jsonl, save_jsonl, strip_thinking
from inference.verify_utils import verify_batch


PROMPT_TEMPLATE = (
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


async def run_revision(client: AsyncOpenAI, model: str,
                       incorrect: List[Dict], args) -> List[Dict]:
    """2-turn revision: critique then revise."""
    semaphore = asyncio.Semaphore(args.concurrency)
    t0 = time.time()

    async def do_one(idx, item):
        prompt_text = PROMPT_TEMPLATE.format(problem=item["prompt"])
        visible_gen = strip_thinking(item["generation"]) if item["generation"] else item["generation"]

        # Turn 1: critique
        messages_t1 = [
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": visible_gen},
            {"role": "user", "content": CRITIQUE_PROMPT},
        ]
        critique = await call_api(client, model, messages_t1,
                                  args.max_tokens, args.temperature, semaphore)

        # Turn 2: revise
        messages_t2 = messages_t1 + [
            {"role": "assistant", "content": critique["content"]},
            {"role": "user", "content": REVISE_PROMPT},
        ]
        revision = await call_api(client, model, messages_t2,
                                  args.max_tokens, args.temperature, semaphore)

        if (idx + 1) % 20 == 0:
            print(f"  Completed {idx + 1}/{len(incorrect)}")

        return idx, item, critique, revision

    tasks = [do_one(i, item) for i, item in enumerate(incorrect)]
    api_results = await asyncio.gather(*tasks)
    api_results.sort(key=lambda x: x[0])

    # Verify final revision
    verify_items = []
    for _, item, _, revision in api_results:
        gold = item.get("answer", "")
        text = strip_thinking(revision["content"]) if revision["content"] else ""
        verify_items.append((gold, text))

    vfy = verify_batch(verify_items)

    results = []
    for (idx, item, critique, revision), (is_correct, status, _) in zip(api_results, vfy):
        results.append({
            "problem_id": item.get("problem_id", ""),
            "prompt": item["prompt"],
            "answer": item.get("answer", ""),
            "incorrect_prefix": item["generation"],
            "critique": critique["content"],
            "revision": revision["content"],
            "recovered": bool(is_correct),
            "model": model,
            "critique_tokens": critique.get("completion_tokens", 0),
            "revision_tokens": revision.get("completion_tokens", 0),
            **{k: item[k] for k in ["category", "subcategory", "source"] if k in item},
        })

    elapsed = time.time() - t0
    num_recovered = sum(1 for r in results if r["recovered"])
    print(f"  Revision: {num_recovered}/{len(results)} = "
          f"{num_recovered/len(results):.1%} ({elapsed:.0f}s)")
    return results


async def async_main(args):
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not set")
        return

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    print(f"Loading baseline from {args.baseline_file}")
    baseline_data = load_jsonl(args.baseline_file)
    print(f"Loaded {len(baseline_data)} baseline results")

    print("Verifying baseline solutions...")
    incorrect = filter_incorrect(baseline_data)
    print(f"Found {len(incorrect)} incorrect out of {len(baseline_data)} "
          f"({len(incorrect)/len(baseline_data):.1%})")

    if not incorrect:
        print("No incorrect solutions. Exiting.")
        return

    print(f"\nRunning 2-turn revision for {args.model}...")
    results = await run_revision(client, args.model, incorrect, args)

    save_jsonl(results, args.output_file)
    print(f"Saved {len(results)} results to {args.output_file}")

    num_recovered = sum(1 for r in results if r["recovered"])
    print(f"\n{'='*60}")
    print(f"Prefix Revision Results (2-turn)")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Baseline total: {len(baseline_data)}")
    print(f"Baseline incorrect: {len(incorrect)} ({len(incorrect)/len(baseline_data):.1%})")
    print(f"Recovered: {num_recovered}/{len(incorrect)} ({num_recovered/len(incorrect):.1%})")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="2-turn prefix revision experiment for API models via OpenRouter"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="OpenRouter model ID")
    parser.add_argument("--baseline_file", type=str, required=True,
                        help="Baseline JSONL output file")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output JSONL file for revision results")
    parser.add_argument("--max_tokens", type=int, default=16384,
                        help="Max tokens per API call")
    parser.add_argument("--temperature", type=float, default=0.9,
                        help="Sampling temperature")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Max concurrent API calls")
    args = parser.parse_args()

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
