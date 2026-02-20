#!/usr/bin/env python3
"""
Generate solution traces from Gemini-2.5-Flash for the strict-filtered SuperGPQA dataset.
Only sends the raw question (no multiple choice options) to get genuine solution traces.
Saves results incrementally to a JSONL file with resume capability.
Supports batched inference using asyncio for concurrent API calls.
"""

import os
import json
import time
import asyncio
import argparse
from pathlib import Path
from tqdm import tqdm
from google import genai
from google.genai import errors


def create_prompt(question: str, field: str, subfield: str) -> str:
    """Create the prompt for the model - raw question without options."""
    return f"""Solve the following problem from {field} ({subfield}). Show your reasoning step by step, then provide the final answer.

**Problem:**
{question}

Please provide a detailed solution with clear step-by-step reasoning, and conclude with your final answer."""


async def generate_with_retry_async(client, prompt: str, max_retries: int = 5) -> tuple[str, dict]:
    """Generate response with exponential backoff retry logic (async version)."""
    for attempt in range(max_retries):
        try:
            response = await client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={
                    "max_output_tokens": 16384,
                }
            )
            # Extract usage metadata if available
            usage_info = {}
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage_info = {
                    "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', None),
                    "completion_tokens": getattr(response.usage_metadata, 'candidates_token_count', None),
                    "total_tokens": getattr(response.usage_metadata, 'total_token_count', None),
                }
            return response.text, usage_info
        except errors.ServerError as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Server error (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            else:
                raise
        except errors.ClientError as e:
            # For client errors (e.g., rate limit), also retry with backoff
            if attempt < max_retries - 1:
                wait_time = 2 ** (attempt + 1)  # Longer backoff for rate limits
                print(f"Client error (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            else:
                raise


async def process_single_example(client, idx: int, example: dict, max_retries: int) -> dict:
    """Process a single example and return the result dict."""
    prompt = create_prompt(
        example['question'],
        example['field'],
        example['subfield']
    )

    try:
        response_text, usage_info = await generate_with_retry_async(
            client, prompt, max_retries=max_retries
        )

        result = {
            "index": idx,
            "uuid": example['uuid'],
            "discipline": example['discipline'],
            "field": example['field'],
            "subfield": example['subfield'],
            "difficulty": example['difficulty'],
            "question": example['question'],
            "ground_truth_answer": example['answer'],
            "ground_truth_letter": example['answer_letter'],
            "options": example['options'],  # Store for later evaluation
            "model": "gemini-2.5-flash",
            "prompt": prompt,
            "response": response_text,
            "usage": usage_info,
            "status": "success",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        }
    except Exception as e:
        result = {
            "index": idx,
            "uuid": example['uuid'],
            "discipline": example['discipline'],
            "field": example['field'],
            "subfield": example['subfield'],
            "difficulty": example['difficulty'],
            "question": example['question'],
            "ground_truth_answer": example['answer'],
            "ground_truth_letter": example['answer_letter'],
            "options": example['options'],
            "model": "gemini-2.5-flash",
            "prompt": prompt,
            "response": None,
            "usage": {},
            "status": "failed",
            "error": str(e),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        }

    return result


async def process_batch(client, batch_items: list, max_retries: int, semaphore: asyncio.Semaphore) -> list:
    """Process a batch of examples concurrently with semaphore-based rate limiting."""
    async def process_with_semaphore(idx, example):
        async with semaphore:
            return await process_single_example(client, idx, example, max_retries)

    tasks = [process_with_semaphore(idx, example) for idx, example in batch_items]
    results = await asyncio.gather(*tasks)
    return results


def load_completed_indices(output_file: Path) -> set:
    """Load indices of already completed examples for resume capability."""
    completed = set()
    if output_file.exists():
        with open(output_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    completed.add(data['index'])
                except (json.JSONDecodeError, KeyError):
                    continue
    return completed


def load_dataset(input_path: str) -> list:
    """Load the filtered SuperGPQA dataset from JSON file."""
    print(f"Loading dataset from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples")
    return data


async def main_async(args):
    """Async main function for batched inference."""
    # Setup paths
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_file = output_dir / args.output_filename

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = load_dataset(str(input_path))
    total_examples = len(dataset)

    # Load completed indices for resume capability
    completed_indices = load_completed_indices(output_file)
    print(f"Found {len(completed_indices)} already completed examples")

    # Initialize Gemini client
    client = genai.Client()

    # Build list of pending examples
    pending_items = []
    for idx in range(total_examples):
        if idx not in completed_indices:
            pending_items.append((idx, dataset[idx]))

    print(f"\nGenerating solution traces from Gemini-2.5-Flash...")
    print(f"NOTE: Sending raw questions WITHOUT multiple choice options")
    print(f"Batch size: {args.batch_size}, Max concurrent requests: {args.max_concurrent}")
    print(f"Pending examples: {len(pending_items)}")
    print(f"Output file: {output_file}")

    successful = len(completed_indices)
    failed = 0

    # Create semaphore for rate limiting concurrent requests
    semaphore = asyncio.Semaphore(args.max_concurrent)

    # Process in batches
    with open(output_file, 'a') as f:
        for batch_start in tqdm(range(0, len(pending_items), args.batch_size), desc="Processing batches"):
            batch_items = pending_items[batch_start:batch_start + args.batch_size]

            # Process batch concurrently
            results = await process_batch(client, batch_items, args.max_retries, semaphore)

            # Write results and update counts
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                if result['status'] == 'success':
                    successful += 1
                else:
                    failed += 1
                    print(f"\nFailed on index {result['index']}: {result.get('error', 'Unknown error')}")

            f.flush()  # Ensure data is written immediately

            # Optional delay between batches
            if args.delay_between_batches > 0:
                await asyncio.sleep(args.delay_between_batches)

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total examples: {total_examples}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Gemini-2.5-Flash solution traces for SuperGPQA (strict math filtered)"
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default="/scratch/dkhasha1/tli104/super_gpqa_math_strict.json",
        help="Path to the filtered SuperGPQA JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/scratch/dkhasha1/tli104/super_gpqa_inference",
        help="Directory to save output results"
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        default="gemini_super_gpqa_math_strict_traces.jsonl",
        help="Output filename"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum number of retries for API calls"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of examples to process per batch"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Maximum number of concurrent API requests"
    )
    parser.add_argument(
        "--delay-between-batches",
        type=float,
        default=0.5,
        help="Delay in seconds between batches to avoid rate limiting"
    )
    args = parser.parse_args()

    # Run async main
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
