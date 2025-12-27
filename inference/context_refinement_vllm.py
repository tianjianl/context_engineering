#!/usr/bin/env python3
"""
Context Refinement Script using vLLM

This script reads prompts from a JSONL file, generates N tokens,
then refines the partial generation using a context refinement prompt.
"""

import argparse
import json
from typing import List, Dict
from pathlib import Path
from vllm import LLM, SamplingParams


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
        assistant_message_prefix = ""  # Accumulates all previous refined contexts

        # Perform multiple rounds of generation and refinement
        for round_num in range(args.rounds):
            print(f"\n{'#'*80}")
            print(f"ROUND {round_num + 1}/{args.rounds}")
            print(f"{'#'*80}")

            # Stage 1: Generate N tokens using chat format
            # User message is always the original prompt
            # Assistant message contains the accumulated prefix from previous rounds
            if round_num == 0:
                # First round: just the user prompt
                prompt_text = original_prompt
            else:
                # Subsequent rounds: user prompt + assistant prefix
                messages = [
                    {"role": "user", "content": original_prompt},
                    {"role": "assistant", "content": assistant_message_prefix}
                ]
                # Format as chat continuation prompt
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

            # Stage 2: Refine based on --accumulate flag
            # Determine what to pass to refinement
            if args.accumulate:
                # Accumulate: assistant prefix + current generation
                if round_num == 0:
                    context_to_refine = current_round_generation
                else:
                    context_to_refine = assistant_message_prefix + current_round_generation
                print(f"\nRefining ACCUMULATED context (prefix + current generation)...")
            else:
                # No accumulation: only refine current round's generation
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

            # Store this round's data
            round_data = {
                "round": round_num + 1,
                "current_round_generation": current_round_generation,
                "refined_context": refined_context
            }
            rounds_data.append(round_data)

            # Update assistant message prefix for next round
            # The prefix grows with each refined context
            if round_num == 0:
                assistant_message_prefix = refined_context
            else:
                assistant_message_prefix = assistant_message_prefix + refined_context

        # Store final results with all rounds
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
    """Process prompts in batches for efficiency."""
    results = []

    print("\n" + "="*80)
    print(f"BATCH MODE: Processing all prompts in batches ({args.rounds} rounds)")
    print("="*80)

    # Extract prompts and filter valid items
    valid_items = [item for item in data if "prompt" in item]
    if len(valid_items) < len(data):
        print(f"Warning: Skipped {len(data) - len(valid_items)} items missing 'prompt' field")

    original_prompts = [item["prompt"] for item in valid_items]

    # Store rounds data for all items
    all_rounds_data = [[] for _ in range(len(valid_items))]

    # Track assistant message prefixes for each item (accumulated refined contexts)
    assistant_message_prefixes = ["" for _ in range(len(valid_items))]

    # Perform multiple rounds
    for round_num in range(args.rounds):
        print(f"\n{'='*80}")
        print(f"ROUND {round_num + 1}/{args.rounds}")
        print(f"{'='*80}")

        # Stage 1: Batch generate N tokens for all prompts
        # Construct prompts based on round number
        if round_num == 0:
            # First round: just the user prompts
            generation_prompts = original_prompts
        else:
            # Subsequent rounds: user prompt + assistant prefix
            generation_prompts = [
                f"{orig}\n\nAssistant: {prefix}"
                for orig, prefix in zip(original_prompts, assistant_message_prefixes)
            ]

        print(f"  Stage 1: Generating {args.num_tokens} tokens for {len(generation_prompts)} prompts...")
        initial_outputs = llm.generate(generation_prompts, initial_sampling_params)
        current_round_generations = [output.outputs[0].text for output in initial_outputs]
        print(f"    ✓ Completed initial generation")

        # Stage 2: Create refinement prompts and batch process
        # Determine what to pass to refinement based on --accumulate flag
        if args.accumulate:
            print(f"  Stage 2: Refining ACCUMULATED contexts (prefix + current generation)...")
            if round_num == 0:
                contexts_to_refine = current_round_generations
            else:
                contexts_to_refine = [
                    prefix + current_gen
                    for prefix, current_gen in zip(assistant_message_prefixes, current_round_generations)
                ]
        else:
            print(f"  Stage 2: Refining CURRENT ROUND generations only...")
            contexts_to_refine = current_round_generations

        refinement_prompts = [
            create_refinement_prompt(orig, context)
            for orig, context in zip(original_prompts, contexts_to_refine)
        ]
        refinement_outputs = llm.generate(refinement_prompts, refinement_sampling_params)
        refined_contexts = [output.outputs[0].text for output in refinement_outputs]
        print(f"    ✓ Completed refinement")

        # Store round data
        for i, (current_gen, refined_ctx) in enumerate(zip(current_round_generations, refined_contexts)):
            round_data = {
                "round": round_num + 1,
                "current_round_generation": current_gen,
                "refined_context": refined_ctx
            }
            all_rounds_data[i].append(round_data)

        # Update assistant message prefixes for next round
        # Each prefix accumulates the refined contexts
        for i in range(len(valid_items)):
            if round_num == 0:
                assistant_message_prefixes[i] = refined_contexts[i]
            else:
                assistant_message_prefixes[i] = assistant_message_prefixes[i] + refined_contexts[i]

    # Combine final results
    for item, orig_prompt, rounds_data, full_message in zip(
        valid_items, original_prompts, all_rounds_data, assistant_message_prefixes
    ):
        result = {
            "original_prompt": orig_prompt,
            "rounds": rounds_data,
            "final_refined_context": rounds_data[-1]["refined_context"],
            "full_assistant_message": full_message,
            **{k: v for k, v in item.items() if k != "prompt"}
        }
        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate and refine contexts using vLLM"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Input JSONL file containing prompts"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen3-4B-Instruct-2507",
        help="Model name or path to use for generation (default: Qwen3-4B-Instruct-2507)"
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
        help="Output JSONL file for results (default: output_refined.jsonl)"
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
        help="Accumulate context across rounds (pass accumulated context to refinement). "
             "If not set, only the newly generated tokens are refined each round."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose output (slower, processes one at a time)"
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
        "--max_refinement_tokens",
        type=int,
        default=512,
        help="Maximum tokens for refinement output (default: 512)"
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs to use for tensor parallelism (default: 1)"
    )

    args = parser.parse_args()

    # Load input data
    print(f"Loading data from {args.input_file}...")
    data = load_jsonl(args.input_file)
    print(f"Loaded {len(data)} prompts")

    # Initialize vLLM
    print(f"Loading model {args.model}...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True
    )

    # Sampling parameters for initial generation
    initial_sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.num_tokens
    )

    # Sampling parameters for refinement
    refinement_sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
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
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(f"\n{'='*80}")
    print(f"Done! Processed {len(results)} prompts.")
    print(f"Results saved to {args.output_file}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
