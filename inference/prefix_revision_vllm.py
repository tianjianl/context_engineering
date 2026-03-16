#!/usr/bin/env python3
"""Prefix Revision Experiment — 2-turn critique-then-revise via vLLM.

Turn 1: Model reviews its own solution.
Turn 2: Model writes a corrected solution based on the review.
No hint is given that the solution is incorrect.

Usage:
    python -m inference.prefix_revision_vllm \
        --model Qwen/Qwen3-4B-Instruct-2507 \
        --baseline_file .../baseline_qwen3_4b_imobench.jsonl \
        --output_file .../revision_qwen3_4b_imobench.jsonl \
        --num_tokens 16384 --temperature 0.9
"""

import argparse
from typing import Dict, List

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

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


def build_multiturn_prompt(tokenizer, messages: List[Dict],
                           add_generation_prompt: bool = True) -> str:
    return tokenizer.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=add_generation_prompt
    )


def main():
    parser = argparse.ArgumentParser(
        description="2-turn prefix revision experiment for local vLLM models"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--baseline_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--num_tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--tensor_parallel_size", type=int, default=None)
    args = parser.parse_args()

    # Load and filter baseline
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

    # Setup vLLM
    num_gpus = torch.cuda.device_count()
    tp = args.tensor_parallel_size or (num_gpus if num_gpus > 0 else 1)
    print(f"Using tensor_parallel_size={tp}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm_kwargs = dict(model=args.model, tensor_parallel_size=tp, trust_remote_code=True)
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.num_tokens,
    )

    # --- Turn 1: Critique ---
    print(f"Turn 1: Generating critiques for {len(incorrect)} problems...")
    critique_prompts = []
    for item in incorrect:
        visible_gen = strip_thinking(item["generation"]) if item["generation"] else item["generation"]
        messages = [
            {"role": "user", "content": PROMPT_TEMPLATE.format(problem=item["prompt"])},
            {"role": "assistant", "content": visible_gen},
            {"role": "user", "content": CRITIQUE_PROMPT},
        ]
        critique_prompts.append(build_multiturn_prompt(tokenizer, messages))

    critique_outputs = llm.generate(critique_prompts, sampling_params)
    critiques = [o.outputs[0].text for o in critique_outputs]
    print("Turn 1 complete.")

    # --- Turn 2: Revise ---
    print(f"Turn 2: Generating revisions for {len(incorrect)} problems...")
    revision_prompts = []
    for item, critique in zip(incorrect, critiques):
        visible_gen = strip_thinking(item["generation"]) if item["generation"] else item["generation"]
        messages = [
            {"role": "user", "content": PROMPT_TEMPLATE.format(problem=item["prompt"])},
            {"role": "assistant", "content": visible_gen},
            {"role": "user", "content": CRITIQUE_PROMPT},
            {"role": "assistant", "content": critique},
            {"role": "user", "content": REVISE_PROMPT},
        ]
        revision_prompts.append(build_multiturn_prompt(tokenizer, messages))

    revision_outputs = llm.generate(revision_prompts, sampling_params)
    revisions = [o.outputs[0].text for o in revision_outputs]
    print("Turn 2 complete.")

    # Verify revisions
    print("Verifying revisions...")
    verify_items = []
    for item, revision in zip(incorrect, revisions):
        gold = item.get("answer", "")
        text = strip_thinking(revision) if revision else ""
        verify_items.append((gold, text))

    vfy = verify_batch(verify_items)

    # Build results
    results = []
    for item, critique, revision, (is_correct, status, _) in zip(
            incorrect, critiques, revisions, vfy):
        results.append({
            "problem_id": item.get("problem_id", ""),
            "prompt": item["prompt"],
            "answer": item.get("answer", ""),
            "incorrect_prefix": item["generation"],
            "critique": critique,
            "revision": revision,
            "recovered": bool(is_correct),
            "model": args.model,
            **{k: item[k] for k in ["category", "subcategory", "source"] if k in item},
        })

    save_jsonl(results, args.output_file)

    num_recovered = sum(1 for r in results if r["recovered"])
    print(f"\n{'='*60}")
    print(f"Prefix Revision Results (2-turn)")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Baseline total: {len(baseline_data)}")
    print(f"Baseline incorrect: {len(incorrect)} ({len(incorrect)/len(baseline_data):.1%})")
    print(f"Recovered: {num_recovered}/{len(incorrect)} ({num_recovered/len(incorrect):.1%})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
