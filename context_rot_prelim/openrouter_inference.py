#!/usr/bin/env python3
"""Context Rot Experiment via OpenRouter API.

Runs the same baseline + sequential multi-turn experiment as run_experiment.py,
but queries models through the OpenRouter API (OpenAI-compatible).

Features:
  - Async concurrent API calls (configurable concurrency)
  - Checkpointing: saves progress after each phase (baseline, each seed)
  - Resumes from checkpoint on restart
  - Exponential backoff on rate limits

Models:
    minimax/minimax-m2.5
    moonshotai/kimi-k2.5
    z-ai/glm-5
    deepseek/deepseek-v3.2
    google/gemini-3-flash-preview
    qwen/qwen3.5-397b-a17b

Usage:
    source ~/.bashrc   # sets OPENROUTER_API_KEY

    python -m context_rot_prelim.openrouter_inference \
        --model minimax/minimax-m2.5 \
        --dataset hmmt \
        --input_file .../hmmt_2025_combined.jsonl \
        --mode both \
        --output_file .../context_rot/minimax-m2.5_hmmt.jsonl
"""

import argparse
import asyncio
import csv as csv_mod
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inference.data_utils import strip_thinking
from inference.verify_utils import verify_batch

SCRATCH = "/scratch/dkhasha1/tli104"

PROMPT_TEMPLATE = (
    "Solve the following math problem. Show your reasoning step by step "
    "and provide your final answer in \\boxed{{}}.\n\n"
    "Problem: {problem}"
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Context Rot via OpenRouter API")
    p.add_argument("--model", required=True,
                   help="OpenRouter model ID (e.g. minimax/minimax-m2.5)")
    p.add_argument("--dataset", required=True, choices=["hmmt", "imobench"])
    p.add_argument("--input_file", default=None)
    p.add_argument("--max_problems", type=int, default=None)
    p.add_argument("--mode", required=True,
                   choices=["baseline", "sequential", "both"])
    p.add_argument("--turns_per_conversation", type=int, default=10)
    p.add_argument("--seeds", type=int, nargs="+",
                   default=[42, 123, 456, 789, 1011])
    p.add_argument("--max_tokens", type=int, default=16384)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--concurrency", type=int, default=10,
                   help="Max concurrent API requests (default: 10)")
    p.add_argument("--output_file", required=True)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading (same as run_experiment.py)
# ---------------------------------------------------------------------------

def load_problems(dataset: str, input_file: str = None,
                  max_problems: int = None) -> List[Dict]:
    problems: List[Dict] = []
    if dataset == "hmmt":
        assert input_file, "--input_file required for hmmt"
        with open(input_file) as f:
            problems = [json.loads(line) for line in f]
        for p in problems:
            if "original_problem" in p:
                p["problem_text"] = p["original_problem"]
            else:
                text = p["prompt"]
                if "Problem:" in text:
                    text = text.split("Problem:", 1)[1]
                if text.strip().endswith("Solution:"):
                    text = text.rsplit("Solution:", 1)[0]
                p["problem_text"] = text.strip()
            if "problem_id" not in p:
                p["problem_id"] = (
                    f"hmmt_{p.get('source', 'unk')}_{p['problem_idx']}")
    elif dataset == "imobench":
        csv_path = input_file or f"{SCRATCH}/imobench/answerbench.csv"
        with open(csv_path) as f:
            for row in csv_mod.DictReader(f):
                ans = row.get("Short Answer", "").strip()
                if ans:
                    problems.append({
                        "problem_id": row["Problem ID"],
                        "problem_text": row["Problem"],
                        "answer": ans,
                        "category": row.get("Category", ""),
                        "source": row.get("Source", ""),
                    })
    if max_problems:
        problems = problems[:max_problems]
    return problems


def create_conversations(problems: List[Dict], turns: int,
                         seed: int) -> List[List[Dict]]:
    rng = random.Random(seed)
    order = list(problems)
    rng.shuffle(order)
    return [order[i:i + turns]
            for i in range(0, len(order), turns)
            if i + turns <= len(order)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_boxed(text: str) -> Optional[str]:
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    start = idx + len("\\boxed{")
    depth, pos = 1, start
    while pos < len(text) and depth > 0:
        if text[pos] == '{':
            depth += 1
        elif text[pos] == '}':
            depth -= 1
        pos += 1
    return text[start:pos - 1].strip() if depth == 0 else None


def _verify_answers(items: List[Tuple[str, str]]):
    if not items:
        return []
    try:
        return verify_batch(items, timeout=10.0)
    except Exception:
        from inference.verify_utils import _verify_single
        return [_verify_single(item) for item in items]


def _make_result(mode, conv_id, turn, prob, resp, is_correct, parsed,
                 prompt_tokens, response_tokens, model, seed,
                 overflow=False):
    return {
        "mode": mode,
        "conversation_id": conv_id,
        "turn": turn,
        "problem_id": prob["problem_id"],
        "ground_truth": prob.get("answer", ""),
        "category": prob.get("category", prob.get("problem_type", "")),
        "source": prob.get("source", ""),
        "generation": resp,
        "extracted_answer": extract_boxed(strip_thinking(resp)) if resp else None,
        "parsed_answer": parsed,
        "is_correct": is_correct,
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "model": model,
        "seed": seed,
        "thinking_enabled": False,
        "context_overflow": overflow,
    }


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def checkpoint_path(output_file: str) -> str:
    return output_file + ".ckpt"


def load_checkpoint(output_file: str) -> Tuple[List[Dict], set]:
    """Load existing results and return (results, completed_phases).

    Phases are strings like "baseline", "seq_42", "seq_123", etc.
    """
    ckpt = checkpoint_path(output_file)
    results = []
    phases = set()

    if Path(ckpt).exists():
        with open(ckpt) as f:
            state = json.load(f)
        results = state.get("results", [])
        phases = set(state.get("completed_phases", []))
        print(f"  Loaded checkpoint: {len(results)} results, "
              f"phases={sorted(phases)}")
    elif Path(output_file).exists():
        # No checkpoint but output file exists — read it as partial results
        with open(output_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
        # Infer completed phases
        if any(r["mode"] == "baseline" for r in results):
            phases.add("baseline")
        for r in results:
            if r["mode"] == "sequential":
                phases.add(f"seq_{r['seed']}")
        if phases:
            print(f"  Inferred checkpoint from output: {len(results)} results, "
                  f"phases={sorted(phases)}")

    return results, phases


def save_checkpoint(output_file: str, results: List[Dict],
                    completed_phases: set):
    ckpt = checkpoint_path(output_file)
    with open(ckpt, "w") as f:
        json.dump({
            "results": results,
            "completed_phases": sorted(completed_phases),
        }, f)


def save_results(results: List[Dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")


def cleanup_checkpoint(output_file: str):
    ckpt = checkpoint_path(output_file)
    if Path(ckpt).exists():
        Path(ckpt).unlink()
        print(f"  Removed checkpoint: {ckpt}")


# ---------------------------------------------------------------------------
# API call with retries
# ---------------------------------------------------------------------------

async def call_api(client: AsyncOpenAI, model: str,
                   messages: List[Dict], max_tokens: int,
                   temperature: float,
                   semaphore: asyncio.Semaphore,
                   max_retries: int = 8) -> Dict:
    """Call OpenRouter API with exponential backoff on errors."""
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
                    print(f"    API error (attempt {attempt+1}/{max_retries}): "
                          f"{type(e).__name__}: {e}, retrying in {wait:.1f}s...")
                    await asyncio.sleep(wait)
                else:
                    print(f"    API error (final): {e}")
                    return {
                        "content": "",
                        "finish_reason": "error",
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "error": str(e),
                    }
    return {"content": "", "finish_reason": "error",
            "prompt_tokens": 0, "completion_tokens": 0}


# ---------------------------------------------------------------------------
# Baseline (parallel single-turn calls)
# ---------------------------------------------------------------------------

async def run_baseline(client: AsyncOpenAI, model: str,
                       problems: List[Dict], args) -> List[Dict]:
    print(f"  Baseline: {len(problems)} problems, "
          f"concurrency={args.concurrency}")
    semaphore = asyncio.Semaphore(args.concurrency)
    t0 = time.time()

    async def do_one(idx, prob):
        messages = [{"role": "user",
                     "content": PROMPT_TEMPLATE.format(
                         problem=prob["problem_text"])}]
        r = await call_api(client, model, messages,
                           args.max_tokens, args.temperature, semaphore)
        return idx, prob, r

    tasks = [do_one(i, p) for i, p in enumerate(problems)]
    api_results = await asyncio.gather(*tasks)
    api_results.sort(key=lambda x: x[0])

    # Batch verify
    vfy = _verify_answers([
        (p.get("answer", ""), strip_thinking(r["content"]))
        for _, p, r in api_results
    ])

    results = []
    for (idx, prob, r), (is_ok, _st, parsed) in zip(api_results, vfy):
        results.append(_make_result(
            "baseline", -1, 0, prob, r["content"], is_ok, parsed,
            r["prompt_tokens"], r["completion_tokens"],
            model, -1))

    c = sum(r["is_correct"] for r in results)
    elapsed = time.time() - t0
    print(f"  Baseline: {c}/{len(results)} = {c/len(results):.1%} "
          f"({elapsed:.0f}s)")
    return results


# ---------------------------------------------------------------------------
# Sequential multi-turn
# ---------------------------------------------------------------------------

async def run_one_conversation(client: AsyncOpenAI, model: str,
                               conv_id: int, conv_problems: List[Dict],
                               args, seed: int,
                               semaphore: asyncio.Semaphore) -> List[Dict]:
    messages: List[Dict] = []
    results: List[Dict] = []

    for turn, prob in enumerate(conv_problems):
        user_msg = {"role": "user",
                    "content": PROMPT_TEMPLATE.format(
                        problem=prob["problem_text"])}
        messages.append(user_msg)

        r = await call_api(client, model, messages,
                           args.max_tokens, args.temperature, semaphore)

        content = r["content"]
        messages.append({"role": "assistant", "content": content})

        vfy = _verify_answers(
            [(prob.get("answer", ""), strip_thinking(content))])
        is_ok, _st, parsed = vfy[0] if vfy else (False, "error", None)

        results.append(_make_result(
            "sequential", conv_id, turn, prob, content, is_ok, parsed,
            r["prompt_tokens"], r["completion_tokens"],
            model, seed))

    return results


async def run_sequential(client: AsyncOpenAI, model: str,
                         conversations: List[List[Dict]],
                         args, seed: int) -> List[Dict]:
    print(f"  Sequential seed={seed}: {len(conversations)} convs x "
          f"{args.turns_per_conversation} turns, concurrency={args.concurrency}")
    semaphore = asyncio.Semaphore(args.concurrency)
    t0 = time.time()

    tasks = [
        run_one_conversation(client, model, ci, conv, args, seed, semaphore)
        for ci, conv in enumerate(conversations)
    ]
    conv_results = await asyncio.gather(*tasks)

    all_results = []
    for r_list in conv_results:
        all_results.extend(r_list)

    # Per-turn summary
    by_turn = defaultdict(list)
    for r in all_results:
        if not r["context_overflow"]:
            by_turn[r["turn"]].append(r["is_correct"])
    for t in sorted(by_turn):
        arr = by_turn[t]
        c = sum(arr)
        ctx_avg = sum(r["prompt_tokens"] for r in all_results
                      if r["turn"] == t and not r["context_overflow"]) / max(len(arr), 1)
        print(f"    Seed {seed} Turn {t+1}: "
              f"{c}/{len(arr)} = {c/len(arr):.1%}  avg_ctx={ctx_avg:,.0f}")

    elapsed = time.time() - t0
    c = sum(r["is_correct"] for r in all_results if not r["context_overflow"])
    n = sum(1 for r in all_results if not r["context_overflow"])
    print(f"  Sequential seed={seed}: {c}/{n} = {c/n:.1%} ({elapsed:.0f}s)")
    return all_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def async_main():
    args = parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set. Run: source ~/.bashrc")
        sys.exit(1)

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    do_bl = args.mode in ("baseline", "both")
    do_seq = args.mode in ("sequential", "both")

    print(f"{'='*70}")
    print(f"Context Rot Experiment (OpenRouter API)")
    print(f"{'='*70}")
    print(f"Model       : {args.model}")
    print(f"Dataset     : {args.dataset}")
    print(f"Mode        : {args.mode}")
    if do_seq:
        print(f"Seeds       : {args.seeds}")
        print(f"Turns/conv  : {args.turns_per_conversation}")
    print(f"Max tokens  : {args.max_tokens}")
    print(f"Temperature : {args.temperature}")
    print(f"Concurrency : {args.concurrency}")

    problems = load_problems(args.dataset, args.input_file, args.max_problems)
    print(f"Problems    : {len(problems)}")

    # Load checkpoint
    all_results, completed = load_checkpoint(args.output_file)
    print(f"{'='*70}\n")

    # Baseline
    if do_bl and "baseline" not in completed:
        bl = await run_baseline(client, args.model, problems, args)
        all_results.extend(bl)
        completed.add("baseline")
        save_checkpoint(args.output_file, all_results, completed)
        save_results(all_results, args.output_file)
        print(f"  Checkpoint saved ({len(all_results)} results)\n")
    elif do_bl:
        print("  Baseline: already completed (skipping)\n")

    # Sequential
    if do_seq:
        for seed in args.seeds:
            phase = f"seq_{seed}"
            if phase in completed:
                print(f"  Sequential seed={seed}: already completed (skipping)")
                continue
            convs = create_conversations(
                problems, args.turns_per_conversation, seed)
            print(f"  Seed {seed}: {len(convs)} conversations")
            sq = await run_sequential(
                client, args.model, convs, args, seed)
            all_results.extend(sq)
            completed.add(phase)
            save_checkpoint(args.output_file, all_results, completed)
            save_results(all_results, args.output_file)
            print(f"  Checkpoint saved ({len(all_results)} results)\n")

    # Final save
    save_results(all_results, args.output_file)
    cleanup_checkpoint(args.output_file)
    print(f"\nSaved {len(all_results)} results -> {args.output_file}")

    # Summary
    bl = [r for r in all_results if r["mode"] == "baseline"]
    sq = [r for r in all_results
          if r["mode"] == "sequential" and not r.get("context_overflow")]
    if bl:
        c = sum(r["is_correct"] for r in bl)
        print(f"Baseline   : {c}/{len(bl)} = {c/len(bl):.1%}")
    if sq:
        c = sum(r["is_correct"] for r in sq)
        print(f"Sequential : {c}/{len(sq)} = {c/len(sq):.1%}")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
