#!/usr/bin/env python3
"""Context Rot Experiment: Multi-turn math problem solving.

Measures whether model accuracy degrades as conversation context accumulates
across turns. Uses random shuffles (seeds) so each problem appears at
different positions across runs; a mixed-effects model then separates position
effects from problem difficulty.

Efficiency:
  - Baseline: all problems batched in a single llm.generate() call
  - Sequential: at each turn, all active conversations batched together
  - Data parallel: conversations sharded across workers, each with own vLLM
  - Verification: batch-parallel via verify_batch

Usage:
    # Auto-detects GPUs, sets TP from model size, DP = remaining GPUs
    python -m context_rot_prelim.run_experiment \
        --model Qwen/Qwen3-4B-Instruct-2507 \
        --dataset hmmt \
        --input_file .../hmmt_2025_combined.jsonl \
        --mode both \
        --output_file .../qwen3_4b_instruct_hmmt.jsonl

    # Override: force TP=2, DP=1 (single-instance, 2 GPUs for model)
    python -m context_rot_prelim.run_experiment \
        --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
        --dataset imobench --max_problems 200 \
        --mode both \
        --tensor_parallel_size 2 --num_dp_workers 1 \
        --output_file .../qwen3_30b_instruct_imobench.jsonl
"""

import argparse
import csv
import json
import multiprocessing as mp
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inference.data_utils import strip_thinking
from inference.verify_utils import verify_batch

SCRATCH = "/scratch/dkhasha1/tli104"
DEFAULT_CACHE = f"{SCRATCH}/hf_model_cache"

PROMPT_TEMPLATE = (
    "Solve the following math problem. Show your reasoning step by step "
    "and provide your final answer in \\boxed{{}}.\n\n"
    "Problem: {problem}"
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Context Rot Experiment")
    # Data
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True, choices=["hmmt", "imobench"])
    p.add_argument("--input_file", default=None)
    p.add_argument("--max_problems", type=int, default=None,
                   help="Cap number of problems (useful for large datasets)")
    # Experiment
    p.add_argument("--mode", required=True,
                   choices=["baseline", "sequential", "both"])
    p.add_argument("--turns_per_conversation", type=int, default=10)
    p.add_argument("--seeds", type=int, nargs="+",
                   default=[42, 123, 456, 789, 1011],
                   help="Random seeds for conversation orderings")
    p.add_argument("--max_tokens", type=int, default=16384)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--enable_thinking", action="store_true",
                   help="Force thinking mode (for Instruct models used as thinking)")
    # Parallelism (auto-detected by default)
    p.add_argument("--tensor_parallel_size", type=int, default=None,
                   help="TP size per worker (default: auto from model size)")
    p.add_argument("--num_dp_workers", type=int, default=None,
                   help="DP workers (default: num_gpus / tp)")
    p.add_argument("--max_model_len", type=int, default=None)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    p.add_argument("--cache_dir", default=DEFAULT_CACHE)
    # Output
    p.add_argument("--output_file", required=True)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_problems(dataset: str, input_file: str = None,
                  max_problems: int = None) -> List[Dict]:
    """Load and normalise to {problem_id, problem_text, answer, ...}."""
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
        problems = []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
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
    """Shuffle and partition into conversations of exactly *turns* problems."""
    rng = random.Random(seed)
    order = list(problems)
    rng.shuffle(order)
    return [order[i:i + turns]
            for i in range(0, len(order), turns)
            if i + turns <= len(order)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_thinking_model(name: str) -> bool:
    return "Thinking" in name or "thinking" in name


def wants_thinking(args) -> bool:
    """Check if thinking mode should be enabled (from model name or flag)."""
    return args.enable_thinking or is_thinking_model(args.model)


def auto_tp(model_name: str) -> int:
    """Infer minimum TP size from model name (parameter count)."""
    name = model_name.lower()
    # MoE models with large total params need ≥2 GPUs for weights
    if "30b" in name or "32b" in name or "70b" in name or "72b" in name:
        return 2
    if "235b" in name:
        return 8
    # Small models fit on 1 GPU
    return 1


def auto_parallel(model_name: str, num_gpus: int,
                  tp_override: int = None,
                  dp_override: int = None):
    """Return (tp, dp_workers, use_dp) based on model size + available GPUs."""
    tp = tp_override if tp_override is not None else auto_tp(model_name)
    tp = min(tp, num_gpus) or 1
    dp = dp_override if dp_override is not None else (num_gpus // tp)
    dp = max(dp, 1)
    use_dp = dp > 1
    return tp, dp, use_dp


def apply_template(tokenizer, messages: List[Dict],
                   enable_thinking: bool = False) -> str:
    """Apply chat template; gracefully falls back if enable_thinking unsupported."""
    try:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
            enable_thinking=enable_thinking)
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)


def extract_boxed(text: str) -> Optional[str]:
    """Extract last \\boxed{...}, handling nested braces."""
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
    """Batch-verify (gold, generated_text) pairs. Returns list of
    (is_correct, status, parsed_answer) tuples."""
    if not items:
        return []
    try:
        return verify_batch(items, timeout=10.0)
    except Exception:
        # Fallback: sequential (avoids nested-multiprocessing issues)
        from inference.verify_utils import _verify_single
        return [_verify_single(item) for item in items]


def _make_result(mode, conv_id, turn, prob, resp, is_correct, parsed,
                 prompt_tokens, response_tokens, model, seed,
                 enable_thinking, overflow=False):
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
        "thinking_enabled": enable_thinking,
        "context_overflow": overflow,
    }


# ---------------------------------------------------------------------------
# Core generation (single vLLM instance)
# ---------------------------------------------------------------------------

def run_baseline_core(llm, tokenizer, problems: List[Dict],
                      args) -> List[Dict]:
    """Baseline: batch all problems in one llm.generate() call."""
    enable_thinking = wants_thinking(args)
    sp = SamplingParams(max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_p=0.95 if args.temperature > 0 else 1.0)

    prompts = [
        apply_template(
            tokenizer,
            [{"role": "user",
              "content": PROMPT_TEMPLATE.format(problem=p["problem_text"])}],
            enable_thinking)
        for p in problems
    ]

    print(f"  Baseline: generating {len(prompts)} responses...")
    outputs = llm.generate(prompts, sp)

    # Batch verify
    vfy = _verify_answers([
        (p.get("answer", ""), strip_thinking(o.outputs[0].text))
        for p, o in zip(problems, outputs)
    ])

    results = [
        _make_result("baseline", -1, 0, prob, out.outputs[0].text,
                     vc[0], vc[2],
                     len(out.prompt_token_ids),
                     len(out.outputs[0].token_ids),
                     args.model, -1, enable_thinking)
        for prob, out, vc in zip(problems, outputs, vfy)
    ]

    c = sum(r["is_correct"] for r in results)
    print(f"  Baseline: {c}/{len(results)} = {c / len(results):.1%}")
    return results


def run_sequential_core(llm, tokenizer, conversations: List[List[Dict]],
                        args, seed: int) -> List[Dict]:
    """Sequential multi-turn: batch across conversations at each turn."""
    enable_thinking = wants_thinking(args)
    sp = SamplingParams(max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_p=0.95 if args.temperature > 0 else 1.0)

    num_convs = len(conversations)
    max_turns = max(len(c) for c in conversations)
    conv_msgs: List[List[Dict]] = [[] for _ in range(num_convs)]
    active = list(range(num_convs))
    results: List[Dict] = []

    # Resolve max context length
    max_ctx = args.max_model_len
    if max_ctx is None:
        try:
            max_ctx = llm.llm_engine.model_config.max_model_len
        except AttributeError:
            max_ctx = 131072

    for turn in range(max_turns):
        prompts: List[str] = []
        batch_ci: List[int] = []

        for ci in active[:]:
            if turn >= len(conversations[ci]):
                active.remove(ci)
                continue

            prob = conversations[ci][turn]
            conv_msgs[ci].append({
                "role": "user",
                "content": PROMPT_TEMPLATE.format(
                    problem=prob["problem_text"]),
            })
            prompt = apply_template(tokenizer, conv_msgs[ci], enable_thinking)
            plen = len(tokenizer.encode(prompt))

            if plen + 256 > max_ctx:
                print(f"    Overflow: conv {ci} turn {turn} "
                      f"({plen} tok > {max_ctx})")
                conv_msgs[ci].pop()  # undo the user message we just appended
                results.append(_make_result(
                    "sequential", ci, turn, prob, "", False, None,
                    plen, 0, args.model, seed, enable_thinking, overflow=True))
                active.remove(ci)
                continue

            prompts.append(prompt)
            batch_ci.append(ci)

        if not prompts:
            break

        # --- batched generation for this turn ---
        outputs = llm.generate(prompts, sp)

        # Update histories
        for ci, out in zip(batch_ci, outputs):
            conv_msgs[ci].append(
                {"role": "assistant", "content": out.outputs[0].text})

        # --- batched verification ---
        vfy = _verify_answers([
            (conversations[ci][turn].get("answer", ""),
             strip_thinking(out.outputs[0].text))
            for ci, out in zip(batch_ci, outputs)
        ])

        for ci, out, (is_ok, _st, parsed) in zip(batch_ci, outputs, vfy):
            prob = conversations[ci][turn]
            results.append(_make_result(
                "sequential", ci, turn, prob, out.outputs[0].text,
                is_ok, parsed,
                len(out.prompt_token_ids), len(out.outputs[0].token_ids),
                args.model, seed, enable_thinking))

        # Turn summary
        t_res = [r for r in results
                 if r["turn"] == turn and r["seed"] == seed
                 and not r["context_overflow"]]
        if t_res:
            c = sum(r["is_correct"] for r in t_res)
            ctx = sum(r["prompt_tokens"] for r in t_res) / len(t_res)
            print(f"    Seed {seed} Turn {turn + 1}/{max_turns}: "
                  f"{c}/{len(t_res)} = {c / len(t_res):.1%}  "
                  f"avg_ctx={ctx:,.0f}")

    return results


# ---------------------------------------------------------------------------
# Data-parallel worker
# ---------------------------------------------------------------------------

def _dp_worker(worker_id: int, baseline_shard: List[Dict],
               conv_by_seed: Dict[int, List[List[Dict]]],
               args, result_queue: mp.Queue):
    """Spawned process: load model on assigned GPUs, run all work."""
    tp = args.tensor_parallel_size
    if tp > 1:
        g0 = worker_id * tp
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(g0 + i) for i in range(tp))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_id)

    print(f"[W{worker_id}] CUDA_VISIBLE_DEVICES="
          f"{os.environ['CUDA_VISIBLE_DEVICES']}  "
          f"baseline={len(baseline_shard)}  "
          f"seq_seeds={list(conv_by_seed.keys())}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, cache_dir=args.cache_dir, trust_remote_code=True)

    llm_kw = dict(model=args.model, tensor_parallel_size=tp,
                  gpu_memory_utilization=args.gpu_memory_utilization,
                  download_dir=args.cache_dir, trust_remote_code=True)
    if args.max_model_len is not None:
        llm_kw["max_model_len"] = args.max_model_len
    llm = LLM(**llm_kw)

    results: List[Dict] = []
    if baseline_shard:
        results.extend(run_baseline_core(llm, tokenizer, baseline_shard, args))
    for seed in sorted(conv_by_seed):
        convs = conv_by_seed[seed]
        if convs:
            results.extend(
                run_sequential_core(llm, tokenizer, convs, args, seed))

    print(f"[W{worker_id}] Done — {len(results)} results")
    result_queue.put((worker_id, results))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _save(results: List[Dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")


def main():
    args = parse_args()
    thinking = wants_thinking(args)
    do_bl = args.mode in ("baseline", "both")
    do_seq = args.mode in ("sequential", "both")

    print(f"{'=' * 70}")
    print(f"Context Rot Experiment")
    print(f"{'=' * 70}")
    print(f"Model       : {args.model} "
          f"({'thinking' if thinking else 'non-thinking'})")
    print(f"Dataset     : {args.dataset}")
    print(f"Mode        : {args.mode}")
    if do_seq:
        print(f"Seeds       : {args.seeds}")
        print(f"Turns/conv  : {args.turns_per_conversation}")
    print(f"Max tokens  : {args.max_tokens}")
    print(f"Temperature : {args.temperature}")

    # Load data
    problems = load_problems(args.dataset, args.input_file, args.max_problems)
    print(f"Problems    : {len(problems)}")

    # Build conversation orderings up-front (cheap, CPU-only)
    conv_by_seed: Dict[int, List[List[Dict]]] = {}
    if do_seq:
        for seed in args.seeds:
            convs = create_conversations(
                problems, args.turns_per_conversation, seed)
            conv_by_seed[seed] = convs
            print(f"  Seed {seed}: {len(convs)} convs × "
                  f"{args.turns_per_conversation} turns")

    num_gpus = torch.cuda.device_count()
    tp, dp, use_dp = auto_parallel(
        args.model, num_gpus, args.tensor_parallel_size, args.num_dp_workers)
    print(f"GPUs        : {num_gpus}  →  TP={tp}, DP={dp}"
          f"{'  (data-parallel)' if use_dp else ''}")
    print(f"{'=' * 70}\n")

    # Store resolved tp on args so workers see it
    args.tensor_parallel_size = tp

    # ------------------------------------------------------------------
    if use_dp:
        # ---- DATA-PARALLEL MODE ----
        num_workers = dp
        print(f"Data parallel: {num_workers} workers × TP={tp}\n")

        # Shard baseline problems round-robin
        bl_shards: List[List[Dict]] = [[] for _ in range(num_workers)]
        if do_bl:
            for i, p in enumerate(problems):
                bl_shards[i % num_workers].append(p)

        # Shard conversations (whole conversations, round-robin)
        c_shards: Dict[int, List[List[List[Dict]]]] = {
            s: [[] for _ in range(num_workers)] for s in conv_by_seed}
        for seed, convs in conv_by_seed.items():
            for i, c in enumerate(convs):
                c_shards[seed][i % num_workers].append(c)

        mp.set_start_method("spawn", force=True)
        q: mp.Queue = mp.Queue()
        procs = []
        for w in range(num_workers):
            w_convs = {s: c_shards[s][w] for s in conv_by_seed}
            p = mp.Process(target=_dp_worker,
                           args=(w, bl_shards[w], w_convs, args, q))
            p.start()
            procs.append(p)

        all_results: List[Dict] = []
        for _ in range(num_workers):
            wid, res = q.get()
            all_results.extend(res)
            print(f"[Main] Worker {wid}: {len(res)} results received")
        for p in procs:
            p.join()

    else:
        # ---- SINGLE-INSTANCE MODE (TP-only) ----
        print(f"Single instance: TP={tp}\n")

        tokenizer = AutoTokenizer.from_pretrained(
            args.model, cache_dir=args.cache_dir, trust_remote_code=True)

        llm_kw = dict(model=args.model, tensor_parallel_size=tp,
                      gpu_memory_utilization=args.gpu_memory_utilization,
                      download_dir=args.cache_dir, trust_remote_code=True)
        if args.max_model_len is not None:
            llm_kw["max_model_len"] = args.max_model_len
        llm = LLM(**llm_kw)

        all_results = []
        if do_bl:
            all_results.extend(
                run_baseline_core(llm, tokenizer, problems, args))
        for seed in sorted(conv_by_seed):
            all_results.extend(
                run_sequential_core(
                    llm, tokenizer, conv_by_seed[seed], args, seed))

    # ------------------------------------------------------------------
    # Save
    _save(all_results, args.output_file)
    print(f"\nSaved {len(all_results)} results → {args.output_file}")

    # Summary
    bl = [r for r in all_results if r["mode"] == "baseline"]
    sq = [r for r in all_results
          if r["mode"] == "sequential" and not r.get("context_overflow")]
    if bl:
        c = sum(r["is_correct"] for r in bl)
        print(f"Baseline   : {c}/{len(bl)} = {c / len(bl):.1%}")
    if sq:
        c = sum(r["is_correct"] for r in sq)
        print(f"Sequential : {c}/{len(sq)} = {c / len(sq):.1%}")


if __name__ == "__main__":
    main()
