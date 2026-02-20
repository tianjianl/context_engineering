#!/usr/bin/env python3
"""
Reconstruct trajectories from Gemini annotations.

For each annotated incorrect sample, truncates the generation at the
Gemini-identified position, runs the refiner to create a summary, then
continues generation with the summary context. This creates better
tool-calling examples from previously-incorrect trajectories.

Output format matches tool_refinement.py output so build_sft_data.py can
process it directly.

Usage (GPU, data-parallel):
    python training/reconstruct_trajectories.py \
        --annotation_files outputs/.../*_annotated.jsonl \
        --trajectory_files outputs/.../tool_refinement_*.jsonl \
        --output /scratch/dkhasha1/tli104/outputs/.../reconstructed_trajectories.jsonl \
        --model Qwen/Qwen3-4B-Instruct-2507 \
        --max_rounds 5
"""

import argparse
import difflib
import json
import os
import signal
import sys
import multiprocessing as mp
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class _ItemTimeout(Exception):
    pass


# Categories worth reconstructing (skip called_too_early — requires merging rounds)
RECONSTRUCT_CATEGORIES = {"never_called", "called_too_late", "should_call_again", "wrong_moment"}


def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def find_position_in_text(text: str, position_quote: str) -> Optional[int]:
    """Find the approximate character offset of a position quote in text.

    Strategy:
    1. Try exact substring match (fast)
    2. Try case-insensitive match
    3. Try matching the longest distinctive phrase (3+ word chunks)
    4. Fall back to keyword-based search with scoring

    Returns the character offset where the match ends (truncation point).
    """
    if not position_quote or not text:
        return None

    quote = position_quote.strip()
    if not quote:
        return None

    # 1. Exact substring match
    idx = text.find(quote)
    if idx >= 0:
        return idx + len(quote)

    # 2. Case-insensitive match
    text_lower = text.lower()
    quote_lower = quote.lower()
    idx = text_lower.find(quote_lower)
    if idx >= 0:
        return idx + len(quote)

    # 3. Try progressively shorter substrings from the quote
    words = quote.split()
    if len(words) >= 3:
        for chunk_size in range(len(words), max(2, len(words) // 2) - 1, -1):
            for start_word in range(len(words) - chunk_size + 1):
                chunk = " ".join(words[start_word:start_word + chunk_size])
                idx = text_lower.find(chunk.lower())
                if idx >= 0:
                    return idx + len(chunk)

    # 4. Keyword scoring
    common = {"that", "this", "with", "from", "then", "have", "been", "will",
              "each", "more", "also", "when", "which", "their", "there"}
    keywords = [w.strip(".,;:!?()[]{}$\\") for w in words
                if len(w.strip(".,;:!?()[]{}$\\")) >= 4
                and w.strip(".,;:!?()[]{}$\\").lower() not in common]

    if not keywords:
        return None

    window = len(quote) * 2
    best_score = 0
    best_pos = None

    for kw in keywords:
        kw_lower = kw.lower()
        start = 0
        while True:
            idx = text_lower.find(kw_lower, start)
            if idx < 0:
                break
            region_start = max(0, idx - window // 2)
            region_end = min(len(text), idx + window // 2)
            region = text_lower[region_start:region_end]
            score = sum(1 for k in keywords if k.lower() in region)
            if score > best_score:
                best_score = score
                best_pos = idx + len(kw)
            start = idx + 1

    if best_score >= max(2, len(keywords) // 3):
        return best_pos

    return None


@dataclass
class ReconstructionTask:
    """A single trajectory to reconstruct."""
    # From annotation file
    line_idx: int
    sample_idx: int
    problem: str  # Full problem text from trajectory file
    answer: str
    annotation_category: str
    annotation_position: str
    annotation_round: int
    annotation_reason: str

    # From trajectory file
    rounds: List[Dict] = field(default_factory=list)
    original_prompt: str = ""
    # Extra metadata from trajectory
    problem_id: str = ""
    category: str = ""
    subcategory: str = ""
    source: str = ""

    # Computed: truncation point
    truncate_round: int = -1
    truncate_offset: int = -1


def build_reconstruction_tasks(
    annotation_files: List[str],
    trajectory_files: List[str],
) -> List[ReconstructionTask]:
    """Match annotations to trajectories and build reconstruction tasks."""
    # Load all trajectory files into a map: (file_idx, line_idx) -> item
    trajectory_maps = []
    for traj_file in trajectory_files:
        data = load_jsonl(traj_file)
        trajectory_maps.append(data)
        print(f"  Loaded {len(data)} problems from {traj_file}")

    tasks = []
    skipped = {"bad_category": 0, "no_annotation": 0, "no_match": 0,
               "no_trajectory": 0, "no_position": 0}

    for ann_file_idx, ann_file in enumerate(annotation_files):
        print(f"\nProcessing annotations: {ann_file}")
        annotations = load_jsonl(ann_file)

        # Match annotation file to trajectory file by index
        if ann_file_idx >= len(trajectory_maps):
            print(f"  Warning: no matching trajectory file for annotation {ann_file_idx}")
            continue
        traj_data = trajectory_maps[ann_file_idx]

        for ann in annotations:
            ann_result = ann.get("annotation")
            if ann_result is None:
                skipped["no_annotation"] += 1
                continue

            line_idx = ann.get("line_idx", -1)
            sample_idx = ann.get("sample_idx", -1)
            answer = ann.get("answer", "")

            # Get trajectory data
            if line_idx < 0 or line_idx >= len(traj_data):
                skipped["no_trajectory"] += 1
                continue

            traj_item = traj_data[line_idx]
            samples = traj_item.get("samples", [])
            if sample_idx < 0 or sample_idx >= len(samples):
                skipped["no_trajectory"] += 1
                continue

            sample = samples[sample_idx]
            rounds = sample.get("rounds", [])
            original_prompt = traj_item.get("original_prompt", "")

            # Process each annotation within this sample
            for sub_ann in ann_result.get("annotations", []):
                cat = sub_ann.get("category", "")
                if cat not in RECONSTRUCT_CATEGORIES:
                    skipped["bad_category"] += 1
                    continue

                position_quote = sub_ann.get("position", "")
                try:
                    ann_round = int(sub_ann.get("round", "0"))
                except (ValueError, TypeError):
                    ann_round = 0

                # Find position in the specified round's generation
                if ann_round < 0 or ann_round >= len(rounds):
                    skipped["no_position"] += 1
                    continue

                gen_text = rounds[ann_round].get("current_round_generation", "")
                offset = find_position_in_text(gen_text, position_quote)
                if offset is None:
                    skipped["no_position"] += 1
                    continue

                task = ReconstructionTask(
                    line_idx=line_idx,
                    sample_idx=sample_idx,
                    problem=original_prompt,
                    answer=answer,
                    annotation_category=cat,
                    annotation_position=position_quote,
                    annotation_round=ann_round,
                    annotation_reason=sub_ann.get("reason", ""),
                    rounds=rounds,
                    original_prompt=original_prompt,
                    problem_id=traj_item.get("problem_id", ""),
                    category=traj_item.get("category", ""),
                    subcategory=traj_item.get("subcategory", ""),
                    source=traj_item.get("source", ""),
                    truncate_round=ann_round,
                    truncate_offset=offset,
                )
                tasks.append(task)

    print(f"\nSkip stats: {skipped}")
    return tasks


def deduplicate_tasks(tasks: List[ReconstructionTask]) -> List[ReconstructionTask]:
    """Deduplicate: keep only the first (earliest) annotation per (line_idx, sample_idx, round)."""
    seen = set()
    deduped = []
    for task in tasks:
        key = (task.line_idx, task.sample_idx, task.truncate_round)
        if key not in seen:
            seen.add(key)
            deduped.append(task)
    return deduped


def worker_process(gpu_id: int, task_shard: List[ReconstructionTask],
                   args, result_queue: mp.Queue, intermediate_path: str):
    """Worker process that reconstructs trajectories on a single GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    def _timeout_handler(signum, frame):
        raise _ItemTimeout()
    signal.signal(signal.SIGALRM, _timeout_handler)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    # Lazy imports for inference helpers
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from inference.tool_refinement import (
        LLM_REFINE_TOOL,
        DEFAULT_SYSTEM_PROMPT,
        LLM_REFINE_USER_HINT,
        build_conversation_messages,
        render_prompt,
        render_summarization_prompt,
    )

    print(f"[GPU {gpu_id}] Starting reconstruction worker with {len(task_shard)} tasks")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    tool_call_token_id = tokenizer.convert_tokens_to_ids("<tool_call>")
    tools = [LLM_REFINE_TOOL]
    system_prompt = DEFAULT_SYSTEM_PROMPT

    # Sampling params
    gen_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.num_tokens,
        n=1,
        stop_token_ids=[tool_call_token_id],
        stop=["<tool_call>"],
    )

    refine_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_refinement_tokens,
    )

    continuation_instructions = (
        "\n\nContinue solving the problem, improving upon the "
        "summary above. You may verify previous conclusions, try a "
        "different approach, or build on the progress so far.\n"
        "Return your final answer in \\boxed{}."
    )

    num_completed = 0
    num_timed_out = 0

    with open(intermediate_path, 'w', encoding='utf-8') as int_f:
        for task_idx, task in enumerate(task_shard):
            if task_idx % 50 == 0:
                print(f"[GPU {gpu_id}] Processing task {task_idx}/{len(task_shard)} "
                      f"(completed={num_completed}, timed_out={num_timed_out})")

            signal.alarm(args.item_timeout)
            try:
                # Step 1: Build conversation up to truncation point
                # Include all rounds before the annotated round as-is
                conversation_turns = []
                existing_summary = ""

                for r_idx in range(task.truncate_round):
                    r = task.rounds[r_idx]
                    gen = r.get("current_round_generation", "")
                    called_tool = r.get("called_tool", False)
                    refined = r.get("refined_context", "") or ""

                    if called_tool:
                        assistant_content = (
                            gen + '\n<tool_call>\n{"name": "llm_refine", "arguments": {}}\n</tool_call>'
                        )
                        conversation_turns.append(
                            {"role": "assistant", "content": assistant_content}
                        )
                        conversation_turns.append(
                            {"role": "user", "content": f"<tool_response>\n{refined}\n</tool_response>"
                             + continuation_instructions}
                        )
                        existing_summary = refined
                    else:
                        conversation_turns.append(
                            {"role": "assistant", "content": gen}
                        )

                # Step 2: Truncate the annotated round's generation
                truncated_gen = task.rounds[task.truncate_round]["current_round_generation"][:task.truncate_offset]

                # Step 3: Run refiner on the truncated text
                refine_prompt = render_summarization_prompt(
                    tokenizer, task.original_prompt,
                    existing_summary, truncated_gen
                )
                refine_outputs = llm.generate([refine_prompt], refine_params)
                summary = refine_outputs[0].outputs[0].text

                # Build the reconstructed trajectory rounds
                reconstructed_rounds = []

                # Add pre-truncation rounds as-is
                for r_idx in range(task.truncate_round):
                    reconstructed_rounds.append(dict(task.rounds[r_idx]))

                # Add the truncated round with tool call
                reconstructed_rounds.append({
                    "round": task.truncate_round + 1,
                    "current_round_generation": truncated_gen,
                    "refined_context": summary,
                    "called_tool": True,
                    "finish_reason": "tool_call",
                })

                # Step 4: Continue generation from summary
                # Build conversation with the truncated round + tool call
                conversation_turns.append(
                    {"role": "assistant", "content": truncated_gen
                     + '\n<tool_call>\n{"name": "llm_refine", "arguments": {}}\n</tool_call>'}
                )
                conversation_turns.append(
                    {"role": "user", "content": f"<tool_response>\n{summary}\n</tool_response>"
                     + continuation_instructions}
                )

                last_summary = summary
                num_tool_calls = 1
                done_reason = ""

                # Continue for up to max_rounds more rounds
                for cont_round in range(args.max_rounds):
                    messages = build_conversation_messages(
                        task.original_prompt, system_prompt,
                        conversation_turns
                    )
                    rendered = render_prompt(tokenizer, messages, tools)
                    cont_outputs = llm.generate([rendered], gen_params)
                    comp = cont_outputs[0].outputs[0]
                    raw_text = comp.text

                    stopped_at_tool = (comp.stop_reason == tool_call_token_id
                                       or comp.stop_reason == "<tool_call>")

                    round_data = {
                        "round": len(reconstructed_rounds) + 1,
                        "current_round_generation": raw_text,
                        "refined_context": None,
                        "called_tool": stopped_at_tool,
                        "finish_reason": "tool_call" if stopped_at_tool else comp.finish_reason,
                    }

                    if stopped_at_tool:
                        # Refine again
                        ref_prompt = render_summarization_prompt(
                            tokenizer, task.original_prompt,
                            last_summary, raw_text
                        )
                        ref_outputs = llm.generate([ref_prompt], refine_params)
                        new_summary = ref_outputs[0].outputs[0].text
                        round_data["refined_context"] = new_summary

                        conversation_turns.append(
                            {"role": "assistant", "content": raw_text
                             + '\n<tool_call>\n{"name": "llm_refine", "arguments": {}}\n</tool_call>'}
                        )
                        conversation_turns.append(
                            {"role": "user", "content": f"<tool_response>\n{new_summary}\n</tool_response>"
                             + continuation_instructions}
                        )
                        last_summary = new_summary
                        num_tool_calls += 1
                    else:
                        done_reason = "completed" if comp.finish_reason == "stop" else "max_tokens"

                    reconstructed_rounds.append(round_data)

                    if not stopped_at_tool:
                        break
                else:
                    done_reason = "max_rounds"

                # Cancel alarm after successful completion
                signal.alarm(0)

                # Build the full assistant message (summary + last generation)
                last_gen = reconstructed_rounds[-1]["current_round_generation"] if reconstructed_rounds else ""
                full_message = (last_summary + last_gen) if last_summary else last_gen

                # Build output in tool_refinement.py format
                result = {
                    "original_prompt": task.original_prompt,
                    "num_samples": 1,
                    "samples": [{
                        "sample_idx": 0,
                        "rounds": reconstructed_rounds,
                        "final_refined_context": last_summary,
                        "full_assistant_message": full_message,
                        "num_tool_calls": num_tool_calls,
                        "done_reason": done_reason,
                    }],
                    # Metadata
                    "problem_id": task.problem_id,
                    "answer": task.answer,
                    "category": task.category,
                    "subcategory": task.subcategory,
                    "source": task.source,
                    # Reconstruction metadata
                    "reconstruction_meta": {
                        "source_line_idx": task.line_idx,
                        "source_sample_idx": task.sample_idx,
                        "annotation_category": task.annotation_category,
                        "annotation_round": task.annotation_round,
                        "truncate_offset": task.truncate_offset,
                    },
                }
                # Save immediately to intermediate file
                int_f.write(json.dumps(result, ensure_ascii=False) + '\n')
                int_f.flush()
                num_completed += 1

            except _ItemTimeout:
                signal.alarm(0)
                print(f"[GPU {gpu_id}] Task {task_idx} timed out after {args.item_timeout}s, skipping")
                num_timed_out += 1
            except Exception as e:
                signal.alarm(0)
                print(f"[GPU {gpu_id}] Task {task_idx} error: {e}, skipping")

    print(f"[GPU {gpu_id}] Done: {num_completed} completed, {num_timed_out} timed out, "
          f"{len(task_shard) - num_completed - num_timed_out} other errors")
    result_queue.put((gpu_id, num_completed))


def load_truncated_tasks(input_path: str) -> List[ReconstructionTask]:
    """Load pre-built truncated tasks from JSONL (output of extract_truncated.py)."""
    tasks = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            task = ReconstructionTask(
                line_idx=record["source_line_idx"],
                sample_idx=record["source_sample_idx"],
                problem=record["original_prompt"],
                answer=record["answer"],
                annotation_category=record["annotation_category"],
                annotation_position=record.get("annotation_position", ""),
                annotation_round=record["annotation_round"],
                annotation_reason=record.get("annotation_reason", ""),
                rounds=record["pre_rounds"] + [{
                    "round": record["truncate_round"] + 1,
                    "current_round_generation": record["truncated_generation"],
                    "refined_context": None,
                    "called_tool": False,
                    "finish_reason": "truncated",
                }],
                original_prompt=record["original_prompt"],
                problem_id=record.get("problem_id", ""),
                category=record.get("category", ""),
                subcategory=record.get("subcategory", ""),
                source=record.get("source", ""),
                truncate_round=record["truncate_round"],
                truncate_offset=len(record["truncated_generation"]),
            )
            tasks.append(task)
    return tasks


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct trajectories from Gemini annotations"
    )
    # Option A: build tasks from annotation + trajectory files
    parser.add_argument(
        "--annotation_files", type=str, nargs="+", default=None,
        help="Gemini annotation JSONL files"
    )
    parser.add_argument(
        "--trajectory_files", type=str, nargs="+", default=None,
        help="Original tool refinement JSONL files (same order as annotation_files)"
    )
    # Option B: load pre-built truncated tasks
    parser.add_argument(
        "--truncated_input", type=str, default=None,
        help="Pre-built truncated tasks JSONL (from extract_truncated.py)"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output JSONL file"
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--num_tokens", type=int, default=8192,
                        help="Max tokens per generation step")
    parser.add_argument("--max_rounds", type=int, default=5,
                        help="Max continuation rounds after truncation")
    parser.add_argument("--max_refinement_tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--num_gpus", type=int, default=None)
    parser.add_argument("--max_model_len", type=int, default=40576)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--item_timeout", type=int, default=600,
                        help="Per-item timeout in seconds (default: 600)")

    args = parser.parse_args()

    # Load tasks from either source
    if args.truncated_input:
        print(f"Loading pre-built truncated tasks from {args.truncated_input}...")
        tasks = load_truncated_tasks(args.truncated_input)
        print(f"Loaded {len(tasks)} tasks")
    elif args.annotation_files and args.trajectory_files:
        if len(args.annotation_files) != len(args.trajectory_files):
            print(f"Error: {len(args.annotation_files)} annotation files but "
                  f"{len(args.trajectory_files)} trajectory files")
            sys.exit(1)
        print("Building reconstruction tasks...")
        tasks = build_reconstruction_tasks(args.annotation_files, args.trajectory_files)
        print(f"Total tasks before dedup: {len(tasks)}")
        tasks = deduplicate_tasks(tasks)
        print(f"After dedup: {len(tasks)}")
    else:
        print("Error: provide either --truncated_input or both --annotation_files and --trajectory_files")
        sys.exit(1)

    if not tasks:
        print("No tasks to reconstruct. Exiting.")
        return

    # Category distribution
    cat_counts = {}
    for t in tasks:
        cat_counts[t.annotation_category] = cat_counts.get(t.annotation_category, 0) + 1
    print(f"Category distribution: {cat_counts}")

    # Detect GPUs
    import torch
    num_gpus = torch.cuda.device_count()
    if args.num_gpus is None:
        args.num_gpus = max(1, num_gpus)
    else:
        args.num_gpus = min(args.num_gpus, num_gpus)

    print(f"\n{'='*60}")
    print(f"Trajectory Reconstruction")
    print(f"{'='*60}")
    print(f"Tasks: {len(tasks)}")
    print(f"GPUs: {args.num_gpus}")
    print(f"Model: {args.model}")
    print(f"Max rounds after truncation: {args.max_rounds}")
    print(f"{'='*60}\n")

    # Shard tasks across GPUs
    shards = [[] for _ in range(args.num_gpus)]
    for i, task in enumerate(tasks):
        shards[i % args.num_gpus].append(task)
    print(f"Shard sizes: {[len(s) for s in shards]}")

    # Compute intermediate file paths (one per GPU)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    intermediate_paths = [
        str(output_path.parent / (output_path.stem + f"_intermediate_gpu{g}" + output_path.suffix))
        for g in range(args.num_gpus)
    ]

    # Launch workers
    mp.set_start_method('spawn', force=True)
    result_queue = mp.Queue()
    processes = []

    for gpu_id in range(args.num_gpus):
        p = mp.Process(
            target=worker_process,
            args=(gpu_id, shards[gpu_id], args, result_queue, intermediate_paths[gpu_id])
        )
        p.start()
        processes.append(p)

    # Collect completion signals (workers now write data directly to intermediate files)
    counts = {}
    for _ in range(args.num_gpus):
        gpu_id, count = result_queue.get()
        counts[gpu_id] = count
        print(f"[Main] GPU {gpu_id} completed {count} reconstructions -> {intermediate_paths[gpu_id]}")

    for p in processes:
        p.join()

    # Merge intermediate files into final output
    merged = []
    for gpu_id in range(args.num_gpus):
        path = intermediate_paths[gpu_id]
        if Path(path).exists():
            gpu_results = load_jsonl(path)
            merged.extend(gpu_results)
            print(f"[Main] Loaded {len(gpu_results)} results from GPU {gpu_id} intermediate file")
        else:
            print(f"[Main] Warning: intermediate file missing for GPU {gpu_id}: {path}")

    # Save
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in merged:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(f"\n{'='*60}")
    print(f"Saved {len(merged)} reconstructed trajectories to {args.output}")

    # Stats
    done_reasons = {}
    total_tool_calls = 0
    for r in merged:
        for s in r["samples"]:
            dr = s.get("done_reason", "unknown")
            done_reasons[dr] = done_reasons.get(dr, 0) + 1
            total_tool_calls += s.get("num_tool_calls", 0)
    print(f"Done reasons: {done_reasons}")
    print(f"Total tool calls: {total_tool_calls}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
