"""Data-parallel orchestration utilities for multi-GPU inference."""

import json
import multiprocessing as mp
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List

import torch


def detect_gpus(requested: int = None) -> int:
    """Detect available GPUs and return how many to use.

    Args:
        requested: User-requested number of GPUs (None = use all).

    Returns:
        Number of GPUs to use.
    """
    num_gpus = torch.cuda.device_count()
    if requested is None:
        return num_gpus
    return min(requested, num_gpus)


def shard_data(data: List[Dict], num_shards: int) -> List[List[Dict]]:
    """Round-robin shard data across num_shards buckets."""
    shards = [[] for _ in range(num_shards)]
    for i, item in enumerate(data):
        shards[i % num_shards].append(item)
    return shards


def run_data_parallel(worker_fn: Callable, shards: List[List[Dict]],
                      num_gpus: int, args) -> List[Dict]:
    """Spawn worker processes, collect results, merge in original order.

    Each worker_fn must have signature:
        worker_fn(gpu_id: int, data_shard: List[Dict], args, result_queue: mp.Queue)
    and put (gpu_id, results_list) into result_queue.

    Returns merged results in the original data order.
    """
    mp.set_start_method('spawn', force=True)
    result_queue = mp.Queue()
    processes = []

    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=worker_fn,
            args=(gpu_id, shards[gpu_id], args, result_queue)
        )
        p.start()
        processes.append(p)

    # Collect results from all workers
    all_results = {}
    for _ in range(num_gpus):
        gpu_id, results = result_queue.get()
        all_results[gpu_id] = results
        print(f"[Main] Received {len(results)} results from GPU {gpu_id}")

    # Wait for all processes to finish
    for p in processes:
        p.join()

    # Merge results maintaining original order (round-robin)
    merged = []
    indices = [0] * num_gpus
    total = sum(len(s) for s in shards)
    for i in range(total):
        gpu_id = i % num_gpus
        if indices[gpu_id] < len(all_results[gpu_id]):
            merged.append(all_results[gpu_id][indices[gpu_id]])
            indices[gpu_id] += 1

    return merged


@dataclass
class SampleState:
    """Tracks the agent loop state for a single (question, sample) pair.

    Used by tool_refinement.py and agentic_refinement.py.
    """
    q_idx: int
    s_idx: int
    original_prompt: str
    conversation_turns: List[Dict] = field(default_factory=list)
    rounds_data: List[Dict] = field(default_factory=list)
    is_done: bool = False
    done_reason: str = ""
    num_tool_calls: int = 0
    last_refined_context: str = ""


def save_intermediate_results(gpu_id: int, valid_items: List[Dict],
                              original_prompts: List[str],
                              all_states: List[List['SampleState']],
                              num_samples: int, round_num: int,
                              output_file: str):
    """Save intermediate results for this GPU after a round completes.

    Args:
        gpu_id: GPU/worker identifier
        valid_items: The data items being processed
        original_prompts: Original problem prompts
        all_states: all_states[q_idx][s_idx] = SampleState
        num_samples: Number of samples per question
        round_num: Current round number (0-indexed)
        output_file: Base output file path (intermediate file is derived from this)
    """
    results = []
    for q_idx, (item, orig_prompt) in enumerate(zip(valid_items, original_prompts)):
        samples = []
        for s_idx in range(num_samples):
            state = all_states[q_idx][s_idx]
            rounds_data = state.rounds_data
            sample_result = {
                "sample_idx": s_idx,
                "rounds": rounds_data,
                "final_refined_context": state.last_refined_context,
                "full_assistant_message": (
                    state.last_refined_context + rounds_data[-1]["current_round_generation"]
                    if rounds_data else ""
                ),
                "num_tool_calls": state.num_tool_calls,
                "done_reason": state.done_reason,
            }
            samples.append(sample_result)

        result = {
            "original_prompt": orig_prompt,
            "num_samples": num_samples,
            "samples": samples,
            **{k: v for k, v in item.items() if k != "prompt"}
        }
        results.append(result)

    output_path = Path(output_file)
    intermediate_file = output_path.parent / f"{output_path.stem}_intermediate_gpu{gpu_id}.jsonl"
    intermediate_file.parent.mkdir(parents=True, exist_ok=True)
    with open(intermediate_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    print(f"[GPU {gpu_id}] Saved intermediate results after round {round_num + 1} to {intermediate_file}")


def save_merged_results(results: List[Dict], output_file: str) -> None:
    """Save merged results to JSONL file."""
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
