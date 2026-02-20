#!/usr/bin/env python3
"""
Tool-Calling LLM Refinement Script for Qwen3-4B-Instruct-2507

Uses the model's native tool-calling format (apply_chat_template with tools=)
to let the model autonomously decide when to call `llm_refine`. When the model
outputs <tool_call>, generation stops, the partial generation is sent to a refiner
(local or API-based), and the compressed summary is injected as a <tool_response>.

Key differences from agentic_refinement.py:
- No thinking mode handling (2507 has no <think> tags)
- Configurable refiner (same model or separate via API)
- Uses tokenizer's built-in chat template with tools= parameter
- Conversation built as structured messages for apply_chat_template
"""

import argparse
import json
import csv
import os
import torch
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path
import urllib.request


# Dataset URLs
IMOBENCH_URL = "https://raw.githubusercontent.com/google-deepmind/superhuman/main/imobench/answerbench.csv"

# Tool definition for llm_refine
LLM_REFINE_TOOL = {
    "type": "function",
    "function": {
        "name": "llm_refine",
        "description": (
            "Summarize your progress and continue with a fresh start. "
            "Call after each major reasoning step."
        ),
        "parameters": {"type": "object", "properties": {}}
    }
}

# Default system prompt
DEFAULT_SYSTEM_PROMPT = """\
You are a mathematical reasoning assistant. You have a tool `llm_refine` that \
summarizes your work so far. After calling it, you continue solving with a fresh \
perspective while retaining key insights. Call it after each major reasoning step. \
Present your final answer using \\boxed{} notation.\
"""

LLM_REFINE_USER_HINT = """

You have `llm_refine` — call it after each major reasoning step to checkpoint your progress."""


# ============================================================
# Data loading utilities
# ============================================================

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


def create_summarization_prompt(original_prompt: str, existing_summary: str,
                                latest_reasoning: str) -> str:
    """Create an RC-style summarization prompt.

    Produces a first-person, descriptive summary (max 2 paragraphs) that
    captures the reasoning progress without adding new analysis.
    """
    return f"""You are given a maths problem and a candidate solution to it. You may also be given a summary of a previous candidate solution to the problem. If this is provided, you may assume that the current candidate solution was generated conditioned on the summary of the previous candidate solution.
Your task is to write a summary of the current candidate solution.

The new summary you generate should possess the following characteristics:
- It should provide a detailed overview of what occurred in the current candidate solution. This may include a summary of the high-level problem-solving strategy, a description of theorems used, verification attempts, calculations and logical deductions etc.
- It should summarize the current candidate solution in light of any previous summaries, if provided. We should be able to understand the relationship between the previous solution and the current solution by reading the summary. Make sure any important information contained in the existing summary is retained in the new one.
- It should be no more than two paragraphs long and written in paragraph form, without headers or subheaders.
- It should be written in the first person, as if though it is being written by the person solving the problem.
- The candidate solution may not be complete. In this case, the summary should still attempt to summarize the partial solution.

IMPORTANT: Do not under any circumstances add any additional reasoning not contained in the latest reasoning step. Your task is only to summarize what is given to you.

### PROBLEM
{original_prompt}

### EXISTING SUMMARY
{existing_summary}

### LATEST CANDIDATE SOLUTION
{latest_reasoning}"""


# ============================================================
# Prompt construction using tokenizer's apply_chat_template
# ============================================================

def build_initial_messages(problem: str, system_prompt: str,
                           include_tool_hint: bool = True) -> List[Dict]:
    """Build the initial message list for a new problem."""
    user_content = problem
    if include_tool_hint:
        user_content += LLM_REFINE_USER_HINT
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def build_conversation_messages(problem: str, system_prompt: str,
                                conversation_turns: List[Dict],
                                include_tool_hint: bool = True) -> List[Dict]:
    """Build full message list including conversation history."""
    messages = build_initial_messages(problem, system_prompt, include_tool_hint)
    messages.extend(conversation_turns)
    return messages


def render_prompt(tokenizer, messages: List[Dict], tools: List[Dict]) -> str:
    """Render messages to a prompt string using the tokenizer's chat template."""
    return tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )


def render_summarization_prompt(tokenizer, problem: str,
                                existing_summary: str,
                                latest_reasoning: str) -> str:
    """Build the summarization prompt (no tools, plain chat template)."""
    summarization_text = create_summarization_prompt(
        problem, existing_summary, latest_reasoning
    )
    messages = [{"role": "user", "content": summarization_text}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ============================================================
# Refiner abstraction
# ============================================================

class LocalRefiner:
    """Uses the same vLLM LLM instance for summarization."""

    def __init__(self, llm, tokenizer, sampling_params):
        self.llm = llm
        self.tokenizer = tokenizer
        self.sampling_params = sampling_params

    def refine_batch(self, batch: List[Dict]) -> List[str]:
        """Summarize a batch using RC-style summarization.

        Args:
            batch: List of dicts with keys 'original_prompt',
                   'existing_summary', and 'latest_reasoning'

        Returns:
            List of summary strings
        """
        prompts = [
            render_summarization_prompt(
                self.tokenizer, item["original_prompt"],
                item["existing_summary"], item["latest_reasoning"]
            )
            for item in batch
        ]
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [output.outputs[0].text for output in outputs]


class APIRefiner:
    """Calls an external vLLM server via OpenAI-compatible API for summarization."""

    def __init__(self, base_url: str, model: str, temperature: float = 0.7,
                 max_tokens: int = 16384):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key="dummy")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def refine_batch(self, batch: List[Dict]) -> List[str]:
        """Summarize a batch by calling the API sequentially.

        Args:
            batch: List of dicts with keys 'original_prompt',
                   'existing_summary', and 'latest_reasoning'

        Returns:
            List of summary strings
        """
        results = []
        for item in batch:
            summarization_text = create_summarization_prompt(
                item["original_prompt"], item["existing_summary"],
                item["latest_reasoning"]
            )
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": summarization_text}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            results.append(response.choices[0].message.content)
        return results


# ============================================================
# State tracking
# ============================================================

@dataclass
class SampleState:
    """Tracks the agent loop state for a single (question, sample) pair."""
    q_idx: int
    s_idx: int
    original_prompt: str
    conversation_turns: List[Dict] = field(default_factory=list)
    rounds_data: List[Dict] = field(default_factory=list)
    is_done: bool = False
    done_reason: str = ""
    num_tool_calls: int = 0
    last_refined_context: str = ""


# ============================================================
# Intermediate saving
# ============================================================

def save_intermediate_results(gpu_id: int, valid_items: List[Dict],
                              original_prompts: List[str],
                              all_states: List[List[SampleState]],
                              num_samples: int, round_num: int,
                              output_file: str):
    """Save intermediate results for this GPU after a round completes."""
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


# ============================================================
# Worker process
# ============================================================

def worker_process(gpu_id: int, data_shard: List[Dict], args, result_queue: mp.Queue):
    """Worker process that runs on a single GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"[GPU {gpu_id}] Starting tool-calling refinement worker with {len(data_shard)} problems")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # Get the <tool_call> special token ID for stopping
    tool_call_token_id = tokenizer.convert_tokens_to_ids("<tool_call>")
    print(f"[GPU {gpu_id}] <tool_call> token ID: {tool_call_token_id}")

    # Sampling params for main generation (stops at <tool_call> token or text)
    generation_sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.num_tokens,
        n=args.num_samples,
        stop_token_ids=[tool_call_token_id],
        stop=["<tool_call>"],
    )

    # Sampling params for continuation (n=1, stops at <tool_call> token or text)
    continuation_sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.num_tokens,
        n=1,
        stop_token_ids=[tool_call_token_id],
        stop=["<tool_call>"],
    )

    # Sampling params for refinement (no tool call stopping)
    refinement_sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_refinement_tokens,
    )

    # Initialize refiner
    if args.refiner_base_url:
        refiner_model = args.refiner_model or args.model
        print(f"[GPU {gpu_id}] Using API refiner: {args.refiner_base_url} model={refiner_model}")
        refiner = APIRefiner(
            base_url=args.refiner_base_url,
            model=refiner_model,
            temperature=args.temperature,
            max_tokens=args.max_refinement_tokens,
        )
    else:
        print(f"[GPU {gpu_id}] Using local refiner (same vLLM instance)")
        refiner = LocalRefiner(
            llm=llm,
            tokenizer=tokenizer,
            sampling_params=refinement_sampling_params,
        )

    system_prompt = args.system_prompt or DEFAULT_SYSTEM_PROMPT
    tools = [LLM_REFINE_TOOL]
    num_samples = args.num_samples

    valid_items = [item for item in data_shard if "prompt" in item and item["prompt"]]
    original_prompts = [item["prompt"] for item in valid_items]

    # Initialize states: all_states[q_idx][s_idx]
    all_states: List[List[SampleState]] = [
        [SampleState(q_idx=q_idx, s_idx=s_idx, original_prompt=prompt)
         for s_idx in range(num_samples)]
        for q_idx, prompt in enumerate(original_prompts)
    ]

    # ============================================================
    # Round 0: Initial generation with n=num_samples
    # ============================================================
    print(f"[GPU {gpu_id}] Round 0: Initial generation")
    initial_prompts = []
    for prompt in original_prompts:
        messages = build_initial_messages(prompt, system_prompt)
        rendered = render_prompt(tokenizer, messages, tools)
        initial_prompts.append(rendered)

    initial_outputs = llm.generate(initial_prompts, generation_sampling_params)

    # Classify each output and populate states
    needs_refinement: List[SampleState] = []
    for q_idx, output in enumerate(initial_outputs):
        for s_idx in range(num_samples):
            state = all_states[q_idx][s_idx]
            comp_output = output.outputs[s_idx]
            raw_text = comp_output.text
            stopped_at_tool = (comp_output.stop_reason == tool_call_token_id
                               or comp_output.stop_reason == "<tool_call>")

            round_data = {
                "round": 1,
                "current_round_generation": raw_text,
                "refined_context": None,
                "called_tool": stopped_at_tool,
                "finish_reason": "tool_call" if stopped_at_tool else comp_output.finish_reason,
            }
            state.rounds_data.append(round_data)

            if stopped_at_tool:
                needs_refinement.append(state)
            else:
                state.is_done = True
                state.done_reason = (
                    "completed" if comp_output.finish_reason == "stop" else "max_tokens"
                )

    # ============================================================
    # Refinement + continuation loop
    # ============================================================
    for round_num in range(1, args.max_rounds):
        # --- Refinement batch ---
        if needs_refinement:
            print(f"[GPU {gpu_id}] Round {round_num}: Refining {len(needs_refinement)} samples")

            refine_batch = []
            for state in needs_refinement:
                last_round = state.rounds_data[-1]
                chunk = last_round["current_round_generation"]

                refine_batch.append({
                    "original_prompt": state.original_prompt,
                    "existing_summary": state.last_refined_context or "",
                    "latest_reasoning": chunk,
                })

            refined_texts = refiner.refine_batch(refine_batch)

            # RC-style continuation instructions
            continuation_instructions = (
                "\n\nContinue solving the problem, improving upon the "
                "summary above. You may verify previous conclusions, try a "
                "different approach, or build on the progress so far.\n"
                "Return your final answer in \\boxed{}."
            )

            for state, refined in zip(needs_refinement, refined_texts):
                # Update state
                state.rounds_data[-1]["refined_context"] = refined
                state.last_refined_context = refined
                state.num_tool_calls += 1

                # Build conversation turns for continuation using structured messages.
                last_round = state.rounds_data[-1]

                if args.compact_context:
                    # Compact mode: fresh start with only summary + continuation hint.
                    # Reset conversation_turns each round (RC-style).
                    # Keep last generation in assistant turn to match training format
                    # (training always has [generation text] + <tool_call> together).
                    state.conversation_turns = [
                        {"role": "assistant", "content": last_round["current_round_generation"] + '\n<tool_call>\n{"name": "llm_refine", "arguments": {}}\n</tool_call>'},
                        {"role": "user", "content": f"<tool_response>\n{refined}\n</tool_response>"
                         + continuation_instructions},
                    ]
                else:
                    # Default: accumulate full generation history
                    assistant_content = (
                        last_round["current_round_generation"]
                        + '\n<tool_call>\n{"name": "llm_refine", "arguments": {}}\n</tool_call>'
                    )
                    state.conversation_turns.append(
                        {"role": "assistant", "content": assistant_content}
                    )
                    state.conversation_turns.append(
                        {"role": "user", "content": f"<tool_response>\n{refined}\n</tool_response>"
                         + continuation_instructions}
                    )

        # Save intermediate results
        save_intermediate_results(
            gpu_id, valid_items, original_prompts,
            all_states, num_samples, round_num - 1, args.output_file
        )

        # --- Continuation generation ---
        active_states = [
            state
            for q_states in all_states
            for state in q_states
            if not state.is_done
        ]
        if not active_states:
            break

        print(f"[GPU {gpu_id}] Round {round_num}: Continuing {len(active_states)} samples")

        continuation_prompts = []
        for state in active_states:
            messages = build_conversation_messages(
                state.original_prompt, system_prompt,
                state.conversation_turns
            )
            rendered = render_prompt(tokenizer, messages, tools)
            continuation_prompts.append(rendered)

        cont_outputs = llm.generate(continuation_prompts, continuation_sampling_params)

        needs_refinement = []
        for state, cont_output in zip(active_states, cont_outputs):
            comp_output = cont_output.outputs[0]
            raw_text = comp_output.text
            stopped_at_tool = (comp_output.stop_reason == tool_call_token_id
                               or comp_output.stop_reason == "<tool_call>")

            round_data = {
                "round": len(state.rounds_data) + 1,
                "current_round_generation": raw_text,
                "refined_context": None,
                "called_tool": stopped_at_tool,
                "finish_reason": "tool_call" if stopped_at_tool else comp_output.finish_reason,
            }
            state.rounds_data.append(round_data)

            if stopped_at_tool:
                needs_refinement.append(state)
            else:
                state.is_done = True
                state.done_reason = (
                    "completed" if comp_output.finish_reason == "stop" else "max_tokens"
                )

    # Mark remaining active samples as done
    for q_states in all_states:
        for state in q_states:
            if not state.is_done:
                state.is_done = True
                state.done_reason = "max_rounds"

    # Save final intermediate
    save_intermediate_results(
        gpu_id, valid_items, original_prompts,
        all_states, num_samples, args.max_rounds - 1, args.output_file
    )

    # ============================================================
    # Build final results
    # ============================================================
    results = []
    for q_idx, (item, orig_prompt) in enumerate(zip(valid_items, original_prompts)):
        samples = []
        for s_idx in range(num_samples):
            state = all_states[q_idx][s_idx]
            rounds_data = state.rounds_data

            # full_assistant_message: refined context + final generation
            last_gen = rounds_data[-1]["current_round_generation"] if rounds_data else ""
            full_message = (
                state.last_refined_context + last_gen
                if state.last_refined_context else last_gen
            )

            sample_result = {
                "sample_idx": s_idx,
                "rounds": rounds_data,
                "final_refined_context": state.last_refined_context,
                "full_assistant_message": full_message,
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

    print(f"[GPU {gpu_id}] Completed: {len(results)} problems, "
          f"{sum(s.num_tool_calls for qs in all_states for s in qs)} total tool calls")
    result_queue.put((gpu_id, results))


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Tool-calling LLM refinement with vLLM Data Parallelism"
    )
    parser.add_argument(
        "--dataset", type=str, choices=["hmmt", "imobench"], required=True,
        help="Dataset to use"
    )
    parser.add_argument("--input_file", type=str, default=None,
                        help="Input JSONL file (required for hmmt)")
    parser.add_argument("--cache_dir", type=str,
                        default="/scratch/dkhasha1/tli104/imobench")
    parser.add_argument("--model", type=str,
                        default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--refiner_model", type=str, default=None,
                        help="Refiner model name (default: same as --model). "
                             "Used with --refiner_base_url for API refiner.")
    parser.add_argument("--refiner_base_url", type=str, default=None,
                        help="Base URL for external refiner vLLM server "
                             "(e.g., http://localhost:8000/v1). "
                             "If not set, uses local vLLM instance.")
    parser.add_argument("--num_tokens", type=int, required=True,
                        help="Max tokens per generation step")
    parser.add_argument("--output_file", type=str, default="output_tool_refinement.jsonl")
    parser.add_argument("--max_rounds", type=int, default=12,
                        help="Maximum number of generation rounds (default: 12)")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of samples per question (default: 1)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_refinement_tokens", type=int, default=None,
                        help="Max tokens for refinement (default: 16384)")
    # Legacy flags (no longer used — summarization is always descriptive)
    parser.add_argument("--preserve_answer", action="store_true", default=True,
                        help="(deprecated, no-op)")
    parser.add_argument("--strip_answer", action="store_true",
                        help="(deprecated, no-op)")
    parser.add_argument("--num_gpus", type=int, default=None)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--compact_context", action="store_true",
                        help="Only keep prompt + latest summary in context "
                             "(removes generation from conversation history)")
    parser.add_argument("--system_prompt", type=str, default=None,
                        help="Override default system prompt")
    parser.add_argument("--system_prompt_file", type=str, default=None,
                        help="Read system prompt from file (takes precedence over --system_prompt)")

    args = parser.parse_args()

    if args.system_prompt_file:
        with open(args.system_prompt_file, "r") as f:
            args.system_prompt = f.read().strip()

    if args.max_refinement_tokens is None:
        args.max_refinement_tokens = 16384

    if args.max_model_len is None:
        args.max_model_len = min(
            args.num_tokens + args.max_refinement_tokens + 8192,
            131072
        )

    if args.dataset == "hmmt" and args.input_file is None:
        parser.error("--input_file is required when using --dataset hmmt")

    # Detect GPUs
    num_gpus = torch.cuda.device_count()
    if args.num_gpus is None:
        args.num_gpus = num_gpus
    else:
        args.num_gpus = min(args.num_gpus, num_gpus)

    print(f"\n{'='*80}")
    print(f"Tool-Calling Refinement - Data Parallel Inference")
    print(f"{'='*80}")
    print(f"Available GPUs: {num_gpus}")
    print(f"Using GPUs: {args.num_gpus}")
    print(f"Model: {args.model}")
    print(f"Refiner: {'API @ ' + args.refiner_base_url if args.refiner_base_url else 'Local (same model)'}")
    if args.refiner_base_url and args.refiner_model:
        print(f"Refiner Model: {args.refiner_model}")
    print(f"Max Model Length: {args.max_model_len}")
    print(f"Num Tokens per step: {args.num_tokens}")
    print(f"Max Refinement Tokens: {args.max_refinement_tokens}")
    print(f"Max Rounds: {args.max_rounds}")
    print(f"Num Samples: {args.num_samples}")
    print(f"Compact Context: {args.compact_context}")
    print(f"{'='*80}\n")

    # Load data
    print(f"Loading Dataset: {args.dataset.upper()}")
    if args.dataset == "imobench":
        if args.input_file:
            data = load_jsonl(args.input_file)
        else:
            csv_path = download_imobench(args.cache_dir)
            data = load_imobench(csv_path)
    elif args.dataset == "hmmt":
        data = load_jsonl(args.input_file)

    print(f"Loaded {len(data)} problems")

    # Shard data across GPUs
    shards = [[] for _ in range(args.num_gpus)]
    for i, item in enumerate(data):
        shards[i % args.num_gpus].append(item)
    print(f"Data sharding: {[len(s) for s in shards]}")

    # Start worker processes
    mp.set_start_method('spawn', force=True)
    result_queue = mp.Queue()
    processes = []

    for gpu_id in range(args.num_gpus):
        p = mp.Process(
            target=worker_process,
            args=(gpu_id, shards[gpu_id], args, result_queue)
        )
        p.start()
        processes.append(p)

    # Collect results
    all_results = {}
    for _ in range(args.num_gpus):
        gpu_id, results = result_queue.get()
        all_results[gpu_id] = results
        print(f"[Main] Received {len(results)} results from GPU {gpu_id}")

    for p in processes:
        p.join()

    # Merge results maintaining original order
    merged_results = []
    indices = [0] * args.num_gpus
    for i in range(len(data)):
        gpu_id = i % args.num_gpus
        if indices[gpu_id] < len(all_results[gpu_id]):
            merged_results.append(all_results[gpu_id][indices[gpu_id]])
            indices[gpu_id] += 1

    # Save results
    print(f"\nSaving {len(merged_results)} results to {args.output_file}...")
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for result in merged_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    # Summary stats
    total_tool_calls = sum(
        s["num_tool_calls"]
        for r in merged_results
        for s in r["samples"]
    )
    done_reasons = {}
    for r in merged_results:
        for s in r["samples"]:
            reason = s.get("done_reason", "unknown")
            done_reasons[reason] = done_reasons.get(reason, 0) + 1

    print(f"\n{'='*80}")
    print(f"Done! Processed {len(merged_results)} problems with {args.num_gpus} GPUs.")
    print(f"Total tool calls: {total_tool_calls}")
    print(f"Done reasons: {done_reasons}")
    print(f"Results saved to {args.output_file}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
