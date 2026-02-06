#!/usr/bin/env python3
"""
Agentic Context Refinement Script using vLLM with Data Parallelism

The model decides when to refine via Qwen3's native tool-calling format.
When the model outputs <tool_call> to call `llm_refine`, generation stops,
the current chunk is refined by the same LLM (without tool system prompt),
and the refined summary is returned as a <tool_response>.
"""

import argparse
import json
import csv
import os
import torch
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import urllib.request

from jinja2 import Template


# Dataset URLs
IMOBENCH_URL = "https://raw.githubusercontent.com/google-deepmind/superhuman/main/imobench/answerbench.csv"

# Tool definition for llm_refine
LLM_REFINE_TOOL = {
    "type": "function",
    "function": {
        "name": "llm_refine",
        "description": (
            "Compress and refine your current reasoning progress. "
            "Returns a concise summary preserving key insights and intermediate results. "
            "Call this when your reasoning is getting long and you want to consolidate "
            "your thoughts before continuing."
        ),
        "parameters": {"type": "object", "properties": {}}
    }
}

# Default system prompt for the agentic solver
DEFAULT_SYSTEM_PROMPT = """\
You are a mathematical reasoning assistant solving competition-level math problems.

You MUST follow this workflow strictly:
1. Do ONE step of reasoning (set up the problem, try one approach, or make partial progress).
2. Then IMMEDIATELY call the `llm_refine` tool to compress and save your progress.
3. After receiving the refined summary, continue with the next reasoning step.
4. Repeat steps 1-3 until you reach a final answer.

CRITICAL RULES:
- You MUST call `llm_refine` after every reasoning step. Do NOT try to solve the entire problem in one pass.
- After writing a chunk of reasoning, STOP and call the tool. Do not keep going.
- When you reach a final answer, present it using \\boxed{} notation (no tool call needed for the final answer).

Example of correct behavior:
1. Write: "Let me set up the problem... [partial work]"
2. Call: llm_refine
3. Receive summary, then write: "Continuing from the summary... [more work]"
4. Call: llm_refine
5. Receive summary, then write: "Now I can conclude... \\boxed{answer}"
"""

# Jinja2 template for Qwen3 tool-calling conversations
# Adapted from training/slime/examples/retool/generate_with_retool.py
TOOL_TEMPLATE = """\
<|im_start|>system
{%- if messages[0]['role'] == 'system' %}
{{ messages[0]['content'] }}
{%- else %}
You are a helpful assistant.
{%- endif %}
{%- if tools %}

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{%- for tool in tools %}
{{ tool | tojson }}
{%- endfor %}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
{%- endif %}
<|im_end|>
{%- for message in messages %}
{%- if message['role'] == 'user' %}
<|im_start|>user
{{ message['content'] }}<|im_end|>
{%- elif message['role'] == 'assistant' %}
<|im_start|>assistant
{{ message['content'] }}<|im_end|>
{%- endif %}
{%- endfor %}
<|im_start|>assistant
"""


# ============================================================
# Data loading utilities (reused from context_refinement_dp.py)
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


def strip_thinking(text: str) -> str:
    """Strip <think>...</think> section from model output."""
    if '<think>' not in text:
        return text
    think_end = text.find('</think>')
    if think_end == -1:
        return ""
    return text[think_end + 8:].strip()


def create_refinement_prompt(original_prompt: str, partial_generation: str, preserve_answer: bool = True) -> str:
    """Create the context refinement prompt."""
    if preserve_answer:
        return f"""Context Refinement Prompt:

Original Prompt:
{original_prompt}

Partial Generation:
{partial_generation}

Your task is to create a compressed summary for another model to continue solving from.

RULES:
1. If a final answer (e.g., \\boxed{{}}) was found, PRESERVE IT at the end of your summary
2. Keep key insights, important calculations, and the reasoning path
3. Remove redundant text, false starts, and unnecessary repetition
4. If the answer seems wrong or unverified, note that verification is needed
5. Be concise but preserve all critical mathematical steps

Output format:
- Key insights and progress made
- Important intermediate results
- If found: "Final Answer: [the answer]" or the \\boxed{{}} expression
- If not solved: what still needs to be done"""
    else:
        return f"""Context Refinement Prompt:

Original Prompt:
{original_prompt}

Partial Generation:
{partial_generation}

Your task is to create a WORK-IN-PROGRESS summary for another model to continue solving from.

CRITICAL RULES:
1. NEVER include any final answer or \\boxed{{}} in your output
2. NEVER conclude or claim the problem is solved
3. Remove any "Final Answer" sections completely
4. Keep only intermediate calculations, key insights, and partial progress
5. End your summary at a natural continuation point where more work is needed
6. If the generation reached a wrong answer, note the approach taken but indicate it needs verification

Output a concise summary of the progress made so far, ending with what still needs to be done. Do NOT provide any final answer."""


# ============================================================
# Prompt construction
# ============================================================

_jinja_template = Template(TOOL_TEMPLATE)


LLM_REFINE_USER_HINT = """

IMPORTANT: You have access to the `llm_refine` tool. If your reasoning is getting long or you want to consolidate your progress before continuing, call it using:
<tool_call>
{"name": "llm_refine", "arguments": {}}
</tool_call>
You will receive a compressed summary of your work so far, then you can continue solving from there."""


def format_tool_prompt(problem: str, conversation_turns: List[Dict],
                       system_prompt: str, tools: List[Dict],
                       include_tool_hint: bool = True) -> str:
    """Build the full prompt string with tool definitions using Jinja2.

    Args:
        problem: The math problem text
        conversation_turns: Prior assistant/user turns (after the initial user message)
        system_prompt: System prompt text
        tools: List of tool definitions
        include_tool_hint: Whether to append tool usage hint to user message
    """
    user_content = problem
    if include_tool_hint and tools:
        user_content += LLM_REFINE_USER_HINT
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    messages.extend(conversation_turns)
    return _jinja_template.render(messages=messages, tools=tools or [])


def format_refinement_prompt(tokenizer, problem: str, context: str,
                             preserve_answer: bool = True,
                             disable_thinking: bool = False) -> str:
    """Build the refinement prompt (no tools, plain chat template)."""
    refinement_text = create_refinement_prompt(problem, context, preserve_answer)
    if disable_thinking:
        refinement_text += "\n\n/no_think"
    messages = [{"role": "user", "content": refinement_text}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


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
    """Save intermediate results for this GPU after a round completes.

    Args:
        all_states: all_states[q_idx][s_idx] = SampleState
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


# ============================================================
# Worker process
# ============================================================

def worker_process(gpu_id: int, data_shard: List[Dict], args, result_queue: mp.Queue):
    """Worker process that runs on a single GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"[GPU {gpu_id}] Starting agentic worker with {len(data_shard)} problems")

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
        top_k=args.top_k,
        max_tokens=args.num_tokens,
        n=args.num_samples,
        stop_token_ids=[tool_call_token_id],
        stop=["<tool_call>"],
    )

    # Sampling params for continuation (n=1, stops at <tool_call> token or text)
    continuation_sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.num_tokens,
        n=1,
        stop_token_ids=[tool_call_token_id],
        stop=["<tool_call>"],
    )

    # Sampling params for refinement (no tool call stopping)
    refinement_sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_refinement_tokens,
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
    initial_prompts = [
        format_tool_prompt(prompt, [], system_prompt, tools)
        for prompt in original_prompts
    ]

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

            if args.strip_thinking_from_generation:
                stripped_text = strip_thinking(raw_text)
            else:
                stripped_text = raw_text

            round_data = {
                "round": 1,
                "current_round_generation": stripped_text,
                "current_round_generation_raw": raw_text,
                "refined_context": None,
                "refined_context_raw": None,
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

            refine_prompts = []
            for state in needs_refinement:
                last_round = state.rounds_data[-1]
                chunk = last_round["current_round_generation"]

                # Accumulate: previous refined context + current chunk
                if state.last_refined_context:
                    context_to_refine = state.last_refined_context + "\n\n" + chunk
                else:
                    context_to_refine = chunk

                refine_prompts.append(format_refinement_prompt(
                    tokenizer, state.original_prompt, context_to_refine,
                    preserve_answer=args.preserve_answer,
                    disable_thinking=args.disable_thinking_for_refinement,
                ))

            refine_outputs = llm.generate(refine_prompts, refinement_sampling_params)

            for state, ref_output in zip(needs_refinement, refine_outputs):
                refined_raw = ref_output.outputs[0].text
                if args.strip_thinking_from_refinement:
                    refined = strip_thinking(refined_raw)
                    if not refined:
                        # Refinement was all thinking; fall back to stripped generation
                        refined = state.rounds_data[-1]["current_round_generation"]
                        print(f"[GPU {gpu_id}] Warning: Refinement incomplete for "
                              f"q{state.q_idx}/s{state.s_idx}/r{round_num}, using generation")
                else:
                    refined = refined_raw

                # Update state
                state.rounds_data[-1]["refined_context"] = refined
                state.rounds_data[-1]["refined_context_raw"] = refined_raw
                state.last_refined_context = refined
                state.num_tool_calls += 1

                # Update conversation turns for continuation
                last_round = state.rounds_data[-1]
                # Assistant message: raw generation + tool call tags
                assistant_content = (
                    last_round["current_round_generation_raw"]
                    + '\n<tool_call>\n{"name": "llm_refine", "arguments": {}}\n</tool_call>'
                )
                state.conversation_turns.append(
                    {"role": "assistant", "content": assistant_content}
                )
                # User message: tool response with refined text
                state.conversation_turns.append(
                    {"role": "user", "content": f"<tool_response>\n{refined}\n</tool_response>"}
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

        continuation_prompts = [
            format_tool_prompt(
                state.original_prompt, state.conversation_turns,
                system_prompt, tools
            )
            for state in active_states
        ]

        cont_outputs = llm.generate(continuation_prompts, continuation_sampling_params)

        needs_refinement = []
        for state, cont_output in zip(active_states, cont_outputs):
            comp_output = cont_output.outputs[0]
            raw_text = comp_output.text
            stopped_at_tool = (comp_output.stop_reason == tool_call_token_id
                               or comp_output.stop_reason == "<tool_call>")

            if args.strip_thinking_from_generation:
                stripped_text = strip_thinking(raw_text)
            else:
                stripped_text = raw_text

            round_data = {
                "round": len(state.rounds_data) + 1,
                "current_round_generation": stripped_text,
                "current_round_generation_raw": raw_text,
                "refined_context": None,
                "refined_context_raw": None,
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
            full_message = state.last_refined_context + last_gen if state.last_refined_context else last_gen

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
        description="Agentic context refinement with tool-calling using vLLM Data Parallelism"
    )
    parser.add_argument(
        "--dataset", type=str, choices=["hmmt", "imobench"], required=True,
        help="Dataset to use"
    )
    parser.add_argument("--input_file", type=str, default=None,
                        help="Input JSONL file (required for hmmt)")
    parser.add_argument("--cache_dir", type=str,
                        default="/scratch/dkhasha1/tli104/imobench")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--num_tokens", type=int, required=True,
                        help="Max tokens per generation step")
    parser.add_argument("--output_file", type=str, default="output_agentic.jsonl")
    parser.add_argument("--max_rounds", type=int, default=12,
                        help="Maximum number of generation rounds (default: 12)")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of samples per question (default: 1)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--max_refinement_tokens", type=int, default=None,
                        help="Max tokens for refinement (default: 16384)")
    parser.add_argument("--preserve_answer", action="store_true", default=True)
    parser.add_argument("--strip_answer", action="store_true")
    parser.add_argument("--disable_thinking_for_refinement", action="store_true")
    parser.add_argument("--strip_thinking_from_refinement", action="store_true", default=True)
    parser.add_argument("--keep_thinking_in_refinement", action="store_true")
    parser.add_argument("--strip_thinking_from_generation", action="store_true", default=True)
    parser.add_argument("--keep_thinking_in_generation", action="store_true")
    parser.add_argument("--num_gpus", type=int, default=None)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--system_prompt", type=str, default=None,
                        help="Override default system prompt")

    args = parser.parse_args()

    if args.max_refinement_tokens is None:
        args.max_refinement_tokens = 16384

    if args.strip_answer:
        args.preserve_answer = False
    if args.keep_thinking_in_refinement:
        args.strip_thinking_from_refinement = False
    if args.keep_thinking_in_generation:
        args.strip_thinking_from_generation = False

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
    print(f"Agentic Refinement - Data Parallel Inference")
    print(f"{'='*80}")
    print(f"Available GPUs: {num_gpus}")
    print(f"Using GPUs: {args.num_gpus}")
    print(f"Model: {args.model}")
    print(f"Max Model Length: {args.max_model_len}")
    print(f"Num Tokens per step: {args.num_tokens}")
    print(f"Max Refinement Tokens: {args.max_refinement_tokens}")
    print(f"Max Rounds: {args.max_rounds}")
    print(f"Num Samples: {args.num_samples}")
    print(f"Preserve Answer: {args.preserve_answer}")
    print(f"Disable Thinking for Refinement: {args.disable_thinking_for_refinement}")
    print(f"Strip Thinking from Generation: {args.strip_thinking_from_generation}")
    print(f"Strip Thinking from Refinement: {args.strip_thinking_from_refinement}")
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
