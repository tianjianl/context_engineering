"""
Custom generate function for tool refinement RL training with slime.

Implements a multi-turn loop where the model generates reasoning, optionally
calls the `llm_refine` tool (detected via <tool_call> tags), and receives a
summary from the same sglang engine as the tool response. Follows the retool
pattern (training/slime/examples/retool/generate_with_retool.py).
"""

import logging
import re

from prompts import (
    CONTINUATION_INSTRUCTIONS,
    LLM_REFINE_TOOL,
    MAX_ROUNDS,
    SYSTEM_PROMPT,
    create_summarization_prompt,
)

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

# Regex to detect a complete <tool_call>...</tool_call> block
TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*\{.*?\}\s*</tool_call>", re.DOTALL)


def postprocess_response(text: str) -> tuple[str, bool]:
    """Truncate response at the last complete <tool_call>...</tool_call> block.

    Returns:
        (truncated_text, has_tool_call)
    """
    matches = list(TOOL_CALL_PATTERN.finditer(text))
    if matches:
        last_match = matches[-1]
        return text[: last_match.end()], True
    return text, False


async def _call_summarizer(
    state: GenerateState,
    url: str,
    problem: str,
    existing_summary: str,
    latest_reasoning: str,
) -> str:
    """Call the same sglang engine to produce a summary of the reasoning.

    Uses a separate prompt (no tools) with lower temperature for consistent summaries.
    """
    summarization_text = create_summarization_prompt(
        problem, existing_summary, latest_reasoning
    )
    messages = [{"role": "user", "content": summarization_text}]
    text = state.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    token_ids = state.tokenizer(text, add_special_tokens=False)["input_ids"]

    summary_params = {
        "temperature": 0.3,
        "max_new_tokens": 2048,
        "top_p": 0.9,
        "no_stop_trim": True,
        "spaces_between_special_tokens": False,
    }

    payload = {
        "input_ids": token_ids,
        "sampling_params": summary_params,
    }
    output = await post(url, payload)

    if output["meta_info"]["finish_reason"]["type"] == "abort":
        logger.warning("Summarizer call aborted, falling back to existing summary")
        return existing_summary or "No summary available."

    return output["text"]


def _compute_observation_tokens(
    tokenizer, messages: list[dict], tools: list[dict], assistant_text: str, tool_response_content: str,
) -> list[int]:
    """Compute the observation tokens using the chat-template delta method.

    This ensures proper turn boundaries (<|im_end|>, <|im_start|>user, etc.)
    are included, matching what the model expects between multi-turn exchanges.
    Follows the tau-bench _get_token_delta pattern.

    Args:
        tokenizer: The tokenizer with apply_chat_template support.
        messages: Messages list BEFORE adding the assistant + tool response turns.
        tools: Tool definitions for apply_chat_template.
        assistant_text: The assistant's response text (reasoning + tool call).
        tool_response_content: The user message content (tool response + instructions).

    Returns:
        Token IDs for the observation (everything between model output and next generation).
    """
    # Text up to the model's output (before turn boundary)
    text_before = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=True
    )
    text_before_with_output = text_before + assistant_text

    # Text after adding assistant + user turns (with generation prompt for next round)
    messages_after = messages + [
        {"role": "assistant", "content": assistant_text},
        {"role": "user", "content": tool_response_content},
    ]
    text_after = tokenizer.apply_chat_template(
        messages_after, tools=tools, tokenize=False, add_generation_prompt=True
    )

    # The observation is the delta: everything after the model's output
    obs_text = text_after[len(text_before_with_output):]
    obs_token_ids = tokenizer(obs_text, add_special_tokens=False)["input_ids"]
    return obs_token_ids, obs_text


async def generate(args, sample: Sample, sampling_params: dict) -> Sample:
    """Custom generation function for tool refinement RL.

    Multi-turn loop:
      1. Generate reasoning (may include <tool_call>llm_refine</tool_call>)
      2. If tool call detected: summarize via same sglang engine, inject as observation
      3. Continue generation from the extended context
      4. Repeat until no tool call, max rounds, or length limit
    """
    assert not args.partial_rollout, (
        "Partial rollout is not supported for tool refinement."
    )

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    # Build initial prompt with tool definitions using native chat template.
    # sample.prompt may be a list of message dicts (from dataset) or a raw string.
    problem_text = sample.prompt
    if isinstance(problem_text, list):
        # Dataset stores prompt as [{"role": "user", "content": "..."}] — extract the text
        problem_text = problem_text[0]["content"]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem_text},
    ]
    tools = [LLM_REFINE_TOOL]
    prompt = state.tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_token_ids = state.tokenizer(prompt, add_special_tokens=False)["input_ids"]

    response = ""
    response_token_ids = []
    loss_masks = []
    tool_call_count = 0
    existing_summary = ""

    if sample.rollout_log_probs is None:
        sample.rollout_log_probs = []

    last_finish_reason = "stop"

    for turn in range(MAX_ROUNDS):
        # Check context length limit (retool lines 234-241)
        total_length = len(prompt_token_ids) + len(response_token_ids)
        if args.rollout_max_context_len is not None:
            max_context_length = args.rollout_max_context_len
        else:
            max_context_length = args.context_parallel_size * args.max_tokens_per_gpu
        if total_length >= max_context_length:
            sample.status = Sample.Status.TRUNCATED
            break

        # Generate reasoning via sglang (retool lines 244-272)
        current_token_ids = prompt_token_ids + response_token_ids
        payload = {
            "input_ids": current_token_ids,
            "sampling_params": sampling_params,
            "return_logprob": True,
        }
        output = await post(url, payload)

        # Handle abort (retool lines 275-277)
        if output["meta_info"]["finish_reason"]["type"] == "abort":
            sample.status = Sample.Status.ABORTED
            break

        last_finish_reason = output["meta_info"]["finish_reason"]["type"]

        # Extract response tokens and logprobs (retool lines 279-290)
        if "output_token_logprobs" in output["meta_info"]:
            cur_response_token_ids = [
                item[1] for item in output["meta_info"]["output_token_logprobs"]
            ]
            cur_response = state.tokenizer.decode(cur_response_token_ids)
            cur_log_probs = [
                item[0] for item in output["meta_info"]["output_token_logprobs"]
            ]
        else:
            cur_response = output["text"]
            cur_response_token_ids = state.tokenizer(
                cur_response, add_special_tokens=False
            )["input_ids"]
            cur_log_probs = [0.0] * len(cur_response_token_ids)

        # Truncate at tool_call boundary if present
        truncated_response, has_tool_call = postprocess_response(cur_response)

        if has_tool_call and truncated_response != cur_response:
            # Find truncation point in the original token sequence using binary
            # search.  Re-tokenizing the truncated text can produce a different
            # token count (BPE round-trip mismatch), so we instead find the
            # smallest prefix of the original tokens whose decoded text covers
            # the truncated response.
            target_len = len(truncated_response)
            lo, hi = 1, len(cur_response_token_ids)
            while lo < hi:
                mid = (lo + hi) // 2
                if len(state.tokenizer.decode(cur_response_token_ids[:mid])) >= target_len:
                    hi = mid
                else:
                    lo = mid + 1
            cur_response_token_ids = cur_response_token_ids[:lo]
            cur_log_probs = cur_log_probs[:lo]
            cur_response = state.tokenizer.decode(cur_response_token_ids)

        # Append model output tokens (loss_mask=1) (retool lines 292-294)
        response += cur_response
        response_token_ids += cur_response_token_ids
        loss_masks += [1] * len(cur_response_token_ids)
        sample.rollout_log_probs += cur_log_probs

        # Check length limit (retool lines 297-298)
        if last_finish_reason == "length" and not has_tool_call:
            break

        # If no tool call, the model produced a final answer — done
        if not has_tool_call:
            break

        # === Tool call detected: perform summarization ===
        tool_call_count += 1

        # Extract reasoning text (before the <tool_call> tag) for summarization
        reasoning_text = cur_response.split("<tool_call>")[0].strip()

        summary = await _call_summarizer(
            state, url, problem_text, existing_summary, reasoning_text
        )
        existing_summary = summary

        # Build tool response content for the user turn
        # Continuation instructions inside <tool_response> tags to match training data format
        tool_response_content = (
            f"<tool_response>\n{summary}"
            f"{CONTINUATION_INSTRUCTIONS}\n</tool_response>"
        )

        # Compute observation tokens using chat-template delta method.
        # This ensures proper turn boundaries (<|im_end|>, <|im_start|>user, etc.)
        obs_token_ids, obs_text = _compute_observation_tokens(
            state.tokenizer, messages, tools, cur_response, tool_response_content
        )
        response += obs_text
        response_token_ids += obs_token_ids
        loss_masks += [0] * len(obs_token_ids)

        # Update messages for next round's delta computation
        messages.append({"role": "assistant", "content": cur_response})
        messages.append({"role": "user", "content": tool_response_content})

        # Placeholder logprobs for observation tokens (retool lines 317-318)
        sample.rollout_log_probs += [0.0] * len(obs_token_ids)

        assert len(response_token_ids) == len(sample.rollout_log_probs), (
            f"Token/logp length mismatch at turn {turn}: "
            f"{len(response_token_ids)} tokens vs {len(sample.rollout_log_probs)} logps"
        )

        if tool_call_count >= MAX_ROUNDS:
            break

    # Assemble final sample (retool lines 327-349)
    sample.tokens = prompt_token_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_mask = loss_masks

    # Store tool call count for reward calculation
    sample.tool_call_count = tool_call_count

    # Set status
    if sample.status == Sample.Status.PENDING:
        match last_finish_reason:
            case "length":
                sample.status = Sample.Status.TRUNCATED
            case "abort":
                sample.status = Sample.Status.ABORTED
            case "stop":
                sample.status = Sample.Status.COMPLETED
            case _:
                sample.status = Sample.Status.COMPLETED

    return sample
