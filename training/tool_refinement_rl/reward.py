"""
Custom reward function for tool refinement RL training.

Uses DAPO math correctness reward with a tool-call bonus to encourage
the model to learn when/how to use the llm_refine tool.

Reward structure (DAPO base: +1 correct, -1 wrong):

Binary mode (TOOL_BONUS_CORRECT/TOOL_BONUS_WRONG, default):
  correct + tool:  +1.0 + 0.3 = +1.3   (best: solved AND used tool)
  correct, no tool: +1.0                 (good: solved without tool)
  wrong + tool:    -1.0 + 0.5 = -0.5    (tried tool, softer penalty)
  wrong, no tool:  -1.0                  (worst: wrong and didn't try)

Per-call mode (TOOL_BONUS_PER_CALL > 0):
  Scales bonus by number of tool calls with diminishing returns.
  correct + N calls: +1.0 + min(N * per_call, cap)
  wrong + N calls:   -1.0 + min(N * per_call, cap)
  Encourages multiple tool calls, not just one.

Format penalties (additive, capped at -0.5 total):
  garbage_tokens_at_start:  -0.3  (non-ASCII noise before reasoning)
  premature_tool_call:      -0.3  (<tool_call> before any reasoning)
  orphan_closing_tag:       -0.3  (starts with </tool_call> or </tool_response>)
  think_tag_bleed:          -0.1  (<think> tag in response)
"""

import os
import re
from slime.rollout.rm_hub.math_dapo_utils import compute_score as math_dapo_compute_score
from slime.utils.types import Sample

# Binary bonus (original mode): same bonus regardless of call count
TOOL_BONUS_CORRECT = float(os.environ.get("TOOL_BONUS_CORRECT", "0.0"))
TOOL_BONUS_WRONG = float(os.environ.get("TOOL_BONUS_WRONG", "0.0"))

# Per-call bonus: scales with number of tool calls (set > 0 to enable)
TOOL_BONUS_PER_CALL = float(os.environ.get("TOOL_BONUS_PER_CALL", "0.0"))
TOOL_BONUS_PER_CALL_CAP_CORRECT = float(os.environ.get("TOOL_BONUS_PER_CALL_CAP_CORRECT", "0.5"))
TOOL_BONUS_PER_CALL_CAP_WRONG = float(os.environ.get("TOOL_BONUS_PER_CALL_CAP_WRONG", "0.5"))

# Scale factor for format penalties (0.0 = disabled, 1.0 = full)
FORMAT_PENALTY_SCALE = float(os.environ.get("FORMAT_PENALTY_SCALE", "1.0"))
MAX_FORMAT_PENALTY = -0.5

# Non-ASCII garbage: CJK, Thai, Arabic, etc. — but allow \n, spaces, common punctuation
_GARBAGE_START_RE = re.compile(r"^[\s]*[^\x00-\x7F\s<(/\\$]")
# Premature tool call: <tool_call> with fewer than 50 chars of reasoning before it
_PREMATURE_TOOL_CALL_RE = re.compile(r"^[\s\S]{0,50}<tool_call>")


def _compute_format_penalty(response: str) -> tuple[float, list[str]]:
    """Check the model's first turn for format violations.

    Only inspects text before the first <tool_response> (i.e. the model's
    own first-turn output), since that's where all observed issues occur.

    Returns (penalty, list_of_issue_names).
    """
    # Isolate first turn: everything before the first tool_response observation
    first_turn = response.split("<tool_response>")[0] if "<tool_response>" in response else response
    stripped = first_turn.strip()

    penalty = 0.0
    issues = []

    # 1. Garbage tokens at start
    if _GARBAGE_START_RE.match(stripped):
        penalty -= 0.3
        issues.append("garbage_tokens_at_start")

    # 2. Premature tool call (calls tool before meaningful reasoning)
    if _PREMATURE_TOOL_CALL_RE.match(stripped):
        penalty -= 0.3
        issues.append("premature_tool_call")

    # 3. Orphan closing tags at start
    if stripped.startswith("</tool_call>") or stripped.startswith("</tool_response>"):
        penalty -= 0.3
        issues.append("orphan_closing_tag")

    # 4. <think> tag bleed (Qwen3 thinking mode leaking)
    if "<think>" in stripped[:200]:
        penalty -= 0.1
        issues.append("think_tag_bleed")

    # Cap total penalty
    penalty = max(penalty, MAX_FORMAT_PENALTY)
    return penalty, issues


async def reward_func(args, sample, **kwargs):
    """Compute reward based on final answer correctness + tool usage bonus + format penalty.

    Primary reward: DAPO binary correctness from \\boxed{} extraction.
    Tool bonus: added when model uses llm_refine, for both correct and wrong
    answers, to give a clear learning signal for tool calling.
    Format penalty: penalizes malformed outputs (garbage tokens, premature
    tool calls, orphan tags, think tag bleed).
    """
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    # Build complete solution string for answer extraction
    prompt_text = sample.prompt
    if isinstance(prompt_text, list):
        prompt_text = prompt_text[0]["content"]
    solution_str = prompt_text + sample.response

    # Ground truth from dataset label field
    ground_truth = sample.label if sample.label is not None else ""

    # DAPO correctness score (+1 correct, -1 wrong)
    result = math_dapo_compute_score(
        solution_str, ground_truth, strict_box_verify=True
    )

    # Store base correctness reward before any modifications
    correctness_reward = result["score"]

    num_tool_calls = getattr(sample, "tool_call_count", 0)
    used_tool = num_tool_calls > 0

    tool_bonus = 0.0
    if TOOL_BONUS_PER_CALL > 0 and num_tool_calls > 0:
        # Per-call mode: bonus scales with number of tool calls
        if result["score"] > 0:
            tool_bonus = min(num_tool_calls * TOOL_BONUS_PER_CALL, TOOL_BONUS_PER_CALL_CAP_CORRECT)
        elif result["score"] < 0:
            tool_bonus = min(num_tool_calls * TOOL_BONUS_PER_CALL, TOOL_BONUS_PER_CALL_CAP_WRONG)
        result["score"] += tool_bonus
    elif result["score"] > 0 and used_tool:
        # Binary mode: correct + tool bonus
        tool_bonus = TOOL_BONUS_CORRECT
        result["score"] += tool_bonus
    elif result["score"] < 0 and used_tool:
        # Binary mode: wrong + tool softer penalty
        tool_bonus = TOOL_BONUS_WRONG
        result["score"] += tool_bonus

    # Format penalty
    format_penalty = 0.0
    if FORMAT_PENALTY_SCALE > 0:
        format_penalty, format_issues = _compute_format_penalty(sample.response)
        format_penalty = format_penalty * FORMAT_PENALTY_SCALE
        result["score"] += format_penalty
        result["format_issues"] = format_issues

    # Store individual reward components for logging
    result["correctness_reward"] = correctness_reward
    result["tool_bonus"] = tool_bonus
    result["format_penalty"] = format_penalty
    result["used_tool"] = 1.0 if used_tool else 0.0
    result["num_tool_calls"] = float(num_tool_calls)

    if result["pred"] is None:
        result["pred"] = ""

    return result
