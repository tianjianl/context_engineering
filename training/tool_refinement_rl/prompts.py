"""
Prompts and tool definitions for tool refinement RL training.

Ported from the RC user inference pipeline (rc/inference/prompts/) which achieves
+5.2% on IMOBench. Key changes from the original RL prompts:
- Richer summarization prompt matching rc/inference/prompts/summarization_prompt.txt
- Explicit improvement strategies in continuation instructions matching
  rc/inference/prompts/reasoning_prompt.txt
- Clearer tool-calling trigger: "when you have completed a full attempt or are stuck"
  instead of "after each major reasoning step"
"""

# Tool definition for llm_refine
LLM_REFINE_TOOL = {
    "type": "function",
    "function": {
        "name": "llm_refine",
        "description": (
            "Summarize your progress and continue with a fresh start. "
            "Call when you have completed a full solution attempt and want to "
            "verify or improve it, or when you are stuck and want to try a "
            "different approach. Do NOT call before doing substantial reasoning."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

# System prompt — clearer trigger for when to call tool
SYSTEM_PROMPT = (
    "You are a mathematical reasoning assistant. You have a tool `llm_refine` that "
    "summarizes your work so far and gives you a fresh context to continue. "
    "Call it when you have completed a full solution attempt and want to verify "
    "or improve it, or when you are stuck and want to try a different approach. "
    "Do NOT call it before you have done substantial reasoning. "
    "Present your final answer using \\boxed{} notation."
)

# Continuation instructions — ported from rc/inference/prompts/reasoning_prompt.txt
CONTINUATION_INSTRUCTIONS = (
    "\n\nYou are given the summary of a previous attempt to solve this problem. "
    "This previous attempt may or may not be correct. Your task is to improve "
    "upon this attempt. You should rely on this summary to guide your thinking.\n"
    "Some strategies you could use include:\n"
    "- Verifying the previous solution.\n"
    "- Proving the result in a different way.\n"
    "- Finding alternative problem-solving strategies.\n"
    "- Continuing from where the previous solution left off, assuming that "
    "the previous solution is incomplete.\n\n"
    "Return your final answer in \\boxed{}."
)

# Max refinement rounds during RL training (lower than inference's 12 to keep sequences manageable)
MAX_ROUNDS = 5


def create_summarization_prompt(
    original_prompt: str, existing_summary: str, latest_reasoning: str
) -> str:
    """Create an RC-style summarization prompt.

    Ported from rc/inference/prompts/summarization_prompt.txt — the richer
    version that produces higher quality summaries in the RC user pipeline.
    """
    return f"""You are given a maths problem and a candidate solution to it. You may also be given a summary of a previous candidate solution to the problem. If this is provided, you may assume that the current candidate solution was generated conditioned on the summary of the previous candidate solution.
Your task is to write a summary of the current candidate solution.

The new summary you generate should possess the following characteristics:
- It should provide a detailed overview of what occurred in the current candidate solution. This may include a summary of the high-level problem-solving strategy, a description of theorems used, verification attempts, calculations and logical deductions etc.
- It should summarize the current candidate solution in light of any previous summaries, if provided. We should be able to understand the relationship between the previous solution and the current solution by reading the summary. Make sure any important information contained in the existing summary is retained in the new one.
- It should be no more than two paragraph long and written in paragraph form, without headers or subheaders.
- It should be written in the first person, as if though it is being written by the person solving the problem.
- The candidate solution may not be complete. In this case, the summary should still attempt to summarize the partial solution.

IMPORTANT: Do not under any circumstances add any additional reasoning not contained in the latest reasoning step. Your task is only to summarize what is given to you.

### PROBLEM
{original_prompt}

### EXISTING SUMMARY
{existing_summary}

### LATEST CANDIDATE SOLUTION
{latest_reasoning}"""
