# Experiment Plan: Teaching Autonomous Tool Calling via RL (Mar 11)

## Problem Statement

The model does not know **when** to call `llm_refine`. Three failure modes:
1. **Skips the tool entirely** — gets +1.0 from correctness on easy problems, rationally avoids tool risk
2. **Calls prematurely** — emits `<tool_call>` before any meaningful reasoning (14% of rollouts)
3. **Spams tool calls** — degenerates into `<tool_call>\n<tool_call>\n<tool_call>` loops (18-call example)

Additionally, the summarizer (same 4B model) often copies context verbatim or loses critical information, so even when the tool is called the continuation is worse.

## Key Evidence: RC User Pipeline Works

The forced RC user inference pipeline achieves **40.81% pass@1 on IMOBench** vs **35.61% baseline** (+5.2%). Step-by-step progression:

| Step | pass@1 |
|------|--------|
| 1    | 34.88% |
| 2    | 38.50% |
| 3    | 40.19% |
| 4    | 40.81% |

The mechanism (summarize-and-restart) is validated. The problem is transferring it from a forced pipeline into autonomous RL behavior.

## Rollout Diagnosis

From `rollout_examples.jsonl` (37 examples, `tool_rl_nobonus` job):

| Condition        | Accuracy | Count |
|------------------|----------|-------|
| Without tool     | **64%**  | 11    |
| With tool        | **27%**  | 26    |

The tool **actively hurts accuracy** under the current RL setup. The model is rationally learning to avoid it.

### Format issues (first-turn)
- garbage_tokens_at_start: 38% of rollouts
- premature_tool_call: 14%
- orphan closing tags: 14%
- think_tag_bleed: 5%

## Root Cause: Three Gaps Between RC User and RL

### Gap 1: Continuation context is degraded

RC user gives a **fresh structured prompt** with explicit improvement strategies:
```
If a summary of a previous attempt is provided, your task is to improve upon this attempt.
Some strategies you could use include:
- Verifying the previous solution.
- Proving the result in a different way.
- Finding alternative problem-solving strategies.
- Continuing from where the previous solution left off.
```

RL gives a terse `<tool_response>` wrapper with:
```
Continue solving the problem, improving upon the summary above.
You may verify previous conclusions, try a different approach, or build on the progress so far.
```

The model gets much less guidance after a tool call in RL than in RC user.

### Gap 2: Summarization prompt is weaker

RC user's summarization prompt (10 lines) explicitly guides: "detailed overview," "relationship between solutions," "theorems used," "retain important information from existing summary." RL version (4 lines) is terse. Same 4B model, less guidance → worse summaries.

### Gap 3: No warmstart for "when to call"

RC user forces refinement. RL expects the model to learn from scratch when to emit `<tool_call>` — but the reward signal is too noisy (tool hurts accuracy → negative signal for calling). Chicken-and-egg: can't learn when to call a tool that doesn't help; tool doesn't help because the context is degraded.

## Plan

### Step 1: Close the prompt gap

Port RC user prompts into the RL pipeline. No retraining, just prompt changes.

**Changes to `training/tool_refinement_rl/prompts.py`:**

1. **Replace `SYSTEM_PROMPT`**: Instead of "call after each major reasoning step" (too vague), use a clearer trigger:
   ```
   You are a mathematical reasoning assistant. You have a tool `llm_refine` that
   summarizes your work so far and gives you a fresh context to continue.
   Call it when you have completed a full solution attempt and want to verify
   or improve it, or when you are stuck and want to try a different approach.
   Do NOT call it before you have done substantial reasoning.
   Present your final answer using \boxed{} notation.
   ```

2. **Replace `CONTINUATION_INSTRUCTIONS`** with RC user's `### INSTRUCTIONS` block:
   ```
   If a summary of a previous attempt is provided, your task is to improve
   upon this attempt. You should rely on this summary to guide your thinking.
   Some strategies you could use include:
   - Verifying the previous solution.
   - Proving the result in a different way.
   - Finding alternative problem-solving strategies.
   - Continuing from where the previous solution left off.
   Return your final answer in \boxed{}.
   ```

3. **Replace `create_summarization_prompt`** with RC user's richer version (the 10-line prompt from `rc/inference/prompts/summarization_prompt.txt`).

**Validation**: Run RL job with updated prompts, compare tool accuracy gap. If tool-using rollouts improve from 27% toward baseline, the prompt gap was a major factor.

### Step 2: SFT warmstart from RC user data

Generate training data that teaches the model both the tool-calling format and **when** to call.

1. **Generate RC user rollouts** on ~1000 hard problems (pass@1 < 50% from dapo-math-17k). Run forced 2-4 step RC pipeline. GPU job.

2. **Grade each step**: Check if step N's \boxed{} answer is correct when step N-1's was wrong. These are "refinement helped" examples.

3. **Convert to tool-calling format**:
   - **Positive examples** (refinement helped): First attempt → `<tool_call>` → summary as `<tool_response>` → improved solution with correct answer
   - **Negative examples** (solved on step 1): Single-turn solution, no tool call
   - Target mix: ~50/50 tool-use vs no-tool-use

4. **SFT** on this dataset using LLaMA-Factory. This teaches format + decision boundary simultaneously.

**Compute**: RC inference ~4 GPU-hours (H200), SFT ~2 GPU-hours.

### Step 3: RL from SFT checkpoint

Start RL from the SFT checkpoint with these changes:

1. **Remove all tool bonuses**: Set `TOOL_BONUS_CORRECT=0`, `TOOL_BONUS_WRONG=0`, `TOOL_BONUS_PER_CALL=0`. The SFT model already knows when to call. Let pure correctness (+1/-1) drive the learning — GRPO will reinforce tool calls that improve answers and suppress ones that don't.

2. **Curriculum filtering**: Only train on problems where base model pass@8 is 10-70%. Too easy → tool unnecessary. Too hard → tool can't help either.

3. **Keep format penalties** at current levels to suppress garbage tokens.

**Hypothesis**: With a proper SFT warmstart and no artificial bonuses, GRPO's within-group normalization will naturally teach the model to call the tool on hard problems (where tool-using rollouts score higher) and skip it on easy problems.

### Step 4: Summarizer quality (if needed)

If Step 1's prompt improvements don't close the summarizer quality gap:

- **Option A**: Snapshot the model before RL, use it as a frozen summarizer. Prevents summarizer degradation during policy updates.
- **Option B**: Use a stronger external model (Qwen3-32B or API) for summarization during RL rollouts. More expensive but guarantees summary quality.

Only pursue if Step 1 + Step 2 show that summarizer quality is still the bottleneck.

## Execution Timeline

| Step | Depends on | Estimated time | Compute |
|------|------------|----------------|---------|
| 1. Prompt gap fix | — | 1 day | 1 RL job (A100 8-GPU, ~24h) |
| 2. SFT warmstart data | — (parallel with 1) | 1-2 days | RC inference (H200 4-GPU, ~4h) + grading (CPU) + SFT (H200, ~2h) |
| 3. RL from SFT | Steps 1 + 2 | 2-3 days | 1 RL job (A100 8-GPU, ~48h) |
| 4. Summarizer fix | Only if 1-3 insufficient | 1 day | varies |

## Success Metrics

- **Primary**: Tool-using rollouts achieve accuracy >= no-tool rollouts (currently 27% vs 64%)
- **Secondary**: Model calls tool on 20-50% of problems (selective, not always/never)
- **Target**: Match or exceed RC user's +5.2% gain on IMOBench, but autonomously
