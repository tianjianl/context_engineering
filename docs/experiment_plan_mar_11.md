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

## Root Cause Analysis (updated Mar 12)

### Three Gaps Between RC User and RL

**Gap 1: Continuation context is degraded** — RC user gives rich structured prompts with improvement strategies; RL gives terse `<tool_response>` wrapper. → Fixed in Step 1 (prompt port).

**Gap 2: Summarization prompt is weaker** — RC user's 10-line prompt vs RL's 4-line version. → Fixed in Step 1 (prompt port). Step 1b confirmed summarizer quality is NOT a bottleneck (+0.75% with API vs self).

**Gap 3: No warmstart for "when to call"** — RL expects the model to learn tool-call timing from scratch, but GRPO can't provide signal because:

### Why RL alone can't learn tool-call timing (Mar 12 analysis)

Analysis of rollout 9 from job 1149793 (no bonus, 20h training):

1. **89-91% of tool calls come AFTER `\boxed{answer}`** — the model solves first, then appends tool calls as afterthoughts. This makes tool calls **reward-neutral**: correct+tool = +1, correct+no-tool = +1, same GRPO advantage.

2. **Difficulty is not the bottleneck.** Group-level stats show 57% of groups are mixed (learnable signal exists), and 50/256 groups (20%) have the ideal setup (tool+correct AND notool+wrong in same group).

3. **But of those 50 "ideal" groups, only 4 called tool proactively (before `\boxed`).** The other 46 are spurious correlation — model happened to get it right AND called tool after.

4. **Net signal: 4/256 groups (1.6%) have genuine proactive tool-call signal** — far too sparse for GRPO to learn from. This is why the SFT warmstart is essential.

## Plan

### Step 1: Close the prompt gap — ✅ DONE

Ported RC user prompts into `training/tool_refinement_rl/prompts.py`:
1. ✅ Replaced `SYSTEM_PROMPT` with clearer trigger ("call when you have completed a full attempt or are stuck")
2. ✅ Replaced `CONTINUATION_INSTRUCTIONS` with RC user's 5-bullet strategy list
3. ✅ Replaced `create_summarization_prompt` with RC user's rich 10-line version

**RL jobs with updated prompts (completed):**

| Job ID | Config | AIME Step 0 → Best | Status |
|--------|--------|---------------------|--------|
| 1149793 | No tool bonus | 8.3% → 23.8% | ✅ Done (54 rollouts) |
| 1149796 | Per-call bonus (0.15/call, cap 0.5) | 9.5% → 47.8% | ✅ Done (47 rollouts) |

**Conclusion:** Per-call bonus is the strongest RL config from base model. But tool calling is still post-`\boxed` afterthoughts, not proactive. SFT warmstart needed.

### Step 1b: Summarizer quality ablation — ✅ COMPLETE

| Round | Self (Qwen3-4B) | API (Gemini 2.5 Flash) | Delta |
|-------|-----------------|----------------------|-------|
| 1 | 35.44% | 35.69% | +0.25% |
| 2 | 38.94% | 38.88% | -0.06% |
| 3 | 40.12% | 41.44% | +1.32% |
| 4 | 40.94% | 41.69% | +0.75% |

**Conclusion: Summarizer quality is NOT the bottleneck.**

### Step 2: SFT warmstart from RC user data — ✅ COMPLETE

**Pipeline:**
1. ✅ Sample hard problems from `polaris_filtered_removed_all_correct` (38K total)
2. ✅ Run 4-step RC user rollouts (n=4 samples/problem) — batch 1 (2K) + batch 2 (2K)
3. ✅ Gemini annotation via OpenRouter
4. ✅ Build SFT dataset: `scripts/build_rc_annotated_sft.py`
5. ✅ Batch 3 (16K problems, n=16, rejection sampling) — in progress

### Step 2b: SFT training — ✅ COMPLETE

**Two SFT runs completed:**

| Job ID | Template | Train Loss | Eval Loss | Status |
|--------|----------|------------|-----------|--------|
| 1151857 | `qwen3` (broken) | 0.1634 | 0.1613 | ✅ Done, 3 epochs, 126 steps |
| 1152164 | `qwen3_nothink` (fixed) | 0.1523 | 0.1577 | ✅ Done, 3 epochs, 156 steps |

**Critical finding:** The `qwen3` template injects empty `<think>\n\n</think>\n\n` into every assistant turn during training (since SFT data has no think tags). This corrupted the model's token distribution, causing garbage outputs and degraded reasoning. Fix: `qwen3_nothink` template.

**HMMT eval results (pass@1, n=16, 12-round tool refinement):**

| Model | HMMT Feb | HMMT Nov | Avg |
|-------|----------|----------|-----|
| Base (Qwen3-4B-Instruct-2507) | 29.58% | 38.54% | 34.06% |
| Old template SFT Step 30 | 10.00% | 16.04% | 13.02% |
| Old template SFT Final | 22.08% | 30.83% | 26.46% |
| **Nothink SFT Step 30** | 29.17% | 42.29% | 35.73% |
| **Nothink SFT Final** | **32.29%** | **44.79%** | **38.54%** |

**Conclusion:** Nothink SFT final improves over base by +4.5% avg on HMMT. Tool call count increased from 92 (step 30) to 246 (final), showing SFT teaches the model to use tools more. This is the best checkpoint for RL.

### Step 3: RL from SFT checkpoint — ⏳ RUNNING

Starting RL from nothink SFT final checkpoint. Three bonus ablations on H200.

| Job ID | Config | Status |
|--------|--------|--------|
| 1158888 | convert_nothink (HF→torch_dist) | ✅ Running |
| 1158889 | No tool bonus — pure correctness | ⏳ Pending (afterok:1158888) |
| 1158890 | Fixed binary bonus (correct+0.3, wrong+0.5) | ⏳ Pending (afterok:1158888) |
| 1158891 | Per-call bonus (0.15/call, cap 0.5) | ⏳ Pending (afterok:1158888) |

**Hypothesis:** With the SFT warmstart teaching proactive tool calling, the GRPO signal should be much stronger than from base model. Per-call bonus was best from base (9.5% → 47.8%), but the SFT model may not need the bonus since it already knows when to call.

**What to watch:**
- Do tool calls happen BEFORE `\boxed` answers? (SFT should fix the post-answer problem)
- Does tool-using accuracy match or exceed no-tool accuracy?
- Which bonus config converges fastest?

### Step 3b: Batch 3 data synthesis — ⏳ IN PROGRESS

Large-scale data generation: 16K problems × n=16 = 256K trajectories. Rejection sampling (no Gemini annotation).

| Shard | Job ID | Status |
|-------|--------|--------|
| s0 (0-1999) | 1151572 | ✅ Done |
| s1 (2000-3999) | 1152144 | ⏳ Pending (MaxGRESPerAccount) |
| s2 (4000-5999) | 1151574 | ✅ Running, step 3/4 |
| s3 (6000-7999) | 1151575 | ✅ Running on c001 |
| s4-s7 | 1151576-1151579 | ⏳ Pending |
| filter | 1151580 | ⏳ Pending (after all shards) |

### Step 4: Summarizer quality — ❌ NOT NEEDED

Step 1b showed API summarizer gives only +0.75% over self-summarization. Eliminated.

## Execution Timeline

| Step | Status | Next action |
|------|--------|-------------|
| 1. Prompt gap fix | ✅ Done | Confirmed prompt fix alone can't teach tool timing |
| 1b. Summarizer ablation | ✅ Done | **Not a bottleneck** — API only +0.75% over self |
| 2. SFT data generation | ✅ Batch 1+2 done, ⏳ Batch 3 | Wait for batch 3 shards to finish |
| 2b. SFT training | ✅ Done | Nothink SFT final is best checkpoint (+4.5% HMMT avg) |
| 3. RL from SFT | ⏳ Running | Monitor 3 bonus ablations (1158889-1158891) |
| 4. Summarizer fix | ❌ Not needed | Eliminated by Step 1b |

## Success Metrics

- **Primary**: Tool-using rollouts achieve accuracy >= no-tool rollouts (was 38% vs 42% in base RL)
- **SFT milestone**: ✅ HMMT accuracy with tool refinement > baseline at nothink SFT final
- **Secondary**: Model calls tool on 20-50% of problems (selective, not always/never)
- **Target**: Match or exceed RC user's +5.2% gain on IMOBench, but autonomously
