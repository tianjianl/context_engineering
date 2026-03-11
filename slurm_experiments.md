## 2026-03-11 RC User Prompt Fix — Tool Refinement RL

Ported RC user inference prompts into RL training pipeline to close the gap between the working
RC user pipeline (+5.2% on IMOBench) and the RL tool-calling setup. Changes to prompts.py:
- Richer summarization prompt (from rc/inference/prompts/summarization_prompt.txt)
- Explicit improvement strategies in continuation instructions (verify, prove differently, find alternatives)
- Clearer tool-calling trigger: "when you have completed a full attempt or are stuck" instead of "after each major reasoning step"
- Better tool description matching the trigger logic

Two ablation runs: one with binary tool bonus, one without, to isolate the effect of reward shaping.

wandb project: tool_rl_2507. Correctness and tool bonus logged separately (reward/correctness_reward, reward/tool_bonus).

| Job ID | Job Name | Partition | Description |
|--------|----------|-----------|-------------|
| 1149793 | rc_user_tool_call | h200 | RC user prompts, **no tool bonus**, no std norm, 4×H200 |
| 1149794 | rc_user_tc_bonus | a100 | RC user prompts, binary tool bonus, 8×A100 — **FAILED** (no CUDA on c012) |
| 1149797 | rc_user_tc_bonus | h200 | RC user prompts, **binary tool bonus** (correct+0.3, wrong+0.5), no std norm, 4×H200 |
| 1149796 | rc_user_per_call | a100 | RC user prompts, **per-call bonus** (0.15/call, cap 0.5), no std norm, 8×A100 |

## 2026-03-11 RC Inference — Summarizer Quality Ablation (NVL)

Tests whether summarizer quality is the bottleneck in the RC pipeline. Reasoning model is
Qwen3-4B-Instruct-2507 in both conditions. Only the summarizer differs.

| Job ID | Job Name | Partition | Description |
|--------|----------|-----------|-------------|
| 1149798 | rc_imobench_self | nvl | IMOBench, self-summarization (Qwen3-4B-Instruct-2507), 4 steps, n=4 |
| 1149799 | rc_imobench_api | nvl | IMOBench, API summarization (Gemini 2.5 Flash via OpenRouter), 4 steps, n=4 |

## 2026-03-11 Tool Refinement RL — Per-Call Bonus (A100)

Same per-call bonus config as job 1140635 but on A100 (8 GPUs). TP=2, CP=4. Tests whether the per-call reward signal works on A100 hardware with higher parallelism.

Reward structure: correct + N calls → +1.0 + min(N×0.15, 0.5), wrong + N calls → -1.0 + min(N×0.15, 0.5).

| Job ID | Job Name | Partition | Description |
|--------|----------|-----------|-------------|
| 1149790 | tool_rl_per_call_a100 | a100 | Per-call bonus (0.15/call, cap 0.5), no SFT, no std norm, 8×A100 — CANCELLED (wandb disabled) |
| 1149792 | tool_rl_per_call_a100 | a100 | Per-call bonus (0.15/call, cap 0.5), no SFT, no std norm, 8×A100, wandb enabled, tool_ref/ panel metrics |

## 2026-03-10 Tool Refinement RL — Per-Call Bonus (H200)

Per-call tool bonus instead of binary. Bonus scales with number of tool calls: 0.15 per call, capped at 0.5. Addresses model converging to exactly 1 tool call (binary bonus gives no gradient for additional calls). Same setup as job 1131512 (no SFT, no std norm, Polaris filtered) but with per-call reward.

Reward structure: correct + N calls → +1.0 + min(N×0.15, 0.5), wrong + N calls → -1.0 + min(N×0.15, 0.5).

| Job ID | Job Name | Partition | Description |
|--------|----------|-----------|-------------|
| 1140635 | tool_rl_per_call_bonus | h200 | Per-call bonus (0.15/call, cap 0.5), no SFT, no std norm |

## 2026-03-10 IMOBench Eval — RL Checkpoints (NVL)

IMOBench eval on latest RL checkpoints. Checkpoint conversion (Megatron distcp → HF) runs first in each script. Params match training eval setup: temp=1.0, top_p=0.7, num_tokens=16384, tp=1.

| Job ID | Job Name | Model | Eval Type | n_samples | max_rounds |
|--------|----------|-------|-----------|-----------|------------|
| 1135636 | eval_grpo_polaris | grpo_polaris_nostd iter39 | baseline | 10 | — |
| 1135637 | eval_tool_bonus | tool_rl_no_sft_bonus iter19 | tool refinement (compact) | 10 | 12 |
| 1135638 | eval_tool_nobonus | tool_rl_no_sft_nobonus iter39 | tool refinement (compact) | 10 | 12 |

## 2026-03-09 Tool Refinement RL — Bonus LR Sweep (H200)

Same as job 1127063 (no SFT, with tool bonus +0.3/+0.5, no std norm, Polaris filtered) but sweeping learning rate.

| Job ID | Job Name | Partition | LR | Description |
|--------|----------|-----------|----|-------------|
| 1132833 | tool_rl_bonus_lr5e7 | h200 | 5e-7 | Tool bonus, lr=5e-7 |
| 1132834 | tool_rl_bonus_lr2e6 | h200 | 2e-6 | Tool bonus, lr=2e-6 |

## 2026-03-09 GRPO Polaris Baseline — No Std Normalization (A100)

Standard GRPO (no tool refinement) on Polaris filtered dataset (38K problems). No GRPO std normalization. Qwen3-4B-Instruct-2507, 8× A100.

| Job ID | Job Name | Partition | Description |
|--------|----------|-----------|-------------|
| 1131500 | grpo_polaris_nostd | a100 | GRPO baseline, Polaris 38K, no std norm, 8×A100 |

## 2026-03-06 Tool Refinement RL — No SFT, Base Qwen3-4B-Instruct-2507 (H200)

RL directly from base model (no SFT checkpoint). Tool description included via custom generate function. No GRPO std normalization. Polaris filtered dataset.

| Job ID | Job Name | Partition | Description |
|--------|----------|-----------|-------------|
| 1131512 | tool_rl_no_sft_bonus | h200 | No SFT, with tool bonus (TOOL_BONUS_CORRECT=0.3, TOOL_BONUS_WRONG=0.5) |
| 1132642 | tool_rl_no_sft_nobonus | h200 | No SFT, no tool bonus |
