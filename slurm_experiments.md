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
