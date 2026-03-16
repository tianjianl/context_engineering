## 2026-03-14 Tool Refinement RL — from nothink SFT final (H200)

RL from nothink SFT final checkpoint (rc_annotated_sft_nothink_checkpoints, job 1152164).
Three bonus ablations. Polaris filtered dataset, no GRPO std norm, lr=1e-6, 4×H200.
wandb project: tool_rl_2507.

| Job ID | Job Name | Partition | Description |
|--------|----------|-----------|-------------|
| 1158885-87 | sft_rl_* | h200 | **FAILED** — race condition on torch_dist conversion |
| 1158888 | convert_nothink | h200 | HF→torch_dist conversion (standalone) |
| 1158889 | sft_rl_no_bonus | h200 | No tool bonus — pure correctness reward (afterok:1158888) |
| 1158890 | sft_rl_fixed_bonus | h200 | Fixed binary bonus (correct+0.3, wrong+0.5) (afterok:1158888) |
| 1158891 | sft_rl_per_call | h200 | Per-call bonus (0.15/call, cap 0.5) (afterok:1158888) |

## 2026-03-13 HMMT Eval — nothink SFT Checkpoints with Tool Refinement (12 rounds)

Eval nothink SFT checkpoints on HMMT Feb + Nov (30 problems each) with autonomous tool refinement.
n=16, temp=1.0, max_rounds=12, compact_context. Script: `eval_sft_hmmt_nothink.sh`.
Checkpoints from job 1152164 (`rc_annotated_sft_nothink_checkpoints`).

| Job ID | Job Name | Partition | Description |
|--------|----------|-----------|-------------|
| 1158801-1158804 | eval/verify_hmmt_nothink | nvl | **CANCELLED** (switched to H200) |
| 1158835-1158842 | eval/verify_hmmt_nothink | h200 | **CANCELLED** (needed GPU count fix) |
| 1158846 | eval_hmmt_nothink | h200 | nothink SFT checkpoint step 30, 4×H200 |
| 1158847 | verify_hmmt_nothink | cpu | Verify step 30 outputs (afterok:1158846) |
| 1158848 | eval_hmmt_nothink | h200 | nothink SFT final (3 epochs, step 156), 4×H200 |
| 1158849 | verify_hmmt_nothink | cpu | Verify final outputs (afterok:1158848) |

## 2026-03-13 SFT Training — qwen3_nothink template fix

Prior SFT runs used `template: qwen3` which injects empty `<think>\n\n</think>\n\n` into every
assistant turn during training (since SFT data has no think tags). This corrupted the model's
token distribution, causing garbage outputs (Thai chars, nonsense) and degraded reasoning.
Fix: switch to `template: qwen3_nothink` which skips think tag injection.

| Job ID | Job Name | Partition | Description |
|--------|----------|-----------|-------------|
| 1152160 | rc_sft_nothink | h200 | **FAILED** — wandb 403 on h204 |
| 1152164 | rc_sft_nothink | h200 | SFT with qwen3_nothink, wandb disabled, exclude h203/h204/h205, output: rc_annotated_sft_nothink_checkpoints |

## 2026-03-13 HMMT Eval — SFT Checkpoints with Tool Refinement (12 rounds)

Eval SFT checkpoints on HMMT Feb + Nov (30 problems each) with autonomous tool refinement.
n=16, temp=1.0, max_rounds=12, compact_context. Script: `eval_sft_hmmt_tool.sh`.

| Job ID | Job Name | Partition | Description |
|--------|----------|-----------|-------------|
| 1151884 | eval_hmmt_tool | nvl | Base model (Qwen3-4B-Instruct-2507) |
| 1151885 | eval_hmmt_tool | nvl | SFT checkpoint step 30 (~epoch 0.72) |
| 1151886 | eval_hmmt_tool | nvl | SFT final (top-level merged weights) — **CANCELLED** (vLLM flash_attn crash) |
| 1152118 | eval_hmmt_tool | nvl | SFT final (resubmit of 1151886) |
| 1152145 | verify_hmmt_eval | cpu | Verify base + step30 eval outputs (fix for missing --dataset arg) |
| 1152146 | verify_hmmt_final | cpu | Verify final eval outputs (afterok:1152118) |

## 2026-03-13 SFT Training — 32K filtered (relaunch with Liger kernel)

Prior runs OOMed on cross_entropy_loss (56 GiB logits allocation for 150K vocab × 32K seq).
Fix: `enable_liger_kernel: true` for fused/chunked cross-entropy.

| Job ID | Job Name | Partition | Description |
|--------|----------|-----------|-------------|
| 1151855 | rc_annotated_sft | h200 | **FAILED** — liger-kernel not installed |
| 1151857 | rc_annotated_sft | h200 | SFT training: 3 epochs, lr=5e-6, cutoff_len=32768, DeepSpeed Z2, Liger kernel, wandb=rc_annotated_sft_liger |

## 2026-03-12 RC SFT Batch 3 — 16K Problems × n=16, Rejection Sampling

Large-scale RC user rollout generation for SFT data. 16K problems from Polaris (excluding 4K from
batches 1+2), n=16 trajectories per problem = 256K total trajectories. 4-step RC with self-summarization
using Qwen3-4B-Instruct-2507.

No Gemini annotation — uses rejection sampling via math-verify: keeps trajectories where round 1 was
wrong but a later round was correct. Filter script: `scripts/filter_rc_rollouts.py`.

| Job ID | Job Name | Partition | Description |
|--------|----------|-----------|-------------|
| 1151571 | rc_sft_prep_b3 | cpu | Data prep: sample 16K problems, exclude batch 1+2 |
| 1151572 | rc_sft_rollout_b3_s0 | a100 | Rollout shard 0 (problems 0-1999), DP=8 TP=1, n=16 |
| 1151573 | rc_sft_rollout_b3_s1 | a100 | Rollout shard 1 (problems 2000-3999) — **FAILED** (no CUDA on c013) |
| 1152144 | rc_sft_rollout_b3_s1 | a100 | Rollout shard 1 resubmit (exclude=c013) |
| 1151574 | rc_sft_rollout_b3_s2 | a100 | Rollout shard 2 (problems 4000-5999) |
| 1151575 | rc_sft_rollout_b3_s3 | a100 | Rollout shard 3 (problems 6000-7999) |
| 1151576 | rc_sft_rollout_b3_s4 | a100 | Rollout shard 4 (problems 8000-9999) |
| 1151577 | rc_sft_rollout_b3_s5 | a100 | Rollout shard 5 (problems 10000-11999) |
| 1151578 | rc_sft_rollout_b3_s6 | a100 | Rollout shard 6 (problems 12000-13999) |
| 1151579 | rc_sft_rollout_b3_s7 | a100 | Rollout shard 7 (problems 14000-15999) |
| 1151580 | rc_sft_filter_b3 | cpu | Rejection sampling + SFT build (after all rollouts) |

## 2026-03-12 SFT Training — 32K filtered (relaunch)

Prior run (job 1151362) OOMed on 63K-token examples. Filtered dataset to ≤32K tokens:
1338 train / 147 val (102 dropped). save_steps/eval_steps changed 14→10. ~41 steps/epoch, 123 total.

| Job ID | Job Name | Partition | Description |
|--------|----------|-----------|-------------|
| 1151495 | rc_annotated_sft | h200 | SFT training: 3 epochs, lr=5e-6, cutoff_len=32768, DeepSpeed Z2, gradient_checkpointing |

## 2026-03-12 SFT Training Pipeline (auto-chained) — FAILED

Build combined SFT dataset from batch 1+2, train, eval checkpoints. Auto-chained via dependencies.
Job 1151362 (SFT) **OOMed** — 105 GiB alloc on logits for long examples. Eval jobs cancelled (DependencyNeverSatisfied).

| Job ID | Job Name | Partition | Description |
|--------|----------|-----------|-------------|
| 1151349 | build_sft_data | cpu | Build combined dataset (after 1151309), then submit SFT + eval jobs |
| 1151362 | rc_annotated_sft | h200 | **FAILED (OOM)** — SFT training, unfiltered data (max 63K tokens) |
| 1151363-1151366 | eval_sft_ckpt ×4 | nvl | **CANCELLED** (DependencyNeverSatisfied) |

## 2026-03-12 RC User + Baseline on ProofBench-HF with Qwen3.5-9B

First RC user eval with a larger model. Qwen3.5-9B on ProofBench-HF (1088 problems).
Params: n=4, temp=1.0, top_p=1.0, max_thinking_tokens=16384, TP=2, use_think_tags.
Dataset: `proofbench_hf_rc_format.json`.

| Job ID | Job Name | Partition | Description |
|--------|----------|-----------|-------------|
| 1151313 | rc_proofbench_9b | nvl | Baseline (1-step), Qwen3.5-9B |
| 1151314 | rc_proofbench_9b | nvl | RC 4-step, Qwen3.5-9B |

## 2026-03-12 RC User Rollouts Batch 2 + SFT Data Build

Batch 2: 2000 more problems (seed=123, excluding batch 1) for more tool-calling SFT examples.
Batch 1 yielded 301 tool + 1000 no-tool examples. Expect ~300 more from batch 2.

SFT conversion script: `scripts/build_rc_annotated_sft.py` — uses Gemini annotations to build
sharegpt-format tool-calling SFT data from RC rollouts.

| Job ID | Job Name | Partition | Description |
|--------|----------|-----------|-------------|
| 1151302 | rc_sft_rollout_b2 | nvl | RC rollouts batch 2 shard 0 (problems 0-499) |
| 1151303 | rc_sft_rollout_b2 | nvl | RC rollouts batch 2 shard 1 (problems 500-999) |
| 1151304 | rc_sft_rollout_b2 | nvl | RC rollouts batch 2 shard 2 (problems 1000-1499) |
| 1151305 | rc_sft_rollout_b2 | nvl | RC rollouts batch 2 shard 3 (problems 1500-1999) |
| 1151306 | rc_sft_annotate_b2 | cpu | Gemini annotation batch 2 shard 0 via OpenRouter (after 1151302) |
| 1151307 | rc_sft_annotate_b2 | cpu | Gemini annotation batch 2 shard 1 via OpenRouter (after 1151303) |
| 1151308 | rc_sft_annotate_b2 | cpu | Gemini annotation batch 2 shard 2 via OpenRouter (after 1151304) |
| 1151309 | rc_sft_annotate_b2 | cpu | Gemini annotation batch 2 shard 3 via OpenRouter (after 1151305) |

## 2026-03-11 RC User Rollouts for SFT Data Synthesis

Generate fresh RC user rollouts on Polaris hard problems (removed all-correct) for SFT warmstart data.
Dual approach: (1) rejection sampling — grade each round to find where refinement helped,
(2) Gemini annotation — API annotates rollout quality per round.

Dataset: 2000 problems sampled from polaris_filtered_removed_all_correct (38K total).
Model: Qwen3-4B-Instruct-2507, 4 steps, n=4, temp=0.7. Rich RC user prompts.

| Job ID | Job Name | Partition | Description |
|--------|----------|-----------|-------------|
| 1150686 | rc_sft_prep | cpu | Data prep: sample 2000 problems, convert to RC format |
| 1150687 | rc_sft_rollout | nvl | RC rollouts shard 0 (problems 0-499), DP=2 TP=1 |
| 1150688 | rc_sft_rollout | nvl | RC rollouts shard 1 (problems 500-999), DP=2 TP=1 |
| 1150689 | rc_sft_rollout | nvl | RC rollouts shard 2 (problems 1000-1499), DP=2 TP=1 |
| 1150690 | rc_sft_rollout | nvl | RC rollouts shard 3 (problems 1500-1999), DP=2 TP=1 |
| 1150691 | rc_sft_annotate | cpu | Gemini annotation shard 0 (after 1150687) — **FAILED** (Gemini API geo-blocked on CPU nodes) |
| 1150692 | rc_sft_annotate | cpu | Gemini annotation shard 1 (after 1150688) — **FAILED** (same) |
| 1150693 | rc_sft_annotate | cpu | Gemini annotation shard 2 (after 1150689) — **FAILED** (same) |
| 1150694 | rc_sft_annotate | cpu | Gemini annotation shard 3 (after 1150690) — **FAILED** (same) |
| 1151296 | rc_sft_annotate | cpu | Gemini annotation shard 0 via OpenRouter (resubmit of 1150691) |
| 1151297 | rc_sft_annotate | cpu | Gemini annotation shard 1 via OpenRouter (resubmit of 1150692) |
| 1151298 | rc_sft_annotate | cpu | Gemini annotation shard 2 via OpenRouter (resubmit of 1150693) |
| 1151299 | rc_sft_annotate | cpu | Gemini annotation shard 3 via OpenRouter (resubmit of 1150694) |

## 2026-03-11 RC User Prompt Fix — Tool Bonus LR Sweep

Same RC user prompts + binary tool bonus, two LR variants submitted to both partitions.
Original job 1150685 (lr=1e-6, H200) cancelled in favor of these.

| Job ID | Job Name | Partition | LR | Description |
|--------|----------|-----------|----|-------------|
| 1150696 | rc_bonus_lr5e7 | h200 | 5e-7 | RC user prompts, binary tool bonus, lr=5e-7, 4×H200 — **CANCELLED** (hung on h203, resubmitted as 1150760) |
| 1150760 | rc_bonus_lr5e7 | h200 | 5e-7 | RC user prompts, binary tool bonus, lr=5e-7, 4×H200, exclude=h203,h205 |
| 1150697 | rc_bonus_lr2e6 | a100 | 2e-6 | RC user prompts, binary tool bonus, lr=2e-6, 8×A100 |

## 2026-03-11 Summarizer Quality Ablation Verification

| Job ID | Job Name | Partition | Description |
|--------|----------|-----------|-------------|
| 1150663 | verify_summarizer | cpu | Verify self vs API summarizer RC IMOBench rollouts |

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
| 1149797 | rc_user_tc_bonus | h200 | RC user prompts, **binary tool bonus** (correct+0.3, wrong+0.5), no std norm, 4×H200, TP=2/CP=2 — **CANCELLED** (resubmitted as 1150089 with TP=1/CP=2) |
| 1150089 | rc_user_tc_bonus | h200 | RC user prompts, **binary tool bonus**, no std norm, 4×H200, TP=1/CP=2 — **CANCELLED** (resubmitted as 1150090, allow h203) |
| 1150090 | rc_user_tc_bonus | h200 | RC user prompts, **binary tool bonus** (correct+0.3, wrong+0.5), no std norm, 4×H200, **TP=1/CP=2**, exclude=h205 only — **CANCELLED** (hung on h203, resubmitted as 1150685) |
| 1150685 | rc_user_tc_bonus | h200 | RC user prompts, **binary tool bonus** (correct+0.3, wrong+0.5), no std norm, 4×H200, TP=1/CP=2, exclude=h203 only |
| 1149796 | rc_user_per_call | a100 | RC user prompts, **per-call bonus** (0.15/call, cap 0.5), no std norm, 8×A100 |

## 2026-03-11 RC Inference — Summarizer Quality Ablation (NVL)

Tests whether summarizer quality is the bottleneck in the RC pipeline. Reasoning model is
Qwen3-4B-Instruct-2507 in both conditions. Only the summarizer differs.

| Job ID | Job Name | Partition | Description |
|--------|----------|-----------|-------------|
| 1149798 | rc_imobench_self | nvl | IMOBench, self-summarization (Qwen3-4B-Instruct-2507), 4 steps, n=4 |
| 1149799 | rc_imobench_api | nvl | IMOBench, API summarization (Gemini 2.5 Flash via OpenRouter), 4 steps, n=4 — **FAILED** (bashrc sourced after conda activate, overrode vllm env) |
| 1149953 | rc_imobench_api | nvl | IMOBench, API summarization (Gemini 2.5 Flash via OpenRouter), 4 steps, n=4 — resubmit of 1149799 |

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
