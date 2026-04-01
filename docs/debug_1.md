# Debug Session 1 — RL Training & Checkpoint Inference (2026-02-24)

## 1. GRPO Norm Removed (iter 219) — IMOBench Inference

**Checkpoint:** `Qwen3-4B-Instruct-2507_grpo_norm_removed_iter219` (from `grpo_norm_removed` RL run, 219 steps)

### Baseline (without tool) — DONE
- **pass@1 = 37.00%** on IMOBench (400 problems, n=16, t=16384)
- File: `baseline_grpo_norm_removed_iter219_imobench_t16384_n16_temp0.7_topp0.9.jsonl`
- Verified: `baseline_grpo_norm_removed_iter219_imobench_t16384_n16_verified.json`

### Tool Refinement (with tool) — PENDING (job 1062431, NVL)
- Original job 1062313 failed: script used `--num_rounds` instead of `--max_rounds`
- Resubmitted as job 1062431 with TWO variants:
  1. **Full context** (accumulated history — matches training format)
  2. **Compact context** (reset each round — mismatched with training)
- Both: n=16, r=12, t=8192
- Blocked waiting for NVL GPUs (polaris baseline chunk01/chunk08 using both slots)

### Training vs Inference Format Mismatch Found
- **Training** always accumulates full multi-turn conversation history (`messages.append()` each round)
- **Inference with `--compact_context`** resets conversation each round (only last turn + summary)
- System prompt, tool definition, summarization prompt, continuation instructions, tool call format — all match
- Only other diff: training max_rounds=5, inference max_rounds=12

## 2. Previous RL Runs — ALL BROKEN (Zero Gradient)

### grpo_tool (job 1061808) — 68 steps, WITH reward normalization
- Output: `tool_ref_rl_Qwen3-4B-Instruct-2507_grpo_tool/`
- WandB: `run-20260223_195300-acfwm00z` (h202)
- Checkpoint at iter 59

### grpo_no_norm_tool (job 1061756) — 79 steps, WITHOUT std normalization
- Output: `tool_ref_rl_Qwen3-4B-Instruct-2507_grpo_no_norm_tool/`
- WandB: `run-20260223_193326-ehieo63a` (h201)
- Checkpoint at iter 79

### Both runs show identical failure:
| Metric | grpo_tool (step 68) | grpo_no_norm_tool (step 79) |
|--------|--------------------|-----------------------------|
| raw_reward | 0.37 → 0.57 | 0.49 → 0.66 |
| rewards (normalized) | ~1e-9 (≈0) | 0.0 (exactly 0) |
| advantages | ~1e-9 (≈0) | ~1e-10 (≈0) |
| pg_loss | ~1e-9 (≈0) | ~1e-9 (≈0) |
| pg_clipfrac | 0.0 | 0.0 |
| ppo_kl | 0.0 | 0.0 |
| grad_norm | 0.11-0.16 | 0.10-0.25 |

**Root cause:** `rewards_normalization: True` in config for BOTH runs. Even `--disable-grpo-std-normalization` only disables the std division but global reward normalization still zeroes out everything. The model learned nothing across all 68-79 steps.

## 3. New Polaris RL Runs — STUCK in Initialization

### polaris_tool_rl (job 1062317, 2×H200 h201-h202)
- Dataset: polaris_filtered (38K problems)
- WITH reward normalization
- Running ~7h, only config printed (1087 lines), no training output
- Ray job submitted: `raysubmit_jZnkzVghbtx2mhHD`
- Training was submitted via `ray job submit` — logs go to Ray, NOT SLURM stdout
- No output directory created, no WandB runs found
- **Likely cause:** Ray job logs are in Ray's temp dir, WandB writes to Ray working dir not `/weka/.../training/wandb/`

### polaris_no_norm (job 1062327, 2×H100 h06-h07)
- Same dataset, WITH `--disable-grpo-std-norm`
- Running ~2.5h, same symptoms (only config, no training output)

### Key discovery: previous runs (grpo_tool, grpo_no_norm_tool) also ran on h201
- New polaris_tool_rl run overwrote the old Ray cluster on h201
- `pkill -9 sglang; pkill -9 slime` in startup script killed old processes
- Old runs' WandB logs (last modified Feb 23 22:44) stopped being updated when new runs took over

## 4. Currently Running Jobs

| Job ID | Name | Partition | Status | Description |
|--------|------|-----------|--------|-------------|
| 1062317 | polaris_tool_rl | h200 (2 nodes) | Running ~7h | RL training, stuck or logs in Ray |
| 1062327 | polaris_no_norm | h100 (2 nodes) | Running ~2.5h | RL training, stuck or logs in Ray |
| 1062304 | pol53k_base_chunk01 | nvl | Running ~7.5h | Polaris 53K baseline, 2665 prompts/worker |
| 1062305 | pol53k_base_chunk08 | nvl | Running ~7.5h | Polaris 53K baseline, 2665 prompts/worker |
| 1062431 | grpo_nr_tr | nvl | Pending | Tool refinement inference (waiting for NVL GPUs) |

## 5. TODO / Next Steps

1. **Debug polaris RL runs**: Check Ray job logs on h201/h06 to see if training actually started or hung
2. **Fix reward normalization**: Need to disable BOTH `rewards_normalization` AND `grpo_std_normalization` for the no_norm variant to actually work
3. **Wait for NVL GPUs**: Once polaris baseline chunks finish, job 1062431 will run tool refinement inference
4. **Create /rldoctor skill**: User requested a diagnostic skill for monitoring RL runs

## 6. Key File Locations

- Training script: `/weka/home/tli104/context_engineering/training/run_tool_refinement_rl.py`
- Generate function: `/weka/home/tli104/context_engineering/training/tool_refinement_rl/generate.py`
- Reward function: `/weka/home/tli104/context_engineering/training/tool_refinement_rl/reward.py`
- Prompts: `/weka/home/tli104/context_engineering/training/tool_refinement_rl/prompts.py`
- Inference: `/weka/home/tli104/context_engineering/inference/tool_refinement.py`
- WandB logs (old runs): `/weka/home/tli104/context_engineering/training/wandb/`
- SLURM scripts: `/scratch/dkhasha1/tli104/slurm_scripts/`
- Model: `/scratch/dkhasha1/tli104/models/Qwen3-4B-Instruct-2507_grpo_norm_removed_iter219`
