# CLAUDE.md

## CRITICAL: Never run heavy scripts on login node

Always submit via SLURM (use CPU partition for non-GPU tasks).

## SLURM

- **H200**: `--partition=h200 --qos=h200_8 --gres=gpu:4`
- **NVL**: `--partition=nvl --gres=gpu:2` (no QOS)
- **A100**: `--partition=a100 --gres=gpu:8` (no QOS)
- **CPU**: `--partition=cpu --mem=64GB --cpus-per-task=8`
- Check jobs: `myjobs` or `sqme`; `checkjobs $jobid` for logs

### Template

```bash
#!/bin/bash
#SBATCH --job-name=<name>
#SBATCH --partition=h200
#SBATCH --qos=h200_8
#SBATCH --gres=gpu:4
#SBATCH --mem=256GB
#SBATCH --cpus-per-task=40
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/dkhasha1/tli104/slurm_logs/<name>_%j.out
#SBATCH --error=/scratch/dkhasha1/tli104/slurm_logs/<name>_%j.err
#SBATCH --exclude=h205

export SCRATCH_DIR=/scratch/dkhasha1/tli104
export HF_HOME=${SCRATCH_DIR}/hf_model_cache
export HF_DATASETS_CACHE=${SCRATCH_DIR}/hf_datasets_cache
source ~/miniconda3/etc/profile.d/conda.sh
# activate conda environment here

```

## Paths

- **Scratch**: `/scratch/dkhasha1/tli104/`
- **Outputs**: `/scratch/dkhasha1/tli104/outputs/`
- **SLURM Scripts**: `/scratch/dkhasha1/tli104/slurm_scripts/`
- **SLURM Logs**: `/scratch/dkhasha1/tli104/slurm_logs/`
- **Datasets**: `/scratch/dkhasha1/tli104/datasets/`

## Conda Envs

- `/scratch/dkhasha1/tli104/vllm` — default (inference)
- `/scratch/dkhasha1/tli104/vllm_0_8_4` — always use for Qwen3-30B-A3B (MoE)
- `/scratch/dkhasha1/tli104/llamafactory` — LLaMA-Factory SFT
- `/scratch/dkhasha1/tli104/slime` — SLIME RL training. Requires extra LD_LIBRARY_PATH:
  ```
  export LD_LIBRARY_PATH=/scratch/dkhasha1/tli104/slime/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:/scratch/dkhasha1/tli104/slime/lib:$LD_LIBRARY_PATH
  ```

## Training

- **SFT**: `/weka/home/tli104/principia/LLaMA-Factory`, launch with `FORCE_TORCHRUN=1 llamafactory-cli train <yaml>`
- **RL**: `training/slime/` (GRPO), launch with `training/run_tool_refinement_rl.py`

## RL Eval Reward Scaling

The `eval/aime` metric is **mean reward**, not accuracy. To convert:

- **No tools**: `acc = (score + 1) / 2`
- **With tool bonus** (TOOL_BONUS_CORRECT=0.3, TOOL_BONUS_WRONG=0.5): `acc = (score + 0.5) / 1.8`

## Model Names

Never abbreviate — always use the full name (e.g. `Qwen3-4B-Instruct-2507`).

## Job Name Prefixes

- **`fixed_`**: RC-style context truncation fix. Prior RL jobs accumulated the full conversation history across tool refinement rounds, causing unbounded context growth. `fixed_` jobs reset the generation context after each tool call to a compact prompt containing only the problem + latest summary (matching RC user inference). Changed in `training/tool_refinement_rl/generate.py` and `training/tinker-cookbook/tinker_cookbook/recipes/tool_refinement/env.py`.

## Git

- Do NOT add `Co-Authored-By: Claude` to commit messages.

## Experiment Tracking

Log every `sbatch` submission in `slurm_experiments.md`.

## Codebase Organization

### Directory Structure

- **`figures/`** — All plots, charts, and visualizations (PNG, PDF, SVG). Never put figures in the repo root or other directories.
- **`results/`** — Model outputs, benchmark CSVs, rollout JSONLs, and other experiment result files. Large outputs (full generation logs) go on scratch at `/scratch/dkhasha1/tli104/outputs/`.
- **`docs/`** — Experiment plans, analysis write-ups, and design documents. Keep the repo root free of planning `.md` files.
- **`scripts/`** — Standalone utility scripts (plotting, data conversion, log parsing, one-off analyses). If a script isn't part of `inference/` or `training/`, it belongs here.
- **`inference/`** — Inference methods and shared utilities (args, data loading, verification).
- **`training/`** — Training code (SFT data building, RL launch scripts, custom RL envs).
- **`analysis/`** — Analysis scripts for specific experiments (error categorization, correction checking).

### Adding New Features

Before writing new code, **always check whether an existing implementation already covers the need**. Specifically:

1. **Search `inference/` and `scripts/`** for similar functionality before creating a new script. Many patterns (data loading, argument parsing, verification, data parallelism) are already implemented in shared utilities (`args_utils.py`, `data_utils.py`, `dp_utils.py`, `verify_utils.py`).
2. **Reuse shared utilities** — don't duplicate argument parsing, dataset loading, or verification logic. Import from the existing modules.
3. **Extend, don't duplicate** — if an existing script does 80% of what you need, add the missing functionality to it (or extract a shared function) rather than writing a new script from scratch.
4. **New standalone scripts** go in `scripts/`, not the repo root.
