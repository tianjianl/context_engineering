# CLAUDE.md

## CRITICAL: Never run heavy scripts on login node

**DO NOT** run Python scripts that use significant memory/compute directly. Always submit via SLURM (use CPU partition for non-GPU tasks). Running on the login node can consume all resources and kill other users' work.

## SLURM

- **H200**: `--partition=h200 --qos=h200_8 --gres=gpu:4`
- **NVL**: `--partition=nvl --gres=gpu:2` (no QOS)
- **A100**: `--partition=a100 --gres=gpu:8` (8 GPUs per node, no QOS)
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

export SCRATCH_DIR=/scratch/dkhasha1/tli104
export HF_HOME=${SCRATCH_DIR}/hf_model_cache
export HF_DATASETS_CACHE=${SCRATCH_DIR}/hf_datasets_cache
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/dkhasha1/tli104/vllm
```

## Paths

- **Scratch**: `/scratch/dkhasha1/tli104/`
- **Outputs**: `/scratch/dkhasha1/tli104/outputs/`
- **SLURM Scripts**: `/scratch/dkhasha1/tli104/slurm_scripts/`
- **SLURM Logs**: `/scratch/dkhasha1/tli104/slurm_logs/`
- **Datasets**: `/scratch/dkhasha1/tli104/datasets/`
- **Model Cache**: `/scratch/dkhasha1/tli104/hf_model_cache`

## Conda Envs

- `/scratch/dkhasha1/tli104/vllm` — default (inference)
- `/scratch/dkhasha1/tli104/vllm_0_8_4` — always use for Qwen3-30B-A3B (MoE)
- `/scratch/dkhasha1/tli104/llamafactory` — LLaMA-Factory SFT

## Training

- **SFT**: `/weka/home/tli104/principia/LLaMA-Factory`, launch with `FORCE_TORCHRUN=1 llamafactory-cli train <yaml>`
- **RL**: `training/slime/` (GRPO), launch with `training/run_tool_refinement_rl.py`

## RL Eval Reward Scaling

The `eval/aime` metric is **mean reward**, not accuracy. To convert:

- **No tools** (GRPO baseline): reward = +1 correct, -1 wrong → `acc = (score + 1) / 2`
- **With tool bonus** (TOOL_BONUS_CORRECT=0.3, TOOL_BONUS_WRONG=0.5): correct+tool = 1.3, wrong+tool = -0.5 → `acc = (score + 0.5) / 1.8` (assuming ~100% tool usage)

## Model Names

Never abbreviate model names — they are unique identifiers. Always use the full name (e.g. `Qwen3-4B-Instruct-2507`, not `Qwen3-4B` or `4B`).

## Experiment Tracking

Log every `sbatch` submission in `slurm_experiments.md`.
