# CLAUDE.md

## CRITICAL RULES

1. **H200 jobs MUST use 4 GPUs**: `--gres=gpu:4` with `--qos=h200_8`
2. **NVL jobs use 2 GPUs**: `--gres=gpu:2` (no QOS needed)
3. **Always specify partition when checking jobs**: `squeue -p h200 -u $USER` or `squeue -p nvl -u $USER`. Use `sqme` alias to check both.
4. **Use `myjobs` to check current running jobs**.

## Project Overview

Context refinement and engineering research project combining vLLM inference with SLIME RL training for iterative context refinement experiments.

## Key Commands

```bash
# Run context refinement (data parallel across GPUs)
python inference/context_refinement_dp.py --dataset imobench --num_tokens 4096 --rounds 1 --accumulate --output_file output.jsonl

# Verify solutions
python inference/verify_solutions.py <input_file.jsonl>

# Generate results summary
python generate_results.py

# Check SLURM job logs (stdout + stderr)
checklogs <job_id>
```

## Environment

- **Outputs**: `/scratch/dkhasha1/tli104/outputs/`
- **SLURM Scripts**: `/scratch/dkhasha1/tli104/slurm_scripts/`
- **SLURM Logs**: `/scratch/dkhasha1/tli104/slurm_logs/`
- **Conda Environment**: `/scratch/dkhasha1/tli104/vllm`
- **Model Cache**: `/scratch/dkhasha1/tli104/hf_model_cache`

## SLURM

### SBATCH Headers (H200) - 4 GPUs
```bash
#!/bin/bash
#SBATCH --job-name=<job_name>
#SBATCH --partition=h200
#SBATCH --qos=h200_8
#SBATCH --gres=gpu:4
#SBATCH --mem=256GB
#SBATCH --cpus-per-task=40
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/dkhasha1/tli104/slurm_logs/<job_name>_%j.out
#SBATCH --error=/scratch/dkhasha1/tli104/slurm_logs/<job_name>_%j.err
```

### SBATCH Headers (NVL) - 2 GPUs
```bash
#!/bin/bash
#SBATCH --job-name=<job_name>
#SBATCH --partition=nvl
#SBATCH --gres=gpu:2
#SBATCH --mem=256GB
#SBATCH --cpus-per-task=40
#SBATCH --time=72:00:00
#SBATCH --output=/scratch/dkhasha1/tli104/slurm_logs/<job_name>_%j.out
#SBATCH --error=/scratch/dkhasha1/tli104/slurm_logs/<job_name>_%j.err
```

### Environment Setup in Scripts
```bash
export SCRATCH_DIR=/scratch/dkhasha1/tli104
export HF_HOME=${SCRATCH_DIR}/hf_model_cache
export HF_DATASETS_CACHE=${SCRATCH_DIR}/hf_datasets_cache

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/dkhasha1/tli104/vllm
```

### Submit Scripts
```bash
for script in /scratch/dkhasha1/tli104/slurm_scripts/<dir>/*.sh; do sbatch "$script"; done
```

## Output File Naming

Format: `<method>_<model>_<benchmark>_<params>.jsonl`

Examples:
- `baseline_qwen3-30b-a3b_imobench_t32768_n16_temp0.7_topp0.9.jsonl`
- `refinement_qwen3-8b_hmmt_t16384_r3_n16_temp0.7.jsonl`

Components: method (`baseline`, `refinement`, `accumulate`), model, benchmark (`imobench`, `hmmt`, `answerbench`), params (`t<tokens>`, `r<rounds>`, `n<samples>`, `temp<temperature>`, `topp<top_p>`)

## SLURM Experiment Tracking

After each `sbatch` submission, log in `slurm_experiments.md`:

| Job ID | Job Name | Partition | GPUs | Description |
|--------|----------|-----------|------|-------------|
| 850474 | baseline_qwen3_30b | nvl | 2 | Qwen3-30B baseline, imobench, t32768, n16 |
