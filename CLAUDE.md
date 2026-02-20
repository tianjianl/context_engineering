# CLAUDE.md

## CRITICAL RULES

1. **H200**: `--partition=h200 --qos=h200_8 --gres=gpu:4`
2. **NVL**: `--partition=nvl --gres=gpu:2` (no QOS)
3. **CPU (verification only)**: `--partition=cpu --mem=64GB --cpus-per-task=8`
4. Check jobs: `myjobs` or `sqme`

## Paths

- **Scratch**: `/scratch/dkhasha1/tli104/`
- **Outputs**: `/scratch/dkhasha1/tli104/outputs/`
- **SLURM Scripts**: `/scratch/dkhasha1/tli104/slurm_scripts/`
- **SLURM Logs**: `/scratch/dkhasha1/tli104/slurm_logs/`
- **Datasets**: `/scratch/dkhasha1/tli104/datasets/`
- **Model Cache**: `/scratch/dkhasha1/tli104/hf_model_cache`
- **Conda Env**: `/scratch/dkhasha1/tli104/vllm` (default)
- **Conda Env (Qwen3-30B-A3B)**: `/scratch/dkhasha1/tli104/vllm_0_8_4` — always use this env for Qwen3-30B-A3B (MoE model)

## Inference Scripts (in `inference/`)

| Script | Purpose |
|--------|---------|
| `baseline_vllm.py` | Single-pass generation |
| `context_refinement_dp.py` | Refinement: `--accumulate`, `--rc`, `--rc_verify`, or `--rc_user` |
| `tool_refinement.py` | Tool-calling refinement (`--compact_context` for prompt+summary only) |
| `verify_solutions.py` | Verify answers against ground truth |
| `verify_by_round.py` | Per-round accuracy breakdown |
| `grade_proofs_gemini.py` | Grade proofs via Gemini API |

## Datasets

| Name | Flag / Input File | Problems |
|------|-------------------|----------|
| IMOBench | `--dataset imobench` | ~343 |
| HMMT Combined | `--dataset hmmt --input_file .../hmmt_2025_combined/hmmt_2025_combined.jsonl` | ~88 |
| ProofBench | `--dataset hmmt --input_file .../proofbench/proofbench.jsonl` | 60 |
| ProofBench HF | `--dataset hmmt --input_file .../proofbench_hf/proofbench_hf.jsonl` | 435 |

## SLURM Script Template

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

## Refinement Methods

| Method | Flag | Description |
|--------|------|-------------|
| **RC** | `--rc` | Summarize previous attempt, inject as assistant prefix (model continues from summary) |
| **RC Verify** | `--rc_verify` | Like RC but summary critically evaluates the solution and suggests improvements |
| **RC User** | `--rc_user` | Like RC but puts summary in user prompt instead of assistant prefix (matches RC reimpl style) |
| **RC Reimpl** | `rc/inference/generate_complete.py` | Original RC paper reimplementation (separate codebase in `rc/`). Summarize-then-regenerate loop with summary in user prompt |

RC User is our reimplementation of the RC reimpl approach within `context_refinement_dp.py`. RC Reimpl is the original external codebase.

## Output File Naming

Format: `<method>_<model>_<benchmark>_<params>.jsonl`

Methods: `baseline`, `refinement`, `rc`, `rc_user`, `rc_reimpl`, `tool_refinement`, `compact_tool_refinement`
Params: `t<tokens>`, `r<rounds>`, `n<samples>`, `temp<T>`, `topp<P>`

## Experiment Tracking

Log every `sbatch` submission in `slurm_experiments.md`.
