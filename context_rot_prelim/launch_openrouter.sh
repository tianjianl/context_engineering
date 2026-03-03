#!/bin/bash
# ==========================================================================
# Context Rot Experiment — OpenRouter API models on CPU SLURM partition
#
# Models:  minimax/minimax-m2.5, moonshotai/kimi-k2.5, z-ai/glm-5,
#          deepseek/deepseek-v3.2, google/gemini-3-flash-preview,
#          qwen/qwen3.5-397b-a17b
# Datasets: HMMT (60 problems), IMOBench (200 problems)
#
# Each job runs mode=both (baseline + sequential with 5 seeds).
# Jobs run on CPU partition (API inference, no GPU needed).
#
# Usage:
#   bash context_rot_prelim/launch_openrouter.sh          # create scripts only
#   bash context_rot_prelim/launch_openrouter.sh --submit  # create + sbatch
# ==========================================================================
set -euo pipefail

SUBMIT=false
[[ "${1:-}" == "--submit" ]] && SUBMIT=true

SCRATCH=/scratch/dkhasha1/tli104
OUTPUT_DIR=${SCRATCH}/outputs/context_rot
SLURM_DIR=${SCRATCH}/slurm_scripts/context_rot
LOG_DIR=${SCRATCH}/slurm_logs
CODE_DIR=/weka/home/tli104/context_engineering

HMMT_FILE=${SCRATCH}/datasets/hmmt_2025_combined/hmmt_2025_combined.jsonl

mkdir -p "$OUTPUT_DIR" "$SLURM_DIR" "$LOG_DIR"

# ---------- model definitions ----------
# Fields: short_name | openrouter_model_id | concurrency
MODELS=(
  "minimax-m2.5|minimax/minimax-m2.5|10"
  "kimi-k2.5|moonshotai/kimi-k2.5|10"
  "glm-5|z-ai/glm-5|10"
  "deepseek-v3.2|deepseek/deepseek-v3.2|10"
  "gemini-3-flash|google/gemini-3-flash-preview|10"
  "qwen3.5-397b-a17b|qwen/qwen3.5-397b-a17b|10"
)

# ---------- dataset definitions ----------
# Fields: name | dataset_flag | extra_args
DATASETS=(
  "hmmt|hmmt|--input_file ${HMMT_FILE}"
  "imobench|imobench|--max_problems 200"
)

for model_cfg in "${MODELS[@]}"; do
  IFS='|' read -r short model_id concurrency <<< "$model_cfg"

  for ds_cfg in "${DATASETS[@]}"; do
    IFS='|' read -r ds_name ds_flag ds_extra <<< "$ds_cfg"

    job="cr_or_${short}_${ds_name}"
    out_file="${OUTPUT_DIR}/${short}_${ds_name}.jsonl"
    script="${SLURM_DIR}/${job}.sh"

    cat > "$script" << SLURM
#!/bin/bash
#SBATCH --job-name=${job}
#SBATCH --partition=cpu
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=${LOG_DIR}/${job}_%j.out
#SBATCH --error=${LOG_DIR}/${job}_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
source ~/.bashrc

cd ${CODE_DIR}

python -m context_rot_prelim.openrouter_inference \\
    --model ${model_id} \\
    --dataset ${ds_flag} \\
    ${ds_extra} \\
    --mode both \\
    --turns_per_conversation 10 \\
    --seeds 42 123 456 789 1011 \\
    --max_tokens 16384 \\
    --temperature 0.0 \\
    --concurrency ${concurrency} \\
    --output_file ${out_file}
SLURM

    chmod +x "$script"
    echo "Created $script"

    if $SUBMIT; then
      sbatch "$script"
    fi
  done
done

echo ""
echo "Scripts in: ${SLURM_DIR}/"
if ! $SUBMIT; then
  echo "To submit all:  for f in ${SLURM_DIR}/cr_or_*.sh; do sbatch \$f; done"
fi
