#!/bin/bash
# ==========================================================================
# Batch Questions Experiment — 10 HMMT Nov 2025 questions in a single prompt
#
# Models:  google/gemini-3-flash-preview, minimax/minimax-m2.5,
#          moonshotai/kimi-k2.5
#
# Each job sends all 10 questions in one prompt with delimiters.
# Runs on CPU partition (API inference, no GPU needed).
# Max generation tokens: 320000
#
# Usage:
#   bash context_rot_prelim/launch_batch_questions.sh          # create scripts only
#   bash context_rot_prelim/launch_batch_questions.sh --submit  # create + sbatch
# ==========================================================================
set -euo pipefail

SUBMIT=false
[[ "${1:-}" == "--submit" ]] && SUBMIT=true

SCRATCH=/scratch/dkhasha1/tli104
OUTPUT_DIR=${SCRATCH}/outputs/context_rot
SLURM_DIR=${SCRATCH}/slurm_scripts/context_rot
LOG_DIR=${SCRATCH}/slurm_logs
CODE_DIR=/weka/home/tli104/context_engineering

HMMT_NOV_FILE=${SCRATCH}/datasets/hmmt_nov_2025/hmmt_nov_2025.jsonl

NUM_QUESTIONS=10
MAX_TOKENS=320000
TEMP=0.9
SEEDS="42 123 456"

mkdir -p "$OUTPUT_DIR" "$SLURM_DIR" "$LOG_DIR"

# ---------- model definitions ----------
# Fields: short_name | openrouter_model_id
MODELS=(
  "gemini-3-flash|google/gemini-3-flash-preview"
  "minimax-m2.5|minimax/minimax-m2.5"
  "kimi-k2.5|moonshotai/kimi-k2.5"
  "glm-5|z-ai/glm-5"
)

for model_cfg in "${MODELS[@]}"; do
  IFS='|' read -r short model_id <<< "$model_cfg"

  job="cr_batch_${short}_hmmt_nov_n${NUM_QUESTIONS}_t${MAX_TOKENS}"
  out_file="${OUTPUT_DIR}/batch_${short}_hmmt_nov_n${NUM_QUESTIONS}_t${MAX_TOKENS}_temp${TEMP}.jsonl"
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

python -m context_rot_prelim.batch_questions_inference \\
    --model ${model_id} \\
    --input_file ${HMMT_NOV_FILE} \\
    --num_questions ${NUM_QUESTIONS} \\
    --seeds ${SEEDS} \\
    --max_tokens ${MAX_TOKENS} \\
    --temperature ${TEMP} \\
    --output_file ${out_file}
SLURM

  chmod +x "$script"
  echo "Created $script"

  if $SUBMIT; then
    sbatch "$script"
  fi
done

echo ""
echo "Scripts in: ${SLURM_DIR}/"
if ! $SUBMIT; then
  echo "To submit all:  for f in ${SLURM_DIR}/cr_batch_*.sh; do sbatch \$f; done"
fi
