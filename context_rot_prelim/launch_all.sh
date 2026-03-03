#!/bin/bash
# ==========================================================================
# Context Rot Experiment — Generate & submit SLURM jobs
#
# Models:  Qwen3-4B-Instruct-2507,  Qwen3-4B-Thinking-2507
#          Qwen3-30B-A3B-Instruct-2507, Qwen3-30B-A3B-Thinking-2507
# Datasets: HMMT (60 problems), IMOBench (200 problems)
#
# Each job runs mode=both (baseline + sequential with 5 seeds).
# TP and DP are auto-detected from model size + available GPUs.
#
# Usage:
#   bash context_rot_prelim/launch_all.sh          # create scripts only
#   bash context_rot_prelim/launch_all.sh --submit  # create + sbatch
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
# Fields: short_name | HF_model | conda_env | max_tokens | extra_flags | partition | qos | gpus
# 4B models → NVL (2 GPUs, TP=1, DP=2)
# 30B models → H200 (4 GPUs, TP=2, DP=2)
MODELS=(
  "qwen3-4b-instruct|Qwen/Qwen3-4B-Instruct-2507|${SCRATCH}/vllm|16384||nvl||2"
  "qwen3-4b-thinking|Qwen/Qwen3-4B-Thinking-2507|${SCRATCH}/vllm|16384||nvl||2"
  "qwen3-30b-a3b-instruct|Qwen/Qwen3-30B-A3B-Instruct-2507|${SCRATCH}/vllm_0_8_4|16384||h200|h200_8|4"
  "qwen3-30b-a3b-thinking|Qwen/Qwen3-30B-A3B-Thinking-2507|${SCRATCH}/vllm_0_8_4|16384||h200|h200_8|4"
)

# ---------- dataset definitions ----------
# Fields: name | dataset_flag | extra_args
DATASETS=(
  "hmmt|hmmt|--input_file ${HMMT_FILE}"
  "imobench|imobench|--max_problems 200"
)

for model_cfg in "${MODELS[@]}"; do
  IFS='|' read -r short hf_model conda max_tok extra_flags partition qos gpus <<< "$model_cfg"

  for ds_cfg in "${DATASETS[@]}"; do
    IFS='|' read -r ds_name ds_flag ds_extra <<< "$ds_cfg"

    job="cr_${short}_${ds_name}"
    out_file="${OUTPUT_DIR}/${short}_${ds_name}.jsonl"
    script="${SLURM_DIR}/${job}.sh"

    # Build QOS line (NVL has no QOS)
    qos_line=""
    [[ -n "$qos" ]] && qos_line="#SBATCH --qos=${qos}"

    cat > "$script" << SLURM
#!/bin/bash
#SBATCH --job-name=${job}
#SBATCH --partition=${partition}
${qos_line}
#SBATCH --gres=gpu:${gpus}
#SBATCH --mem=256GB
#SBATCH --cpus-per-task=40
#SBATCH --time=48:00:00
#SBATCH --output=${LOG_DIR}/${job}_%j.out
#SBATCH --error=${LOG_DIR}/${job}_%j.err

export SCRATCH_DIR=${SCRATCH}
export HF_HOME=${SCRATCH}/hf_model_cache
export HF_DATASETS_CACHE=${SCRATCH}/hf_datasets_cache
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ${conda}

cd ${CODE_DIR}

python -m context_rot_prelim.run_experiment \\
    --model ${hf_model} \\
    --dataset ${ds_flag} \\
    ${ds_extra} \\
    --mode both \\
    --turns_per_conversation 10 \\
    --seeds 42 123 456 789 1011 \\
    --max_tokens ${max_tok} \\
    --temperature 0.0 \\
    ${extra_flags} \\
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
  echo "To submit all:  for f in ${SLURM_DIR}/cr_*.sh; do sbatch \$f; done"
fi
