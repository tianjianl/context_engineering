#!/bin/bash
# Launch RC SFT batch 3 pipeline: data prep → rollouts (A100 × 8) → filter
# 16K problems, n=16 per problem, rejection sampling via math-verify
#
# Usage: bash scripts/launch_rc_sft_b3.sh

set -euo pipefail

SCRATCH=/scratch/dkhasha1/tli104
WORK=/weka/home/tli104/context_engineering
LOGS=${SCRATCH}/slurm_logs
OUTPUT_DIR=${SCRATCH}/outputs/rc_sft_rollouts_b3
DATASET=${SCRATCH}/datasets/polaris_rc_sft_batch3_16000.json
EXCLUDE=${SCRATCH}/datasets/polaris_rc_sft_exclude_b1b2.json
MODEL=${SCRATCH}/models/Qwen3-4B-Instruct-2507

NUM_SHARDS=8
SHARD_SIZE=2000
NUM_PROBLEMS=16000
N_SAMPLES=16

mkdir -p ${LOGS} ${OUTPUT_DIR}

echo "=== RC SFT Batch 3: ${NUM_PROBLEMS} problems × n=${N_SAMPLES} ==="
echo ""

# ─── Write SLURM scripts ──────────────────────────────────────────────────

# Step 1: Data Prep
cat > /tmp/rc_sft_prep_b3.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=rc_sft_prep_b3
#SBATCH --partition=cpu
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00

export SCRATCH_DIR=/scratch/dkhasha1/tli104
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ${SCRATCH_DIR}/vllm
cd /weka/home/tli104/context_engineering

# Merge batch 1+2 into exclude file
python3 -c "
import json
b1 = json.load(open('${SCRATCH_DIR}/datasets/polaris_rc_sft_2000.json'))
b2 = json.load(open('${SCRATCH_DIR}/datasets/polaris_rc_sft_batch2_2000.json'))
combined = b1 + b2
print(f'Combined exclude list: {len(combined)} problems')
json.dump(combined, open('${SCRATCH_DIR}/datasets/polaris_rc_sft_exclude_b1b2.json', 'w'))
"

python scripts/prepare_rc_sft_rollouts.py \
    --input ${SCRATCH_DIR}/datasets/polaris_filtered_removed_all_correct/polaris_filtered_removed_all_correct.jsonl \
    --output ${SCRATCH_DIR}/datasets/polaris_rc_sft_batch3_16000.json \
    --num_problems 16000 \
    --seed 456 \
    --exclude_json ${SCRATCH_DIR}/datasets/polaris_rc_sft_exclude_b1b2.json
EOF

# Step 2: Rollout (parameterized by SHARD env var)
cat > /tmp/rc_sft_rollout_b3.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=rc_sft_rollout_b3
#SBATCH --partition=a100
#SBATCH --gres=gpu:8
#SBATCH --mem=256GB
#SBATCH --cpus-per-task=40
#SBATCH --time=48:00:00

SHARD=${SHARD:-0}
SHARD_SIZE=2000
START_IDX=$((SHARD * SHARD_SIZE))
END_IDX=$(((SHARD + 1) * SHARD_SIZE))

export SCRATCH_DIR=/scratch/dkhasha1/tli104
export HF_HOME=${SCRATCH_DIR}/hf_model_cache
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ${SCRATCH_DIR}/vllm
cd /weka/home/tli104/context_engineering

MODEL=${SCRATCH_DIR}/models/Qwen3-4B-Instruct-2507
DATASET=${SCRATCH_DIR}/datasets/polaris_rc_sft_batch3_16000.json
OUTPUT_DIR=${SCRATCH_DIR}/outputs/rc_sft_rollouts_b3
mkdir -p ${OUTPUT_DIR}

python rc/inference/inference/generate_complete.py \
    --dataset_path ${DATASET} \
    --reasoning_prompt_path rc/inference/prompts/reasoning_prompt.txt \
    --summarization_prompt_path rc/inference/prompts/summarization_prompt.txt \
    --output_path ${OUTPUT_DIR}/shard_${SHARD}.json \
    --model_path ${MODEL} \
    --tp_size 1 \
    --num_gpus 8 \
    --n 16 \
    --max_steps 4 \
    --max_thinking_tokens 16384 \
    --max_summarization_tokens 2048 \
    --temperature 0.7 \
    --top_p 0.95 \
    --gpu_memory_utilization 0.85 \
    --model_class qwen \
    --use_think_tags \
    --start_index ${START_IDX} \
    --end_index ${END_IDX} \
    --seed $((456 + SHARD))
EOF

# Step 3: Filter
cat > /tmp/rc_sft_filter_b3.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=rc_sft_filter_b3
#SBATCH --partition=cpu
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00

export SCRATCH_DIR=/scratch/dkhasha1/tli104
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ${SCRATCH_DIR}/vllm
cd /weka/home/tli104/context_engineering

python scripts/filter_rc_rollouts.py \
    --rollout_dirs ${SCRATCH_DIR}/outputs/rc_sft_rollouts_b3 \
    --output_dir ${SCRATCH_DIR}/datasets/rc_sft_b3_filtered \
    --no_tool_ratio 0.5 \
    --max_tokens 32768 \
    --tokenizer_path ${SCRATCH_DIR}/models/Qwen3-4B-Instruct-2507 \
    --num_workers 8 \
    --best_per_problem \
    --seed 42
EOF

# ─── Submit pipeline ──────────────────────────────────────────────────────

# Step 1: Data prep
PREP_JOB=$(sbatch --parsable \
    --output=${LOGS}/rc_sft_prep_b3_%j.out \
    --error=${LOGS}/rc_sft_prep_b3_%j.err \
    /tmp/rc_sft_prep_b3.sh)
echo "Step 1: Data prep → job ${PREP_JOB}"

# Step 2: Rollouts (8 A100 shards, depend on prep)
ROLLOUT_JOBS=""
for SHARD in $(seq 0 $((NUM_SHARDS - 1))); do
    START=$((SHARD * SHARD_SIZE))
    END=$(((SHARD + 1) * SHARD_SIZE))

    JOB=$(sbatch --parsable \
        --dependency=afterok:${PREP_JOB} \
        --export=SHARD=${SHARD} \
        --output=${LOGS}/rc_sft_rollout_b3_s${SHARD}_%j.out \
        --error=${LOGS}/rc_sft_rollout_b3_s${SHARD}_%j.err \
        /tmp/rc_sft_rollout_b3.sh)
    ROLLOUT_JOBS="${ROLLOUT_JOBS}:${JOB}"
    echo "Step 2: Rollout shard ${SHARD} (${START}-${END}) → job ${JOB}"
done

# Step 3: Filter (depends on all rollouts)
FILTER_JOB=$(sbatch --parsable \
    --dependency=afterok${ROLLOUT_JOBS} \
    --output=${LOGS}/rc_sft_filter_b3_%j.out \
    --error=${LOGS}/rc_sft_filter_b3_%j.err \
    /tmp/rc_sft_filter_b3.sh)
echo "Step 3: Filter + build SFT → job ${FILTER_JOB}"

echo ""
echo "=== Pipeline Summary ==="
echo "  Data prep:  ${PREP_JOB}"
echo "  Rollouts:   ${ROLLOUT_JOBS#:} (8 shards on A100, 8 GPUs each)"
echo "  Filter:     ${FILTER_JOB}"
echo ""
echo "  Input:      ${NUM_PROBLEMS} problems × n=${N_SAMPLES} = $((NUM_PROBLEMS * N_SAMPLES)) trajectories"
echo "  Output:     ${SCRATCH}/datasets/rc_sft_b3_filtered/"
echo ""
echo "Monitor: squeue -u \$USER | grep rc_sft"
