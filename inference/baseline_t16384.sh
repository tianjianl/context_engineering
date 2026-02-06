# Configuration
NUM_TOKENS=16384
MODEL="zai-org/GLM-4.7-Flash"

# Environment setup
export SCRATCH_DIR=/scratch/dkhasha1/tli104
export HF_HOME=${SCRATCH_DIR}/hf_model_cache
export HF_DATASETS_CACHE=${SCRATCH_DIR}/hf_datasets_cache
# export HF_TOKEN=<your_token_here>

# Paths
OUTPUT_DIR=${SCRATCH_DIR}/outputs/imobench_glm47_flash
mkdir -p ${OUTPUT_DIR}
mkdir -p ${SCRATCH_DIR}/slurm_logs

OUTPUT_FILE=${OUTPUT_DIR}/baseline_t${NUM_TOKENS}_temp0.7_topp0.9.jsonl 

echo "========================================"
echo "IMOBench Baseline - GLM-4.7-Flash"
echo "========================================"
echo "Model: $MODEL"
echo "Num Tokens: $NUM_TOKENS"
echo "Mode: Baseline (No Refinement)"
echo "Output: $OUTPUT_FILE"
echo "========================================"

cd /weka/home/tli104/context_engineering/inference

python baseline_vllm.py \
    --dataset imobench \
    --output_file ${OUTPUT_FILE} \
    --model ${MODEL} \
    --num_tokens ${NUM_TOKENS} \
    --temperature 0.7 \
    --top_p 0.9

echo "Done! Output saved to: $OUTPUT_FILE"
