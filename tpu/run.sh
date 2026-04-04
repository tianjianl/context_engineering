#!/bin/bash
# Run Qwen3.5-4B inference on IMOBench v2 using TPU v6e-8
set -e

cd ~/context_engineering

python -m tpu.inference \
    --dataset imobench_v2 \
    --model Qwen/Qwen3.5-4B \
    --num_tokens 16384 \
    --num_samples 1 \
    --temperature 0.7 \
    --top_p 0.9 \
    --tensor_parallel_size 8 \
    --output_file results/tpu_qwen3.5_4b_imobench_v2.jsonl
