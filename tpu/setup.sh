#!/bin/bash
# Setup script for vLLM inference on TPU v6e
# Run this on the TPU VM after SSH-ing in:
#   gcloud compute tpus tpu-vm ssh tli104-v6e-8 --project=tianjian-project --zone=us-east1-d
set -e

echo "=== Installing vLLM for TPU ==="

# The v2-alpha-tpuv6e runtime has PyTorch/XLA pre-installed.
# Install vLLM with TPU support.
pip install vllm

# Install additional dependencies
pip install transformers accelerate math_verify

echo "=== Cloning repo ==="
if [ ! -d ~/context_engineering ]; then
    git clone https://github.com/tianjianl/context_engineering.git ~/context_engineering
else
    cd ~/context_engineering && git pull
fi

echo "=== Setup complete ==="
echo "Run inference with: cd ~/context_engineering && bash tpu/run.sh"
