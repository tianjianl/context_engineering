#!/bin/bash
# Setup script for vLLM inference on TPU v6e
# Run this on the TPU VM after SSH-ing in:
#   gcloud compute tpus tpu-vm ssh tli104-v6e-8 --project=tianjian-project --zone=us-east1-d
set -e

echo "=== Installing vLLM for TPU ==="

# Clean up any existing vllm installs
pip uninstall -y vllm vllm-tpu 2>/dev/null || true

# Install vllm-tpu (the TPU-specific package)
pip install vllm-tpu

# Do NOT upgrade transformers — newer versions break vllm-tpu internals
pip install math_verify

echo "=== Cloning repo ==="
if [ ! -d ~/context_engineering ]; then
    git clone https://github.com/tianjianl/context_engineering.git ~/context_engineering
else
    cd ~/context_engineering && git pull
fi

echo "=== Setup complete ==="
echo "Verify TPU detection: python -c \"from vllm.platforms import current_platform; print(current_platform)\""
echo "Run inference with: cd ~/context_engineering && bash tpu/run.sh"
