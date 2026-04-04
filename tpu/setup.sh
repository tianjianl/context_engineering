#!/bin/bash
# Setup script for vLLM inference on TPU v6e
# Run this on the TPU VM after SSH-ing in:
#   gcloud compute tpus tpu-vm ssh tli104-v6e-8 --project=tianjian-project --zone=us-east1-d
set -e

echo "=== Installing vLLM for TPU (from source) ==="

# Clean up any existing vllm installs
pip uninstall -y vllm vllm-tpu 2>/dev/null || true

# Install uv for faster installs
pip install uv

# Clone tpu-inference to get the pinned compatible vLLM commit
if [ ! -d ~/tpu-inference ]; then
    git clone https://github.com/vllm-project/tpu-inference.git ~/tpu-inference
else
    cd ~/tpu-inference && git pull
fi
VLLM_COMMIT_HASH="$(cat ~/tpu-inference/.buildkite/vllm_lkg.version)"
echo "Using vLLM commit: $VLLM_COMMIT_HASH"

# Clone vLLM at the compatible commit
if [ ! -d ~/vllm ]; then
    git clone https://github.com/vllm-project/vllm.git ~/vllm
fi
cd ~/vllm
git fetch origin
git checkout "$VLLM_COMMIT_HASH"

# Install vLLM with TPU target
uv pip install --system -r requirements/tpu.txt
VLLM_TARGET_DEVICE="tpu" uv pip install --system -e .

# Install tpu-inference plugin
cd ~/tpu-inference
uv pip install --system -e .

# Install latest transformers for Qwen3.5 support
uv pip install --system --upgrade transformers accelerate math_verify

echo "=== Cloning repo ==="
if [ ! -d ~/context_engineering ]; then
    git clone https://github.com/tianjianl/context_engineering.git ~/context_engineering
else
    cd ~/context_engineering && git pull
fi

echo "=== Setup complete ==="
echo "Verify TPU detection: python -c \"from vllm.platforms import current_platform; print(current_platform)\""
echo "Run inference with: cd ~/context_engineering && bash tpu/run.sh"
