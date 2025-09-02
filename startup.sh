#!/bin/bash
set -e

# Download model weights
python3 /app/tools/download_model.py

# Set up environment for DotsOCR model registration
export hf_model_path=/app/weights/DotsOCR
export PYTHONPATH=/app/weights:$PYTHONPATH

# Register DotsOCR model with vLLM
sed -i '/^from vllm\.entrypoints\.cli\.main import main$/a\from DotsOCR import modeling_dots_ocr_vllm' $(which vllm)

# Start vLLM server with DotsOCR
HF_TOKEN=$(cat /secrets/hf_access_token) vllm serve ${hf_model_path} \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.95 \
  --chat-template-content-format string \
  --served-model-name DotsOCR \
  --trust-remote-code \
  --enable-prefix-caching \
  --enable-chunked-prefill