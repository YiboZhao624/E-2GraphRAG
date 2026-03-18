#!/usr/bin/env bash

set -euo pipefail

# Use four GPUs; change this if you need a different set.
export CUDA_VISIBLE_DEVICES=4,5

MODEL_BASE="/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3-32B"
REF_HASH="$(cat "${MODEL_BASE}/refs/main")"
MODEL_PATH="${MODEL_BASE}/snapshots/${REF_HASH}"

vllm serve "${MODEL_PATH}" \
  --tensor-parallel-size 2 \
  --host 127.0.0.1 \
  --port 8001 \
  --served-model-name qwen3-32b \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.9