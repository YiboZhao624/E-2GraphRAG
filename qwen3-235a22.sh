#!/usr/bin/env bash

set -euo pipefail

# Use four GPUs; change this if you need a different set.
export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_BASE="/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3-235B-A22B"
REF_HASH="$(cat "${MODEL_BASE}/refs/main")"
MODEL_PATH="${MODEL_BASE}/snapshots/${REF_HASH}"

vllm serve "${MODEL_PATH}" \
  --tensor-parallel-size 4 \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name qwen3-235b-a22b \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.9