#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/sft.yaml}"
shift || true

GPUS="${GPUS:-}"
NPROC="${NPROC:-}"

if [[ -n "$GPUS" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPUS"
  if [[ -z "$NPROC" ]]; then
    IFS=',' read -r -a GPU_LIST <<< "$GPUS"
    NPROC="${#GPU_LIST[@]}"
  fi
fi

NPROC="${NPROC:-1}"

uv run torchrun \
  --standalone \
  --nproc_per_node="$NPROC" \
  -m llm_emotion_test.main \
  train-sft \
  --config "$CONFIG" \
  "$@"
