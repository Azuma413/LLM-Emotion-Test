#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/rl_grpo.yaml}"
shift || true

GPUS="${GPUS:-}"

if [[ -n "$GPUS" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPUS"
fi

uv run llm-emotion-test \
  train-rl \
  --config "$CONFIG" \
  "$@"
