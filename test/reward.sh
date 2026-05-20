#!/usr/bin/env bash
set -x

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR=${LOG_DIR:-"$SCRIPT_DIR/log"}
REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-/path/to/judge-model}
VLLM_PORT=${VLLM_PORT:-8355}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-4}

mkdir -p "$LOG_DIR"

nohup python3 -u -m vllm.entrypoints.openai.api_server \
    --model "$REWARD_MODEL_PATH" \
    --trust-remote-code \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --dtype bfloat16 \
    --port "$VLLM_PORT" \
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.8}" \
    --uvicorn-log-level "error" \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --disable-custom-all-reduce \
    > "$LOG_DIR/vllm_server.log" 2>&1 &
