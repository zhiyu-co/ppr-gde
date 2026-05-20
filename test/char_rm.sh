#!/usr/bin/env bash
set -x

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export CHAR_RM_MODEL_PATH=${CHAR_RM_MODEL_PATH:-/path/to/BaichuanCharRM}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR=${LOG_DIR:-"$SCRIPT_DIR/log"}
CHAR_RM_HOST=${CHAR_RM_HOST:-0.0.0.0}
CHAR_RM_PORT=${CHAR_RM_PORT:-8001}

mkdir -p "$LOG_DIR"

nohup uvicorn verl.utils.char_rm.api_server:app \
    --host "$CHAR_RM_HOST" \
    --port "$CHAR_RM_PORT" \
    --log-level info \
    > "$LOG_DIR/char_rm_server.log" 2>&1 &
