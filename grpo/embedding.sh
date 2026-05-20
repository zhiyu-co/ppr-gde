#!/usr/bin/env bash
set -x

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
export EMBEDDING_MODEL_PATH=${EMBEDDING_MODEL_PATH:-/path/to/embedding-model}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR=${LOG_DIR:-"$SCRIPT_DIR/log"}
EMBEDDING_HOST=${EMBEDDING_HOST:-0.0.0.0}
EMBEDDING_PORT=${EMBEDDING_PORT:-8356}

mkdir -p "$LOG_DIR"

nohup uvicorn verl.utils.embeddings.api_server:app \
    --host "$EMBEDDING_HOST" \
    --port "$EMBEDDING_PORT" \
    --log-level info \
    > "$LOG_DIR/embedding_server.log" 2>&1 &
